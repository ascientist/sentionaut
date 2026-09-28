"""Canonical form: adaptive tracking over a population of random linear dynamical systems.

The neurostimulation problem, stripped to its mathematical skeleton::

    θ_p ~ P_Θ                                   patient, sampled once
    φ_{t+1} = ρ φ_t + η_t                       slow drift within the patient
    x_{t+1} = A_θ x_t + (B_θ + φ_t) u_t + w_t   fast neural state, controlled
    o_t     = C x_t + v_t                       what the implant measures
    z_t     = D x_t                             percept
    z*      = E(r),  r ~ P_R                    target, independent of the patient

    J(π) = E_{θ, φ, r, w, v} [ Σ_t ‖D x_t − z*‖² + λ ‖u_t‖² ],   u_t ∈ U_safe

A policy sees only ``(z*, o_{≤t}, u_{<t})``. The population mean dynamics
``(Ā, B̄)`` are fixed by a seed; ``σ_A``, ``σ_B`` set the inter-patient spread.

Controllers, from least to most informed:

- ``ZeroController``: no stimulation.
- ``PopulationController``: LQG designed for the mean patient ``(Ā, B̄)``,
  one controller for everybody.
- ``AdaptiveController``: recursive least squares on an ARX model of
  ``o`` (with forgetting to follow drift), then certainty-equivalent LQ
  tracking on that model. It knows nothing about the patient.
- ``OracleController`` (privileged): LQG with the true ``(A_θ, B_θ + φ_t)``.
  For a known linear-Gaussian system this is optimal (separation principle),
  so it is the floor that every other controller is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import solve_discrete_are


@dataclass(frozen=True)
class PopulationConfig:
    n: int = 6  # neural state dimension
    m: int = 4  # electrodes
    dz: int = 3  # percept dimension (z = D x = first dz latents)
    horizon: int = 400
    rho_A: float = 0.9  # spectral radius of the mean dynamics Ā (damped oscillations)
    sigma_A: float = 0.05  # inter-patient spread of A
    b_scale: float = 0.4  # entry std of the mean recruitment B̄
    sigma_B: float = 0.5  # inter-patient spread of B, relative to b_scale
    drift: float = 0.0  # std of the per-step drift increment η_t (on B)
    drift_decay: float = 0.999  # ρ in φ_{t+1} = ρ φ_t + η_t
    process_noise: float = 0.05
    obs_noise: float = 0.05
    observe: str = "full"  # "full": o = x + v ; "percept": o = D x + v (partial)
    lam_u: float = 0.1  # control penalty; smaller saturates u_max and chatters
    # Nonlinearity (regimes C, D). x' = A s(x) + B_t g(u) + w
    recruit: str = "linear"  # "linear": g(u)=u | "tanh": saturating | "even": u^2 - 1/4
    saturation: float = 0.0  # s(x) = sat*tanh(x/sat): firing-rate saturation; 0 = linear
    u_max: float = 2.0
    n_targets: int = 100  # the common "image" set, shared by all patients
    target_std: float = 0.5
    seed: int = 0  # population seed: Ā, B̄ and the target set


@dataclass
class Patient:
    A: np.ndarray
    B: np.ndarray


def recruit(cfg: PopulationConfig, u):
    """Electrode recruitment ``g(u)``, elementwise.

    ``tanh`` saturates (smooth nonlinearity). ``even`` is polarity-insensitive
    (``g(u) = g(-u)``), as for charge-balanced biphasic pulses, with a tonic
    offset (-1/4) so drive can be negative. Every electrode then has two equally good
    settings ``±u``: ``2^m`` optimal stimulation patterns, a genuinely
    multimodal control problem.
    """
    if cfg.recruit == "linear":
        return u
    if cfg.recruit == "tanh":
        return np.tanh(u)
    if cfg.recruit == "even":
        return u**2 - 0.25
    raise ValueError(cfg.recruit)


def recruit_grad(cfg: PopulationConfig, u):
    if cfg.recruit == "linear":
        return np.ones_like(u)
    if cfg.recruit == "tanh":
        return 1 - np.tanh(u) ** 2
    return 2 * u


def rate(cfg: PopulationConfig, x):
    """Firing-rate saturation ``s(x) = sat·tanh(x / sat)`` (identity when ``sat = 0``)."""
    return cfg.saturation * np.tanh(x / cfg.saturation) if cfg.saturation > 0 else x


def step_mean(cfg: PopulationConfig, A, B_t, x, u):
    """Noise-free ``f(x, u) = A s(x) + B_t g(u)``; broadcasts over leading batch dims."""
    return rate(cfg, x) @ A.T + recruit(cfg, u) @ B_t.T


def step_jacobians(cfg: PopulationConfig, A, B_t, x, u):
    """``(∂f/∂x, ∂f/∂u)`` at a single ``(x, u)``."""
    ds = 1 - np.tanh(x / cfg.saturation) ** 2 if cfg.saturation > 0 else np.ones_like(x)
    fx = A * ds[None, :]
    fu = B_t * recruit_grad(cfg, u)[None, :]
    return fx, fu


class Population:
    """``P_Θ``: fixed mean system plus Gaussian inter-patient variation."""

    def __init__(self, cfg: PopulationConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        n, m, dz = cfg.n, cfg.m, cfg.dz
        assert n % 2 == 0, "n must be even (2x2 rotation blocks)"
        # Ā: block-diagonal damped rotations = oscillatory neural modes.
        A = np.zeros((n, n))
        for i, w in enumerate(rng.uniform(0.1, 0.6, n // 2)):
            c, s = np.cos(w), np.sin(w)
            A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, -s], [s, c]]
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))  # mix modes across latents
        self.A_bar = cfg.rho_A * Q @ A @ Q.T
        self.B_bar = cfg.b_scale * rng.normal(size=(n, m))
        self.D = np.eye(n)[:dz]
        self.C = np.eye(n) if cfg.observe == "full" else self.D.copy()
        self.targets = cfg.target_std * rng.normal(size=(cfg.n_targets, dz))

    def sample(self, rng: np.random.Generator) -> Patient:
        cfg = self.cfg
        A = self.A_bar + cfg.sigma_A * rng.normal(size=self.A_bar.shape) / np.sqrt(cfg.n)
        r = np.max(np.abs(np.linalg.eigvals(A)))
        if r > 0.97:
            A *= 0.97 / r
        B = self.B_bar + cfg.sigma_B * cfg.b_scale * rng.normal(size=self.B_bar.shape)
        return Patient(A, B)


# ------------------------------------------------------------------ control
def lq_tracker(A, B, Dz, c, z_star, lam_u, u_max):
    """LQ tracking of a constant target for ``x' = A x + B u + c``, ``z = Dz x``.

    Steady state: ``x_ss = (I − A)^{-1}(B u_ss + c)`` with ``u_ss`` the ridge
    least-squares solution of ``Dz x_ss ≈ z*``. Feedback: infinite-horizon LQR
    gain on deviations, ``u = u_ss − K (x − x_ss)``. Returns ``(x_ss, u_ss, K)``.
    """
    n, m = B.shape
    M = np.linalg.pinv(np.eye(n) - A)
    G, g = Dz @ M @ B, Dz @ M @ c
    u_ss = np.linalg.solve(G.T @ G + lam_u * np.eye(m), G.T @ (z_star - g))
    u_ss = np.clip(u_ss, -u_max, u_max)
    x_ss = M @ (B @ u_ss + c)
    try:
        P = solve_discrete_are(A, B, Dz.T @ Dz + 1e-6 * np.eye(n), lam_u * np.eye(m))
        K = np.linalg.solve(lam_u * np.eye(m) + B.T @ P @ B, B.T @ P @ A)
    except (np.linalg.LinAlgError, ValueError):
        K = np.zeros((m, n))  # estimated model not stabilisable: steady-state only
    return x_ss, u_ss, K


class KalmanFilter:
    def __init__(self, A, B, C, q, r):
        self.A, self.B, self.C = A, B, C
        self.Q = max(q, 1e-3) ** 2 * np.eye(A.shape[0])
        self.R = max(r, 1e-3) ** 2 * np.eye(C.shape[0])
        self.x, self.P = np.zeros(A.shape[0]), np.eye(A.shape[0]) * 0.1

    def update(self, o):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (o - self.C @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.C) @ self.P
        return self.x

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q


class Controller:
    name = "controller"
    privileged = False

    def reset(self, pop: Population, z_star: np.ndarray, info: dict) -> None:
        self.pop, self.cfg, self.z_star = pop, pop.cfg, z_star

    def act(self, o: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError


class ZeroController(Controller):
    name = "zero"

    def act(self, o, info):
        return np.zeros(self.cfg.m)


class _LQG(Controller):
    """Kalman filter + LQ tracker for a given model ``(A, B)``."""

    refresh = 0  # re-design every k steps (0: never)

    def _design(self, A, B):
        self.kf.A, self.kf.B = A, B
        c = np.zeros(A.shape[0])
        self.x_ss, self.u_ss, self.K = lq_tracker(
            A, B, self.pop.D, c, self.z_star, self.cfg.lam_u, self.cfg.u_max
        )

    def act(self, o, info):
        x_hat = self.kf.update(o)
        u = np.clip(self.u_ss - self.K @ (x_hat - self.x_ss), -self.cfg.u_max, self.cfg.u_max)
        self.kf.predict(u)
        return u


class PopulationController(_LQG):
    """One LQG for everybody, designed on the mean patient ``(Ā, B̄)``."""

    name = "population LQG"

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        c = pop.cfg
        self.kf = KalmanFilter(pop.A_bar, pop.B_bar, pop.C, c.process_noise, c.obs_noise)
        self._design(pop.A_bar, pop.B_bar)


class OracleController(_LQG):
    """Privileged LQG on the true, current ``(A_θ, B_θ + φ_t)``: the optimum."""

    name = "oracle LQG *"
    privileged = True

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        c = pop.cfg
        self.kf = KalmanFilter(info["A"], info["B_t"], pop.C, c.process_noise, c.obs_noise)
        self._design(info["A"], info["B_t"])
        self.t = 0

    def act(self, o, info):
        if self.cfg.drift > 0 and self.t % 10 == 0:
            self._design(info["A"], info["B_t"])
        self.t += 1
        return super().act(o, info)


class AdaptiveController(Controller):
    """Online identification + certainty-equivalent LQ tracking (no patient knowledge).

    ARX model ``o_{t+1} = Σ_{i<L} (Â_i o_{t-i}) + Σ_{i<L} (B̂_i u_{t-i}) + ĉ``
    fitted by RLS with forgetting ``λ_f`` (memory ≈ 1/(1−λ_f) steps). With
    partial observation (``o = D x``) a lag ``L ≥ ⌈n / dim o⌉`` is needed for an
    exact input–output description (observability index). The model is
    rewritten in state-space form on the stacked regressor ``ξ_t`` and
    handed to ``lq_tracker``.
    """

    name = "adaptive (RLS + CE)"

    def __init__(
        self,
        forget: float = 0.99,
        n_explore: int = 30,
        probe: float = 0.5,
        dither: float = 0.05,
        lags: int | None = None,
        refresh: int = 10,
        seed: int = 0,
    ):
        self.forget, self.n_explore, self.probe = forget, n_explore, probe
        self.dither, self.lags, self.refresh = dither, lags, refresh
        self.rng = np.random.default_rng(seed)

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        c = self.cfg
        self.p = pop.C.shape[0]
        self.L = self.lags or int(np.ceil(c.n / self.p))
        self.nxi = self.L * self.p + (self.L - 1) * c.m
        k = self.nxi + c.m + 1
        self.theta = np.zeros((k, self.p))
        self.P = 1e3 * np.eye(k)
        self.obs_hist = [np.zeros(self.p)] * self.L
        self.u_hist = [np.zeros(c.m)] * self.L
        self.phi = None
        self.t = 0
        # target lives in percept space; map the observation onto it
        self.Dz_obs = pop.D if c.observe == "full" else np.eye(self.p)

    def _xi(self):
        return np.concatenate(self.obs_hist[::-1][: self.L] + self.u_hist[::-1][: self.L - 1])

    def _rls(self, y):
        phi, P = self.phi, self.P
        k = P @ phi / (self.forget + phi @ P @ phi)
        self.theta += np.outer(k, y - phi @ self.theta)
        self.P = (P - np.outer(k, phi @ P)) / self.forget

    def _design(self):
        c, p, L, m = self.cfg, self.p, self.L, self.cfg.m
        Th = self.theta.T  # (p, nxi + m + 1)
        A = np.zeros((self.nxi, self.nxi))
        B = np.zeros((self.nxi, m))
        cvec = np.zeros(self.nxi)
        A[:p] = Th[:, : self.nxi]
        B[:p] = Th[:, self.nxi : self.nxi + m]
        cvec[:p] = Th[:, -1]
        for i in range(1, L):  # shift past observations
            A[i * p : (i + 1) * p, (i - 1) * p : i * p] = np.eye(p)
        if L > 1:  # shift past inputs; newest u enters the first u slot
            u0 = L * p
            B[u0 : u0 + m] = np.eye(m)
            for i in range(1, L - 1):
                A[u0 + i * m : u0 + (i + 1) * m, u0 + (i - 1) * m : u0 + i * m] = np.eye(m)
        Dz = np.zeros((self.Dz_obs.shape[0], self.nxi))
        Dz[:, :p] = self.Dz_obs
        self.x_ss, self.u_ss, self.K = lq_tracker(A, B, Dz, cvec, self.z_star, c.lam_u, c.u_max)

    def act(self, o, info):
        c = self.cfg
        if self.phi is not None:
            self._rls(o)
        self.obs_hist = (self.obs_hist + [o])[-self.L :]
        xi = self._xi()
        if self.t < self.n_explore:
            u = self.probe * self.rng.normal(size=c.m)
        else:
            if (self.t - self.n_explore) % self.refresh == 0:
                self._design()
            u = self.u_ss - self.K @ (xi - self.x_ss) + self.dither * self.rng.normal(size=c.m)
        u = np.clip(u, -c.u_max, c.u_max)
        self.phi = np.concatenate([xi, u, [1.0]])
        self.u_hist = (self.u_hist + [u])[-self.L :]
        self.t += 1
        return u


# ------------------------------------------------------------------ simulate
def run_episode(pop: Population, patient: Patient, ctrl: Controller, z_star, rng):
    """One patient, one target. Returns per-step percept error and total cost."""
    c = pop.cfg
    x = np.zeros(c.n)
    phi = np.zeros_like(patient.B)
    info = {"A": patient.A, "B_t": patient.B + phi, "x": x}
    ctrl.reset(pop, z_star, info)
    err, cost = np.zeros(c.horizon), np.zeros(c.horizon)
    for t in range(c.horizon):
        o = pop.C @ x + c.obs_noise * rng.normal(size=pop.C.shape[0])
        info["B_t"] = patient.B + phi
        info["x"] = x  # privileged: only oracle controllers may read it
        u = ctrl.act(o, info)
        x = step_mean(c, patient.A, patient.B + phi, x, u) + c.process_noise * rng.normal(size=c.n)
        if c.drift > 0:
            phi = c.drift_decay * phi + c.drift * rng.normal(size=phi.shape)
        e = pop.D @ x - z_star
        err[t] = e @ e
        cost[t] = err[t] + c.lam_u * u @ u
    return err, cost


def evaluate(
    cfg: PopulationConfig, make_ctrl, n_patients: int = 30, seed: int = 1000, tail: float = 0.5
) -> dict:
    """Held-out patients ``θ ~ P_Θ`` (seeded), one target from the common set each."""
    pop = Population(cfg)
    errs, costs, ref = [], [], []
    for i in range(n_patients):
        rng = np.random.default_rng(seed + i)
        patient = pop.sample(rng)
        z_star = pop.targets[rng.integers(len(pop.targets))]
        state = rng.bit_generator.state
        e, c = run_episode(pop, patient, make_ctrl(), z_star, rng)
        errs.append(e)
        costs.append(c)
        # Same patient, target and noise draws, no stimulation: the failure line.
        rng0 = np.random.default_rng()
        rng0.bit_generator.state = state
        ref.append(run_episode(pop, patient, ZeroController(), z_star, rng0)[0])
    errs, costs = np.array(errs), np.array(costs)
    k = int(cfg.horizon * (1 - tail))
    ss = errs[:, k:].mean(1)
    ss_zero = np.array(ref)[:, k:].mean(1)
    return {
        "cost": float(costs.mean()),
        "ss_error": float(ss.mean()),
        "ss_median": float(np.median(ss)),
        # "failure": the controller ends up worse than not stimulating at all
        "fail_rate": float(np.mean(ss > ss_zero)),
        "error_curve": np.median(errs, 0),
    }


LEVELS: dict[str, PopulationConfig] = {
    "L0 known patient": PopulationConfig(sigma_A=0.0, sigma_B=0.0),
    "L1 random patients": PopulationConfig(),
    "L2 + partial observation": PopulationConfig(observe="percept"),
    "L3 + drift": PopulationConfig(observe="percept", drift=0.01, horizon=1000),
}


def level(name: str, **overrides) -> PopulationConfig:
    return replace(LEVELS[name], **overrides)


# The "ARIMA test": each regime is where a class of methods should win or tie.
REGIMES: dict[str, PopulationConfig] = {
    # A: LTI + Gaussian, one known patient. LQG is optimal.
    "A": PopulationConfig(sigma_A=0.0, sigma_B=0.0, horizon=150),
    # B: unknown LTI patient. Online identification should be hard to beat.
    "B": PopulationConfig(horizon=150),
    # C: smooth nonlinear (saturating recruitment and firing rates) + drift.
    "C": PopulationConfig(recruit="tanh", saturation=1.0, drift=0.01, horizon=150),
    # D: partial observation + heterogeneity + multimodal (polarity-insensitive) recruitment.
    "D": PopulationConfig(recruit="even", saturation=1.0, observe="percept", horizon=150),
}


def regime(name: str, **overrides) -> PopulationConfig:
    return replace(REGIMES[name], **overrides)
