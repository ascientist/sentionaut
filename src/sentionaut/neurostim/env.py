"""NeuroStim: a toy constrained POMDP for closed-loop neurostimulation.

Latent state ``s_t = (x_t, h_t, B_t)``:

- ``x_t in R^d``  neural state (only seen through ``o_t = C x_t + eps``)
- ``h_t in R^m``  per-electrode adaptation / fatigue
- ``B_t in R^{d x m}``  unknown electrode recruitment matrix (per "patient",
  optionally drifting)

Dynamics::

    x_{t+1} = A x_t + f(x_t) + B_t g(a_{t-delta}, h_t) + w_t,   f(x) = -alpha x^3
    g(a, h) = tanh(a) / (1 + lambda h)
    h_{t+1} = rho h_t + kappa |a_t|
    B_{t+1} = B_t + theta (B_0 - B_t) + eta_t

Reward ``r_t = -||H x_{t+1} - z*||^2 - w_E ||a_t||^2 - w_D ||a_t - a_{t-1}||^2``.
Safety is *not* folded into the reward: each step also returns a constraint
cost (charge ``sum|a| <= q_max`` and adaptation ``max h <= h_max``) in ``info``,
so the problem is a CMDP ``max E[sum gamma^t r] s.t. E[sum c_k] <= d_k``.

The API follows Gymnasium (``reset -> (obs, info)``,
``step -> (obs, r, terminated, truncated, info)``) without depending on it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

# Recruitment matrix from the design note: no electrode controls one latent alone.
DEFAULT_B = np.array(
    [
        [0.8, 0.1, 0.4, -0.2],
        [0.3, 0.7, -0.4, 0.1],
        [-0.1, 0.4, 0.8, 0.3],
    ]
)
# Stable (spectral radius ~0.9), weakly coupled intrinsic dynamics.
DEFAULT_A = np.array(
    [
        [0.90, 0.05, 0.00],
        [-0.05, 0.90, 0.05],
        [0.00, -0.05, 0.85],
    ]
)
# Partial observation: x1 directly, x2 and x3 only through one mixed biomarker.
PARTIAL_C = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.7, 0.7],
    ]
)


@dataclass(frozen=True)
class Box:
    """Minimal stand-in for ``gymnasium.spaces.Box``."""

    low: np.ndarray
    high: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class NeuroStimConfig:
    """Every benchmark axis is a field; presets live in ``PRESETS``."""

    horizon: int = 200
    # Intrinsic dynamics.
    A: np.ndarray = field(default_factory=lambda: DEFAULT_A.copy())
    cubic: float = 0.02  # alpha in f(x) = -alpha x^3
    process_noise: float = 0.02
    x0_std: float = 0.3
    randomize_A: bool = False
    # Stimulation mapping.
    B: np.ndarray = field(default_factory=lambda: DEFAULT_B.copy())
    mapping: str = "overlap"  # "overlap" (B above) | "diagonal"
    stim_gain: float = 0.5  # global recruitment scale applied to B
    randomize_B: bool = False  # new "patient" B ~ p(B) every episode
    delay: int = 0
    # Adaptation.
    adaptation: bool = True
    adapt_decay: float = 0.95  # rho
    adapt_rate: float = 0.05  # kappa  (steady state h = kappa/(1-rho) |a| = |a|)
    adapt_strength: float = 2.0  # lambda
    # Nonstationarity.
    drift_std: float = 0.0
    drift_revert: float = 0.01
    # Observation.
    observation: str = "full"  # "full" | "partial"
    obs_noise: float = 0.05
    # Task.
    target: tuple[float, ...] = (1.0, 0.0, -1.0)
    w_energy: float = 0.01
    w_smooth: float = 0.01
    # Safety (CMDP constraints).
    a_max: float = 1.0
    q_max: float = 2.0  # per-step total charge sum_i |a_i|
    h_max: float = 0.8

    @property
    def n_latent(self) -> int:
        return self.A.shape[0]

    @property
    def n_electrodes(self) -> int:
        return self.B.shape[1]

    def C(self) -> np.ndarray:
        if self.observation == "full":
            return np.eye(self.n_latent)
        if self.observation == "partial":
            return PARTIAL_C.copy()
        raise ValueError(f"unknown observation mode {self.observation!r}")

    def nominal_B(self) -> np.ndarray:
        if self.mapping == "overlap":
            return self.B.copy()
        if self.mapping == "diagonal":
            d, m = self.n_latent, self.n_electrodes
            B = np.zeros((d, m))
            B[np.arange(m) % d, np.arange(m)] = 0.8
            return B
        raise ValueError(f"unknown mapping {self.mapping!r}")


PRESETS: dict[str, NeuroStimConfig] = {
    # Every hard axis switched off: essentially a noisy linear regulator.
    "NeuroStim-Easy-v0": NeuroStimConfig(mapping="diagonal", adaptation=False, obs_noise=0.0),
    # Canonical: 3 latents, 4 overlapping electrodes, unknown per-patient B,
    # adaptation, safety constraints, noisy full observation.
    "NeuroStim-v0": NeuroStimConfig(randomize_B=True),
    # All axes on: partial obs, delay, random A and B, drifting B.
    "NeuroStim-Hard-v0": NeuroStimConfig(
        randomize_B=True, randomize_A=True, delay=2, drift_std=0.01, observation="partial"
    ),
}


def make(name: str = "NeuroStim-v0", **overrides) -> "NeuroStimEnv":
    """Build a preset, optionally overriding any ``NeuroStimConfig`` field."""
    if name not in PRESETS:
        raise KeyError(f"unknown env {name!r}; choose from {sorted(PRESETS)}")
    return NeuroStimEnv(replace(PRESETS[name], **overrides))


def sample_patient_B(rng: np.random.Generator, d: int, m: int, min_sv: float = 0.4) -> np.ndarray:
    """p(B): Gaussian recruitment, rejected until the target is controllable."""
    while True:
        B = rng.normal(0.0, 0.5, size=(d, m))
        if np.linalg.svd(B, compute_uv=False)[-1] >= min_sv:
            return B


def sample_A(rng: np.random.Generator, d: int) -> np.ndarray:
    """p(A): diagonally dominant, spectral radius < 0.97."""
    while True:
        A = np.diag(rng.uniform(0.8, 0.95, size=d)) + rng.normal(0.0, 0.05, size=(d, d))
        if np.max(np.abs(np.linalg.eigvals(A))) < 0.97:
            return A


class NeuroStimEnv:
    """Gymnasium-style CMDP. ``info`` carries costs and (privileged) latents."""

    def __init__(self, config: NeuroStimConfig | None = None):
        self.cfg = config or NeuroStimConfig()
        c = self.cfg
        d, m = c.n_latent, c.n_electrodes
        self.C = c.C()
        self.target = np.asarray(c.target, dtype=float)
        assert self.target.shape == (d,), "target must match the latent dimension"
        self.action_space = Box(np.full(m, -c.a_max), np.full(m, c.a_max))
        n_obs = self.C.shape[0]
        self.observation_space = Box(np.full(n_obs, -np.inf), np.full(n_obs, np.inf))
        self.rng = np.random.default_rng()
        self.reset(seed=0)

    # ------------------------------------------------------------------ core
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, d, m = self.cfg, self.cfg.n_latent, self.cfg.n_electrodes
        self.A = sample_A(self.rng, d) if c.randomize_A else c.A.copy()
        B0 = sample_patient_B(self.rng, d, m) if c.randomize_B else c.nominal_B()
        self.B0 = c.stim_gain * B0
        self.B = self.B0.copy()
        self.x = self.rng.normal(0.0, c.x0_std, size=d)
        self.h = np.zeros(m)
        self.a_prev = np.zeros(m)
        # Actions in flight: stimulation delivered at t reaches x at t + delay.
        self.pending: deque[np.ndarray] = deque([np.zeros(m)] * c.delay, maxlen=c.delay + 1)
        self.t = 0
        return self._observe(), self._info()

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=float), -c.a_max, c.a_max)
        self.pending.append(a)
        a_eff = self.pending.popleft()

        x_next = mean_dynamics(self.A, self.B, c, self.x, self.h, a_eff)
        x_next = x_next + self.rng.normal(0.0, c.process_noise, size=self.x.shape)
        h_next = adapt(c, self.h, a)

        err = self.target_error(x_next)
        reward = -(err + c.w_energy * a @ a + c.w_smooth * np.sum((a - self.a_prev) ** 2))

        charge = float(np.abs(a).sum())
        cost_terms = {
            "charge": max(0.0, charge - c.q_max),
            "adaptation": max(0.0, float(h_next.max()) - c.h_max),
        }
        cost = float(any(v > 1e-9 for v in cost_terms.values()))

        if c.drift_std > 0:
            self.B = (
                self.B
                + c.drift_revert * (self.B0 - self.B)
                + self.rng.normal(0.0, c.drift_std, size=self.B.shape)
            )
        self.x, self.h, self.a_prev = x_next, h_next, a
        self.t += 1
        truncated = self.t >= c.horizon
        info = self._info()
        info.update(cost=cost, cost_terms=cost_terms, error=err, charge=charge)
        return self._observe(), float(reward), False, truncated, info

    # --------------------------------------------------------------- helpers
    def target_error(self, x: np.ndarray) -> float:
        return float(np.sum((x - self.target) ** 2))

    @property
    def target_obs(self) -> np.ndarray:
        """Target expressed in observation space (what a non-privileged agent may know)."""
        return self.C @ self.target

    def _observe(self) -> np.ndarray:
        noise = self.rng.normal(0.0, self.cfg.obs_noise, size=self.C.shape[0])
        return self.C @ self.x + noise

    def _info(self) -> dict:
        # Privileged latent state, for oracles and plotting only.
        return {
            "x": self.x.copy(),
            "h": self.h.copy(),
            "B": self.B.copy(),
            "A": self.A.copy(),
            "pending": [p.copy() for p in self.pending],
            "t": self.t,
        }


def adapt(c: NeuroStimConfig, h: np.ndarray, a: np.ndarray) -> np.ndarray:
    if not c.adaptation:
        return h
    return c.adapt_decay * h + c.adapt_rate * np.abs(a)


def effective_gain(c: NeuroStimConfig, h: np.ndarray) -> np.ndarray:
    """Per-electrode multiplier ``1 / (1 + lambda h)`` on ``tanh(a)``."""
    if not c.adaptation:
        return np.ones_like(h)
    return 1.0 / (1.0 + c.adapt_strength * h)


def mean_dynamics(A, B, c: NeuroStimConfig, x, h, a_eff) -> np.ndarray:
    """Noise-free ``A x + f(x) + B g(a, h)``."""
    return A @ x - c.cubic * x**3 + B @ (np.tanh(a_eff) * effective_gain(c, h))
