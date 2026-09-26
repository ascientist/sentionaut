"""Reference controllers and an evaluation harness for ``NeuroStimEnv``.

Each policy implements ``reset(env, obs, info)`` and ``act(obs, info) -> a``.
Non-privileged policies only read public quantities from ``env`` (action
bounds, charge limit, ``target_obs``); ``privileged = True`` marks policies that
read the latent ``x``, ``h``, ``A``, ``B`` from ``info`` (upper-bound references,
not solutions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PRESETS, NeuroStimEnv, mean_dynamics


def box_ridge_lsq(M, e, mu, lo, hi, iters: int = 50) -> np.ndarray:
    """``argmin_v ||M v - e||^2 + mu ||v||^2`` s.t. ``lo <= v <= hi`` (projected gradient)."""
    L = 2.0 * (np.linalg.norm(M, 2) ** 2 + mu)
    v = np.clip(np.linalg.solve(M.T @ M + mu * np.eye(M.shape[1]), M.T @ e), lo, hi)
    for _ in range(iters):
        v = np.clip(v - (2.0 * (M.T @ (M @ v - e) + mu * v)) / L, lo, hi)
    return v


def charge_shield(a: np.ndarray, q_max: float) -> np.ndarray:
    """Scale ``a`` down so that ``sum |a| <= q_max`` (the one constraint any agent can check)."""
    q = np.abs(a).sum()
    return a * (q_max / q) if q > q_max else a


class Policy:
    name = "policy"
    privileged = False

    def reset(self, env: NeuroStimEnv, obs: np.ndarray, info: dict) -> None:
        self.env = env

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError


class ZeroPolicy(Policy):
    """No stimulation: shows the intrinsic dynamics relaxing to the origin."""

    name = "zero"

    def act(self, obs, info):
        return np.zeros(self.env.action_space.shape)


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, scale: float = 0.5, seed: int = 0):
        self.scale, self.rng = scale, np.random.default_rng(seed)

    def act(self, obs, info):
        return self.scale * self.rng.uniform(-1, 1, self.env.action_space.shape)


class SingleElectrodeBandit(Policy):
    """UCB1 over {off, +-s on one electrode}: treats stimulation as a stateless bandit.

    This is what a Bayesian-optimisation / bandit formulation reduces to; it
    ignores state persistence, delay and adaptation.
    """

    name = "bandit (UCB)"

    def __init__(self, amplitude: float = 0.8, c: float = 0.5):
        self.amplitude, self.c = amplitude, c

    def reset(self, env, obs, info):
        super().reset(env, obs, info)
        m = env.action_space.shape[0]
        eye = np.eye(m) * self.amplitude
        self.arms = np.vstack([np.zeros(m), eye, -eye])
        self.n = np.zeros(len(self.arms))
        self.mean = np.zeros(len(self.arms))
        self.last = None

    def act(self, obs, info):
        if self.last is not None and "reward" in info:
            k = self.last
            self.n[k] += 1
            self.mean[k] += (info["reward"] - self.mean[k]) / self.n[k]
        untried = np.flatnonzero(self.n == 0)
        if untried.size:
            k = int(untried[0])
        else:
            k = int(np.argmax(self.mean + self.c * np.sqrt(np.log(self.n.sum()) / self.n)))
        self.last = k
        return self.arms[k]


class NominalPI(Policy):
    """ "Universal" feedback controller tuned on the nominal (population) B.

    ``a = pinv(B_nom) (Kp e + Ki sum e)`` with ``e`` the observed target error.
    Works when the patient matches the population, fails when B differs.
    """

    name = "nominal PI"

    def __init__(self, kp: float = 0.6, ki: float = 0.08):
        self.kp, self.ki = kp, ki

    def reset(self, env, obs, info):
        super().reset(env, obs, info)
        c = env.cfg
        self.Binv = np.linalg.pinv(c.stim_gain * c.nominal_B())
        self.Cinv = np.linalg.pinv(env.C)
        self.integral = np.zeros(env.cfg.n_latent)

    def act(self, obs, info):
        e = self.env.target - self.Cinv @ obs
        self.integral = np.clip(self.integral + e, -10, 10)
        a = self.Binv @ (self.kp * e + self.ki * self.integral)
        a = np.clip(a, -self.env.cfg.a_max, self.env.cfg.a_max)
        return charge_shield(a, self.env.cfg.q_max)


class OnlineSysID(Policy):
    """Identify the individual online, then control (certainty equivalence).

    1. Cautious random probing for ``n_explore`` steps.
    2. Recursive least squares with forgetting on ``o_{t+1} ~ A_hat o_t + B_hat a_t + c``.
       Forgetting lets ``B_hat`` track adaptation and drift.
    3. Greedy one-step control on the estimated model + small dither for
       persistent excitation; charge shield on top.

    Assumes no delay and linear recruitment; it is given only ``target_obs``.
    """

    name = "online sysID"

    def __init__(
        self,
        n_explore: int = 20,
        probe: float = 0.3,
        forget: float = 0.98,
        ridge: float = 0.05,
        dither: float = 0.05,
        seed: int = 0,
    ):
        self.n_explore, self.probe, self.forget = n_explore, probe, forget
        self.ridge, self.dither = ridge, dither
        self.rng = np.random.default_rng(seed)

    def reset(self, env, obs, info):
        super().reset(env, obs, info)
        n, m = obs.shape[0], env.action_space.shape[0]
        p = n + m + 1
        self.theta = np.zeros((p, n))
        self.P = 100.0 * np.eye(p)
        self.phi = None
        self.t = 0

    def _update(self, obs):
        phi, P = self.phi, self.P
        k = P @ phi / (self.forget + phi @ P @ phi)
        self.theta += np.outer(k, obs - phi @ self.theta)
        self.P = (P - np.outer(k, phi @ P)) / self.forget

    def act(self, obs, info):
        if self.phi is not None:
            self._update(obs)
        c = self.env.cfg
        n, m = obs.shape[0], c.n_electrodes
        if self.t < self.n_explore:
            a = self.rng.uniform(-self.probe, self.probe, m)
        else:
            A_hat = self.theta[:n].T
            B_hat = self.theta[n : n + m].T
            bias = self.theta[-1]
            e = self.env.target_obs - A_hat @ obs - bias
            a = box_ridge_lsq(B_hat, e, self.ridge, -c.a_max, c.a_max)
            a = a + self.dither * self.rng.standard_normal(m)
        a = charge_shield(np.clip(a, -c.a_max, c.a_max), c.q_max)
        self.phi = np.concatenate([obs, a, [1.0]])
        self.t += 1
        return a


class OracleGreedy(Policy):
    """Privileged reference: knows x, h, A, B_t and actions in flight.

    Rolls the noise-free model forward through the delay, then picks the
    stimulation whose (adaptation-attenuated) drive best reaches the target in
    one step. Enforces both constraints exactly (it can see h). Myopic: it does
    not plan around future adaptation, so it is a strong reference, not the optimum.
    """

    name = "oracle greedy"
    privileged = True

    def __init__(self, ridge: float = 0.02, safe: bool = True):
        self.ridge, self.safe = ridge, safe

    def act(self, obs, info):
        c = self.env.cfg
        x, h, A, B = info["x"], info["h"], info["A"], info["B"]
        for a_in_flight in info["pending"]:
            x = mean_dynamics(A, B, c, x, h, a_in_flight)
        gain = 1.0 / (1.0 + c.adapt_strength * h) if c.adaptation else np.ones_like(h)
        a_lim = np.full_like(h, c.a_max)
        if self.safe and c.adaptation:
            a_lim = np.clip((c.h_max - c.adapt_decay * h) / c.adapt_rate, 0.0, c.a_max)
        e = self.env.target - (A @ x - c.cubic * x**3)
        drive_lim = np.tanh(a_lim) * gain
        # Solve in drive space v = tanh(a) * gain (linear in v), then invert.
        v = box_ridge_lsq(B, e, self.ridge, -drive_lim, drive_lim)
        a = np.arctanh(np.clip(v / gain, -0.999999, 0.999999))
        a = np.clip(a, -a_lim, a_lim)
        return charge_shield(a, c.q_max) if self.safe else a


# ---------------------------------------------------------------- evaluation
@dataclass
class Trajectory:
    obs: np.ndarray  # (T+1, n_obs)
    x: np.ndarray  # (T+1, d) latent
    h: np.ndarray  # (T+1, m) adaptation
    actions: np.ndarray  # (T, m)
    rewards: np.ndarray  # (T,)
    costs: np.ndarray  # (T,)
    errors: np.ndarray  # (T,)
    target: np.ndarray


def rollout(env: NeuroStimEnv, policy: Policy, seed: int | None = None) -> Trajectory:
    obs, info = env.reset(seed=seed)
    policy.reset(env, obs, info)
    keys = ("obs", "x", "h", "actions", "rewards", "costs", "errors")
    log = {k: [] for k in keys}
    log["obs"].append(obs), log["x"].append(info["x"]), log["h"].append(info["h"])
    done = False
    while not done:
        a = np.asarray(policy.act(obs, info), dtype=float)
        obs, r, term, trunc, info = env.step(a)
        info["reward"] = r  # lets reward-driven policies (bandit) learn online
        done = term or trunc
        log["obs"].append(obs), log["x"].append(info["x"]), log["h"].append(info["h"])
        log["actions"].append(np.clip(a, -env.cfg.a_max, env.cfg.a_max))
        (
            log["rewards"].append(r),
            log["costs"].append(info["cost"]),
            log["errors"].append(info["error"]),
        )
    return Trajectory(**{k: np.asarray(log[k]) for k in keys}, target=env.target.copy())


def summarize(trajs: list[Trajectory], tail: int = 50) -> dict[str, float]:
    """Return, steady-state error (last ``tail`` steps), violation rate and charge."""
    ret = np.array([t.rewards.sum() for t in trajs])
    return {
        "return": float(ret.mean()),
        "return_se": float(ret.std(ddof=1) / np.sqrt(len(ret))) if len(ret) > 1 else 0.0,
        "ss_error": float(np.mean([t.errors[-tail:].mean() for t in trajs])),
        "violation_rate": float(np.mean([t.costs.mean() for t in trajs])),
        "charge": float(np.mean([np.abs(t.actions).sum(1).mean() for t in trajs])),
    }


def evaluate(
    env: NeuroStimEnv, policy: Policy, episodes: int = 20, seed: int = 1000
) -> dict[str, float]:
    """Held-out patients: seeds ``seed .. seed + episodes - 1``."""
    return summarize([rollout(env, policy, seed=seed + i) for i in range(episodes)])


def default_baselines() -> list[Policy]:
    return [
        ZeroPolicy(),
        RandomPolicy(),
        SingleElectrodeBandit(),
        NominalPI(),
        OnlineSysID(),
        OracleGreedy(),
    ]


__all__ = [
    "PRESETS",
    "Policy",
    "ZeroPolicy",
    "RandomPolicy",
    "SingleElectrodeBandit",
    "NominalPI",
    "OnlineSysID",
    "OracleGreedy",
    "Trajectory",
    "rollout",
    "evaluate",
    "summarize",
    "default_baselines",
    "box_ridge_lsq",
    "charge_shield",
]
