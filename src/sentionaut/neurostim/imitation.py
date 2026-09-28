"""Supervised baseline: imitate the privileged oracle with DAgger (no RL).

The teacher (``OracleGreedy``) sees the latent ``x, h, A, B``. The student sees
only the ``(o_t, a_{t-1})`` history window the PPO baseline gets, and
regresses the teacher's action with MSE. This is asymmetric (privileged)
imitation: the student has to do the identification implicitly, but the
learning signal is a dense per-step target instead of a scalar return.

Plain behaviour cloning (iteration 0 only) trains on states the teacher
visits. After a few student mistakes those states stop covering what the
student sees (covariate shift). DAgger (Ross et al., 2011) fixes this. Each
iteration rolls out a mix of student and teacher, labels every visited state
with the teacher's action, adds the pairs to the dataset and refits.

The teacher never probes, so its trajectories carry little information about
``B``. With ``calibration > 0`` every episode (student and teacher alike)
starts with a fixed probe sequence: each electrode is pulsed once at
``+-probe``, as in a clinical calibration sweep. The recorded responses
stay in the student's input for the whole episode. Identification then
becomes supervised regression on a designed experiment, not something the
student must infer from whatever the teacher happened to do.

The student can be no better than its teacher. Here the teacher is a myopic
greedy controller, so this baseline measures "how much of the oracle's
performance survives without privileged state", not the optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .baselines import OracleGreedy, Policy, charge_shield
from .env import NeuroStimEnv
from .ppo import History


@dataclass
class DAggerConfig:
    iterations: int = 8  # iteration 0 is behaviour cloning
    episodes_per_iter: int = 40
    history: int = 16
    hidden: int = 128
    epochs: int = 20  # passes over the aggregated dataset per iteration
    batch: int = 512
    lr: float = 1e-3
    beta_decay: float = 0.5  # P(teacher acts) = beta_decay ** iteration
    calibration: bool = False  # fixed probe sweep at episode start, kept as context
    probe: float = 0.5
    seed: int = 0


class StudentNet(nn.Module):
    def __init__(self, n_in: int, n_act: int, hidden: int, a_max: float):
        super().__init__()
        self.a_max = a_max
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_act),
        )

    def forward(self, z):
        return self.a_max * torch.tanh(self.net(z))


def probe_sequence(n_act: int, amplitude: float) -> np.ndarray:
    """``+a e_1, ..., +a e_m, -a e_1, ..., -a e_m``: one pulse per electrode and sign."""
    eye = amplitude * np.eye(n_act)
    return np.vstack([eye, -eye])


class StudentPolicy(Policy):
    """History-conditioned regression policy, plus the public charge shield.

    Input: ``(o, a_prev)`` window of length ``history``. With calibration, it
    also gets the ``(o, a)`` record of the probe sweep, which is zero until the
    sweep ends.
    """

    name = "DAgger student"

    def __init__(self, net: StudentNet, cfg: DAggerConfig, name: str | None = None):
        self.net, self.cfg = net, cfg
        if name:
            self.name = name

    def reset(self, env, obs, info):
        super().reset(env, obs, info)
        n_obs, n_act = obs.shape[0], env.action_space.shape[0]
        self.hist = History(self.cfg.history, n_obs, n_act)
        self.probes = probe_sequence(n_act, self.cfg.probe) if self.cfg.calibration else []
        self.calib = np.zeros((len(self.probes), n_obs + n_act), np.float32)
        self.t = 0
        self.z = self._features(self.hist.reset(obs, info), obs)

    @property
    def calibrating(self) -> bool:
        return self.t < len(self.probes)

    def _features(self, window: np.ndarray, obs: np.ndarray) -> np.ndarray:
        if 0 < self.t <= len(self.probes):  # response to probe t-1 just arrived
            self.calib[self.t - 1] = np.concatenate([obs, self.probes[self.t - 1]])
        return np.concatenate([window, self.calib.ravel()]).astype(np.float32)

    def act(self, obs, info):
        if self.t > 0:
            self.z = self._features(self.hist.push(obs, self.a_prev, info), obs)
        if self.calibrating:
            a = self.probes[self.t]
        else:
            with torch.no_grad():
                a = self.net(torch.as_tensor(self.z)).numpy().astype(float)
            a = charge_shield(a, self.env.cfg.q_max)
        self.a_prev = a
        self.t += 1
        return a


def _collect(env: NeuroStimEnv, student: StudentPolicy, teacher: Policy, beta, rng, episodes):
    """Roll out the mixture; label every visited history with the teacher's action."""
    Z, Y = [], []
    for _ in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(1 << 31)))
        student.reset(env, obs, info)
        teacher.reset(env, obs, info)
        # One coin per episode: the rollout stays coherent (an all-teacher or
        # all-student episode) instead of switching controllers every step.
        use_teacher = rng.random() < beta
        done = False
        while not done:
            calibrating = student.calibrating
            a_student = student.act(obs, info)  # updates the student's history
            a_teacher = teacher.act(obs, info)
            if calibrating:  # the probe sweep is fixed, not learned
                a = a_student
            else:
                Z.append(student.z.copy())
                Y.append(a_teacher)
                a = a_teacher if use_teacher else a_student
            student.a_prev = a  # the history records what was actually delivered
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
    return np.asarray(Z, np.float32), np.asarray(Y, np.float32)


def train_dagger(
    make_env,
    cfg: DAggerConfig | None = None,
    teacher: Policy | None = None,
    verbose: bool = True,
) -> tuple[StudentPolicy, list[dict]]:
    """Train a student on ``make_env()`` patients by DAgger from ``teacher``."""
    from .baselines import evaluate

    cfg = cfg or DAggerConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    teacher = teacher or OracleGreedy()
    env = make_env()
    n_obs, n_act = env.observation_space.shape[0], env.action_space.shape[0]
    n_calib = 2 * n_act if cfg.calibration else 0
    n_in = (cfg.history + n_calib) * (n_obs + n_act)
    net = StudentNet(n_in, n_act, cfg.hidden, env.cfg.a_max)
    name = "DAgger + calibration" if cfg.calibration else "DAgger student"
    student = StudentPolicy(net, cfg, name=name)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    Z = np.zeros((0, n_in), np.float32)
    Y = np.zeros((0, n_act), np.float32)
    log = []

    for it in range(cfg.iterations):
        beta = cfg.beta_decay**it
        z_new, y_new = _collect(env, student, teacher, beta, rng, cfg.episodes_per_iter)
        Z, Y = np.concatenate([Z, z_new]), np.concatenate([Y, y_new])
        zt, yt = torch.as_tensor(Z), torch.as_tensor(Y)
        for _ in range(cfg.epochs):
            for idx in torch.randperm(len(zt)).split(cfg.batch):
                loss = (net(zt[idx]) - yt[idx]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            mse = float((net(zt) - yt).pow(2).mean())
        m = evaluate(make_env(), student, episodes=10, seed=500)  # not the test seeds
        entry = {"iteration": it, "beta": beta, "samples": len(Z), "mse": mse, **m}
        log.append(entry)
        if verbose:
            print(
                f"  iter {it}  beta {beta:.2f}  n {len(Z):>6d}  mse {mse:.4f}  "
                f"return {m['return']:8.1f}  viol {m['violation_rate']:.3f}"
            )
    return student, log
