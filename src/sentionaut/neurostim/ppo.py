"""Compact PPO(-Lagrangian) baseline for ``NeuroStimEnv``.

The policy sees a short history of ``(o_t, a_{t-1})`` pairs, the minimum needed
to act in a POMDP with unknown ``B`` and adaptation without a recurrent net.
Trained across many sampled patients it is the "offline universal policy": it
can only personalise to the extent that the history window lets it infer ``B``.

With ``cost_limit`` set, a Lagrange multiplier on the per-episode violation
count is learned by dual ascent (``r - lambda * c``), turning PPO into a
constrained-MDP solver.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .baselines import Policy
from .env import NeuroStimEnv


@dataclass
class PPOConfig:
    total_steps: int = 300_000
    n_envs: int = 8
    n_steps: int = 256  # per env per iteration
    history: int = 4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    lr: float = 3e-4
    epochs: int = 10
    minibatch: int = 512
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    hidden: int = 64
    reward_scale: float = 0.1
    # Diagnostic: append the true (privileged) B_t to the input. Separates
    # "cannot identify the patient" from "cannot control a known patient".
    privileged_B: bool = False
    cost_limit: float | None = None  # allowed violating steps per episode
    lagrange_lr: float = 0.05
    seed: int = 0


class History:
    """Rolling window of ``(o_t, a_{t-1})``, zero-padded at episode start.

    With ``privileged_B`` the flattened true ``B_t`` from ``info`` is appended.
    """

    def __init__(self, k: int, n_obs: int, n_act: int, privileged_B: bool = False):
        self.k, self.n_obs, self.n_act, self.privileged_B = k, n_obs, n_act, privileged_B

    def reset(self, obs: np.ndarray, info: dict) -> np.ndarray:
        self.buf = deque([np.zeros(self.n_obs + self.n_act)] * self.k, maxlen=self.k)
        return self.push(obs, np.zeros(self.n_act), info)

    def push(self, obs: np.ndarray, a_prev: np.ndarray, info: dict) -> np.ndarray:
        self.buf.append(np.concatenate([obs, a_prev]))
        z = list(self.buf) + ([info["B"].ravel()] if self.privileged_B else [])
        return np.concatenate(z).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, n_in: int, n_act: int, hidden: int):
        super().__init__()

        def mlp(n_out):
            return nn.Sequential(
                nn.Linear(n_in, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_out),
            )

        self.mu, self.v = mlp(n_act), mlp(1)
        self.log_std = nn.Parameter(torch.full((n_act,), -0.7))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp())


class PPOPolicy(Policy):
    """Deterministic (mean) evaluation wrapper around a trained ``ActorCritic``."""

    name = "PPO"

    def __init__(self, net: ActorCritic, cfg: PPOConfig, name: str | None = None):
        self.net, self.cfg = net, cfg
        self.privileged = cfg.privileged_B
        if name:
            self.name = name

    def reset(self, env, obs, info):
        super().reset(env, obs, info)
        n_act = env.action_space.shape[0]
        self.hist = History(self.cfg.history, obs.shape[0], n_act, self.cfg.privileged_B)
        self.z = self.hist.reset(obs, info)
        self.first = True

    def act(self, obs, info):
        if not self.first:
            self.z = self.hist.push(obs, self.a_prev, info)
        self.first = False
        with torch.no_grad():
            a = self.net.mu(torch.as_tensor(self.z)).numpy().astype(float)
        self.a_prev = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        return self.a_prev


def train_ppo(
    make_env, cfg: PPOConfig | None = None, log_every: int = 10, verbose: bool = True
) -> tuple[PPOPolicy, list[dict]]:
    """Train on ``cfg.n_envs`` copies of ``make_env()``; every reset is a new patient."""
    cfg = cfg or PPOConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    envs: list[NeuroStimEnv] = [make_env() for _ in range(cfg.n_envs)]
    n_obs, n_act = envs[0].observation_space.shape[0], envs[0].action_space.shape[0]
    hists = [History(cfg.history, n_obs, n_act, cfg.privileged_B) for _ in envs]
    z = np.stack([h.reset(*e.reset(seed=int(rng.integers(1 << 31)))) for h, e in zip(hists, envs)])

    net = ActorCritic(z.shape[1], n_act, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    lam = 0.0
    ep_ret = np.zeros(cfg.n_envs)
    ep_cost = np.zeros(cfg.n_envs)
    done_ret, done_cost, log = [], [], []
    N, E = cfg.n_steps, cfg.n_envs
    n_iters = cfg.total_steps // (N * E)

    for it in range(n_iters):
        Z = np.zeros((N, E, z.shape[1]), np.float32)
        A = np.zeros((N, E, n_act), np.float32)
        LP, R, V, D = (np.zeros((N, E), np.float32) for _ in range(4))
        for t in range(N):
            with torch.no_grad():
                zt = torch.as_tensor(z)
                dist = net.dist(zt)
                a = dist.sample()
                LP[t] = dist.log_prob(a).sum(-1).numpy()
                V[t] = net.v(zt).squeeze(-1).numpy()
            Z[t], A[t] = z, a.numpy()
            for i, env in enumerate(envs):
                act = np.clip(A[t, i].astype(float), -env.cfg.a_max, env.cfg.a_max)
                obs, r, term, trunc, info = env.step(act)
                ep_ret[i] += r
                ep_cost[i] += info["cost"]
                R[t, i] = cfg.reward_scale * (r - lam * info["cost"])
                z[i] = hists[i].push(obs, act, info)
                if term or trunc:
                    if trunc and not term:  # bootstrap through time-limit truncation
                        with torch.no_grad():
                            R[t, i] += cfg.gamma * net.v(torch.as_tensor(z[i])).item()
                    D[t, i] = 1.0
                    done_ret.append(ep_ret[i])
                    done_cost.append(ep_cost[i])
                    ep_ret[i] = ep_cost[i] = 0.0
                    z[i] = hists[i].reset(*env.reset(seed=int(rng.integers(1 << 31))))

        with torch.no_grad():
            last_v = net.v(torch.as_tensor(z)).squeeze(-1).numpy()
        adv = np.zeros_like(R)
        gae = np.zeros(E, np.float32)
        for t in reversed(range(N)):
            next_v = last_v if t == N - 1 else V[t + 1]
            nonterm = 1.0 - D[t]
            delta = R[t] + cfg.gamma * next_v * nonterm - V[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + V

        bz, ba, blp = (torch.as_tensor(x.reshape(N * E, -1)) for x in (Z, A, LP))
        blp = blp.squeeze(-1)
        badv = torch.as_tensor(adv.reshape(-1))
        bret = torch.as_tensor(ret.reshape(-1))
        for _ in range(cfg.epochs):
            for idx in torch.randperm(N * E).split(cfg.minibatch):
                dist = net.dist(bz[idx])
                lp = dist.log_prob(ba[idx]).sum(-1)
                ratio = (lp - blp[idx]).exp()
                a_n = badv[idx]
                a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8)
                pg = -torch.min(ratio * a_n, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_n).mean()
                vf = (net.v(bz[idx]).squeeze(-1) - bret[idx]).pow(2).mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if cfg.cost_limit is not None and done_cost:
            recent = float(np.mean(done_cost[-2 * E :]))
            lam = max(0.0, lam + cfg.lagrange_lr * (recent - cfg.cost_limit))

        if done_ret:
            entry = {
                "step": (it + 1) * N * E,
                "return": float(np.mean(done_ret[-2 * E :])),
                "cost": float(np.mean(done_cost[-2 * E :])),
                "lambda": lam,
            }
            log.append(entry)
            if verbose and (it % log_every == 0 or it == n_iters - 1):
                print(
                    f"  step {entry['step']:>7d}  return {entry['return']:8.1f}  "
                    f"violations/ep {entry['cost']:6.1f}  lambda {lam:.3f}"
                )

    name = "PPO-Lagrangian" if cfg.cost_limit is not None else "PPO"
    return PPOPolicy(net, cfg, name=name), log
