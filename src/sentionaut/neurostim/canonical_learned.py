"""Learned baselines for the canonical random-LDS problem (torch).

Compact implementations of each method's core idea, sized for a 6-D system on
a CPU. They are **not** the reference implementations, and they are named
"-style" where that matters.

- ``train_sac``: soft actor-critic (Haarnoja et al. 2018). Model-free.
- ``train_tdmpc``: TD-MPC2-style (Hansen et al. 2024). A latent world model
  (encoder, latent dynamics, reward and Q heads, policy prior) trained by
  latent consistency + TD, with MPPI planning in latent space at test time.
- ``train_jepa``: JEPA-style predictive representation (LeCun 2022; V-JEPA,
  Bardes et al. 2024). The encoder and predictor are trained to predict
  *embeddings* of future observation histories (EMA target, VICReg
  anti-collapse), never pixels or observations. A linear probe then reads
  the percept, and MPPI plans through the predictor.
- ``train_diffusion``: diffusion policy (Chi et al. 2023). A DDPM over action
  chunks ``p(u_{t:t+H} | history, z*)``, trained on demonstrations.
- ``train_bc``: behaviour cloning with an MSE loss on the same demonstrations.
  It averages the modes that diffusion keeps apart.

Every learned policy sees the same input: the last ``k`` pairs
``(o_t, u_{t-1})`` and the target ``z*``. The demonstrations come from a
batched MPPI teacher that knows the true model (privileged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import solve_discrete_are

from .canonical import Controller, Population, PopulationConfig, lq_tracker, recruit_grad

HISTORY = 8


def mlp(n_in, n_out, hidden=256, layers=2, act=nn.Mish):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, n_out))


# ------------------------------------------------------------------ simulator
class TorchSim:
    """Batched episodes on ``n_envs`` patients from ``P_Θ``, same dynamics as ``run_episode``."""

    def __init__(self, pop: Population, n_envs: int, seed: int, history: int = HISTORY):
        self.pop, self.cfg, self.E, self.k = pop, pop.cfg, n_envs, history
        self.rng = np.random.default_rng(seed)
        self.C = torch.tensor(pop.C, dtype=torch.float32)
        self.D = torch.tensor(pop.D, dtype=torch.float32)
        self.p = pop.C.shape[0]

    @property
    def feat_dim(self):
        return self.k * (self.p + self.cfg.m) + self.cfg.dz

    def reset(self):
        c, E = self.cfg, self.E
        pats = [self.pop.sample(self.rng) for _ in range(E)]
        self.A = torch.tensor(np.stack([p.A for p in pats]), dtype=torch.float32)
        self.B = torch.tensor(np.stack([p.B for p in pats]), dtype=torch.float32)
        self.phi = torch.zeros_like(self.B)
        self.z = torch.tensor(
            self.pop.targets[self.rng.integers(len(self.pop.targets), size=E)], dtype=torch.float32
        )
        self.x = torch.zeros(E, c.n)
        self.t = 0
        self.hist = torch.zeros(E, self.k, self.p + c.m)
        self._push(self._observe(), torch.zeros(E, c.m))
        return self.feature()

    def _noise(self, *shape):
        return torch.tensor(self.rng.normal(size=shape), dtype=torch.float32)

    def _observe(self):
        return self.x @ self.C.T + self.cfg.obs_noise * self._noise(self.E, self.p)

    def _push(self, o, u_prev):
        self.hist = torch.cat([self.hist[:, 1:], torch.cat([o, u_prev], -1)[:, None]], 1)

    def feature(self):
        return torch.cat([self.hist.flatten(1), self.z], -1)

    def recruit(self, u):
        c = self.cfg
        if c.recruit == "linear":
            return u
        if c.recruit == "tanh":
            return torch.tanh(u)
        return u**2 - 0.25

    def rate(self, x):
        s = self.cfg.saturation
        return s * torch.tanh(x / s) if s > 0 else x

    def step_mean(self, x, u, A=None, B=None):
        A = self.A if A is None else A
        B = self.B + self.phi if B is None else B
        return torch.einsum("...ij,...j->...i", A, self.rate(x)) + torch.einsum(
            "...ij,...j->...i", B, self.recruit(u)
        )

    def step(self, u):
        c = self.cfg
        u = u.clamp(-c.u_max, c.u_max)
        self.x = self.step_mean(self.x, u) + c.process_noise * self._noise(self.E, c.n)
        if c.drift > 0:
            self.phi = c.drift_decay * self.phi + c.drift * self._noise(*self.phi.shape)
        e = self.x @ self.D.T - self.z
        err = (e**2).sum(-1)
        reward = -(err + c.lam_u * (u**2).sum(-1))
        self._push(self._observe(), u)
        self.t += 1
        return self.feature(), reward, err, self.t >= c.horizon


class TorchPolicyController(Controller):
    """Runs a torch policy on the numpy harness, rebuilding the same history features."""

    def __init__(self, act_fn, name: str, chunk: int = 1, privileged: bool = False):
        self.act_fn, self.name, self.chunk, self.privileged = act_fn, name, chunk, privileged

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        self.p = pop.C.shape[0]
        self.hist = np.zeros((HISTORY, self.p + pop.cfg.m))
        self.u_prev = np.zeros(pop.cfg.m)
        self.queue = []
        if hasattr(self.act_fn, "reset"):
            self.act_fn.reset()

    def act(self, o, info):
        self.hist = np.vstack([self.hist[1:], np.concatenate([o, self.u_prev])])
        if not self.queue:
            feat = torch.tensor(
                np.concatenate([self.hist.ravel(), self.z_star]), dtype=torch.float32
            )
            with torch.no_grad():
                out = self.act_fn(feat[None])[0].numpy().astype(float)
            self.queue = list(out.reshape(-1, self.cfg.m)[: self.chunk])
        u = np.clip(self.queue.pop(0), -self.cfg.u_max, self.cfg.u_max)
        self.u_prev = u
        return u


# ------------------------------------------------------------------ SAC
class Actor(nn.Module):
    def __init__(self, n_in, m, u_max):
        super().__init__()
        self.net, self.u_max = mlp(n_in, 2 * m), u_max

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s).chunk(2, -1)
        log_std = log_std.clamp(-5, 1)
        if deterministic:
            return self.u_max * torch.tanh(mu), None
        eps = torch.randn_like(mu)
        pre = mu + log_std.exp() * eps
        a = torch.tanh(pre)
        logp = (-0.5 * eps**2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(self.u_max * (1 - a**2) + 1e-6).sum(-1)
        return self.u_max * a, logp


@dataclass
class SACConfig:
    steps: int = 150_000
    n_envs: int = 32
    updates_per_step: int = 4
    batch: int = 256
    gamma: float = 0.95
    tau: float = 0.01
    lr: float = 3e-4
    start: int = 5_000
    reward_scale: float = 1.0
    seed: int = 0


def train_sac(cfg: PopulationConfig, sc: SACConfig | None = None, verbose=True):
    sc = sc or SACConfig()
    torch.manual_seed(sc.seed)
    pop = Population(cfg)
    sim = TorchSim(pop, sc.n_envs, seed=10_000 + sc.seed)
    n, m = sim.feat_dim, cfg.m
    actor = Actor(n, m, cfg.u_max)
    qs = nn.ModuleList([mlp(n + m, 1) for _ in range(2)])
    qt = nn.ModuleList([mlp(n + m, 1) for _ in range(2)])
    qt.load_state_dict(qs.state_dict())
    log_alpha = torch.zeros((), requires_grad=True)
    oa = torch.optim.Adam(actor.parameters(), lr=sc.lr)
    oq = torch.optim.Adam(qs.parameters(), lr=sc.lr)
    oal = torch.optim.Adam([log_alpha], lr=sc.lr)
    cap = 400_000
    S, Au, R, S2 = (torch.zeros(cap, d) for d in (n, m, 1, n))
    ptr = size = 0
    s = sim.reset()
    iters = sc.steps // sc.n_envs
    for it in range(iters):
        with torch.no_grad():
            if size < sc.start:
                a = cfg.u_max * (2 * torch.rand(sc.n_envs, m) - 1) * 0.5
            else:
                a, _ = actor(s)
        s2, r, _, done = sim.step(a)
        idx = torch.arange(ptr, ptr + sc.n_envs) % cap
        S[idx], Au[idx], R[idx, 0], S2[idx] = s, a, sc.reward_scale * r, s2
        ptr, size = (ptr + sc.n_envs) % cap, min(size + sc.n_envs, cap)
        s = sim.reset() if done else s2
        if size < sc.start:
            continue
        for _ in range(sc.updates_per_step):
            b = torch.randint(0, size, (sc.batch,))
            s_b, a_b, r_b, s2_b = S[b], Au[b], R[b], S2[b]
            alpha = log_alpha.exp().detach()
            with torch.no_grad():  # time-limit truncation only: always bootstrap
                a2, logp2 = actor(s2_b)
                sa2 = torch.cat([s2_b, a2], -1)
                q_next = torch.min(qt[0](sa2), qt[1](sa2)) - alpha * logp2[:, None]
                target = r_b + sc.gamma * q_next
            sa = torch.cat([s_b, a_b], -1)
            lq = sum(F.mse_loss(q(sa), target) for q in qs)
            oq.zero_grad()
            lq.backward()
            oq.step()
            a_new, logp = actor(s_b)
            san = torch.cat([s_b, a_new], -1)
            la = (alpha * logp[:, None] - torch.min(qs[0](san), qs[1](san))).mean()
            oa.zero_grad()
            la.backward()
            oa.step()
            lal = -(log_alpha * (logp.detach() + m).mean())
            oal.zero_grad()
            lal.backward()
            oal.step()
            with torch.no_grad():
                for p, pt in zip(qs.parameters(), qt.parameters()):
                    pt.lerp_(p, sc.tau)
        if verbose and it % max(1, iters // 5) == 0:
            print(f"    SAC step {it * sc.n_envs:>7d}  reward {r.mean().item():.3f}", flush=True)
    actor.eval()
    return TorchPolicyController(lambda f: actor(f, deterministic=True)[0], "SAC")


# ------------------------------------------------------------------ TD-MPC2-style
def simnorm(z, group=8):
    shp = z.shape
    return F.softmax(z.view(*shp[:-1], -1, group), -1).view(shp)


class TDMPCModel(nn.Module):
    def __init__(self, n_in, m, u_max, latent=64):
        super().__init__()
        self.enc = mlp(n_in, latent)
        self.dyn = mlp(latent + m, latent)
        self.rew = mlp(latent + m, 1)
        self.qs = nn.ModuleList([mlp(latent + m, 1) for _ in range(2)])
        self.pi = Actor(latent, m, u_max)

    def encode(self, s):
        return simnorm(self.enc(s))

    def next(self, z, a):
        return simnorm(self.dyn(torch.cat([z, a], -1)))

    def q(self, z, a, qs=None):
        za = torch.cat([z, a], -1)
        qs = qs or self.qs
        return torch.min(qs[0](za), qs[1](za))


@dataclass
class TDMPCConfig:
    steps: int = 100_000
    n_envs: int = 32
    updates_per_step: int = 2
    batch: int = 256
    horizon: int = 3  # model rollout length in training and planning
    gamma: float = 0.95
    tau: float = 0.01
    lr: float = 3e-4
    target_entropy_scale: float = 1.0  # auto-tuned entropy weight, target -scale*m (as SAC)
    rho: float = 0.5  # temporal discount of the multi-step losses
    start: int = 5_000
    seed: int = 0


class LatentPlanner:
    """MPPI over action sequences in a learned latent model, with a terminal value.

    ``model_step(z, a) -> (z', reward)``, ``terminal(z, a) -> value`` and
    ``prior(z) -> actions`` (optional policy samples).
    """

    def __init__(
        self,
        encode,
        model_step,
        terminal,
        m,
        u_max,
        horizon,
        prior=None,
        samples=256,
        n_prior=24,
        iters=5,
        elites=32,
        temp=0.5,
        init_std=0.5,
        stationarity=0.0,
        knots=None,
    ):
        # knots: perturbations piecewise constant over this many blocks (as in MPPI*);
        # sampling all H x m numbers independently is hopeless for long horizons.
        self.knots = knots
        # Terminal penalty w * ||z_H - z_{H-1}||^2: a held set-point is an equilibrium.
        # A value-free substitute for a terminal cost when no Q function is learned.
        self.init_std, self.stationarity = init_std, stationarity
        self.encode, self.model_step, self.terminal, self.prior = (
            encode,
            model_step,
            terminal,
            prior,
        )
        self.m, self.u_max, self.H = m, u_max, horizon
        self.samples, self.n_prior, self.iters, self.elites, self.temp = (
            samples,
            n_prior,
            iters,
            elites,
            temp,
        )
        self.reset()

    def reset(self):
        self.mean = torch.zeros(self.H, self.m)

    @torch.no_grad()
    def __call__(self, feat):
        z0 = self.encode(feat)  # (1, latent)
        mean, std = self.mean.clone(), torch.full((self.H, self.m), self.init_std * self.u_max)
        for _ in range(self.iters):
            if self.knots:
                eps = torch.randn(self.samples, self.knots, self.m)
                eps = eps.repeat_interleave(-(-self.H // self.knots), 1)[:, : self.H]
            else:
                eps = torch.randn(self.samples, self.H, self.m)
            U = (mean + std * eps).clamp(-self.u_max, self.u_max)
            if self.prior is not None:
                zp, Up = z0.expand(self.n_prior, -1), []
                for _k in range(self.H):
                    a = self.prior(zp)
                    Up.append(a)
                    zp, _ = self.model_step(zp, a)
                U = torch.cat([U, torch.stack(Up, 1)], 0)
            z, G = z0.expand(U.shape[0], -1), torch.zeros(U.shape[0])
            for k in range(self.H):
                z_prev = z
                z, r = self.model_step(z, U[:, k])
                G += r
            G += self.terminal(z, U[:, -1])
            if self.stationarity:
                G -= self.stationarity * ((z - z_prev) ** 2).sum(-1)
            top = G.topk(self.elites).indices
            w = torch.softmax((G[top] - G[top].max()) / (self.temp * (G[top].std() + 1e-6)), 0)
            mean = (w[:, None, None] * U[top]).sum(0)
            std = ((w[:, None, None] * (U[top] - mean) ** 2).sum(0)).sqrt().clamp(0.05, self.u_max)
        self.mean = torch.cat([mean[1:], mean[-1:]], 0)
        return mean[:1]


def train_tdmpc(cfg: PopulationConfig, tc: TDMPCConfig | None = None, verbose=True):
    tc = tc or TDMPCConfig()
    torch.manual_seed(tc.seed)
    pop = Population(cfg)
    sim = TorchSim(pop, tc.n_envs, seed=20_000 + tc.seed)
    n, m, T, H = sim.feat_dim, cfg.m, cfg.horizon, tc.horizon
    model = TDMPCModel(n, m, cfg.u_max)
    target_qs = nn.ModuleList([mlp(64 + m, 1) for _ in range(2)])
    target_qs.load_state_dict(model.qs.state_dict())
    opt = torch.optim.Adam(
        [p for k, p in model.named_parameters() if not k.startswith("pi.")], lr=tc.lr
    )
    opt_pi = torch.optim.Adam(model.pi.parameters(), lr=tc.lr)
    log_alpha = torch.zeros((), requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=tc.lr)
    # Episode buffer: states (N_ep, T+1, n), actions/rewards (N_ep, T, .)
    eps_S, eps_A, eps_R = [], [], []
    s = sim.reset()
    cur_S, cur_A, cur_R = [s], [], []
    iters = tc.steps // tc.n_envs
    for it in range(iters):
        with torch.no_grad():
            if it * tc.n_envs < tc.start:
                a = cfg.u_max * 0.5 * (2 * torch.rand(tc.n_envs, m) - 1)
            else:
                a, _ = model.pi(model.encode(s))
        s, r, _, done = sim.step(a)
        cur_S.append(s)
        cur_A.append(a)
        cur_R.append(r)
        if done:
            eps_S.append(torch.stack(cur_S, 1))
            eps_A.append(torch.stack(cur_A, 1))
            eps_R.append(torch.stack(cur_R, 1))
            BS, BA, BR = torch.cat(eps_S), torch.cat(eps_A), torch.cat(eps_R)
            s = sim.reset()
            cur_S, cur_A, cur_R = [s], [], []
        if not eps_S or it * tc.n_envs < tc.start:
            continue
        for _ in range(tc.updates_per_step):
            e = torch.randint(0, BS.shape[0], (tc.batch,))
            t0 = torch.randint(0, T - H, (tc.batch,))
            ts = t0[:, None] + torch.arange(H + 1)
            obs = BS[e[:, None], ts]  # (b, H+1, n)
            act = BA[e[:, None], ts[:, :H]]
            rew = BR[e[:, None], ts[:, :H]]
            with torch.no_grad():
                z_next = model.encode(obs[:, 1:])
                a_next, logp_next = model.pi(z_next)
                alpha = log_alpha.exp()
                q_target = rew + tc.gamma * (
                    model.q(z_next, a_next, target_qs)[..., 0] - alpha * logp_next
                )
            z = model.encode(obs[:, 0])
            loss, zs = 0.0, []
            for k in range(H):
                w = tc.rho**k
                za = torch.cat([z, act[:, k]], -1)
                q_pred = sum(F.mse_loss(q(za)[:, 0], q_target[:, k]) for q in model.qs)
                loss = loss + w * (
                    20 * F.mse_loss(model.next(z, act[:, k]), z_next[:, k])
                    + 0.1 * F.mse_loss(model.rew(za)[:, 0], rew[:, k])
                    + 0.1 * q_pred
                )
                zs.append(z.detach())
                z = model.next(z, act[:, k])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20)
            opt.step()
            zd = torch.cat(zs)
            a_pi, logp = model.pi(zd)
            lp = (log_alpha.exp().detach() * logp - model.q(zd, a_pi)[:, 0]).mean()
            opt_pi.zero_grad()
            lp.backward()
            opt_pi.step()
            la = -(log_alpha * (logp.detach() + tc.target_entropy_scale * m).mean())
            opt_alpha.zero_grad()
            la.backward()
            opt_alpha.step()
            with torch.no_grad():
                for p, pt in zip(model.qs.parameters(), target_qs.parameters()):
                    pt.lerp_(p, tc.tau)
        if verbose and it % max(1, iters // 5) == 0:
            print(f"    TD-MPC step {it * tc.n_envs:>7d}  reward {r.mean().item():.3f}", flush=True)
    model.eval()

    def model_step(z, a):
        za = torch.cat([z, a], -1)
        return model.next(z, a), model.rew(za)[:, 0]

    planner = LatentPlanner(
        model.encode,
        model_step,
        terminal=lambda z, a: model.q(z, model.pi(z, deterministic=True)[0])[:, 0],
        m=m,
        u_max=cfg.u_max,
        horizon=tc.horizon,
        prior=lambda z: model.pi(z)[0],
        init_std=0.2,
    )
    ctrl = TorchPolicyController(planner, "TD-MPC2-style")
    ctrl.model = model
    ctrl.prior_controller = TorchPolicyController(
        lambda f: model.pi(model.encode(f), deterministic=True)[0], "TD-MPC2-style (policy prior)"
    )
    return ctrl


# ------------------------------------------------------------------ data for offline methods
def collect_random(cfg: PopulationConfig, episodes: int, seed: int, n_envs: int = 100):
    """Exploration episodes: i.i.d., held and random-walk stimulation (as in NeuroStim-Percept)."""
    pop = Population(cfg)
    sim = TorchSim(pop, n_envs, seed=seed)
    g = torch.Generator().manual_seed(seed)
    S, A, Obs = [], [], []
    for _ in range(max(1, episodes // n_envs)):
        s = sim.reset()
        amp = torch.rand(n_envs, 1, generator=g) * cfg.u_max * 0.6
        held = 2 * torch.rand(n_envs, cfg.m, generator=g) - 1
        walk = torch.zeros(n_envs, cfg.m)
        kind = torch.randint(0, 3, (n_envs, 1), generator=g)
        ss, aa, oo = [s], [], []
        for _t in range(cfg.horizon):
            walk = (walk + 0.3 * torch.randn(n_envs, cfg.m, generator=g)).clamp(-1, 1)
            iid = 2 * torch.rand(n_envs, cfg.m, generator=g) - 1
            a = amp * torch.where(kind == 0, iid, torch.where(kind == 1, held, walk))
            s, _, _, _ = sim.step(a)
            ss.append(s)
            aa.append(a)
            oo.append(sim.hist[:, -1, : sim.p].clone())  # newest observation
        S.append(torch.stack(ss, 1))
        A.append(torch.stack(aa, 1))
        Obs.append(torch.stack(oo, 1))
    return sim, torch.cat(S), torch.cat(A), torch.cat(Obs)


class MPPITeacher:
    """Batched MPPI on the true model of each env (privileged), used to make demonstrations."""

    def __init__(
        self, sim: TorchSim, horizon=15, samples=256, sigma=0.6, temp=0.1, iters=4, knots=3
    ):
        self.sim, self.H, self.K, self.sigma, self.temp = sim, horizon, samples, sigma, temp
        self.iters, self.knots = iters, knots

    def reset(self):
        c, sim = self.sim.cfg, self.sim
        self.U = torch.zeros(sim.E, self.H, c.m)
        Ps, xs = [], []
        u_op = np.full(c.m, 0.5 if c.recruit == "even" else 0.0)
        D = sim.pop.D
        for e in range(sim.E):
            A, B = sim.A[e].double().numpy(), sim.B[e].double().numpy()
            Bl = B * recruit_grad(c, u_op)[None, :]
            Ps.append(
                solve_discrete_are(A, Bl, D.T @ D + 1e-6 * np.eye(c.n), c.lam_u * np.eye(c.m))
            )
            xs.append(
                lq_tracker(A, Bl, D, np.zeros(c.n), sim.z[e].double().numpy(), c.lam_u, c.u_max)[0]
            )
        self.P = torch.tensor(np.stack(Ps), dtype=torch.float32)
        self.xss = torch.tensor(np.stack(xs), dtype=torch.float32)

    @torch.no_grad()
    def act(self):
        c, sim, E, K = self.sim.cfg, self.sim, self.sim.E, self.K
        A = sim.A[:, None].expand(E, K, c.n, c.n)
        B = (sim.B + sim.phi)[:, None].expand(E, K, c.n, c.m)
        for i in range(self.iters):
            eps = self.sigma * 0.5**i * torch.randn(E, K, self.knots, c.m)
            eps = eps.repeat_interleave(-(-self.H // self.knots), 2)[:, :, : self.H]
            eps[:, 0] = 0
            Us = (self.U[:, None] + eps).clamp(-c.u_max, c.u_max)
            X = sim.x[:, None].expand(E, K, c.n)
            J = c.lam_u * (Us**2).sum((2, 3))
            for k in range(self.H):
                X = sim.step_mean(X, Us[:, :, k], A, B)
                J += ((X @ sim.D.T - sim.z[:, None]) ** 2).sum(-1)
            dx = X - self.xss[:, None]
            J += torch.einsum("eki,eij,ekj->ek", dx, self.P, dx)
            Jn = J - J.min(1, keepdim=True).values
            w = torch.softmax(-Jn / (self.temp * (J.std(1, keepdim=True) + 1e-9)), 1)
            self.U = (w[:, :, None, None] * Us).sum(1)
        u = self.U[:, 0].clone()
        self.U = torch.cat([self.U[:, 1:], self.U[:, -1:]], 1)
        return u


def collect_demos(cfg: PopulationConfig, episodes: int, seed: int, n_envs: int = 32):
    """Teacher trajectories: (features, actions) with actions as chunks aligned per step."""
    pop = Population(cfg)
    sim = TorchSim(pop, n_envs, seed=seed)
    teacher = MPPITeacher(sim)
    S, A = [], []
    for _ in range(max(1, episodes // n_envs)):
        s = sim.reset()
        teacher.reset()
        ss, aa = [], []
        for _t in range(cfg.horizon):
            a = teacher.act()
            ss.append(s)
            aa.append(a)
            s, _, _, _ = sim.step(a)
        S.append(torch.stack(ss, 1))
        A.append(torch.stack(aa, 1))
    return torch.cat(S), torch.cat(A)


def chunk_dataset(S, A, chunk: int):
    """(feature_t, a_{t:t+chunk}) pairs within episodes."""
    N, T, _ = S.shape
    t = torch.arange(T - chunk + 1)
    feats = S[:, t].reshape(-1, S.shape[-1])
    acts = torch.stack([A[:, t + i] for i in range(chunk)], 2).reshape(len(feats), -1)
    return feats, acts


# ------------------------------------------------------------------ BC and diffusion policy
def train_bc(cfg, demos, chunk=8, execute=4, steps=5000, verbose=True, seed=0):
    torch.manual_seed(seed)
    X, Y = chunk_dataset(*demos, chunk)
    net = mlp(X.shape[1], Y.shape[1], hidden=512)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(steps):
        b = torch.randint(0, len(X), (256,))
        loss = F.mse_loss(net(X[b]), Y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    if verbose:
        print(f"    BC mse {loss.item():.4f}", flush=True)
    net.eval()
    return TorchPolicyController(net, "BC (MSE)", chunk=execute)


class Denoiser(nn.Module):
    def __init__(self, n_cond, n_act, hidden=512):
        super().__init__()
        self.cond = mlp(n_cond, 128)
        self.net = mlp(n_act + 128 + 64, n_act, hidden=hidden, layers=3)

    def forward(self, a, t, c):
        freqs = torch.exp(torch.linspace(0, math.log(1000), 32))
        emb = t[:, None].float() * freqs[None]
        temb = torch.cat([emb.sin(), emb.cos()], -1)
        return self.net(torch.cat([a, temb, self.cond(c)], -1))


class DiffusionSampler:
    """Deterministic DDIM sampling (Song et al. 2021) on a subsequence of the DDPM steps.

    The initial noise alone picks the mode; each denoising step is then
    deterministic, which removes the per-step action jitter of ancestral
    sampling.
    """

    def __init__(self, net, n_act, betas, u_max, steps: int = 10):
        self.net, self.n_act, self.u_max = net, n_act, u_max
        self.abar = torch.cumprod(1 - betas, 0)
        self.ts = torch.linspace(len(betas) - 1, 0, steps).round().long()

    @torch.no_grad()
    def __call__(self, c):
        a = torch.randn(c.shape[0], self.n_act)
        for i, t in enumerate(self.ts):
            ab = self.abar[t]
            eps = self.net(a, torch.full((c.shape[0],), int(t)), c)
            a0 = ((a - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            ab_prev = self.abar[self.ts[i + 1]] if i + 1 < len(self.ts) else torch.tensor(1.0)
            a = ab_prev.sqrt() * a0 + (1 - ab_prev).sqrt() * eps
        return a.clamp(-1, 1) * self.u_max


def train_diffusion(cfg, demos, chunk=8, execute=4, steps=15000, n_diff=50, verbose=True, seed=0):
    """DDPM (ε-prediction) over normalised action chunks, conditioned on the history features."""
    torch.manual_seed(seed)
    X, Y = chunk_dataset(*demos, chunk)
    Y = Y / cfg.u_max
    net = Denoiser(X.shape[1], Y.shape[1])
    betas = torch.linspace(1e-4, 0.05, n_diff)
    abar = torch.cumprod(1 - betas, 0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    for step in range(steps):
        b = torch.randint(0, len(X), (256,))
        t = torch.randint(0, n_diff, (256,))
        eps = torch.randn_like(Y[b])
        noisy = abar[t].sqrt()[:, None] * Y[b] + (1 - abar[t]).sqrt()[:, None] * eps
        loss = F.mse_loss(net(noisy, t, X[b]), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if verbose:
        print(f"    diffusion eps-mse {loss.item():.4f}", flush=True)
    net.eval()
    return TorchPolicyController(
        DiffusionSampler(net, Y.shape[1], betas, cfg.u_max), "Diffusion policy", chunk=execute
    )


# ------------------------------------------------------------------ JEPA-style + MPPI
def vicreg_terms(z, eps: float = 1e-4):
    """VICReg regularisers on a batch of embeddings ``(B, L)``.

    Variance: hinge keeping each dimension's std at >= 1 (prevents collapse to a
    constant). Covariance: squared off-diagonal covariance / L (prevents
    dimensional collapse, i.e. all dimensions encoding the same thing).
    """
    z = z - z.mean(0)
    std = (z.var(0) + eps).sqrt()
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return F.relu(1 - std).mean(), (off**2).sum() / z.shape[1]


def sigreg(z, n_dirs: int = 256, n_t: int = 17, t_max: float = 3.0, generator=None):
    """SIGReg (LeJEPA, Balestriero & LeCun 2025): push embeddings towards N(0, I).

    Sketched: project on ``n_dirs`` random unit directions, then run the
    Epps-Pulley test on each 1-D projection. That test compares the empirical
    characteristic function with the standard normal one, exp(-t^2/2):

        EP = n * integral |phi_n(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt

    (trapezoid rule on [0, t_max]; the integrand is even in t). A projection that
    is constant (collapse) or non-Gaussian (clusters, heavy tails, wrong scale)
    is penalised. VICReg constrains only the first two moments. Fresh directions
    each call make it cover all of them in expectation.
    """
    B, L = z.shape
    dirs = torch.randn(L, n_dirs, generator=generator)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    x = z @ dirs  # (B, n_dirs)
    t = torch.linspace(0, t_max, n_t)
    xt = x[..., None] * t  # (B, n_dirs, n_t)
    gauss = torch.exp(-0.5 * t**2)
    err = (torch.cos(xt).mean(0) - gauss) ** 2 + torch.sin(xt).mean(0) ** 2
    return (B * torch.trapezoid(err * gauss, t, dim=-1)).mean()


@torch.no_grad()
def collapse_stats(z):
    """Collapse diagnostics of embeddings ``(B, L)``.

    ``eff_rank`` is the exponential of the entropy of the normalised singular
    values (Roy & Vetterli 2007): L for isotropic embeddings, 1 when every
    dimension carries the same signal.
    """
    z = z - z.mean(0)
    sv = torch.linalg.svdvals(z)
    p = sv / sv.sum()
    zs = z / (z.std(0) + 1e-8)  # scale-free Gaussianity: shape only, not variance
    return {
        "std_min": float(z.std(0).min()),
        "std_mean": float(z.std(0).mean()),
        "eff_rank": float(torch.exp(-(p * torch.log(p + 1e-12)).sum())),
        # Epps-Pulley on fixed directions of the standardised embedding: ~0-1 for a
        # Gaussian sample, large for clustered or heavy-tailed ones.
        "nongauss": float(sigreg(zs, generator=torch.Generator().manual_seed(0))),
    }


@dataclass
class JEPAConfig:
    episodes: int = 1500
    steps: int = 6000
    latent: int = 16
    rollout: int = 5
    # None: stop-gradient on the online encoder (VICReg-style, as in PLDM). A float
    # gives an EMA target encoder (I-JEPA style).
    ema: float | None = None
    # VICReg weights (Bardes et al. 2022 use 25 / 25 / 1). The regularisers are applied
    # to EVERY embedding the loss touches: the encoder output at each of the
    # rollout + 1 timesteps (the prediction targets included) and each predictor
    # output. Regularising only the first encoding lets targets and predictions
    # shrink, which is exactly how a predictive loss collapses.
    # reg: "vicreg" (variance/covariance, 2nd moments), "sigreg" (LeJEPA: the full
    # distribution matched to an isotropic Gaussian), or "none" (the control:
    # nothing prevents collapse).
    reg: str = "vicreg"
    pred_w: float = 25.0
    var_w: float = 25.0
    cov_w: float = 1.0
    sigreg_w: float = 0.05  # LeJEPA's lambda: loss = (1 - lambda) pred + lambda SIGReg
    # Targets without gradient (PLDM / BYOL-style). LeJEPA drops it: SIGReg alone
    # prevents collapse, with no stop-gradient or teacher-student heuristics.
    stop_grad: bool = True
    reg_predictions: bool = True
    reg_all_encodings: bool = True  # False: only the first encoding (the old setup)
    # Read-out of the percept from the frozen latent, used by the planner's cost.
    # "mlp": the latent holds the percept nonlinearly (held-out MSE 0.010 with an MLP
    # probe vs 0.41 with a linear one, regime A), so a linear read-out misleads MPPI.
    probe: str = "mlp"
    probe_steps: int = 3000
    plan_horizon: int = 10
    plan_knots: int | None = None
    plan_stationarity: float = 0.0
    log_every: int = 1000
    seed: int = 0


def train_jepa(cfg: PopulationConfig, jc: JEPAConfig | None = None, verbose=True, log=None):
    """Train the JEPA-style model; ``log`` (a list) receives collapse diagnostics."""
    jc = jc or JEPAConfig()
    torch.manual_seed(jc.seed)
    sim, S, A, Obs = collect_random(cfg, jc.episodes, seed=30_000 + jc.seed)
    kh, p, m, L = sim.k, sim.p, cfg.m, jc.latent

    def obs_only(f):
        """Observation history only. With past actions in the input, a latent that
        just copies the action history is perfectly predictable from the next
        action: a shortcut that satisfies the JEPA loss without modelling the brain."""
        return f[..., : kh * (p + m)].unflatten(-1, (kh, p + m))[..., :p].flatten(-2)

    hist = obs_only(S)
    n = hist.shape[-1]
    enc, pred = mlp(n, L), mlp(L + m, L)
    enc_t = mlp(n, L)
    enc_t.load_state_dict(enc.state_dict())
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-4)
    N, T1, _ = hist.shape
    K = jc.rollout
    for step in range(jc.steps):
        e = torch.randint(0, N, (256,))
        t0 = torch.randint(0, T1 - K - 1, (256,))
        ts = t0[:, None] + torch.arange(K + 1)
        Z = enc(hist[e[:, None], ts])  # (B, K+1, L): every encoding, with gradient
        if jc.ema:
            with torch.no_grad():
                tgt = enc_t(hist[e[:, None], ts[:, 1:]])
        else:
            tgt = Z[:, 1:].detach() if jc.stop_grad else Z[:, 1:]
        s, preds = Z[:, 0], []
        for k in range(K):
            s = pred(torch.cat([s, A[e, t0 + k]], -1))
            preds.append(s)
        preds = torch.stack(preds, 1)  # (B, K, L)
        pred_loss = F.mse_loss(preds, tgt)
        embs = [Z[:, k] for k in range(K + 1 if jc.reg_all_encodings else 1)]
        if jc.reg_predictions:
            embs += [preds[:, k] for k in range(K)]
        zero = torch.zeros(())
        var_loss = cov_loss = sig_loss = zero
        if jc.reg == "vicreg":
            regs = [vicreg_terms(z_) for z_ in embs]
            var_loss = torch.stack([r[0] for r in regs]).mean()
            cov_loss = torch.stack([r[1] for r in regs]).mean()
            loss = jc.pred_w * pred_loss + jc.var_w * var_loss + jc.cov_w * cov_loss
        elif jc.reg == "sigreg":
            sig_loss = torch.stack([sigreg(z_) for z_ in embs]).mean()
            loss = (1 - jc.sigreg_w) * pred_loss + jc.sigreg_w * sig_loss
        elif jc.reg == "none":
            loss = pred_loss
        else:
            raise ValueError(jc.reg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if jc.ema:
            with torch.no_grad():
                for w_on, w_tgt in zip(enc.parameters(), enc_t.parameters()):
                    w_tgt.lerp_(w_on, 1 - jc.ema)
        if log is not None and (step % jc.log_every == 0 or step == jc.steps - 1):
            entry = {
                "step": step,
                "pred": pred_loss.item(),
                "var": var_loss.item(),
                "cov": cov_loss.item(),
                "sigreg": sig_loss.item(),
                **collapse_stats(Z[:, 0].detach()),
                **{f"pred_{k_}": v for k_, v in collapse_stats(preds[:, -1].detach()).items()},
            }
            log.append(entry)
            if verbose:
                print(
                    "    JEPA " + "  ".join(f"{k_} {v:.3g}" for k_, v in entry.items()), flush=True
                )
    # Probe latent -> observed percept, on the frozen encoder (the only place the
    # percept is used). Fitted on 90 % of the episodes, scored on the rest.
    with torch.no_grad():
        Z = enc(hist[:, 1:]).reshape(-1, L)
        P3 = Obs @ (sim.D.T if cfg.observe == "full" else torch.eye(Obs.shape[-1]))
        Pobs = P3.reshape(-1, P3.shape[-1])
    n_fit = int(0.9 * N) * (T1 - 1)
    if jc.probe == "linear":
        with torch.no_grad():
            Z1 = torch.cat([Z, torch.ones(len(Z), 1)], 1)
            W = torch.linalg.lstsq(Z1[:n_fit], Pobs[:n_fit]).solution

        def probe(z):
            return torch.cat([z, torch.ones(len(z), 1)], 1) @ W
    else:
        probe_net = mlp(L, Pobs.shape[1], hidden=128)
        opt_p = torch.optim.Adam(probe_net.parameters(), lr=1e-3)
        for _ in range(jc.probe_steps):
            b = torch.randint(0, n_fit, (256,))
            lp = F.mse_loss(probe_net(Z[b]), Pobs[b])
            opt_p.zero_grad()
            lp.backward()
            opt_p.step()
        probe_net.eval()
        probe = probe_net
    with torch.no_grad():
        probe_mse = F.mse_loss(probe(Z[n_fit:]), Pobs[n_fit:]).item()
    if verbose:
        print(f"    JEPA pred loss {pred_loss.item():.4f}  probe mse {probe_mse:.4f}", flush=True)
    if log is not None:
        log.append({"probe_mse": probe_mse, "percept_var": float(Pobs.var(0).mean())})
    enc.eval()
    pred.eval()

    class Plan:
        def __init__(self):
            self.z_star = None
            self.planner = LatentPlanner(
                lambda f: enc(obs_only(f)),
                self.model_step,
                terminal=lambda z, a: torch.zeros(z.shape[0]),
                m=m,
                u_max=cfg.u_max,
                horizon=jc.plan_horizon,
                knots=jc.plan_knots,
                stationarity=jc.plan_stationarity,
            )

        def model_step(self, s, a):
            s2 = pred(torch.cat([s, a], -1))
            zp = probe(s2)
            return s2, -((zp - self.z_star) ** 2).sum(-1) - cfg.lam_u * (a**2).sum(-1)

        def reset(self):
            self.planner.reset()

        def __call__(self, feat):
            self.z_star = feat[0, -cfg.dz :]
            return self.planner(feat)

    ctrl = TorchPolicyController(Plan(), "JEPA-style + MPPI")
    ctrl.enc, ctrl.pred, ctrl.probe, ctrl.obs_only = enc, pred, probe, obs_only  # diagnostics
    return ctrl
