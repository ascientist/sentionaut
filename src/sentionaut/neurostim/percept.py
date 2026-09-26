"""NeuroStim-Percept: learn stimulation for a target image through a learned world model.

Pipeline::

    target image y ─► policy π ─► a_{1:T} ─► brain (NeuroStim dynamics) ─► x_t ─► D(x_t) = percept
                                     └─────► world model W (learned) ─► ẑ_t ─► D(ẑ_t)

- **Brain** (ground truth, a black box to the learner): the NeuroStim dynamics
  ``x_{t+1} = A x_t - αx³ + B diag(1/(1+λh)) tanh(a) + w`` lifted to a
  ``d``-dimensional latent that is the code space of a frozen MNIST
  autoencoder. The percept is ``D(x_t)``. ``B`` is the patient.
- **World model** ``W``: GRU trained only on observed (stimulation, percept)
  episodes. From the resting percept (and, for random patients, a calibration
  probe) it predicts the latent trajectory, decoded by the same ``D``.
  ``D`` stands in for a known ``brain2vision`` readout, like the repo's percept
  models. The patient-specific part, stimulation → neural state, is learned.
- **Policy** ``π(E(y), context) → a_{1:T}``: open-loop stimulation sequence,
  trained by backprop through ``W`` on the reconstruction loss
  ``Σ_{t ≥ t_hold} ||D(ẑ_t) − y||²``. No RL and no gradient through the brain.
  It is evaluated on the true brain.

Reference: the same policy trained by backprop through the *true* brain
(privileged, differentiable simulator). The gap between the two is the price
of the world model.
"""

from __future__ import annotations

import gzip
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def load_mnist(cache: str | Path = "data/mnist") -> dict[str, torch.Tensor]:
    """MNIST as float images in [0, 1] ``(N, 784)`` and int labels; cached on disk."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, fname in MNIST_FILES.items():
        path = cache / fname
        if not path.exists():
            urllib.request.urlretrieve(MNIST_URL + fname, path)
        with gzip.open(path, "rb") as f:
            raw = np.frombuffer(f.read(), np.uint8)
        if key.endswith("_x"):
            out[key] = torch.tensor(raw[16:].reshape(-1, 784), dtype=torch.float32) / 255.0
        else:
            out[key] = torch.tensor(raw[8:], dtype=torch.long)
    return out


# ------------------------------------------------------------ vision models
class AutoEncoder(nn.Module):
    """MNIST autoencoder whose code is the brain's latent neural space.

    Trained with Gaussian noise on the code and a small L2 penalty. The
    noise keeps codes O(1) and makes the decoder smooth around the data
    manifold, which matters because stimulation lands the latent off-manifold.
    """

    def __init__(self, d: int = 16, hidden: int = 512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(784, hidden), nn.GELU(), nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, d)
        )
        self.dec = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, hidden), nn.GELU(), nn.Linear(hidden, 784)
        )

    def encode(self, y):
        return self.enc(y)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))


class Classifier(nn.Module):
    """Frozen digit classifier used as a "is the percept recognisable" readout."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, y):
        return self.net(y)


def _fit(model, loss_fn, x, y, epochs, batch=256, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=g).split(batch):
            loss = loss_fn(model, x[idx], y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval().requires_grad_(False)
    return model


def train_autoencoder(
    x: torch.Tensor, d: int = 16, epochs: int = 10, code_noise: float = 0.3, l2: float = 1e-2
) -> AutoEncoder:
    def loss(m, xb, _):
        z = m.encode(xb)
        rec = m.decode(z + code_noise * torch.randn_like(z))
        return F.binary_cross_entropy(rec, xb) + l2 * z.pow(2).mean()

    return _fit(AutoEncoder(d), loss, x, x, epochs)


def train_classifier(x: torch.Tensor, y: torch.Tensor, epochs: int = 3) -> Classifier:
    return _fit(Classifier(), lambda m, xb, yb: F.cross_entropy(m(xb), yb), x, y, epochs)


# ------------------------------------------------------------ ground truth
@dataclass
class BrainConfig:
    d: int = 16  # latent neural dimensions (= autoencoder code size)
    m: int = 24  # electrodes
    horizon: int = 30
    hold_from: int = 10  # reconstruction loss / evaluation window t >= hold_from
    persistence: float = 0.8  # diagonal of A
    coupling: float = 0.03  # off-diagonal std of A
    cubic: float = 0.02
    b_std: float = 0.3  # recruitment matrix entry std
    process_noise: float = 0.02
    obs_noise: float = 0.02  # pixel noise on observed percepts
    x0_std: float = 0.1
    adapt_decay: float = 0.95
    adapt_rate: float = 0.05
    adapt_strength: float = 2.0
    a_max: float = 1.0
    q_max: float = 8.0  # total charge per step, sum_i |a_i|
    h_max: float = 0.8
    probe_amp: float = 1.0  # calibration sweep: one pulse per electrode, from rest


class Brain:
    """Batched, differentiable ground-truth dynamics; one ``B`` per batch row."""

    def __init__(self, cfg: BrainConfig, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        d = cfg.d
        A = cfg.persistence * torch.eye(d) + cfg.coupling * torch.randn(d, d, generator=g)
        self.A = A * min(1.0, 0.95 / torch.linalg.eigvals(A).abs().max().item())
        # Calibration sweep, the same for every patient: pulse each electrode once.
        # m pulses give d*m response numbers, enough to identify the d x m matrix B.
        self.probe_actions = cfg.probe_amp * torch.eye(cfg.m)

    def sample_patients(self, n: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return self.cfg.b_std * torch.randn(n, self.cfg.d, self.cfg.m, generator=g)

    def rollout(self, B, actions, seed: int | None = None, noise: bool = True):
        """``actions (N, T, m)`` → latents ``(N, T+1, d)`` and adaptation ``(N, T+1, m)``."""
        c = self.cfg
        N, T, _ = actions.shape
        g = torch.Generator().manual_seed(seed) if seed is not None else None

        def randn(*shape):
            return torch.randn(*shape, generator=g) if noise else torch.zeros(*shape)

        x = c.x0_std * randn(N, c.d)
        h = torch.zeros(N, c.m)
        xs, hs = [x], [h]
        for t in range(T):
            a = actions[:, t]
            drive = torch.tanh(a) / (1 + c.adapt_strength * h)
            x = x @ self.A.T - c.cubic * x**3 + torch.einsum("ndm,nm->nd", B, drive)
            x = x + c.process_noise * randn(N, c.d)
            h = c.adapt_decay * h + c.adapt_rate * a.abs()
            xs.append(x)
            hs.append(h)
        return torch.stack(xs, 1), torch.stack(hs, 1)

    def observe(self, ae: AutoEncoder, xs, seed: int | None = None):
        """Noisy percept images ``(N, T+1, 784)``: what a learner is allowed to see."""
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        img = ae.decode(xs)
        return (img + self.cfg.obs_noise * torch.randn(img.shape, generator=g)).clamp(0, 1)

    def calibrate(self, ae: AutoEncoder, B, seed: int | None = None) -> torch.Tensor:
        """Probe sweep: pulse each electrode once from rest (resting in between).

        Returns the encoded percept change per pulse, ``E(o_after) - E(o_rest)``,
        flattened to ``(N, m*d)``. Each entry is a noisy view of one column of ``B``.
        """
        m = self.cfg.m
        out = []
        for i, Bc in enumerate(B.split(1024)):  # chunked: N*m decoded images add up fast
            n = Bc.shape[0]
            Bm = Bc.repeat_interleave(m, 0)  # (n*m, d, m): one 1-step rollout per pulse
            P = self.probe_actions.repeat(n, 1)[:, None]  # (n*m, 1, m)
            s = None if seed is None else seed + i
            with torch.no_grad():
                xs, _ = self.rollout(Bm, P, seed=s)
                z = ae.encode(self.observe(ae, xs, seed=s))
            out.append((z[:, 1] - z[:, 0]).view(n, m * self.cfg.d))
        return torch.cat(out)


def project_charge(a: torch.Tensor, q_max: float) -> torch.Tensor:
    """Differentiably scale each step so ``sum_i |a_i| <= q_max``."""
    q = a.abs().sum(-1, keepdim=True)
    return a * torch.clamp(q_max / (q + 1e-8), max=1.0)


def exploration_actions(cfg: BrainConfig, n: int, seed: int) -> torch.Tensor:
    """Mixture that covers what a policy may emit: i.i.d. noise, held patterns, smooth drifts."""
    g = torch.Generator().manual_seed(seed)
    T, m = cfg.horizon, cfg.m
    amp = torch.rand(n, 1, 1, generator=g)
    iid = torch.rand(n, T, m, generator=g) * 2 - 1
    held = (torch.rand(n, 1, m, generator=g) * 2 - 1).expand(n, T, m)
    walk = torch.cumsum(0.3 * torch.randn(n, T, m, generator=g), 1).clamp(-1, 1)
    kind = torch.randint(0, 3, (n, 1, 1), generator=g)
    a = torch.where(kind == 0, iid, torch.where(kind == 1, held, walk)) * amp
    return project_charge(a * cfg.a_max, cfg.q_max)


# ------------------------------------------------------------ world model
class ContextHyper(nn.Module):
    """Calibration record → a patient-specific ``m x k`` action embedding (+ a vector).

    Stimulation acts through ``B a``: the effect of an action is *multiplied*
    by the patient. Concatenating the context with the action makes an MLP
    learn that product from scratch. A hypernetwork emits the product's
    matrix directly, which is the natural inductive bias.
    """

    def __init__(self, n_ctx: int, m: int, k: int, hidden: int = 256):
        super().__init__()
        self.m, self.k = m, k
        self.trunk = nn.Sequential(nn.Linear(n_ctx, hidden), nn.GELU())
        self.W = nn.Linear(hidden, m * k)
        self.vec = nn.Linear(hidden, 64)

    def forward(self, ctx):
        f = self.trunk(ctx)
        return self.W(f).view(-1, self.m, self.k) / self.m**0.5, self.vec(f)


class WorldModel(nn.Module):
    """``(z_0, context, a_{1:T}) → ẑ_{1:T}``: open-loop latent prediction with a GRU.

    The GRU state carries the neural state and the adaptation. With a
    context, actions enter through the patient-specific embedding
    ``a W(context)``.
    """

    def __init__(self, d: int, m: int, n_ctx: int = 0, hidden: int = 256, k: int = 64):
        super().__init__()
        self.n_ctx = n_ctx
        self.hyper = ContextHyper(n_ctx, m, k) if n_ctx else None
        c = 64 if n_ctx else 0
        self.init = nn.Sequential(nn.Linear(d + c, hidden), nn.Tanh())
        self.cell = nn.GRUCell((k if n_ctx else m) + c, hidden)
        self.readout = nn.Linear(hidden, d)

    def forward(self, z0, actions, ctx=None):
        if self.hyper is not None:
            W, c = self.hyper(ctx)
            emb = torch.einsum("ntm,nmk->ntk", actions, W)
            actions = torch.cat([emb, c[:, None].expand(-1, actions.shape[1], -1)], -1)
        else:
            c = z0[:, :0]
        h = self.init(torch.cat([z0, c], -1))
        zs = []
        for t in range(actions.shape[1]):
            h = self.cell(actions[:, t], h)
            zs.append(self.readout(h))
        return torch.stack(zs, 1)


def collect(brain: Brain, ae: AutoEncoder, B, actions, seed: int, calibrate: bool):
    """Stimulate the true brain; keep only what is observable (percepts, actions, probe)."""
    with torch.no_grad():
        xs, hs = brain.rollout(B, actions, seed=seed)
        obs = brain.observe(ae, xs, seed=seed + 1)
    ctx = brain.calibrate(ae, B, seed=seed + 2) if calibrate else None
    return {"obs": obs, "actions": actions, "ctx": ctx}


def train_world_model(
    data: dict,
    ae: AutoEncoder,
    cfg: BrainConfig,
    n_ctx: int = 0,
    steps: int = 3000,
    batch: int = 256,
    lr: float = 1e-3,
    wm: WorldModel | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> WorldModel:
    """Fit ``W`` by image-space MSE of its decoded open-loop predictions."""
    torch.manual_seed(seed)
    wm = wm or WorldModel(cfg.d, cfg.m, n_ctx)
    wm.train().requires_grad_(True)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    obs, act, ctx = data["obs"], data["actions"], data["ctx"]
    z0 = ae.encode(obs[:, 0])
    n = len(obs)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,))
        pred = ae.decode(wm(z0[idx], act[idx], None if ctx is None else ctx[idx]))
        loss = F.mse_loss(pred, obs[idx, 1:])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (step % 1000 == 0 or step == steps - 1):
            print(f"    world model step {step:>5d}  image mse {loss.item():.4f}")
    wm.eval().requires_grad_(False)
    return wm


# ------------------------------------------------------------ policy
class StimPolicy(nn.Module):
    """``(E(y), context) → a_{1:T}``: open-loop stimulation sequence for a target image.

    With a context, the network outputs ``k`` features per step and the
    patient-specific matrix ``W(context)`` maps them to the ``m`` electrodes.
    This mirrors the world model: ``a ≈ W u``, with ``W`` playing the role of
    a learned pseudo-inverse of ``B``.
    """

    def __init__(self, cfg: BrainConfig, n_ctx: int = 0, hidden: int = 512, k: int = 64):
        super().__init__()
        self.cfg, self.n_ctx, self.k = cfg, n_ctx, k
        self.hyper = ContextHyper(n_ctx, cfg.m, k) if n_ctx else None
        n_out = k if n_ctx else cfg.m
        self.net = nn.Sequential(
            nn.Linear(cfg.d + (64 if n_ctx else 0), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.horizon * n_out),
        )

    def forward(self, z_target, ctx=None):
        T, m = self.cfg.horizon, self.cfg.m
        if self.hyper is None:
            u = self.net(z_target).view(-1, T, m)
        else:
            W, c = self.hyper(ctx)
            feats = self.net(torch.cat([z_target, c], -1)).view(-1, T, self.k)
            u = torch.einsum("ntk,nmk->ntm", feats, W)
        return project_charge(self.cfg.a_max * torch.tanh(u), self.cfg.q_max)


def recon_loss(percepts, y, hold_from: int):
    """Mean pixel MSE over the hold window; ``percepts (N, T, 784)`` for t = 1..T."""
    return (percepts[:, hold_from - 1 :] - y[:, None]).pow(2).mean()


def train_policy(
    simulate,
    ae: AutoEncoder,
    images: torch.Tensor,
    cfg: BrainConfig,
    sample_patient,
    n_ctx: int = 0,
    steps: int = 2000,
    batch: int = 256,
    lr: float = 1e-3,
    charge_weight: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> StimPolicy:
    """Backprop the reconstruction loss through ``simulate(B, ctx, actions) → percepts``.

    ``simulate`` is either the world model (the method) or the true brain (the
    privileged reference). ``sample_patient(batch, step) → (B, ctx)`` supplies
    the patient: fixed, or a fresh random one per sample.
    """
    torch.manual_seed(seed)
    pi = StimPolicy(cfg, n_ctx)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        idx = torch.randint(0, len(images), (batch,))
        y = images[idx]
        B, ctx = sample_patient(batch, step)
        a = pi(ae.encode(y), ctx)
        loss = recon_loss(simulate(B, ctx, a), y, cfg.hold_from)
        loss = loss + charge_weight * a.abs().sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print(f"    policy step {step:>5d}  loss {loss.item():.4f}")
    pi.eval().requires_grad_(False)
    return pi


def wm_simulator(wm: WorldModel, brain: Brain, ae: AutoEncoder):
    """Percepts predicted by ``W`` from the (observed) resting percept."""

    def simulate(B, ctx, a):
        rest = ae.decode(brain.cfg.x0_std * torch.randn(a.shape[0], brain.cfg.d))
        return ae.decode(wm(ae.encode(rest), a, ctx))

    return simulate


def true_simulator(brain: Brain, ae: AutoEncoder):
    """Privileged: differentiate through the true brain (noise included)."""

    def simulate(B, ctx, a):
        xs, _ = brain.rollout(B, a)
        return ae.decode(xs[:, 1:])

    return simulate


@torch.no_grad()
def evaluate(pi, brain: Brain, ae: AutoEncoder, clf: Classifier, y, labels, B, ctx, seed=0):
    """Stimulate the true brain with ``π``; score the percepts it produces."""
    a = (
        pi(ae.encode(y), ctx)
        if pi is not None
        else torch.zeros(len(y), brain.cfg.horizon, brain.cfg.m)
    )
    xs, hs = brain.rollout(B, a, seed=seed)
    percepts = ae.decode(xs[:, 1:])
    hold = percepts[:, brain.cfg.hold_from - 1 :]
    return {
        "mse": float((hold - y[:, None]).pow(2).mean()),
        "accuracy": float((clf(hold.mean(1)).argmax(-1) == labels).float().mean()),
        "charge": float(a.abs().sum(-1).mean()),
        "h_violation": float((hs[:, 1:] > brain.cfg.h_max).any(-1).float().mean()),
        "percepts": percepts,
        "actions": a,
    }
