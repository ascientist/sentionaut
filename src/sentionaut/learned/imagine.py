"""Train a stimulus encoder through a learned world model (SuperMimic-style).

The loop follows Liao et al. 2026 ("SuperMimic", arXiv 2610.28480):

1. collect teacher transitions with the current encoder plus action noise into
   a replay buffer;
2. fit a differentiable world model on k-step autoregressive windows seeded at
   logged states;
3. improve the encoder by backpropagating the imagined episode loss through
   the frozen world model.

The teacher is one of the differentiable percept models (``axonmap`` or
``dynaphos``). Because the ports are differentiable, the encoder can also be
trained by backpropagating through the teacher itself, which the robotics
paper could not do; ``teacher-bptt`` is that control. ``reinforce-sim`` and
``reinforce-wm`` are score-function baselines on the teacher and on the world
model.

The world model is not the paper's MLP. Electrodes are tokens with a recurrent
memory each (Dynaphos charge and activation are per-electrode states), and
image patches cross-attend to the electrode tokens with a learned null key.

Docs: https://ascientist.github.io/sentionaut/imagination/
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..core.config import Config
from ..core.registry import build_components
from ..encoders.vision import ClosedLoopVisionEncoder, PhospheneMapEncoder, probe_centroids


# --------------------------------------------------------------------- task
@dataclass
class TaskSpec:
    model: str = "axonmap"
    implant: str = "argusii"
    xrange: tuple[float, float] = (-15.0, 15.0)
    yrange: tuple[float, float] = (-10.0, 10.0)
    xystep: float = 1.0
    a_max: float = 3.0  # axon map: x threshold; dynaphos: µA
    a_min: float = 0.5  # executed amplitude below this is set to 0
    cutoff: float | None = 5.0  # axon map: sparse path, max rel. error ~2.5e-6
    horizon: int = 10  # control steps per target (20 ms each)
    charge_weight: float = 0.02
    loss_from: int = 2  # score percepts from this step on

    @staticmethod
    def preset(name: str) -> "TaskSpec":
        if name == "axonmap":
            return TaskSpec()
        if name == "dynaphos":
            return TaskSpec(
                model="dynaphos",
                implant="orion",
                xrange=(-4.0, 1.0),
                yrange=(-2.5, 2.5),
                xystep=0.2,
                a_max=300.0,
                a_min=30.0,
            )
        raise ValueError(name)


class Teacher:
    """Batched differentiable percept model behind one ``step(state, amp)`` call."""

    def __init__(self, spec: TaskSpec, device: torch.device | None = None):
        self.spec = spec
        cfg = Config(
            model=spec.model,
            implant=spec.implant,
            xrange=spec.xrange,
            yrange=spec.yrange,
            xystep=spec.xystep,
        )
        self.device = device or torch.device("cpu")
        self.implant, self.topo, self.model = build_components(cfg, device=self.device)
        self.n_elec = self.implant.n_electrodes
        self.grid = tuple(self.topo.grid_shape)
        self.n_phys = 3 if spec.model == "dynaphos" else 0
        self.cutoff = spec.cutoff
        self.scale = 1.0
        self.scale = self._calibrate_scale()

    def execute(self, amp: torch.Tensor) -> torch.Tensor:
        """Amplitudes the implant delivers: clipped and gated below ``a_min``."""
        amp = amp.clamp(0.0, self.spec.a_max)
        return amp * (amp >= self.spec.a_min).to(amp.dtype)

    def reset(self, n: int):
        if self.spec.model == "dynaphos":
            return self.model.initial_state_batch(n)
        return torch.zeros(n, *self.grid, device=self.device)

    def step(self, state, amp: torch.Tensor):
        """``amp`` is the raw encoder action; returns ``(state, image / scale)``."""
        a = self.execute(amp)
        if self.spec.model == "dynaphos":
            state, img = self.model.step_batch(state, a)
        elif torch.is_grad_enabled():
            # Recompute the axon-map step in the backward pass: storing every
            # (batch, electrode, pixel, axon segment) tensor needs GBs per step.
            state = checkpoint(
                self.model.step_batch, state, a, cutoff=self.cutoff, use_reentrant=False
            )
            img = state
        else:
            state = self.model.step_batch(state, a, cutoff=self.cutoff)
            img = state
        return state, img / self.scale

    def phys(self, state, n: int) -> torch.Tensor:
        """Per-electrode physical state ``(n, E, n_phys)`` used to seed the world model."""
        if self.n_phys == 0:
            return torch.zeros(n, self.n_elec, 0, device=self.device)
        m = self.model
        return torch.stack([state["A"] / m.a50, state["Q"] / m.rheobase, state["sigma"]], dim=-1)

    @torch.no_grad()
    def _calibrate_scale(self) -> float:
        g = torch.Generator(device="cpu").manual_seed(0)
        amp = torch.rand(64, self.n_elec, generator=g) * self.spec.a_max
        amp = amp * (torch.rand(64, self.n_elec, generator=g) < 0.3)
        state = self.reset(64)
        for _ in range(self.spec.horizon):
            state, img = self.step(state, amp.to(self.device))
        return float(torch.quantile(img.flatten(), 0.99).clamp(min=1e-6))

    @torch.no_grad()
    def phosphene_centroids(self) -> torch.Tensor:
        """Phosphene centre of each electrode in grid pixels ``(row, col)``; silent ones at 0."""

        def render(amp):
            state = self.reset(amp.shape[0])
            for _ in range(3):
                state, img = self.step(state, amp)
            return img

        c = probe_centroids(render, self.n_elec, self.spec.a_max * 0.8)
        return torch.nan_to_num(c, nan=0.0)

    def electrode_xy(self) -> torch.Tensor:
        """Phosphene centroids normalised to [-1, 1] (used as electrode positions by the model)."""
        c = self.phosphene_centroids()
        h, w = self.grid
        return torch.stack([c[:, 1] / (w - 1), c[:, 0] / (h - 1)], -1) * 2 - 1


def episode_loss(spec: TaskSpec, imgs: torch.Tensor, acts: torch.Tensor, target: torch.Tensor):
    """Per-sample loss: percept MSE over steps ``t >= loss_from`` + charge.

    ``imgs (n, T, H, W)`` are the percepts after each of the ``T`` actions.
    """
    mse = ((imgs[:, spec.loss_from :] - target[:, None]) ** 2).mean((1, 2, 3))
    charge = acts.clamp(0, spec.a_max).mean((1, 2)) / spec.a_max
    return mse + spec.charge_weight * charge, mse, charge


# ----------------------------------------------------------------- targets
def load_targets(
    grid: tuple[int, int], cache: str | Path = "data/mnist"
) -> dict[str, torch.Tensor]:
    """MNIST digits resized to fit the percept grid, centred, in [0, 1]."""
    from ..neurostim.percept import load_mnist

    d = load_mnist(cache)
    h, w = grid
    side = min(h, w)

    def fit(x):
        x = F.interpolate(x.view(-1, 1, 28, 28), size=(side, side), mode="bilinear")
        out = torch.zeros(x.shape[0], h, w)
        r0, c0 = (h - side) // 2, (w - side) // 2
        out[:, r0 : r0 + side, c0 : c0 + side] = x[:, 0].clamp(0, 1)
        return out

    return {"train": fit(d["train_x"]), "test": fit(d["test_x"][:256])}


# ---------------------------------------------------------------- networks
Encoder = ClosedLoopVisionEncoder  # ``vision2stim``: (target, current percept) -> amplitudes


def _sincos(xy: torch.Tensor, dim: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(dim // 4, device=xy.device, dtype=xy.dtype) * math.pi / 2
    ang = xy[..., :, None] * freqs  # (..., 2, dim/4)
    return torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)


class TokenWorldModel(nn.Module):
    """``stim2vision`` surrogate: ``(percept_t, memory_t, amp_t) -> (percept_{t+1}, memory_{t+1})``.

    Each electrode has a recurrent memory (GRU cell, shared weights) driven by
    its own amplitude. Image patches carry the previous percept and query the
    electrode tokens by cross-attention, with a learned null key so a patch can
    receive nothing. The output is a residual update of the percept.
    """

    def __init__(
        self,
        grid: tuple[int, int],
        elec_xy: torch.Tensor,
        a_max: float,
        n_phys: int = 0,
        dim: int = 64,
        heads: int = 4,
        depth: int = 2,
        patch: int = 3,
    ):
        super().__init__()
        self.grid, self.patch, self.a_max, self.dim = grid, patch, a_max, dim
        h, w = grid
        self.ph, self.pw = math.ceil(h / patch), math.ceil(w / patch)
        self.register_buffer("elec_pos", _sincos(elec_xy.float(), dim))
        py, px = torch.meshgrid(
            torch.linspace(-1, 1, self.ph), torch.linspace(-1, 1, self.pw), indexing="ij"
        )
        self.register_buffer("patch_pos", _sincos(torch.stack([px, py], -1).view(-1, 2), dim))
        self.amp_in = nn.Linear(2, dim)
        self.cell = nn.GRUCell(dim, dim)
        self.seed = nn.Linear(n_phys, dim) if n_phys else None
        self.mem0 = nn.Parameter(torch.zeros(dim))
        self.patch_in = nn.Linear(patch * patch, dim)
        self.null_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "nq": nn.LayerNorm(dim),
                    "nk": nn.LayerNorm(dim),
                    "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                    "nm": nn.LayerNorm(dim),
                    "mlp": nn.Sequential(
                        nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
                    ),
                }
            )
            for _ in range(depth)
        )
        self.out = nn.Linear(dim, patch * patch)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def init_memory(self, phys: torch.Tensor) -> torch.Tensor:
        n, e = phys.shape[:2]
        if self.seed is not None:
            return self.seed(phys)
        return self.mem0.expand(n, e, self.dim).contiguous()

    def _patches(self, img: torch.Tensor) -> torch.Tensor:
        h, w = self.grid
        p = self.patch
        x = F.pad(img, (0, self.pw * p - w, 0, self.ph * p - h))
        x = x.view(-1, self.ph, p, self.pw, p).permute(0, 1, 3, 2, 4)
        return x.reshape(-1, self.ph * self.pw, p * p)

    def _unpatch(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.grid
        p = self.patch
        x = x.view(-1, self.ph, self.pw, p, p).permute(0, 1, 3, 2, 4)
        return x.reshape(-1, self.ph * p, self.pw * p)[:, :h, :w]

    def forward(self, img: torch.Tensor, mem: torch.Tensor, amp: torch.Tensor):
        n, e = amp.shape
        a = amp / self.a_max
        u = self.amp_in(torch.stack([a, (a > 0).to(a.dtype)], -1))
        mem = self.cell(u.reshape(n * e, -1), mem.reshape(n * e, -1)).view(n, e, -1)
        tok = mem + self.elec_pos + u
        kv = torch.cat([self.null_k.expand(n, 1, -1), tok], 1)
        q = self.patch_in(self._patches(img)) + self.patch_pos
        for b in self.blocks:
            k = b["nk"](kv)
            q = q + b["attn"](b["nq"](q), k, k, need_weights=False)[0]
            q = q + b["mlp"](b["nm"](q))
        nxt = F.relu(img + self._unpatch(self.out(q)))
        return nxt, mem


# ------------------------------------------------------------------ buffer
class Replay:
    """Whole episodes: percepts ``(n, T+1, H, W)``, physics ``(n, T+1, E, k)``, actions ``(n, T, E)``."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.imgs = self.phys = self.acts = None

    def add(self, imgs, phys, acts):
        if self.imgs is None:
            self.imgs, self.phys, self.acts = imgs, phys, acts
        else:
            self.imgs = torch.cat([self.imgs, imgs])[-self.capacity :]
            self.phys = torch.cat([self.phys, phys])[-self.capacity :]
            self.acts = torch.cat([self.acts, acts])[-self.capacity :]

    def __len__(self):
        return 0 if self.imgs is None else len(self.imgs)

    def windows(self, n: int, k: int, g: torch.Generator):
        idx = torch.randint(0, len(self), (n,), generator=g)
        t0 = torch.randint(0, self.acts.shape[1] - k + 1, (n,), generator=g)
        ar = t0[:, None] + torch.arange(k + 1)
        imgs = self.imgs[idx[:, None], ar]
        acts = self.acts[idx[:, None], ar[:, :k]]
        phys0 = self.phys[idx, t0]
        return imgs, phys0, acts


# ----------------------------------------------------------------- rollouts
def teacher_rollout(teacher: Teacher, enc: Encoder, target, noise: float = 0.0, g=None):
    """Run the encoder in the teacher; differentiable unless called under ``no_grad``."""
    n = target.shape[0]
    state = teacher.reset(n)
    img = torch.zeros(n, *teacher.grid)
    imgs, phys, acts = [img], [teacher.phys(state, n)], []
    for _ in range(teacher.spec.horizon):
        a = enc(target, img)
        if noise:
            a = a + noise * teacher.spec.a_max * torch.randn(a.shape, generator=g)
        state, img = teacher.step(state, a)
        imgs.append(img)
        phys.append(teacher.phys(state, n))
        acts.append(a)
    return torch.stack(imgs, 1), torch.stack(phys, 1), torch.stack(acts, 1)


def model_rollout(wm: TokenWorldModel, teacher: Teacher, enc: Encoder, target):
    n = target.shape[0]
    img = torch.zeros(n, *teacher.grid)
    mem = wm.init_memory(teacher.phys(teacher.reset(n), n))
    imgs, acts = [img], []
    for _ in range(teacher.spec.horizon):
        a = enc(target, img)
        img, mem = wm(img, mem, a.clamp(0, teacher.spec.a_max))
        imgs.append(img)
        acts.append(a)
    return torch.stack(imgs, 1), torch.stack(acts, 1)


def fit_world_model(wm, opt, buf: Replay, steps: int, batch: int, k: int, g) -> float:
    total = 0.0
    for _ in range(steps):
        imgs, phys0, acts = buf.windows(batch, k, g)
        img, mem = imgs[:, 0], wm.init_memory(phys0)
        loss = 0.0
        for j in range(k):
            img, mem = wm(img, mem, acts[:, j].clamp(0, wm.a_max))
            loss = loss + F.mse_loss(img, imgs[:, j + 1])
        loss = loss / k
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        total += float(loss)
    return total / max(steps, 1)


def reinforce_loss(spec: TaskSpec, enc: Encoder, step_fn, target, sigma: float, g):
    """Score-function (REINFORCE) loss with a per-step batch-mean baseline.

    ``step_fn(state, amp) -> (state, image)`` is the teacher or the world model.
    Gaussian noise is added to the encoder logits, so the executed amplitude is
    ``a_max * sigmoid(logits + noise)``.
    """
    n = target.shape[0]
    state, img = None, torch.zeros(n, *target.shape[1:])
    logps, rewards = [], []
    for t in range(spec.horizon):
        mu = enc.logits(target, img)
        z = mu.detach() + sigma * torch.randn(mu.shape, generator=g)
        logps.append((-((z - mu) ** 2) / (2 * sigma**2)).sum(-1))
        a = spec.a_max * torch.sigmoid(z)
        with torch.no_grad():
            state, img = step_fn(state, a)
            r = -spec.charge_weight * a.mean(-1) / spec.a_max / spec.horizon
            if t >= spec.loss_from:
                r = r - ((img - target) ** 2).mean((1, 2)) / (spec.horizon - spec.loss_from)
        rewards.append(r)
    R = torch.stack(rewards, 1).flip(1).cumsum(1).flip(1)  # reward-to-go
    adv = R - R.mean(0, keepdim=True)
    adv = adv / adv.std().clamp(min=1e-6)
    return -(torch.stack(logps, 1) * adv).mean()


# --------------------------------------------------------------- training
@dataclass
class RunConfig:
    arm: str = "wm-bptt"  # teacher-bptt | wm-bptt | reinforce-sim | reinforce-wm | baselines
    preset: str = "axonmap"
    budget_s: float = 600.0  # wall-clock training budget (evaluation excluded)
    batch: int = 64
    lr: float = 1e-3
    seed: int = 0
    eval_every_s: float = 120.0
    eval_n: int = 64  # test digits for in-loop evaluation; the final one uses all 256
    # world-model loop
    collect: int = 64  # teacher episodes per cycle
    action_noise: float = 0.05  # std, fraction of a_max
    wm_steps: int = 40
    policy_steps: int = 20
    wm_window: int = 8
    capacity: int = 4096
    warmup_episodes: int = 512
    # model-free
    sigma: float = 0.5
    threads: int = 1
    out: str = "data/imagine"
    tags: dict = field(default_factory=dict)


@torch.no_grad()
def evaluate(teacher: Teacher, enc: Encoder, targets: torch.Tensor) -> dict:
    imgs, _, acts = teacher_rollout(teacher, enc, targets)
    loss, mse, charge = episode_loss(teacher.spec, imgs[:, 1:], acts, targets)
    return {"loss": float(loss.mean()), "mse": float(mse.mean()), "charge": float(charge.mean())}


@torch.no_grad()
def evaluate_baselines(teacher: Teacher, targets: torch.Tensor) -> dict:
    """Non-learned references on the teacher: no stimulation, and the phosphene map.

    The phosphene-map encoder drives each electrode with the target sampled at
    its phosphene centre, the same amplitudes at every step (open loop). Its
    ``on`` level and gain are swept and the best is kept, so it is a fair
    rather than a straw baseline.
    """
    spec = teacher.spec
    n = targets.shape[0]
    zero = torch.zeros(n, spec.horizon, teacher.n_elec)

    def score(acts):
        state, imgs = teacher.reset(n), []
        for t in range(spec.horizon):
            state, img = teacher.step(state, acts[:, t])
            imgs.append(img)
        loss, mse, charge = episode_loss(spec, torch.stack(imgs, 1), acts, targets)
        return {
            "loss": float(loss.mean()),
            "mse": float(mse.mean()),
            "charge": float(charge.mean()),
        }

    out = {"zero": score(zero)}
    cent = teacher.phosphene_centroids()
    best = None
    for on in (0.1, 0.3, 0.5):
        for frac in (0.25, 0.5, 1.0):
            enc = PhospheneMapEncoder(
                cent,
                teacher.grid,
                a_max=spec.a_min + frac * (spec.a_max - spec.a_min),
                a_min=spec.a_min,
                on=on,
            )
            amp = enc(targets)
            r = score(amp[:, None].expand(-1, spec.horizon, -1))
            r.update(on=on, gain=frac)
            if best is None or r["loss"] < best["loss"]:
                best = r
    out["phosphene_map"] = best
    return out


def run(rc: RunConfig, teacher: Teacher | None = None, targets: dict | None = None) -> dict:
    torch.set_num_threads(rc.threads)
    if rc.arm == "baselines":
        spec = TaskSpec.preset(rc.preset)
        teacher = teacher or Teacher(spec)
        targets = targets or load_targets(teacher.grid)
        result = {"spec": asdict(spec), "baselines": evaluate_baselines(teacher, targets["test"])}
        out = Path(rc.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rc.preset}_baselines.json").write_text(json.dumps(result, indent=1))
        return result
    torch.manual_seed(rc.seed)
    g = torch.Generator().manual_seed(rc.seed)
    spec = TaskSpec.preset(rc.preset)
    teacher = teacher or Teacher(spec)
    targets = targets or load_targets(teacher.grid)
    train, test = targets["train"], targets["test"]
    enc = Encoder(teacher.grid, teacher.n_elec, spec.a_max)
    opt = torch.optim.Adam(enc.parameters(), lr=rc.lr)
    use_wm = rc.arm in ("wm-bptt", "reinforce-wm")
    if use_wm:
        wm = TokenWorldModel(teacher.grid, teacher.electrode_xy(), spec.a_max, teacher.n_phys)
        wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        buf = Replay(rc.capacity)

    def batch_targets(n):
        return train[torch.randint(0, len(train), (n,), generator=g)]

    log = []
    clock, teacher_steps, imagined_steps, next_eval = 0.0, 0, 0, 0.0
    wm_loss = float("nan")

    def record(final: bool = False):
        r = evaluate(teacher, enc, test if final else test[: rc.eval_n])
        r["final"] = final
        r.update(
            t=clock, teacher_steps=teacher_steps, imagined_steps=imagined_steps, wm_loss=wm_loss
        )
        log.append(r)

    record()
    while clock < rc.budget_s:
        t0 = time.perf_counter()
        if rc.arm == "teacher-bptt":
            y = batch_targets(rc.batch)
            imgs, _, acts = teacher_rollout(teacher, enc, y)
            loss = episode_loss(spec, imgs[:, 1:], acts, y)[0].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            teacher_steps += rc.batch * spec.horizon
        elif rc.arm == "reinforce-sim":
            y = batch_targets(rc.batch)
            loss = reinforce_loss(spec, enc, teacher.step, y, rc.sigma, g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            teacher_steps += rc.batch * spec.horizon
        else:
            n_new = rc.warmup_episodes if len(buf) == 0 else rc.collect
            with torch.no_grad():
                for s in range(0, n_new, rc.batch):
                    y = batch_targets(min(rc.batch, n_new - s))
                    imgs, phys, acts = teacher_rollout(teacher, enc, y, rc.action_noise, g)
                    buf.add(imgs, phys, acts.clamp(0, spec.a_max))
            teacher_steps += n_new * spec.horizon
            wm.train()
            wm_loss = fit_world_model(wm, wm_opt, buf, rc.wm_steps, rc.batch, rc.wm_window, g)
            wm.requires_grad_(False)
            for _ in range(rc.policy_steps):
                y = batch_targets(rc.batch)
                if rc.arm == "wm-bptt":
                    imgs, acts = model_rollout(wm, teacher, enc, y)
                    loss = episode_loss(spec, imgs[:, 1:], acts, y)[0].mean()
                else:

                    def wm_step(state, a):
                        if state is None:
                            state = (
                                torch.zeros(a.shape[0], *teacher.grid),
                                wm.init_memory(teacher.phys(teacher.reset(a.shape[0]), a.shape[0])),
                            )
                        img, mem = wm(state[0], state[1], a)
                        return (img, mem), img

                    loss = reinforce_loss(spec, enc, wm_step, y, rc.sigma, g)
                opt.zero_grad()
                loss.backward()
                opt.step()
                imagined_steps += rc.batch * spec.horizon
            wm.requires_grad_(True)
        clock += time.perf_counter() - t0
        if clock >= next_eval:
            record()
            next_eval = clock + rc.eval_every_s
    record(final=True)
    result = {"config": asdict(rc), "spec": asdict(spec), "scale": teacher.scale, "log": log}
    out = Path(rc.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{rc.preset}_{rc.arm}_s{rc.seed}.json"
    (out / name).write_text(json.dumps(result, indent=1))
    torch.save(enc.state_dict(), out / name.replace(".json", "_encoder.pt"))
    return result


def cli():  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f, v in asdict(RunConfig()).items():
        if f == "tags":
            continue
        p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    rc = RunConfig(**vars(p.parse_args()))
    r = run(rc)
    if rc.arm == "baselines":
        print(json.dumps(r["baselines"]))
        return
    last = r["log"][-1]
    print(json.dumps({k: last[k] for k in ("t", "teacher_steps", "loss", "mse", "charge")}))


if __name__ == "__main__":  # pragma: no cover
    cli()
