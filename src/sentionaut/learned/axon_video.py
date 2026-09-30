"""Genie-style video world model for axon-map percepts.

``AxonMapWorld`` keeps the analytical fade and learns only the spatial drive.
This model hard-codes nothing temporal: it reads a causal window of past frames
and stimulations and predicts the next frame, so the fade, the rise under held
stimulation, and the percept threshold have to be learned from the teacher's
video.

The block layout is Genie's spatiotemporal transformer (Bruce et al., 2024):
attention across patches inside a frame, then causal attention across frames at
each patch, so cost is ``T·N² + N·T²`` rather than ``(T·N)²``. Two Genie parts
are left out on purpose. Genie infers latent actions because game video is
unlabelled; the stimulation here is known, so it enters as electrode tokens, as
in ``AxonMapWorld``. Genie predicts VQ tokens with MaskGIT; brightness is a
smooth scalar field, so frames stay continuous and the loss is MSE on them.
Context frames get noise during training (GameNGen, Valevski et al., 2024) so
the model tolerates its own errors when it runs free.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base import Action, State
from ..core.device import get_device
from .axon_world import AdaLN, AxonMapWorld, distill_hdf5
from .model import sincos_2d_pos_embed


def _sincos_1d(n: int, dim: int, device, dtype) -> torch.Tensor:
    half = dim // 2
    omega = 1.0 / (10000.0 ** (torch.arange(half, device=device, dtype=dtype) / half))
    pos = torch.arange(n, device=device, dtype=dtype)[:, None] * omega[None, :]
    return torch.cat([pos.sin(), pos.cos()], dim=1)


class STBlock(nn.Module):
    """Stimulation cross-attention, spatial attention, causal temporal attention, MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, cond_dim: int):
        super().__init__()
        self.adaln_cross = AdaLN(cond_dim, dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln_space = AdaLN(cond_dim, dim)
        self.space = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_time = nn.LayerNorm(dim)
        self.time = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln_mlp = AdaLN(cond_dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, elec, elec_mask, cond, t: int) -> torch.Tensor:
        h = self.adaln_cross(x, cond)
        x = x + self.cross(h, elec, elec, key_padding_mask=elec_mask, need_weights=False)[0]
        h = self.adaln_space(x, cond)
        x = x + self.space(h, h, h, need_weights=False)[0]
        bt, n, d = x.shape
        b = bt // t
        xt = x.reshape(b, t, n, d).transpose(1, 2).reshape(b * n, t, d)
        h = self.norm_time(xt)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        xt = xt + self.time(h, h, h, attn_mask=causal, need_weights=False)[0]
        x = xt.reshape(b, n, t, d).transpose(1, 2).reshape(bt, n, d)
        return x + self.mlp(self.adaln_mlp(x, cond))


class AxonVideoWorld(nn.Module):
    def __init__(
        self,
        grid_shape: tuple[int, int],
        n_electrodes: int,
        *,
        dim: int = 96,
        depth: int = 4,
        heads: int = 4,
        patch_size: int = 4,
        mlp_ratio: float = 2.0,
        context: int = 16,
        context_noise: float = 0.02,
        dt_ms: float = 20.0,
    ):
        super().__init__()
        if dim % 4 != 0 or dim % heads != 0:
            raise ValueError("dim must be divisible by 4 and by heads")
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.n_electrodes = int(n_electrodes)
        self.dim = dim
        self.patch_size = patch_size
        self.context = int(context)
        self.context_noise = float(context_noise)
        # Only used to express the learned fade as a time constant; nothing steps with it.
        self.dt_ms = float(dt_ms)
        self.arch = {
            "grid_shape": list(self.grid_shape),
            "n_electrodes": self.n_electrodes,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "patch_size": patch_size,
            "mlp_ratio": mlp_ratio,
            "context": self.context,
            "context_noise": self.context_noise,
            "dt_ms": self.dt_ms,
        }
        self.register_buffer("elec_xy", torch.zeros(self.n_electrodes, 2))
        self.elec_embed = nn.Linear(5, dim)
        # Always-valid key, so a frame with every electrode off still has something to attend to.
        self.null_elec = nn.Parameter(torch.zeros(1, 1, dim))
        self.cond_mlp = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.patch_embed = nn.Linear(patch_size * patch_size, dim)
        self.blocks = nn.ModuleList([STBlock(dim, heads, mlp_ratio, dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.patch_unembed = nn.Linear(dim, patch_size * patch_size)
        # Start as "the frame persists" (tau = infinity), so any fade is learned, not initialised.
        nn.init.zeros_(self.patch_unembed.weight)
        nn.init.zeros_(self.patch_unembed.bias)

    bind = AxonMapWorld.bind
    _to_patches = AxonMapWorld._to_patches
    _from_patches = AxonMapWorld._from_patches

    def forward(self, frames, amp, freq, phase_dur, rho, axlambda) -> torch.Tensor:
        """``frames`` ``(B, T, H, W)`` and actions ``(B, T, ...)`` → next frames ``(B, T, H, W)``.

        Output ``t`` depends only on frames and actions ``0..t``.
        """
        b, t, h, w = frames.shape
        device, dtype = frames.device, frames.dtype
        patches, shape = self._to_patches(frames.reshape(b * t, h, w))
        x = self.patch_embed(patches)
        x = x + sincos_2d_pos_embed(shape[2], shape[3], self.dim, device, dtype)[None]
        te = _sincos_1d(t, self.dim, device, dtype)
        x = x + te[None].expand(b, -1, -1).reshape(b * t, 1, self.dim)

        n_e = self.n_electrodes
        xy = self.elec_xy.to(dtype)[None, None].expand(b, t, -1, -1) / 1000.0
        feat = torch.cat(
            [xy, amp[..., None], freq[..., None] / 100.0, phase_dur[..., None]], dim=-1
        ).reshape(b * t, n_e, 5)
        elec = torch.cat([self.null_elec.expand(b * t, -1, -1), self.elec_embed(feat)], dim=1)
        mask = torch.cat(
            [
                torch.zeros(b * t, 1, dtype=torch.bool, device=device),
                (amp == 0).reshape(b * t, n_e),
            ],
            dim=1,
        )
        subject = torch.stack([rho, axlambda], dim=-1).to(dtype).reshape(b * t, 2) / 1000.0
        cond = self.cond_mlp(subject)
        for block in self.blocks:
            x = block(x, elec, mask, cond, t)
        delta = self._from_patches(self.patch_unembed(self.norm(x)), shape)
        return F.relu(frames + delta.reshape(b, t, h, w))

    def rollout(self, first, amp, freq, phase_dur, rho, axlambda) -> torch.Tensor:
        """Free-running: ``first`` ``(B, H, W)``, then only its own predictions, sliding window."""
        frames = [first]
        out = []
        for i in range(amp.shape[1]):
            lo = max(0, i + 1 - self.context)
            ctx = torch.stack(frames[lo : i + 1], dim=1)
            acts = [a[:, lo : i + 1] for a in (amp, freq, phase_dur, rho, axlambda)]
            nxt = self.forward(ctx, *acts)[:, -1]
            frames.append(nxt)
            out.append(nxt)
        return torch.stack(out, dim=1)

    def step(self, state: State | None, action: Action) -> State:
        """Same contract as the teacher. The context window rides in ``State.aux``."""
        device, dtype = self.elec_xy.device, self.elec_xy.dtype
        if state is None:
            image = torch.zeros(self.grid_shape, device=device, dtype=dtype)
            hist: dict = {}
        else:
            image, hist = state.image.to(device=device, dtype=dtype), state.aux
        amp = action.amp.to(device=device, dtype=dtype)
        new = {
            "frames": image,
            "amp": amp,
            "freq": (action.freq if action.freq is not None else torch.zeros_like(amp)).to(amp),
            "phase_dur": (
                action.phase_dur if action.phase_dur is not None else torch.zeros_like(amp)
            ).to(amp),
            "rho": torch.as_tensor(200.0 if action.rho is None else action.rho).to(amp),
            "axlambda": torch.as_tensor(500.0 if action.axlambda is None else action.axlambda).to(
                amp
            ),
        }
        window = {}
        for key, value in new.items():
            past = hist.get(key)
            seq = value[None] if past is None else torch.cat([past, value[None]], dim=0)
            window[key] = seq[-self.context :]
        nxt = self.forward(
            window["frames"][None],
            *(window[k][None] for k in ("amp", "freq", "phase_dur", "rho", "axlambda")),
        )[0, -1]
        return State(image=nxt, aux=window)


def video_loss(model: AxonVideoWorld, batch: dict) -> torch.Tensor:
    """Teacher-forced next-frame MSE in training; free-running rollout MSE in eval."""
    target = batch["target"]
    acts = (batch["amp"], batch["freq"], batch["phase_dur"], batch["rho"], batch["axlambda"])
    if not model.training:
        return F.mse_loss(model.rollout(batch["s0"], *acts), target)
    frames = torch.cat([batch["s0"][:, None], target[:, :-1]], dim=1)
    if model.context_noise > 0:
        frames = (frames + model.context_noise * torch.randn_like(frames)).clamp_min(0.0)
    return F.mse_loss(model(frames, *acts), target)


def train_video(
    dataset_path: str | Path,
    *,
    context: int = 16,
    dim: int = 96,
    depth: int = 4,
    heads: int = 4,
    patch_size: int = 4,
    context_noise: float = 0.02,
    **kwargs,
) -> dict:
    def factory(grid, n_elec, dt_ms):
        return AxonVideoWorld(
            grid,
            n_elec,
            dim=dim,
            depth=depth,
            heads=heads,
            patch_size=patch_size,
            context=context,
            context_noise=context_noise,
            dt_ms=dt_ms,
        )

    return distill_hdf5(
        dataset_path, rollout_k=context, model_factory=factory, loss_fn=video_loss, **kwargs
    )


def load_video(ckpt_path: str | Path, device: torch.device | None = None) -> AxonVideoWorld:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    model = AxonVideoWorld(**ckpt["arch"])
    model.load_state_dict(ckpt["model"])
    return model.to(device or torch.device("cpu")).eval()


# ---------------------------------------------------------------- report


def _episode_arrays(dataset_path: Path, rows: list[int], scale: float) -> dict[str, torch.Tensor]:
    from .axon_report import _rows_batch

    return {k: v[None] for k, v in _rows_batch(dataset_path, rows, scale).items()}


def _phases(amp: np.ndarray) -> list[str]:
    """``on`` for a fresh stimulation, ``hold`` when it repeats the previous one, ``off`` for none."""
    labels = []
    for t in range(amp.shape[0]):
        if not amp[t].any():
            labels.append("off")
        elif t > 0 and np.array_equal(amp[t], amp[t - 1]):
            labels.append("hold")
        else:
            labels.append("on")
    return labels


_ACT_KEYS = ("amp", "freq", "phase_dur", "rho", "axlambda")


def _val_stack(dataset_path: Path) -> dict[str, torch.Tensor]:
    """Every held-out episode as one ``(episodes, T, ...)`` batch."""
    from .axon_report import _dataset_meta, _rows_batch, _val_episodes

    scale = _dataset_meta(dataset_path)["scale"]
    episodes = list(_val_episodes(dataset_path).values())
    # ponytail: stream episodes share one length; ragged ones are cut to the shortest.
    n = min(len(rows) for rows in episodes)
    parts = [_rows_batch(dataset_path, rows[:n], scale) for rows in episodes]
    return {k: torch.stack([p[k] for p in parts]) for k in parts[0]}


def _chunks(stack: dict, size: int, device):
    for i in range(0, stack["amp"].shape[0], size):
        yield {k: v[i : i + size].to(device) for k, v in stack.items()}


@torch.no_grad()
def _teacher_forced(model: AxonVideoWorld, ep: dict) -> torch.Tensor:
    """Next-frame prediction at every ``t`` from the teacher's own last ``context`` frames."""
    c, t_len = model.context, ep["amp"].shape[1]
    acts = [ep[k] for k in _ACT_KEYS]
    head = model(ep["s_t"][:, :c], *(a[:, :c] for a in acts))
    tail = [
        model(ep["s_t"][:, t - c + 1 : t + 1], *(a[:, t - c + 1 : t + 1] for a in acts))[:, -1]
        for t in range(c, t_len)
    ]
    return torch.cat([head, torch.stack(tail, dim=1)], dim=1) if tail else head


@torch.no_grad()
def fade_constant(
    model, dataset_path: Path, device, min_signal: float = 0.05, chunk: int = 32
) -> dict:
    """Effective fade time constant on silent steps, teacher-forced so only the fade is tested.

    For an off frame the teacher is ``B' = (1 - dt/tau) B``, so ``r = <B', B> / <B, B>`` gives
    ``tau = dt / (1 - r)``. The same ratio on the model's prediction is its learned ``tau``.
    """
    ratios_s, ratios_t = [], []
    for ep in _chunks(_val_stack(Path(dataset_path)), chunk, device):
        pred = _teacher_forced(model, ep)
        now = ep["s_t"]
        silent = ~(ep["amp"] != 0).any(dim=-1) & (now.amax(dim=(-2, -1)) > min_signal)
        denom = (now * now).sum(dim=(-2, -1))
        ratios_s += ((pred * now).sum(dim=(-2, -1)) / denom.clamp_min(1e-12))[silent].tolist()
        ratios_t += ((ep["s_tp1"] * now).sum(dim=(-2, -1)) / denom.clamp_min(1e-12))[
            silent
        ].tolist()
    dt = model.dt_ms

    def tau(r):
        return dt / np.clip(1.0 - np.asarray(r), 1e-6, None)

    tau_s, tau_t = tau(ratios_s), tau(ratios_t)
    return {
        "n_silent_steps": len(ratios_s),
        "teacher_decay_ratio_median": float(np.median(ratios_t)),
        "student_decay_ratio_median": float(np.median(ratios_s)),
        "teacher_tau_ms_median": float(np.median(tau_t)),
        "student_tau_ms_median": float(np.median(tau_s)),
        "student_tau_ms_iqr": [float(np.percentile(tau_s, 25)), float(np.percentile(tau_s, 75))],
    }


@torch.no_grad()
def horizon_errors(model, dataset_path: Path, device, baseline=None, chunk: int = 32) -> dict:
    """Free-running MSE by frame index over whole held-out episodes, longer than the context."""
    errs, base_errs = [], []
    for ep in _chunks(_val_stack(Path(dataset_path)), chunk, device):
        acts = [ep[k] for k in _ACT_KEYS]
        pred = model.rollout(ep["s_t"][:, 0], *acts)
        errs.append(((pred - ep["s_tp1"]) ** 2).mean(dim=(-2, -1)).cpu())
        if baseline is not None:
            base = _markov_rollout(baseline, ep, device)
            base_errs.append(((base - ep["s_tp1"]) ** 2).mean(dim=(-2, -1)).cpu())
    err = torch.cat(errs)
    out = {
        "mse_by_frame": err.mean(dim=0).tolist(),
        "mse": float(err.mean()),
        "n_episodes": int(err.shape[0]),
        "frames": int(err.shape[1]),
    }
    if baseline is not None:
        base = torch.cat(base_errs)
        out["baseline_mse_by_frame"] = base.mean(dim=0).tolist()
        out["baseline_mse"] = float(base.mean())
    return out


def _markov_rollout(model: AxonMapWorld, ep: dict, device) -> torch.Tensor:
    brightness = ep["s_t"][:, 0].to(device)
    out = []
    for t in range(ep["amp"].shape[1]):
        acts = (ep[k][:, t].to(device) for k in _ACT_KEYS)
        brightness = model.predict_next(brightness, *acts)
        out.append(brightness)
    return torch.stack(out, dim=1)


def _episode_videos(model, dataset_path: Path, device, n: int, baseline=None):
    from .axon_report import _dataset_meta, _val_episodes

    scale = _dataset_meta(dataset_path)["scale"]
    for ep_id, rows in list(_val_episodes(dataset_path).items())[:n]:
        ep = _episode_arrays(dataset_path, rows, scale)
        acts = [ep[k].to(device) for k in _ACT_KEYS]
        with torch.no_grad():
            student = model.rollout(ep["s_t"][:, 0].to(device), *acts)[0].cpu().numpy()
            base = None
            if baseline is not None:
                base = _markov_rollout(baseline, ep, device)[0].cpu().numpy()
        yield ep_id, ep["s_tp1"][0].numpy(), student, base, _phases(ep["amp"][0].numpy())


def plot_timeline(model, dataset_path, out_path, device, n_episodes=3, baseline=None) -> Path:
    """Mean brightness per frame, with stimulation phases shaded: rise, plateau, fade."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    videos = list(_episode_videos(model, Path(dataset_path), device, n_episodes, baseline))
    fig, axes = plt.subplots(len(videos), 1, figsize=(10, 2.3 * len(videos)), sharex=True)
    axes = np.atleast_1d(axes)
    colours = {"on": "#fde68a", "hold": "#fca5a5", "off": None}
    for ax, (ep_id, teacher, student, base, phases) in zip(axes, videos):
        t = np.arange(1, len(phases) + 1)
        for i, p in enumerate(phases):
            if colours[p]:
                ax.axvspan(i + 0.5, i + 1.5, color=colours[p], alpha=0.5, lw=0)
        ax.plot(t, teacher.mean(axis=(1, 2)), color="black", lw=2, label="teacher (axon map)")
        ax.plot(t, student.mean(axis=(1, 2)), color="#1d4ed8", lw=1.5, ls="--", label="video model")
        if base is not None:
            ax.plot(
                t, base.mean(axis=(1, 2)), color="#c2410c", lw=1, ls=":", label="drive + exact fade"
            )
        ax.axvline(model.context + 0.5, color="grey", lw=0.8, ls="-.")
        ax.set_ylabel(f"ep {ep_id}\nmean B")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(
        "Held-out streams, free-running. Yellow: new stimulation, red: held, white: off. "
        "Dash-dot: end of context window.",
        fontsize=9,
    )
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_frames(model, dataset_path, out_path, device, n_cols: int = 10) -> Path:
    """Teacher, video model, and error on one held-out stream, frames sampled across it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep_id, teacher, student, _, phases = next(_episode_videos(model, Path(dataset_path), device, 1))
    idx = np.linspace(0, len(phases) - 1, n_cols).round().astype(int)
    # A single hot pixel would otherwise set the scale and leave the rest black.
    vmax = max(float(np.percentile(teacher, 99.9)), 1e-6)
    emax = max(float(np.abs(student - teacher).max()), 1e-6)
    fig, axes = plt.subplots(3, n_cols, figsize=(1.6 * n_cols, 5.2))
    for c, t in enumerate(idx):
        for r, (img, label, cmap, hi) in enumerate(
            [
                (teacher[t], "teacher", "magma", vmax),
                (student[t], "video model", "magma", vmax),
                (np.abs(student[t] - teacher[t]), "|error|", "viridis", emax),
            ]
        ):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0:
                ax.set_title(f"t={t + 1} {phases[t]}", fontsize=8)
    fig.suptitle(f"Held-out stream {ep_id}, free-running from its first frame", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def write_gif(model, dataset_path, out_path, device, upscale: int = 4, fps: int = 5) -> Path:
    """Teacher | video model | error, one held-out stream, one GIF frame per percept frame."""
    import imageio.v2 as imageio
    from matplotlib import colormaps

    ep_id, teacher, student, _, phases = next(_episode_videos(model, Path(dataset_path), device, 1))
    vmax = max(float(teacher.max()), float(student.max()), 1e-6)
    magma, viridis = colormaps["magma"], colormaps["viridis"]
    gap = np.ones((teacher.shape[1], 2, 3))
    frames = []
    for t in range(len(phases)):
        tiles = [
            magma(np.clip(teacher[t] / vmax, 0, 1))[..., :3],
            gap,
            magma(np.clip(student[t] / vmax, 0, 1))[..., :3],
            gap,
            viridis(np.clip(np.abs(student[t] - teacher[t]) / vmax, 0, 1))[..., :3],
        ]
        img = np.concatenate(tiles, axis=1)
        img = img.repeat(upscale, axis=0).repeat(upscale, axis=1)
        frames.append((img * 255).astype(np.uint8))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    return out_path


def plot_curves(train_timing, baseline_timing, horizon, context, out_path) -> Path:
    """Per-epoch validation for both students, and free-running error by frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    for timing, name, colour in [
        (train_timing, "video model", "#1d4ed8"),
        (baseline_timing, "drive + exact fade", "#c2410c"),
    ]:
        if timing.get("val_history"):
            left.plot(
                range(1, len(timing["val_history"]) + 1),
                timing["val_history"],
                color=colour,
                label=f"{name}, held-out free-running",
            )
    if train_timing.get("loss_history"):
        left.plot(
            range(1, len(train_timing["loss_history"]) + 1),
            train_timing["loss_history"],
            color="#1d4ed8",
            ls=":",
            label="video model, train teacher-forced",
        )
    left.set_yscale("log")
    left.set_xlabel("epoch")
    left.set_ylabel("16-frame MSE (p99-scaled)")
    left.set_title("Training and validation")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    frames = np.arange(1, len(horizon["mse_by_frame"]) + 1)
    right.plot(frames, horizon["mse_by_frame"], color="#1d4ed8", marker=".", label="video model")
    if "baseline_mse_by_frame" in horizon:
        right.plot(
            frames,
            horizon["baseline_mse_by_frame"],
            color="#c2410c",
            marker=".",
            label="drive + exact fade",
        )
    right.axvline(context + 0.5, color="grey", lw=0.8, ls="-.", label="end of context window")
    right.set_xlabel("frame of a 40-frame held-out stream")
    right.set_ylabel("free-running MSE")
    right.set_title("Error vs horizon")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def bench_step(model, dataset_path, device, calls: int = 20) -> dict:
    """Per-frame cost with a full context window, next to the teacher's own ``step``."""
    from ..core.config import Config
    from ..core.registry import build_components
    from .axon_report import _dataset_meta

    meta = _dataset_meta(Path(dataset_path))
    cfg = Config.from_dict({**meta["config"], "device": str(device)})
    _, _, teacher = build_components(cfg, device)
    n_e = model.n_electrodes
    amp = torch.zeros(n_e, device=device)
    amp[:3] = 2.0
    action = Action(
        amp=amp,
        freq=torch.where(amp > 0, 40.0, 0.0),
        phase_dur=torch.where(amp > 0, 0.2, 0.0),
        rho=cfg.rho,
        axlambda=cfg.axlambda,
    )

    def _timed(fn, warm):
        state = warm()
        t0 = time.perf_counter()
        for _ in range(calls):
            state = fn(state)
        return (time.perf_counter() - t0) / calls * 1000.0

    def warm_video():
        state = None
        for _ in range(model.context):
            state = model.step(state, action)
        return state

    with torch.no_grad():
        video_ms = _timed(lambda s: model.step(s, action), warm_video)
        teacher_ms = _timed(lambda s: teacher.step(s, action), lambda: teacher.step(None, action))
        b = 8
        frames = torch.zeros(b, model.context, *model.grid_shape, device=device)
        acts = [
            a[None, None].expand(b, model.context, -1).contiguous()
            for a in (action.amp, action.freq, action.phase_dur)
        ]
        subj = [
            torch.full((b, model.context), float(v), device=device) for v in (cfg.rho, cfg.axlambda)
        ]
        batched_ms = _timed(lambda s: model(frames, *acts, *subj), lambda: None)
    return {
        "device": str(device),
        "grid_shape": list(model.grid_shape),
        "context": model.context,
        "calls": calls,
        "teacher_step_ms": teacher_ms,
        "video_step_ms": video_ms,
        "video_batch8_forward_ms": batched_ms,
        "video_batch8_ms_per_stream": batched_ms / b,
        "video_parameters": sum(p.numel() for p in model.parameters()),
    }


@click.group()
def cli():
    """Genie-style axon-map video world model."""


@cli.command()
@click.option("--dataset", "dataset_path", type=click.Path(path_type=Path), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(path_type=Path), required=True)
@click.option("--timing", "timing_path", type=click.Path(path_type=Path), default=None)
@click.option("--epochs", type=int, default=1, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=5e-4, show_default=True)
@click.option("--context", type=int, default=16, show_default=True)
@click.option("--dim", type=int, default=96, show_default=True)
@click.option("--depth", type=int, default=4, show_default=True)
@click.option("--heads", type=int, default=4, show_default=True)
@click.option("--patch-size", type=int, default=4, show_default=True)
@click.option("--context-noise", type=float, default=0.02, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--train-stride", type=int, default=8, show_default=True)
@click.option("--val-stride", type=int, default=16, show_default=True)
@click.option("--implant", "implant_name", type=str, default="argusii", show_default=True)
@click.option("--device", type=str, default=None)
def train(dataset_path, ckpt_path, timing_path, device, **kwargs):
    """Teacher-forced next-frame training; validation is free-running."""
    out = train_video(
        dataset_path,
        ckpt_path=ckpt_path,
        timing_path=timing_path,
        device=get_device(device),
        **kwargs,
    )
    click.echo(f"val_mse={out['val_mse']:.6f} samples_per_s={out['samples_per_s']:.2f}")


@cli.command()
@click.option("--dataset", "dataset_path", type=click.Path(path_type=Path), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(path_type=Path), required=True)
@click.option("--baseline-ckpt", type=click.Path(path_type=Path), default=None)
@click.option("--timing-train", type=click.Path(path_type=Path), default=None)
@click.option("--timing-gen", type=click.Path(path_type=Path), default=None)
@click.option("--timing-baseline", type=click.Path(path_type=Path), default=None)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("docs/assets/axon-video"))
@click.option("--device", type=str, default=None)
def report(
    dataset_path,
    ckpt_path,
    baseline_ckpt,
    timing_train,
    timing_gen,
    timing_baseline,
    out_dir,
    device,
):
    """Figures, GIF and JSON: learned fade, horizon error, timings."""
    from .axon_report import _dataset_meta, _read_json, load_student

    dev = get_device(device)
    model = load_video(ckpt_path, dev)
    meta = _dataset_meta(dataset_path)
    baseline = None
    if baseline_ckpt is not None:
        baseline = load_student(baseline_ckpt, device=dev, grid_shape=meta["grid_shape"])
    horizon = horizon_errors(model, dataset_path, dev, baseline)
    fade = fade_constant(model, dataset_path, dev)
    train_timing = _read_json(timing_train)
    baseline_timing = _read_json(timing_baseline)
    out_dir = Path(out_dir)
    figures = [
        plot_timeline(model, dataset_path, out_dir / "timeline.png", dev, baseline=baseline),
        plot_frames(model, dataset_path, out_dir / "frames.png", dev),
        write_gif(model, dataset_path, out_dir / "stream.gif", dev),
        plot_curves(
            train_timing, baseline_timing, horizon, model.context, out_dir / "validation.png"
        ),
    ]
    payload = {
        "dataset": meta,
        "arch": model.arch,
        "checkpoint_bytes": Path(ckpt_path).stat().st_size,
        "fade": fade,
        "horizon": horizon,
        "generation_timing": _read_json(timing_gen),
        "training_timing": train_timing,
        "baseline_training_timing": baseline_timing,
        "hardware": bench_step(model, dataset_path, dev),
        "figures": [str(f) for f in figures],
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(payload, indent=2))
    click.echo(
        f"free-running mse={horizon['mse']:.6f} "
        f"baseline={horizon.get('baseline_mse', float('nan')):.6f} "
        f"tau student={fade['student_tau_ms_median']:.1f} ms teacher={fade['teacher_tau_ms_median']:.1f} ms"
        f"\n{path}"
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
