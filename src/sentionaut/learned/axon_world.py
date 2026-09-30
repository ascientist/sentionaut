"""Specialist world model for axon-map percepts.

The network predicts the spatial drive. The leaky brightness update stays the
exact ``FadingTemporalTorch`` step used by ``BiphasicAxonMapTorch``. Electrodes
are a variable set, so patch tokens cross-attend to electrode tokens instead of
reading a padded action vector. ``rho`` and ``axlambda`` enter through adaptive
layer norm.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Callable

import click
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..core.base import Action, Implant, State
from ..core.device import get_device
from ..models.fading import FadingTemporalTorch
from .model import sincos_2d_pos_embed


class AdaLN(nn.Module):
    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.fc(cond).chunk(2, dim=-1)
        return self.norm(tokens) * (1 + scale[:, None, :]) + shift[:, None, :]


class DriveBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, cond_dim: int):
        super().__init__()
        self.adaln_attn = AdaLN(cond_dim, dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln_mlp = AdaLN(cond_dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens, elec, key_padding_mask, cond) -> torch.Tensor:
        h = self.adaln_attn(tokens, cond)
        attn, _ = self.cross(h, elec, elec, key_padding_mask=key_padding_mask, need_weights=False)
        tokens = tokens + attn
        return tokens + self.mlp(self.adaln_mlp(tokens, cond))


class AxonMapWorld(nn.Module):
    def __init__(
        self,
        grid_shape: tuple[int, int],
        n_electrodes: int,
        *,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        patch_size: int = 4,
        mlp_ratio: float = 2.0,
        dt_ms: float = 20.0,
        fade_tau_ms: float = 100.0,
        thresh_percept: float = 0.0,
    ):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4 for 2D sin/cos positions")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.n_electrodes = int(n_electrodes)
        self.dim = dim
        self.patch_size = patch_size
        self.dt_ms = float(dt_ms)
        self.fading = FadingTemporalTorch(tau_ms=fade_tau_ms, thresh_percept=thresh_percept)
        self.register_buffer("elec_xy", torch.zeros(self.n_electrodes, 2))

        self.elec_embed = nn.Linear(5, dim)
        self.cond_mlp = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.patch_embed = nn.Linear(patch_size * patch_size, dim)
        self.patch_unembed = nn.Linear(dim, patch_size * patch_size)
        self.blocks = nn.ModuleList([DriveBlock(dim, heads, mlp_ratio, dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def bind(self, implant: Implant) -> "AxonMapWorld":
        xy = implant.electrode_coords().detach()
        xy = xy.to(device=self.elec_xy.device, dtype=self.elec_xy.dtype)
        if xy.shape != self.elec_xy.shape:
            raise ValueError(
                f"implant electrodes {tuple(xy.shape)} != model {tuple(self.elec_xy.shape)}"
            )
        self.elec_xy.copy_(xy)
        return self

    def _to_patches(self, image: torch.Tensor):
        b, h, w = image.shape
        p = self.patch_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        n_h, n_w = hp // p, wp // p
        x = image.reshape(b, n_h, p, n_w, p).permute(0, 1, 3, 2, 4)
        return x.reshape(b, n_h * n_w, p * p), (h, w, n_h, n_w)

    def _from_patches(self, tokens: torch.Tensor, shape) -> torch.Tensor:
        h, w, n_h, n_w = shape
        b = tokens.shape[0]
        p = self.patch_size
        x = tokens.reshape(b, n_h, n_w, p, p).permute(0, 1, 3, 2, 4)
        return x.reshape(b, n_h * p, n_w * p)[:, :h, :w]

    def spatial_drive(self, brightness, amp, freq, phase_dur, rho, axlambda) -> torch.Tensor:
        """Positive drive field. ``brightness`` is ``(H, W)`` or ``(B, H, W)``."""
        single = brightness.ndim == 2
        if single:
            brightness = brightness[None]
            amp = amp[None]
            freq = freq[None]
            phase_dur = phase_dur[None]
        device, dtype = brightness.device, brightness.dtype
        b = brightness.shape[0]
        xy = self.elec_xy.to(dtype=dtype)[None].expand(b, -1, -1) / 1000.0
        feat = torch.cat(
            [xy, amp[..., None], freq[..., None] / 100.0, phase_dur[..., None]], dim=-1
        )
        elec = self.elec_embed(feat)
        # A fully masked key row makes softmax NaN. Keep one zero token valid and
        # zero that row's drive after the blocks.
        silent = amp == 0
        all_silent = silent.all(dim=1)
        if all_silent.any():
            silent = silent.clone()
            silent[all_silent, 0] = False
            elec = elec.clone()
            elec[all_silent] = 0

        patches, shape = self._to_patches(brightness)
        tokens = self.patch_embed(patches)
        pe = sincos_2d_pos_embed(shape[2], shape[3], self.dim, device, dtype)
        tokens = tokens + pe[None]
        rho_b = _batch_scalar(rho, b, device, dtype) / 1000.0
        axl_b = _batch_scalar(axlambda, b, device, dtype) / 1000.0
        cond = self.cond_mlp(torch.stack([rho_b, axl_b], dim=-1))
        for block in self.blocks:
            tokens = block(tokens, elec, silent, cond)
        tokens = self.norm(tokens)
        drive = F.relu(self._from_patches(self.patch_unembed(tokens), shape))
        if all_silent.any():
            drive = torch.where(all_silent[:, None, None], torch.zeros_like(drive), drive)
        return drive[0] if single else drive

    def predict_next(self, brightness, amp, freq, phase_dur, rho, axlambda) -> torch.Tensor:
        drive = self.spatial_drive(brightness, amp, freq, phase_dur, rho, axlambda)
        return self.fading.step(brightness, -drive, self.dt_ms)

    def step(self, state: State | None, action: Action) -> State:
        device = self.elec_xy.device
        dtype = self.elec_xy.dtype
        if state is None:
            brightness = torch.zeros(self.grid_shape, device=device, dtype=dtype)
        else:
            brightness = state.image.to(device=device, dtype=dtype)
        amp = action.amp.to(device=device, dtype=dtype)
        freq = action.freq if action.freq is not None else torch.zeros_like(action.amp)
        pdur = action.phase_dur if action.phase_dur is not None else torch.zeros_like(action.amp)
        freq = freq.to(device=device, dtype=dtype)
        pdur = pdur.to(device=device, dtype=dtype)
        rho = 200.0 if action.rho is None else action.rho
        axlambda = 500.0 if action.axlambda is None else action.axlambda
        nxt = self.predict_next(brightness, amp, freq, pdur, rho, axlambda)
        return State(image=nxt)


def _batch_scalar(value, batch: int, device, dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(batch)
    return tensor.reshape(batch)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class AxonSequenceDataset(Dataset):
    """K-step windows. The HDF5 file stays open for the life of the dataset."""

    def __init__(self, path: str | Path, windows: list[list[int]], scale: float):
        self.path = str(path)
        self.windows = windows
        self.scale = float(scale) if scale else 1.0
        self._h5 = None

    def __len__(self) -> int:
        return len(self.windows)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rows = self.windows[i]
        g = self._file()["world"]
        scale = self.scale
        s_t = g["s_t"][rows].astype(np.float32) / scale
        s_tp1 = g["s_tp1"][rows].astype(np.float32) / scale
        return {
            "s0": torch.from_numpy(s_t[0]),
            "target": torch.from_numpy(s_tp1),
            "amp": torch.from_numpy(g["amp"][rows].astype(np.float32)),
            "freq": torch.from_numpy(g["freq"][rows].astype(np.float32)),
            "phase_dur": torch.from_numpy(g["phase_dur"][rows].astype(np.float32)),
            "rho": torch.from_numpy(g["rho"][rows].astype(np.float32)),
            "axlambda": torch.from_numpy(g["axlambda"][rows].astype(np.float32)),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


def _episode_windows(
    path: Path, rollout_k: int
) -> tuple[list[list[int]], list[list[int]], float, tuple, int, float]:
    with h5py.File(path, "r") as h5:
        g = h5["world"]
        episode_id = g["episode_id"][:].astype(np.int64)
        step_id = g["step_in_episode"][:].astype(np.int64)
        grid = tuple(int(v) for v in h5["metadata"].attrs["grid_shape"])
        n_elec = int(h5["metadata"].attrs["max_electrodes"])
        dt_ms = float(h5["metadata"].attrs["dt_ms"])
        scale_raw = h5["metadata"].attrs.get("percept_scale")
        scale = 1.0
        if scale_raw:
            parsed = json.loads(scale_raw)
            scale = float(next(iter(parsed.values()))) or 1.0
    train_rows, val_rows = _holdout_rows(episode_id)
    train = _windows_in(episode_id, step_id, train_rows, rollout_k)
    val = _windows_in(episode_id, step_id, val_rows, rollout_k)
    if not train:
        train = _windows_in(episode_id, step_id, train_rows, 1)
    if not val:
        val = _windows_in(episode_id, step_id, val_rows, 1)
    if not train or not val:
        raise RuntimeError(f"no train/val windows in {path}")
    return train, val, scale, grid, n_elec, dt_ms


def _holdout_rows(episode_id: np.ndarray) -> tuple[set[int], set[int]]:
    episodes = np.unique(episode_id)
    if len(episodes) == 1:
        # One episode has no episode to hold out. Cut the last 20% of its steps.
        n = len(episode_id)
        cut = max(1, int(n * 0.8))
        if cut >= n:
            cut = n - 1
        return set(range(cut)), set(range(cut, n))
    n_val = max(1, int(round(len(episodes) * 0.2)))
    val_eps = set(int(e) for e in episodes[-n_val:])
    train, val = set(), set()
    for i, ep in enumerate(episode_id):
        (val if int(ep) in val_eps else train).add(i)
    return train, val


def _windows_in(episode_id, step_id, allowed: set[int], k: int) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for i, ep in enumerate(episode_id):
        groups.setdefault(int(ep), []).append(i)
    windows = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda i: int(step_id[i]))
        usable = [i for i in rows if i in allowed]
        if not usable:
            continue
        pos = [rows.index(i) for i in usable]
        contiguous = pos == list(range(pos[0], pos[0] + len(pos)))
        if not contiguous:
            continue
        kk = min(k, len(usable))
        for start in range(0, len(usable) - kk + 1):
            windows.append(usable[start : start + kk])
    return windows


def _rollout_loss(model: AxonMapWorld, batch: dict) -> torch.Tensor:
    brightness = batch["s0"]
    k = batch["target"].shape[1]
    loss = brightness.new_zeros(())
    for t in range(k):
        brightness = model.predict_next(
            brightness,
            batch["amp"][:, t],
            batch["freq"][:, t],
            batch["phase_dur"][:, t],
            batch["rho"][:, t],
            batch["axlambda"][:, t],
        )
        loss = loss + F.mse_loss(brightness, batch["target"][:, t])
    return loss / k


def _save_ckpt(
    path: Path, model, opt, epoch: int, history: list[float], val_history: list[float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "loss_history": history,
            "val_history": val_history,
            # Width, depth and electrode count are recoverable from the weights;
            # these three are not.
            "grid_shape": model.grid_shape,
            "heads": model.blocks[0].cross.num_heads,
            "dt_ms": model.dt_ms,
            "arch": getattr(model, "arch", None),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        },
        path,
    )


def _load_ckpt(path: Path, model, opt) -> tuple[int, list[float], list[float]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    return int(ckpt["epoch"]), list(ckpt["loss_history"]), list(ckpt.get("val_history", []))


def distill_hdf5(
    dataset_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    rollout_k: int = 4,
    device: torch.device | None = None,
    ckpt_path: str | Path | None = None,
    timing_path: str | Path | None = None,
    implant_name: str = "argusii",
    dim: int = 64,
    depth: int = 2,
    heads: int = 4,
    patch_size: int = 4,
    num_workers: int = 0,
    bf16: bool = False,
    model_factory: Callable[[tuple, int, float], nn.Module] | None = None,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor] | None = None,
    train_stride: int = 1,
    val_stride: int = 1,
) -> dict:
    """Train on axon-map transitions. Default is ``AxonMapWorld``, K-step MSE after the exact fade.

    ``model_factory(grid, n_electrodes, dt_ms)`` and ``loss_fn(model, batch)`` swap in
    another student over the same windows, checkpoints, and timings. A stride keeps every
    ``stride``-th K-step window; windows overlap by ``K - 1`` frames otherwise.
    """
    dataset_path = Path(dataset_path)
    device = device or get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_w, val_w, scale, grid, n_elec, dt_ms = _episode_windows(dataset_path, rollout_k)
    train_w, val_w = train_w[:: max(1, train_stride)], val_w[:: max(1, val_stride)]
    if model_factory is None:
        model = AxonMapWorld(
            grid, n_elec, dim=dim, depth=depth, heads=heads, patch_size=patch_size, dt_ms=dt_ms
        )
    else:
        model = model_factory(grid, n_elec, dt_ms)
    model = model.to(device)
    loss_fn = loss_fn or _rollout_loss
    from ..core.config import Config
    from ..implants.registry import build_implant

    model.bind(build_implant(Config(model="axonmap", implant=implant_name), device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    history: list[float] = []
    val_history: list[float] = []
    ckpt = Path(ckpt_path) if ckpt_path else None
    if ckpt is not None and ckpt.exists():
        start_epoch, history, val_history = _load_ckpt(ckpt, model, opt)
        model.to(device)

    train_ds = AxonSequenceDataset(dataset_path, train_w, scale)
    val_ds = AxonSequenceDataset(dataset_path, val_w, scale)
    pin = device.type == "cuda"
    loader_kw = {"batch_size": batch_size, "pin_memory": pin, "num_workers": num_workers}
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    stop = {"flag": False}
    previous = signal.getsignal(signal.SIGTERM)

    def _on_term(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)
    t_load = 0.0
    t_step = 0.0
    t_val = 0.0
    n_samples = 0
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total, count = 0.0, 0
            tick = time.perf_counter()
            for batch in train_loader:
                t_load += time.perf_counter() - tick
                batch = {k: v.to(device, non_blocking=pin) for k, v in batch.items()}
                t0 = time.perf_counter()
                opt.zero_grad()
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                    loss = loss_fn(model, batch)
                loss.backward()
                opt.step()
                _sync(device)
                t_step += time.perf_counter() - t0
                n_samples += batch["s0"].shape[0]
                total += float(loss.detach().cpu())
                count += 1
                tick = time.perf_counter()
                if stop["flag"]:
                    break
            history.append(total / max(count, 1))
            t0 = time.perf_counter()
            val_history.append(_eval_mse(model, val_loader, device, pin, bf16, loss_fn))
            t_val += time.perf_counter() - t0
            if ckpt is not None:
                _save_ckpt(ckpt, model, opt, epoch + 1, history, val_history)
            if stop["flag"]:
                break
        if not val_history:
            val_history.append(_eval_mse(model, val_loader, device, pin, bf16, loss_fn))
    finally:
        signal.signal(signal.SIGTERM, previous)
        train_ds.close()
        val_ds.close()

    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    timing = {
        "data_load_s": t_load,
        "forward_backward_s": t_step,
        "validation_s": t_val,
        "n_samples": n_samples,
        "samples_per_s": n_samples / max(t_load + t_step, 1e-9),
        "peak_gpu_mem_bytes": int(peak),
        "val_mse": val_history[-1],
        "loss_history": history,
        "val_history": val_history,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "rollout_k": rollout_k,
        "grid_shape": list(grid),
        "n_electrodes": n_elec,
        "n_train_windows": len(train_w),
        "n_val_windows": len(val_w),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "bf16": bool(bf16),
        "device": str(device),
    }
    if timing_path is not None:
        path = Path(timing_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(timing, indent=2))
    timing["model"] = model
    return timing


@torch.no_grad()
def _eval_mse(model, loader, device, pin: bool, bf16: bool = False, loss_fn=None) -> float:
    loss_fn = loss_fn or _rollout_loss
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=pin) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = loss_fn(model, batch)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def distill_online(
    teacher,
    student: AxonMapWorld,
    *,
    steps: int = 10,
    rollout_k: int = 4,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """One-rollout MSE against a built axon-map teacher. Local smoke path."""
    from ..core.config import Config
    from ..generate import ActionRanges, sample_action

    device = student.elec_xy.device
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    ranges = ActionRanges()
    cfg = Config(model="axonmap", implant="argusii")
    n_e = student.n_electrodes
    history = []
    for _ in range(steps):
        actions = []
        for t in range(rollout_k):
            action, _ = sample_action(cfg, n_e, rng, ranges, device, silent=t == rollout_k - 1)
            actions.append(action)
        state = None
        targets = []
        for action in actions:
            state = teacher.step(state, action)
            targets.append(state.image.detach())
        brightness = torch.zeros(student.grid_shape, device=device, dtype=student.elec_xy.dtype)
        loss = brightness.new_zeros(())
        for action, target in zip(actions, targets):
            brightness = student.step(State(image=brightness), action).image
            loss = loss + F.mse_loss(brightness, target)
        loss = loss / rollout_k
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))
    return history


@click.command()
@click.option("--dataset", "dataset_path", type=click.Path(path_type=Path), required=True)
@click.option("--epochs", type=int, default=1, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--rollout-k", type=int, default=4, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--ckpt", "ckpt_path", type=click.Path(path_type=Path), required=True)
@click.option("--timing", "timing_path", type=click.Path(path_type=Path), default=None)
@click.option("--dim", type=int, default=64, show_default=True)
@click.option("--depth", type=int, default=2, show_default=True)
@click.option("--implant", "implant_name", type=str, default="argusii", show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--bf16", is_flag=True, default=False)
@click.option("--train-stride", type=int, default=1, show_default=True)
@click.option("--val-stride", type=int, default=1, show_default=True)
def cli(
    dataset_path,
    epochs,
    batch_size,
    lr,
    rollout_k,
    device,
    ckpt_path,
    timing_path,
    dim,
    depth,
    implant_name,
    num_workers,
    bf16,
    train_stride,
    val_stride,
):
    dev = get_device(device)
    if dev.type != "cuda" and device == "cuda":
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    out = distill_hdf5(
        dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        rollout_k=rollout_k,
        device=dev,
        ckpt_path=ckpt_path,
        timing_path=timing_path,
        implant_name=implant_name,
        dim=dim,
        depth=depth,
        num_workers=num_workers,
        bf16=bf16,
        train_stride=train_stride,
        val_stride=val_stride,
    )
    click.echo(
        f"val_mse={out['val_mse']:.6f} samples_per_s={out['samples_per_s']:.2f} "
        f"loss_history={out['loss_history']}"
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
