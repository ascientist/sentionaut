"""Held-out samples, validation curves, and teacher/student timings for ``AxonMapWorld``.

Reads a distillation checkpoint plus the HDF5 the teacher wrote, and emits the
figures and the JSON numbers quoted on the docs page. The validation episodes are
the same last-20% split ``distill_hdf5`` trains against, so the metrics here and
the ``val_mse`` in ``timing_train.json`` measure the same rollouts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..core.base import Action  # noqa: E402
from ..core.config import Config  # noqa: E402
from ..core.device import get_device  # noqa: E402
from .axon_world import AxonMapWorld, _episode_windows, _holdout_rows  # noqa: E402


def load_student(
    ckpt_path: str | Path,
    *,
    device: torch.device | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> AxonMapWorld:
    """Rebuild the student from a checkpoint. Width and depth come from the weights."""
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    grid = tuple(ckpt.get("grid_shape") or grid_shape or ())
    if len(grid) != 2:
        raise ValueError("checkpoint has no grid_shape; pass grid_shape from the dataset")
    model = AxonMapWorld(
        grid,
        int(sd["elec_xy"].shape[0]),
        dim=int(sd["norm.weight"].shape[0]),
        depth=1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")),
        heads=int(ckpt.get("heads", 4)),
        patch_size=int(round(sd["patch_embed.weight"].shape[1] ** 0.5)),
        dt_ms=float(ckpt.get("dt_ms", 20.0)),
    )
    model.load_state_dict(sd)
    return model.to(device or torch.device("cpu")).eval()


def _dataset_meta(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        meta = h5["metadata"].attrs
        scale_raw = meta.get("percept_scale")
        scale = 1.0
        if scale_raw:
            scale = float(next(iter(json.loads(scale_raw).values()))) or 1.0
        return {
            "grid_shape": tuple(int(v) for v in meta["grid_shape"]),
            "n_electrodes": int(meta["max_electrodes"]),
            "dt_ms": float(meta["dt_ms"]),
            "config": json.loads(meta["config_table"])[0],
            "scale": scale,
            "n_transitions": int(h5["world"]["s_t"].shape[0]),
        }


def _val_episodes(path: Path) -> dict[int, list[int]]:
    """Held-out episodes as ``episode id -> rows ordered by step``."""
    with h5py.File(path, "r") as h5:
        g = h5["world"]
        episode_id = g["episode_id"][:].astype(np.int64)
        step_id = g["step_in_episode"][:].astype(np.int64)
    _, val_rows = _holdout_rows(episode_id)
    episodes: dict[int, list[int]] = {}
    for row in sorted(val_rows):
        episodes.setdefault(int(episode_id[row]), []).append(row)
    for ep, rows in episodes.items():
        episodes[ep] = sorted(rows, key=lambda i: int(step_id[i]))
    return episodes


def _rows_batch(path: Path, rows: list[int], scale: float) -> dict[str, torch.Tensor]:
    with h5py.File(path, "r") as h5:
        g = h5["world"]
        return {
            "s_t": torch.from_numpy(g["s_t"][rows].astype(np.float32) / scale),
            "s_tp1": torch.from_numpy(g["s_tp1"][rows].astype(np.float32) / scale),
            "amp": torch.from_numpy(g["amp"][rows].astype(np.float32)),
            "freq": torch.from_numpy(g["freq"][rows].astype(np.float32)),
            "phase_dur": torch.from_numpy(g["phase_dur"][rows].astype(np.float32)),
            "rho": torch.from_numpy(g["rho"][rows].astype(np.float32)),
            "axlambda": torch.from_numpy(g["axlambda"][rows].astype(np.float32)),
        }


@torch.no_grad()
def _free_rollout(model: AxonMapWorld, batch: dict, device: torch.device) -> torch.Tensor:
    """Student frames for a whole episode, fed only its own previous prediction."""
    brightness = batch["s_t"][:1].to(device)
    frames = []
    for t in range(batch["s_tp1"].shape[0]):
        brightness = model.predict_next(
            brightness,
            batch["amp"][t : t + 1].to(device),
            batch["freq"][t : t + 1].to(device),
            batch["phase_dur"][t : t + 1].to(device),
            batch["rho"][t : t + 1].to(device),
            batch["axlambda"][t : t + 1].to(device),
        )
        frames.append(brightness[0].cpu())
    return torch.stack(frames)


@torch.no_grad()
def holdout_metrics(
    model: AxonMapWorld,
    dataset_path: str | Path,
    *,
    rollout_k: int = 4,
    device: torch.device | None = None,
    max_windows: int | None = None,
) -> dict:
    """K-step rollout error on the held-out episodes, in the training's scaled units."""
    dataset_path = Path(dataset_path)
    device = device or torch.device("cpu")
    _, val_windows, scale, _, _, _ = _episode_windows(dataset_path, rollout_k)
    if max_windows is not None:
        val_windows = val_windows[:max_windows]
    k = max(len(w) for w in val_windows)
    sq_per_step = np.zeros(k)
    abs_per_step = np.zeros(k)
    count_per_step = np.zeros(k)
    teacher_all: list[np.ndarray] = []
    student_all: list[np.ndarray] = []
    peak_abs_err = 0.0
    for rows in val_windows:
        batch = _rows_batch(dataset_path, rows, scale)
        student = _free_rollout(model, batch, device).numpy()
        teacher = batch["s_tp1"].numpy()
        err = student - teacher
        for t in range(len(rows)):
            sq_per_step[t] += float((err[t] ** 2).mean())
            abs_per_step[t] += float(np.abs(err[t]).mean())
            count_per_step[t] += 1
        peak_abs_err = max(peak_abs_err, float(np.abs(teacher.max() - student.max())))
        teacher_all.append(teacher.ravel())
        student_all.append(student.ravel())
    teacher_flat = np.concatenate(teacher_all)
    student_flat = np.concatenate(student_all)
    mse = float(((student_flat - teacher_flat) ** 2).mean())
    teacher_range = float(teacher_flat.max() - teacher_flat.min())
    return {
        "n_val_windows": len(val_windows),
        "rollout_k": k,
        "percept_scale": scale,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(student_flat - teacher_flat).mean()),
        "nrmse_vs_teacher_range": float(np.sqrt(mse) / max(teacher_range, 1e-12)),
        "pearson_r": float(np.corrcoef(student_flat, teacher_flat)[0, 1]),
        "teacher_max": float(teacher_flat.max()),
        "peak_brightness_abs_error": peak_abs_err,
        "mse_per_step": (sq_per_step / np.maximum(count_per_step, 1)).tolist(),
        "mae_per_step": (abs_per_step / np.maximum(count_per_step, 1)).tolist(),
    }


def bench_forward(
    dataset_path: str | Path,
    model: AxonMapWorld,
    *,
    device: torch.device | None = None,
    calls: int = 20,
    batch_sizes: tuple[int, ...] = (1, 8, 64),
) -> dict:
    """Teacher vs student spatial cost on the dataset's own grid, plus the build the student skips."""
    from ..core.registry import build_components

    device = device or torch.device("cpu")
    meta = _dataset_meta(Path(dataset_path))
    cfg = Config.from_dict({**meta["config"], "device": str(device)})

    t0 = time.perf_counter()
    implant, _, teacher = build_components(cfg, device)
    build_s = time.perf_counter() - t0

    n_e = implant.n_electrodes
    amp = torch.zeros(n_e, device=device)
    amp[: min(3, n_e)] = 2.0
    freq = torch.where(amp > 0, torch.full_like(amp, 40.0), amp)
    pdur = torch.where(amp > 0, torch.full_like(amp, 0.2), amp)
    action = Action(amp=amp, freq=freq, phase_dur=pdur, rho=cfg.rho, axlambda=cfg.axlambda)

    def _timed(fn) -> float:
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(calls):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / calls * 1000.0

    out: dict = {
        "device": str(device),
        "grid_shape": list(meta["grid_shape"]),
        "n_electrodes": n_e,
        "axon_samples": int(teacher.topography.coords.shape[1]),
        "calls": calls,
        "teacher_topography_build_ms": build_s * 1000.0,
        "teacher_axon_tensor_bytes": int(
            teacher.topography.coords.numel() * teacher.topography.coords.element_size()
        ),
        "student_parameters": sum(p.numel() for p in model.parameters()),
        "teacher_spatial_forward_ms": _timed(lambda: teacher.spatial_forward(action)),
        "student_spatial_drive_ms": {},
    }
    with torch.no_grad():
        for b in batch_sizes:
            brightness = torch.zeros(b, *meta["grid_shape"], device=device)
            amp_b = amp[None].expand(b, -1).contiguous()
            freq_b = freq[None].expand(b, -1).contiguous()
            pdur_b = pdur[None].expand(b, -1).contiguous()
            rho_b = torch.full((b,), cfg.rho, device=device)
            axl_b = torch.full((b,), cfg.axlambda, device=device)
            ms = _timed(
                lambda: model.spatial_drive(brightness, amp_b, freq_b, pdur_b, rho_b, axl_b)
            )
            out["student_spatial_drive_ms"][str(b)] = ms
            out["student_spatial_drive_ms_per_sample"] = ms / b
    return out


def plot_samples(
    model: AxonMapWorld,
    dataset_path: str | Path,
    out_path: str | Path,
    *,
    device: torch.device | None = None,
    n_episodes: int = 2,
) -> Path:
    """Teacher frames, free-running student frames, and their difference, on held-out episodes."""
    dataset_path = Path(dataset_path)
    device = device or torch.device("cpu")
    scale = _dataset_meta(dataset_path)["scale"]
    episodes = list(_val_episodes(dataset_path).items())[:n_episodes]
    n_steps = max(len(rows) for _, rows in episodes)
    fig, axes = plt.subplots(
        3 * len(episodes), n_steps, figsize=(1.9 * n_steps, 2.0 * 3 * len(episodes))
    )
    axes = np.atleast_2d(axes)
    for e, (ep, rows) in enumerate(episodes):
        batch = _rows_batch(dataset_path, rows, scale)
        teacher = batch["s_tp1"].numpy()
        student = _free_rollout(model, batch, device).numpy()
        vmax = max(float(teacher.max()), float(student.max()), 1e-6)
        emax = max(float(np.abs(student - teacher).max()), 1e-6)
        for t in range(n_steps):
            active = int((batch["amp"][t] != 0).sum())
            for r, (img, label, cmap, hi) in enumerate(
                [
                    (teacher[t], "teacher", "magma", vmax),
                    (student[t], "student", "magma", vmax),
                    (np.abs(student[t] - teacher[t]), "|error|", "viridis", emax),
                ]
            ):
                ax = axes[3 * e + r, t]
                im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=hi)
                ax.set_xticks([])
                ax.set_yticks([])
                if t == 0:
                    ax.set_ylabel(f"ep {ep}\n{label}", fontsize=8)
                if r == 0:
                    ax.set_title(f"t={t + 1}, {active} on", fontsize=8)
                if t == n_steps - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Held-out episodes: axon-map teacher vs free-running student", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_validation(train_timing: dict, metrics: dict, out_path: str | Path) -> Path:
    """Loss and held-out MSE per epoch, and how the error grows across the unrolled steps."""
    loss = train_timing.get("loss_history") or []
    val = train_timing.get("val_history") or []
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = np.arange(1, len(loss) + 1)
    left.plot(epochs, loss, label="train K-step MSE", color="#c2410c")
    if val:
        left.plot(np.arange(1, len(val) + 1), val, label="held-out K-step MSE", color="#1d4ed8")
    left.set_yscale("log")
    left.set_xlabel("epoch")
    left.set_ylabel("MSE (percept units, p99-scaled)")
    left.set_title("Training and validation")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    per_step = metrics.get("mse_per_step") or []
    right.bar(np.arange(1, len(per_step) + 1), per_step, color="#1d4ed8")
    right.set_xlabel("unrolled step")
    right.set_ylabel("held-out MSE")
    right.set_title("Error across the rollout")
    right.set_xticks(np.arange(1, len(per_step) + 1))
    right.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _read_json(path: str | Path | None) -> dict:
    if path is None:
        return {}
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


@click.command()
@click.option("--dataset", "dataset_path", type=click.Path(path_type=Path), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(path_type=Path), required=True)
@click.option("--timing-gen", type=click.Path(path_type=Path), default=None)
@click.option("--timing-train", type=click.Path(path_type=Path), default=None)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("artifacts"))
@click.option("--prefix", type=str, default="axon_world", show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--rollout-k", type=int, default=4, show_default=True)
@click.option("--episodes", type=int, default=2, show_default=True)
@click.option("--bench-calls", type=int, default=20, show_default=True)
@click.option("--max-windows", type=int, default=None)
def cli(
    dataset_path,
    ckpt_path,
    timing_gen,
    timing_train,
    out_dir,
    prefix,
    device,
    rollout_k,
    episodes,
    bench_calls,
    max_windows,
):
    """Write the samples figure, the validation figure, and the report JSON."""
    dev = get_device(device)
    meta = _dataset_meta(dataset_path)
    model = load_student(ckpt_path, device=dev, grid_shape=meta["grid_shape"])
    metrics = holdout_metrics(
        model, dataset_path, rollout_k=rollout_k, device=dev, max_windows=max_windows
    )
    train_timing = _read_json(timing_train)
    samples = plot_samples(
        model,
        dataset_path,
        Path(out_dir) / f"{prefix}_samples.png",
        device=dev,
        n_episodes=episodes,
    )
    curves = plot_validation(train_timing, metrics, Path(out_dir) / f"{prefix}_validation.png")
    report = {
        "dataset": meta,
        "checkpoint_bytes": Path(ckpt_path).stat().st_size,
        "holdout": metrics,
        "generation_timing": _read_json(timing_gen),
        "training_timing": train_timing,
        "hardware": bench_forward(dataset_path, model, device=dev, calls=bench_calls),
        "figures": [str(samples), str(curves)],
    }
    report_path = Path(out_dir) / f"{prefix}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    click.echo(
        f"holdout_mse={metrics['mse']:.6f} pearson_r={metrics['pearson_r']:.4f} "
        f"nrmse={metrics['nrmse_vs_teacher_range']:.4f}\n{report_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
