"""Render one animation per analytical physics model on the local device (MPS).

Each clip has dual synced panels: the GPU-rendered percept and a tissue-geometry
schematic. Axon Map sweeps rho/axlambda (phase-offset) and translates the
implant; the cortical Scoreboard and Dynaphos sweep the implant toward the
periphery to expose cortical-magnification growth (Dynaphos additionally shows
temporal charge buildup across frames).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

# pulse2percept forces "TkAgg" on macOS at import time; import it first, then
# force a headless backend so figure rendering works without a display.
import pulse2percept  # noqa: E402, F401

matplotlib.use("Agg", force=True)
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .core.base import Action, Pose  # noqa: E402
from .core.config import Config  # noqa: E402
from .core.device import get_device  # noqa: E402
from .core.registry import build_components  # noqa: E402


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[..., :3].copy()


def _write(frames: list[np.ndarray], outdir: Path, name: str, fps: int = 12) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    gif = outdir / f"{name}.gif"
    mp4 = outdir / f"{name}.mp4"
    imageio.mimsave(gif, frames, duration=1.0 / fps, loop=0)
    paths = [gif]
    try:
        imageio.mimsave(mp4, frames, fps=fps)
        paths.append(mp4)
    except Exception:  # mp4 needs ffmpeg; gif is the guaranteed deliverable
        pass
    return paths


def _percept_extent(cfg: Config):
    return [cfg.xrange[0], cfg.xrange[1], cfg.yrange[0], cfg.yrange[1]]


def animate_axonmap(outdir: Path, device: torch.device, n_frames: int = 48) -> list[Path]:
    cfg = Config(model="axonmap", implant="argusii", xrange=(-12, 12), yrange=(-12, 12), xystep=0.5)
    implant, topo, model = build_components(cfg, device)
    names = implant.names
    sel = [names[i] for i in (20, 28, 36) if i < len(names)]
    sel_idx = [implant.names.index(s) for s in sel]
    N = implant.n_electrodes

    coords = topo.coords.reshape(-1, 2).cpu().numpy()
    coords = coords[np.random.default_rng(0).choice(coords.shape[0], size=4000, replace=False)]
    elec = implant.electrode_coords().cpu().numpy()

    pulse_frames = n_frames // 2
    state = None
    frames = []
    for t in range(n_frames):
        phase = 2 * math.pi * t / n_frames
        rho = 200.0 * (1.0 + 0.6 * math.sin(phase))
        axl = 500.0 * (1.0 + 0.5 * math.sin(phase + math.pi / 2))
        dx = 400.0 * math.sin(phase)
        amp = torch.zeros(N, device=device)
        freq = torch.zeros(N, device=device)
        pdur = torch.zeros(N, device=device)
        if t < pulse_frames:
            for e in sel_idx:
                amp[e] = 2.0
                freq[e] = 30.0
                pdur[e] = 0.45
        action = Action(amp=amp, freq=freq, phase_dur=pdur, rho=rho, axlambda=axl, pose=Pose(dx=dx))
        state = model.step(state, action)
        img = state.image.detach().cpu().numpy()

        fig, (axp, axt) = plt.subplots(1, 2, figsize=(9, 4.5))
        axp.imshow(img, cmap="inferno", extent=_percept_extent(cfg), origin="lower")
        axp.set_title(
            f"Axon Map percept\nrho={rho:.0f}  axlambda={axl:.0f}"
            + (" (fade)" if t >= pulse_frames else "")
        )
        axp.set_xlabel("x (dva)")
        axp.set_ylabel("y (dva)")
        axt.scatter(coords[:, 0], coords[:, 1], s=0.5, c="0.7", alpha=0.4)
        axt.scatter(elec[:, 0] + dx, elec[:, 1], s=18, c="tab:blue")
        for e in sel_idx:
            axt.scatter(elec[e, 0] + dx, elec[e, 1], s=60, c="tab:red")
        axt.set_title("retinal tissue (axon bundles + array)")
        axt.set_xlabel("x (microns)")
        axt.set_ylabel("y (microns)")
        axt.set_aspect("equal")
        fig.tight_layout()
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)
    return _write(frames, outdir, "axonmap")


def _animate_cortical(
    model_name: str, outdir: Path, device: torch.device, n_frames: int, thread_state: bool
) -> list[Path]:
    cfg = Config(
        model=model_name,
        implant="orion",
        xrange=(-6, 6),
        yrange=(-6, 6),
        xystep=0.2,
        regions=("v1",),
    )
    implant, topo, model = build_components(cfg, device)
    N = implant.n_electrodes
    sel_idx = [0, 25, 50]
    amp_val = 200.0 if model_name == "dynaphos" else 250.0
    pulse_frames = n_frames // 2

    cortex = topo.cortex_xy["v1"].cpu().numpy()
    cortex = cortex[np.isfinite(cortex).all(axis=1)]
    elec_base = implant.electrode_coords().cpu().numpy()

    state = None
    frames = []
    for t in range(n_frames):
        frac = t / max(n_frames - 1, 1)
        dx = 5000.0 * frac
        amp = torch.zeros(N, device=device)
        if t < pulse_frames:
            for e in sel_idx:
                amp[e] = amp_val
        rho = None if model_name == "dynaphos" else 1000.0
        action = Action(amp=amp, rho=rho, pose=Pose(dx=dx))
        state = model.step(state, action)
        img = state.image.detach().cpu().numpy()

        fig, (axp, axt) = plt.subplots(1, 2, figsize=(9, 4.5))
        axp.imshow(img, cmap="inferno", extent=_percept_extent(cfg), origin="lower")
        title = f"{model_name} percept"
        if t >= pulse_frames:
            title += " (fade)"
        elif thread_state:
            title += f"\nframe {t} (charge buildup)"
        axp.set_title(title)
        axp.set_xlabel("x (dva)")
        axp.set_ylabel("y (dva)")
        axt.scatter(cortex[:, 0] / 1000.0, cortex[:, 1] / 1000.0, s=0.5, c="0.7", alpha=0.4)
        elec = elec_base.copy()
        elec[:, 0] += dx
        axt.scatter(elec[:, 0] / 1000.0, elec[:, 1] / 1000.0, s=14, c="tab:blue")
        for e in sel_idx:
            axt.scatter(elec[e, 0] / 1000.0, elec[e, 1] / 1000.0, s=55, c="tab:red")
        axt.set_title("cortical tissue (V1 map + electrodes)")
        axt.set_xlabel("x (mm)")
        axt.set_ylabel("y (mm)")
        axt.set_aspect("equal")
        fig.tight_layout()
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)
    return _write(frames, outdir, model_name)


def animate_scoreboard(outdir: Path, device: torch.device, n_frames: int = 48) -> list[Path]:
    return _animate_cortical("scoreboard", outdir, device, n_frames, thread_state=True)


def animate_dynaphos(outdir: Path, device: torch.device, n_frames: int = 48) -> list[Path]:
    return _animate_cortical("dynaphos", outdir, device, n_frames, thread_state=True)


ANIMATORS = {
    "axonmap": animate_axonmap,
    "scoreboard": animate_scoreboard,
    "dynaphos": animate_dynaphos,
}


def main(
    model: str = "all", outdir: str = "artifacts", n_frames: int = 48, device: str | None = None
) -> list[Path]:
    dev = get_device(device)
    out = Path(outdir)
    names = list(ANIMATORS) if model == "all" else [model]
    written: list[Path] = []
    for name in names:
        written += ANIMATORS[name](out, dev, n_frames)
    return written


def cli():  # pragma: no cover - thin click wrapper
    import click

    @click.command()
    @click.option(
        "--model",
        type=click.Choice(["axonmap", "scoreboard", "dynaphos", "all"]),
        default="all",
        show_default=True,
    )
    @click.option("--outdir", type=str, default="artifacts", show_default=True)
    @click.option("--frames", "n_frames", type=int, default=48, show_default=True)
    @click.option("--device", type=str, default=None)
    def _cli(model, outdir, n_frames, device):
        paths = main(model, outdir, n_frames, device)
        for p in paths:
            click.echo(str(p))

    _cli()


if __name__ == "__main__":  # pragma: no cover
    cli()
