"""Render one animation per analytical physics model on the local device (MPS).

Each clip has dual synced panels: the GPU-rendered percept and a tissue-geometry
schematic. Two scenarios:

- ``sweep``: Axon Map sweeps rho/axlambda (phase-offset) and translates the
  implant; the cortical Scoreboard and Dynaphos sweep the implant toward the
  periphery to expose cortical-magnification growth.
- ``sequence``: the implant stays still and a small zone of neighbouring
  electrodes receives a stimulation sequence (single electrodes, pairs, an
  amplitude ramp, then a repeated pulse train), so the clip shows how each model
  responds to stimulation over time at one retinal/cortical location.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import NamedTuple

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

RETINAL_LEVELS = (1.0, 2.0, 3.0)  # x threshold
CORTICAL_LEVELS = (100.0, 200.0, 300.0)  # uA


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[..., :3].copy()


def _write(
    frames: list[np.ndarray], outdir: Path, name: str, fps: int = 12, mp4: bool = True
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    gif = outdir / f"{name}.gif"
    imageio.mimsave(gif, frames, duration=1.0 / fps, loop=0)
    paths = [gif]
    if mp4:
        try:
            imageio.mimsave(outdir / f"{name}.mp4", frames, fps=fps)
            paths.append(outdir / f"{name}.mp4")
        except Exception:  # mp4 needs ffmpeg; gif is the guaranteed deliverable
            pass
    return paths


class Scene(NamedTuple):
    cfg: Config
    implant: object
    topo: object
    model: object
    tissue: np.ndarray
    tissue_title: str
    unit: str
    scale: float


def _scene(
    model_name: str,
    device: torch.device,
    window: tuple[tuple[float, float], tuple[float, float], float] = ((-6, 6), (-6, 6), 0.2),
) -> Scene:
    """Build the components and tissue scatter; ``window`` only applies to cortical models."""
    if model_name == "axonmap":
        cfg = Config(
            model="axonmap", implant="argusii", xrange=(-12, 12), yrange=(-12, 12), xystep=0.5
        )
        implant, topo, model = build_components(cfg, device)
        coords = topo.coords.reshape(-1, 2).cpu().numpy()
        rng = np.random.default_rng(0)
        tissue = coords[rng.choice(coords.shape[0], size=4000, replace=False)]
        return Scene(
            cfg,
            implant,
            topo,
            model,
            tissue,
            "retinal tissue (axon bundles + array)",
            "microns",
            1.0,
        )
    xrange, yrange, xystep = window
    cfg = Config(
        model=model_name,
        implant="orion",
        xrange=xrange,
        yrange=yrange,
        xystep=xystep,
        regions=("v1",),
    )
    implant, topo, model = build_components(cfg, device)
    cortex = topo.cortex_xy["v1"].cpu().numpy()
    tissue = cortex[np.isfinite(cortex).all(axis=1)]
    return Scene(
        cfg, implant, topo, model, tissue, "cortical tissue (V1 map + electrodes)", "mm", 1000.0
    )


def _draw_frame(
    scene: Scene,
    img: np.ndarray,
    title: str,
    elec: np.ndarray,
    highlight: list[int],
    active: tuple[int, ...] = (),
    vmax: float | None = None,
) -> np.ndarray:
    cfg = scene.cfg
    s = scene.scale
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(8, 4), dpi=80)
    extent = [cfg.xrange[0], cfg.xrange[1], cfg.yrange[0], cfg.yrange[1]]
    axp.imshow(img, cmap="inferno", extent=extent, origin="lower", vmin=0.0, vmax=vmax)
    axp.set_title(title, fontsize=10)
    axp.set_xlabel("x (dva)")
    axp.set_ylabel("y (dva)")
    axt.scatter(scene.tissue[:, 0] / s, scene.tissue[:, 1] / s, s=0.5, c="0.7", alpha=0.4)
    axt.scatter(elec[:, 0] / s, elec[:, 1] / s, s=16, c="tab:blue")
    for e in highlight:
        axt.scatter(elec[e, 0] / s, elec[e, 1] / s, s=60, c="tab:red")
    for e in active:
        axt.scatter(elec[e, 0] / s, elec[e, 1] / s, s=140, facecolors="none", edgecolors="yellow")
    axt.set_title(scene.tissue_title, fontsize=10)
    axt.set_xlabel(f"x ({scene.unit})")
    axt.set_ylabel(f"y ({scene.unit})")
    axt.set_aspect("equal")
    fig.tight_layout()
    frame = _fig_to_rgb(fig)
    plt.close(fig)
    return frame


def animate_axonmap(outdir: Path, device: torch.device, n_frames: int = 48) -> list[Path]:
    scene = _scene("axonmap", device)
    implant, model = scene.implant, scene.model
    sel_idx = [i for i in (20, 28, 36) if i < implant.n_electrodes]
    N = implant.n_electrodes
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
        title = f"Axon Map percept\nrho={rho:.0f}  axlambda={axl:.0f}" + (
            " (fade)" if t >= pulse_frames else ""
        )
        shifted = elec + np.array([dx, 0.0])
        frames.append(
            _draw_frame(scene, state.image.detach().cpu().numpy(), title, shifted, sel_idx)
        )
    return _write(frames, outdir, "axonmap")


def _animate_cortical(model_name: str, outdir: Path, device: torch.device, n_frames: int):
    scene = _scene(model_name, device)
    implant, model = scene.implant, scene.model
    N = implant.n_electrodes
    sel_idx = [0, 25, 50]
    amp_val = 200.0 if model_name == "dynaphos" else 250.0
    pulse_frames = n_frames // 2
    elec_base = implant.electrode_coords().cpu().numpy()

    state = None
    frames = []
    for t in range(n_frames):
        dx = 5000.0 * t / max(n_frames - 1, 1)
        amp = torch.zeros(N, device=device)
        if t < pulse_frames:
            for e in sel_idx:
                amp[e] = amp_val
        rho = None if model_name == "dynaphos" else 1000.0
        state = model.step(state, Action(amp=amp, rho=rho, pose=Pose(dx=dx)))
        title = f"{model_name} percept" + (
            " (fade)" if t >= pulse_frames else f"\nframe {t} (charge buildup)"
        )
        shifted = elec_base + np.array([dx, 0.0])
        frames.append(
            _draw_frame(scene, state.image.detach().cpu().numpy(), title, shifted, sel_idx)
        )
    return _write(frames, outdir, model_name)


def animate_scoreboard(outdir: Path, device: torch.device, n_frames: int = 48) -> list[Path]:
    return _animate_cortical("scoreboard", outdir, device, n_frames)


def animate_dynaphos(outdir: Path, device: torch.device, n_frames: int = 48) -> list[Path]:
    return _animate_cortical("dynaphos", outdir, device, n_frames)


ANIMATORS = {
    "axonmap": animate_axonmap,
    "scoreboard": animate_scoreboard,
    "dynaphos": animate_dynaphos,
}


class Step(NamedTuple):
    stage: str
    electrodes: tuple[int, ...]
    amp: float


def stimulation_zone(elec_xy: np.ndarray, size: int = 4) -> list[int]:
    """Electrode nearest the array centroid, followed by its nearest neighbours."""
    centre = int(np.argmin(np.linalg.norm(elec_xy - elec_xy.mean(axis=0), axis=1)))
    order = np.argsort(np.linalg.norm(elec_xy - elec_xy[centre], axis=1))
    return [int(i) for i in order[:size]]


def sequence_schedule(
    zone: list[int],
    levels: tuple[float, ...],
    on: int = 5,
    rest: int = 4,
    train_pulses: int = 4,
    train_on: int = 3,
    train_rest: int = 3,
) -> list[Step]:
    """One ``Step`` per frame: patterns (singles, pairs, ramp), then a pulse train.

    Rest frames carry no electrodes so the fading of each response is visible
    before the next step starts. Pairs always include the zone centre, which
    guarantees both electrodes are neighbours.
    """
    mid = levels[len(levels) // 2]
    patterns = [("single", (e,), mid) for e in zone]
    patterns += [("pair", (zone[0], e), mid) for e in zone[1:]]
    patterns += [("ramp", tuple(zone), a) for a in levels]

    steps: list[Step] = []
    for stage, elecs, amp in patterns:
        steps += [Step(stage, elecs, amp)] * on
        steps += [Step(stage, (), 0.0)] * rest
    for k in range(1, train_pulses + 1):
        stage = f"pulse train {k}/{train_pulses}"
        steps += [Step(stage, tuple(zone), mid)] * train_on
        steps += [Step(stage, (), 0.0)] * train_rest
    return steps


def animate_sequence(
    model_name: str,
    outdir: Path,
    device: torch.device,
    zone_size: int = 4,
    **schedule_kwargs,
) -> list[Path]:
    """Fixed implant pose; a stimulation sequence over one zone of electrodes."""
    scene = _scene(model_name, device)
    elec = scene.implant.electrode_coords().cpu().numpy()
    zone = stimulation_zone(elec, zone_size)
    retinal = model_name == "axonmap"
    if not retinal:
        # Near the fovea cortical magnification makes phosphenes a fraction of a
        # degree wide, so zoom the percept onto the zone's visual-field location.
        zx = torch.as_tensor(elec[zone, 0], dtype=torch.float64)
        zy = torch.as_tensor(elec[zone, 1], dtype=torch.float64)
        px, py = scene.topo.polimeni.v1_to_dva(zx, zy)
        cx, cy = float(px.mean()), float(py.mean())
        scene = _scene(model_name, device, ((cx - 1.0, cx + 1.0), (cy - 1.0, cy + 1.0), 0.02))
    implant, model = scene.implant, scene.model
    names = implant.names
    N = implant.n_electrodes
    levels = RETINAL_LEVELS if retinal else CORTICAL_LEVELS
    unit = "x threshold" if retinal else "uA"
    steps = sequence_schedule(zone, levels, **schedule_kwargs)

    state = None
    images = []
    for st in steps:
        amp = torch.zeros(N, device=device)
        amp[list(st.electrodes)] = st.amp
        if retinal:
            on = amp > 0
            action = Action(
                amp=amp,
                freq=torch.where(on, 30.0, 0.0),
                phase_dur=torch.where(on, 0.45, 0.0),
                rho=200.0,
                axlambda=500.0,
                pose=Pose(),
            )
        else:
            rho = 1000.0 if model_name == "scoreboard" else None
            action = Action(amp=amp, rho=rho, pose=Pose())
        state = model.step(state, action)
        images.append(state.image.detach().cpu().numpy())

    # One colour scale for the whole clip, otherwise every fading frame is
    # re-normalised to full brightness and the temporal dynamics disappear.
    vmax = max(float(img.max()) for img in images) or 1.0
    frames = []
    for st, img in zip(steps, images):
        if st.electrodes:
            detail = f"{' + '.join(names[e] for e in st.electrodes)} @ {st.amp:g} {unit}"
        else:
            detail = "rest"
        title = f"{model_name} percept, fixed implant\n{st.stage}: {detail}"
        frames.append(_draw_frame(scene, img, title, elec, zone, st.electrodes, vmax=vmax))
    return _write(frames, outdir, f"{model_name}_sequence", mp4=False)


def main(
    model: str = "all",
    outdir: str = "artifacts",
    n_frames: int = 48,
    device: str | None = None,
    scenario: str = "sweep",
) -> list[Path]:
    dev = get_device(device)
    out = Path(outdir)
    names = list(ANIMATORS) if model == "all" else [model]
    written: list[Path] = []
    for name in names:
        if scenario == "sequence":
            written += animate_sequence(name, out, dev)
        else:
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
    @click.option(
        "--scenario",
        type=click.Choice(["sweep", "sequence"]),
        default="sweep",
        show_default=True,
        help="sweep: moving implant; sequence: fixed implant, stimulation sequence over one zone.",
    )
    @click.option("--outdir", type=str, default="artifacts", show_default=True)
    @click.option("--frames", "n_frames", type=int, default=48, show_default=True)
    @click.option("--device", type=str, default=None)
    def _cli(model, scenario, outdir, n_frames, device):
        paths = main(model, outdir, n_frames, device, scenario)
        for p in paths:
            click.echo(str(p))

    _cli()


if __name__ == "__main__":  # pragma: no cover
    cli()
