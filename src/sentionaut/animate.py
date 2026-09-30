"""Render one animation per analytical physics model on the local device (MPS).

Each clip has dual synced panels: the GPU-rendered percept and a tissue-geometry
schematic. Two scenarios:

- ``sweep``: Axon Map sweeps rho/axlambda (phase-offset) and translates the
  implant; the cortical Scoreboard and Dynaphos sweep the implant toward the
  periphery to expose cortical-magnification growth.
  Dynaphos x axon map sweeps lambda (streak length), then the current (rho =
  sqrt(I/K) grows and washes the streak out), then shows one pulse fading.
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
from .core.config import AXON_MAP_MODELS, UA_AMP_MODELS, Config  # noqa: E402
from .core.device import get_device  # noqa: E402
from .core.registry import build_components  # noqa: E402

RETINAL_LEVELS = (1.0, 2.0, 3.0)  # x threshold
# Argus II corner zone (centre first) for the Dynaphos x axon map periphery demo,
# and a percept window framing its phosphenes and their axonal streaks.
PERIPHERAL_ZONE = ["B2", "A2", "B1", "C2"]
PERIPHERAL_WINDOW = ((-18.0, 2.0), (-4.0, 16.0), 0.5)
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
    window: tuple[tuple[float, float], tuple[float, float], float] | None = None,
) -> Scene:
    """Build the components and tissue scatter for a percept ``window`` (dva).

    Defaults: 24 x 24 dva for retinal models, 12 x 12 dva for cortical ones.
    """
    if model_name in AXON_MAP_MODELS:
        xrange, yrange, xystep = window or ((-12, 12), (-12, 12), 0.5)
        cfg = Config(
            model=model_name, implant="argusii", xrange=xrange, yrange=yrange, xystep=xystep
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
    xrange, yrange, xystep = window or ((-6, 6), (-6, 6), 0.2)
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
    spread: dict[int, float] | None = None,
) -> np.ndarray:
    """``spread`` draws a circle of that radius (tissue units) around electrodes,
    e.g. the Dynaphos current spread rho = sqrt(I/K) on the retina."""
    cfg = scene.cfg
    s = scene.scale
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(8, 4), dpi=80)
    extent = [cfg.xrange[0], cfg.xrange[1], cfg.yrange[0], cfg.yrange[1]]
    axp.imshow(img, cmap="inferno", extent=extent, origin="upper", vmin=0.0, vmax=vmax)
    axp.set_title(title, fontsize=10)
    axp.set_xlabel("x (dva)")
    axp.set_ylabel("y (dva)")
    axt.scatter(scene.tissue[:, 0] / s, scene.tissue[:, 1] / s, s=0.5, c="0.7", alpha=0.4)
    axt.scatter(elec[:, 0] / s, elec[:, 1] / s, s=16, c="tab:blue")
    for e in highlight:
        axt.scatter(elec[e, 0] / s, elec[e, 1] / s, s=60, c="tab:red")
    for e in active:
        axt.scatter(elec[e, 0] / s, elec[e, 1] / s, s=140, facecolors="none", edgecolors="yellow")
    for e, radius in (spread or {}).items():
        axt.add_patch(
            plt.Circle(
                (elec[e, 0] / s, elec[e, 1] / s), radius / s, fill=False, color="orange", lw=1
            )
        )
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


def animate_dynaphos_axonmap(outdir: Path, device: torch.device, n_frames: int = 48) -> list[Path]:
    """Three acts on a fixed Argus II (corner A1, centre C5, corner F10):

    1. lambda sweep at 100 uA: the round Dynaphos blob grows an axonal streak;
    2. current sweep at lambda = 1500 um: rho = sqrt(I/K) grows and washes it out;
    3. one 150 uA pulse, then rest: the streak lingers ~200 ms and vanishes.

    Acts 1-2 restart the model for every frame (5 x 20 ms of stimulation, enough
    for A to cross threshold) so each frame isolates one parameter value; act 3
    threads the state through time.
    """
    scene = _scene("dynaphos_axonmap", device, ((-16, 16), (-16, 16), 0.5))
    implant, model = scene.implant, scene.model
    names = implant.names
    sel = [names.index(n) for n in ("A1", "C5", "F10")]
    N = implant.n_electrodes
    elec = implant.electrode_coords().cpu().numpy()
    n_sweep = max(n_frames // 3, 2)
    n_pulse, n_rest = 5, max(n_frames - 2 * n_sweep - 5, 1)

    def amp_of(current: float) -> torch.Tensor:
        amp = torch.zeros(N, device=device)
        amp[sel] = current
        return amp

    def settled(current: float, lam: float):
        state = None
        for _ in range(5):
            state = model.step(state, Action(amp=amp_of(current), axlambda=lam))
        return state

    acts = [("lambda sweep", 100.0, float(lam)) for lam in np.geomspace(50, 2000, n_sweep)]
    acts += [("current sweep", float(c), 1500.0) for c in np.linspace(60, 300, n_sweep)]
    renders = []
    for stage, current, lam in acts:
        renders.append((stage, current, lam, settled(current, lam), current))
    state = None
    for t in range(n_pulse + n_rest):
        current = 150.0 if t < n_pulse else 0.0
        state = model.step(state, Action(amp=amp_of(current), axlambda=1500.0))
        stage = "pulse" if t < n_pulse else f"rest +{(t - n_pulse + 1) * 20} ms"
        renders.append((stage, current, 1500.0, state, 150.0))

    frames = []
    for stage, current, lam, st, shown in renders:
        rho = float(st.aux["sigma"][sel[0]])
        title = (
            f"Dynaphos x axon map: {stage}\nI={shown:.0f} uA  rho={rho:.0f} um  lambda={lam:.0f} um"
        )
        active = tuple(sel) if current > 0 else ()
        spread = {e: float(st.aux["sigma"][e]) for e in sel}
        img = st.image.detach().cpu().numpy()
        frames.append(_draw_frame(scene, img, title, elec, sel, active, 1.0, spread))
    return _write(frames, outdir, "dynaphos_axonmap", fps=8)


ANIMATORS = {
    "axonmap": animate_axonmap,
    "scoreboard": animate_scoreboard,
    "dynaphos": animate_dynaphos,
    "dynaphos_axonmap": animate_dynaphos_axonmap,
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


STAGE_COLOURS = {
    "single": "tab:blue",
    "pair": "tab:green",
    "ramp": "tab:orange",
    "train": "tab:purple",
}


def _kind(stage: str) -> str:
    return "train" if stage.startswith("pulse train") else stage


def key_frames(steps: list[Step]) -> list[int]:
    """Frames that summarise the sequence: end of the first single pulse, the first
    pair, each ramp level, and the last rest frame before the pulse train."""
    ends = [
        i
        for i, st in enumerate(steps)
        if st.electrodes and (i + 1 == len(steps) or steps[i + 1] != st)
    ]
    single = next(i for i in ends if steps[i].stage == "single")
    pair = next(i for i in ends if steps[i].stage == "pair")
    ramp = [i for i in ends if steps[i].stage == "ramp"]
    rest = next(i for i in range(ramp[-1] + 1, len(steps)) if steps[i].electrodes) - 1
    return [single, pair, *ramp, rest]


def _step_label(st: Step, names, unit: str) -> str:
    if not st.electrodes:
        return "rest"
    return f"{' + '.join(names[e] for e in st.electrodes)} @ {st.amp:g} {unit}"


def _sequence_figures(
    scene: Scene,
    steps: list[Step],
    images: list[np.ndarray],
    vmax: float,
    unit: str,
    dt_ms: float,
    outdir: Path,
    name: str,
    hidden: dict | None = None,
) -> list[Path]:
    """Key-frame strip and peak-brightness timeline used by the docs walkthrough.

    ``hidden`` optionally adds a panel with per-frame hidden state of one electrode
    (Dynaphos activation and memory trace), whose brightness output saturates and
    so hides the dynamics that the sequence is meant to show.
    """
    cfg = scene.cfg
    extent = [cfg.xrange[0], cfg.xrange[1], cfg.yrange[0], cfg.yrange[1]]
    picks = key_frames(steps)

    fig, axes = plt.subplots(1, len(picks), figsize=(2.2 * len(picks), 2.7), dpi=100)
    for n, (ax, i) in enumerate(zip(axes, picks), start=1):
        st = steps[i]
        ax.imshow(images[i], cmap="inferno", extent=extent, origin="upper", vmin=0.0, vmax=vmax)
        head = f"({n}) {_kind(st.stage) if st.electrodes else 'rest'}, t={i * dt_ms:.0f} ms"
        ax.set_title(f"{head}\n{_step_label(st, scene.implant.names, unit)}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    stages = outdir / f"{name}_stages.png"
    fig.savefig(stages)
    plt.close(fig)

    t = np.arange(len(images)) * dt_ms
    peak = np.array([float(img.max()) for img in images])
    rows = 2 if hidden else 1
    fig, axs = plt.subplots(rows, 1, figsize=(8, 2.8 * rows), dpi=100, sharex=True, squeeze=False)
    for axis in axs[:, 0]:
        start = None
        for i, st in enumerate(steps):
            if st.electrodes and start is None:
                start = i
            if start is not None and (i + 1 == len(steps) or steps[i + 1] != st):
                axis.axvspan(
                    start * dt_ms,
                    (i + 1) * dt_ms,
                    color=STAGE_COLOURS[_kind(st.stage)],
                    alpha=0.2,
                )
                start = None
    ax = axs[0, 0]
    ax.plot(t, peak, color="black", lw=1.2)
    for n, i in enumerate(picks, start=1):
        ax.annotate(
            f"({n})", (t[i], peak[i]), textcoords="offset points", xytext=(0, 5), fontsize=8
        )
        ax.plot(t[i], peak[i], "o", color="black", ms=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in STAGE_COLOURS.values()]
    labels = [f"{k} stimulation" for k in STAGE_COLOURS]
    ax.legend(handles, labels, fontsize=7, loc="upper left", ncol=4, frameon=False)
    ax.set_ylim(0, peak.max() * 1.25 or 1.0)
    ax.set_ylabel("peak percept brightness")
    if hidden:
        hx = axs[1, 0]
        hx.plot(t, hidden["A"] * 1e7, color="tab:red", lw=1.2, label="activation A")
        hx.axhline(hidden["a_thr"] * 1e7, color="tab:red", ls=":", lw=1, label="threshold")
        hx.axhline(hidden["a50"] * 1e7, color="0.4", ls="--", lw=1, label="half brightness")
        hx.set_ylabel(f"A of electrode {hidden['electrode']} (x1e-7)")
        hx.legend(fontsize=7, loc="upper left", ncol=3, frameon=False)
        qx = hx.twinx()
        qx.plot(t, hidden["Q"], color="tab:cyan", lw=1.2)
        qx.set_ylabel("memory trace Q (uA)", color="tab:cyan")
    axs[-1, 0].set_xlabel("time (ms)")
    fig.tight_layout()
    timeline = outdir / f"{name}_timeline.png"
    fig.savefig(timeline)
    plt.close(fig)
    return [stages, timeline]


def animate_sequence(
    model_name: str,
    outdir: Path,
    device: torch.device,
    zone_size: int = 4,
    zone: list[str] | None = None,
    axlambda: float = 500.0,
    suffix: str = "",
    window: tuple[tuple[float, float], tuple[float, float], float] | None = None,
    **schedule_kwargs,
) -> list[Path]:
    """Fixed implant pose; a stimulation sequence over one zone of electrodes.

    ``zone`` names the electrodes (centre first) instead of the default zone at
    the array centre; ``suffix`` is appended to the output file names and
    ``window`` overrides the retinal percept window (dva).
    """
    scene = _scene(model_name, device, window)
    elec = scene.implant.electrode_coords().cpu().numpy()
    if zone is None:
        zone = stimulation_zone(elec, zone_size)
    else:
        zone = [scene.implant.names.index(n) for n in zone]
    retinal = model_name in AXON_MAP_MODELS
    ua = model_name in UA_AMP_MODELS
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
    levels = CORTICAL_LEVELS if ua else RETINAL_LEVELS
    unit = "uA" if ua else "x threshold"
    steps = sequence_schedule(zone, levels, **schedule_kwargs)

    state = None
    images = []
    trace_a: list[float] = []
    trace_q: list[float] = []
    spreads: list[dict[int, float]] = []
    for st in steps:
        amp = torch.zeros(N, device=device)
        amp[list(st.electrodes)] = st.amp
        if retinal and ua:
            # Dynaphos pulse defaults (300 Hz, 170 us); lambda as in the axon-map demo.
            action = Action(amp=amp, axlambda=axlambda, pose=Pose())
        elif retinal:
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
        if "A" in state.aux:
            trace_a.append(float(state.aux["A"][zone[0]]))
            trace_q.append(float(state.aux["Q"][zone[0]]))
        if retinal and ua:  # retinal current spread (microns) of stimulated electrodes
            spreads.append({e: float(state.aux["sigma"][e]) for e in st.electrodes})

    # One colour scale for the whole clip, otherwise every fading frame is
    # re-normalised to full brightness and the temporal dynamics disappear.
    vmax = max(float(img.max()) for img in images) or 1.0
    spreads = spreads if spreads else [None] * len(steps)
    frames = []
    for st, img, spread in zip(steps, images, spreads):
        title = f"{model_name} percept, fixed implant\n{st.stage}: {_step_label(st, names, unit)}"
        frames.append(
            _draw_frame(scene, img, title, elec, zone, st.electrodes, vmax=vmax, spread=spread)
        )
    name = f"{model_name}_sequence{suffix}"
    paths = _write(frames, outdir, name, mp4=False)
    dt_ms = float(getattr(model, "dt_ms", getattr(model, "dt", 20.0)))
    hidden = None
    if trace_a:
        hidden = {
            "A": np.array(trace_a),
            "Q": np.array(trace_q),
            "a_thr": model.a_thr,
            "a50": model.a50,
            "electrode": names[zone[0]],
        }
    return paths + _sequence_figures(scene, steps, images, vmax, unit, dt_ms, outdir, name, hidden)


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
            if name == "dynaphos_axonmap":
                # Same schedule on a peripheral zone, where the axon streaks show.
                written += animate_sequence(
                    name,
                    out,
                    dev,
                    zone=PERIPHERAL_ZONE,
                    axlambda=1500.0,
                    suffix="_periphery",
                    window=PERIPHERAL_WINDOW,
                )
        else:
            written += ANIMATORS[name](out, dev, n_frames)
    return written


def cli():  # pragma: no cover - thin click wrapper
    import click

    @click.command()
    @click.option(
        "--model",
        type=click.Choice(["axonmap", "scoreboard", "dynaphos", "dynaphos_axonmap", "all"]),
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
