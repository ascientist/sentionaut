"""Didactic figures for the Dynaphos x axon map docs page.

Writes to docs/assets/demos/:

- ``dynaphos_axonmap_anatomy.png``: one phosphene, step by step (axon bundles
  and current spread on the retina, Dynaphos A/Q dynamics, the isotropic render
  without axons, the full render with axons).
- ``dynaphos_axonmap_streaks.png``: lambda sweep on three electrodes.
- ``dynaphos_axonmap_lambda_current.png``: lambda x current grid, showing that
  the ratio lambda / rho sets the elongation.
- ``dynaphos_axonmap_samples.png``: random stimulations drawn with the dataset
  generator's action sampler.

Uses the JAX backend when JAX is installed (``uv sync --extra jax``), else the
torch one; it also checks that the two backends agree when both are available.

    uv run python examples/dynaphos_axonmap_figures.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentionaut.core.base import Action  # noqa: E402
from sentionaut.core.config import Config  # noqa: E402
from sentionaut.core.registry import build_components  # noqa: E402
from sentionaut.generate import ActionRanges, sample_action  # noqa: E402

OUT = Path("docs/assets/demos")
HAS_JAX = importlib.util.find_spec("jax") is not None
MODEL = "dynaphos_axonmap_jax" if HAS_JAX else "dynaphos_axonmap"
WINDOW = dict(xrange=(-16, 16), yrange=(-16, 16), xystep=0.5)
EXTENT = (-16, 16, -16, 16)  # row 0 of every percept is the top (y = +16)
N_FRAMES = 5  # 20 ms frames: enough for 60 uA to push A over threshold


def build(model: str = MODEL):
    cfg = Config(model=model, implant="argusii", **WINDOW)
    return build_components(cfg, torch.device("cpu"))


def run(model, electrodes, current, lam, n_frames=N_FRAMES):
    """Stimulate ``electrodes`` for ``n_frames``; return the final state and A/Q traces."""
    amp = torch.zeros(model.implant.n_electrodes)
    amp[electrodes] = current
    state, trace = None, []
    for _ in range(n_frames):
        state = model.step(state, Action(amp=amp, axlambda=lam))
        trace.append((float(state.aux["A"][electrodes[0]]), float(state.aux["Q"][electrodes[0]])))
    return state, np.array(trace)


def elongation(img: np.ndarray) -> float:
    """Major/minor axis ratio of the intensity-weighted second moments."""
    yy, xx = np.indices(img.shape, dtype=float)
    w = img / img.sum()
    mx, my = (w * xx).sum(), (w * yy).sum()
    cxx = (w * (xx - mx) ** 2).sum()
    cyy = (w * (yy - my) ** 2).sum()
    cxy = (w * (xx - mx) * (yy - my)).sum()
    ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
    return float(np.sqrt(ev[1] / ev[0]))


def percept(ax, img, title):
    ax.imshow(img, cmap="inferno", vmin=0, vmax=1, extent=EXTENT)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (dva)")
    ax.set_ylabel("y (dva)")


def anatomy(implant, topo, model) -> Path:
    e = implant.names.index("B2")
    current, lam = 150.0, 1500.0
    state, trace = run(model, [e], current, lam)
    iso, _ = run(model, [e], current, 100.0)
    rho = float(state.aux["sigma"][e])
    xy = implant.electrode_coords().cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), dpi=100)
    ax = axs[0]
    coords = topo.coords.cpu().numpy()
    mask = topo.mask.cpu().numpy().astype(bool)
    for p in range(0, coords.shape[0], 23):  # a subsample of the axon bundles
        ax.plot(coords[p, mask[p], 0], coords[p, mask[p], 1], color="0.75", lw=0.4)
    gx, gy = np.meshgrid(np.linspace(-4500, 2500, 200), np.linspace(-3500, 3500, 200))
    g = np.exp(-((gx - xy[e, 0]) ** 2 + (gy - xy[e, 1]) ** 2) / (2 * rho**2))
    ax.contourf(gx, gy, g, levels=[0.135, 0.607, 1.0], colors=["#fdd49e", "#fc8d59"], alpha=0.7)
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c="tab:blue")
    ax.scatter(*xy[e], s=40, c="tab:red")
    ax.set_xlim(-4500, 2500)
    ax.set_ylim(-3500, 3500)
    ax.set_aspect("equal")
    ax.set_title(
        f"(a) retina: axon bundles + current spread\nrho = sqrt(I/K) = {rho:.0f} um", fontsize=9
    )
    ax.set_xlabel("x (um, retina)")
    ax.set_ylabel("y (um, retina)")

    ax = axs[1]
    t = np.arange(1, N_FRAMES + 1) * model.dt
    ax.plot(t, trace[:, 0] * 1e7, "o-", color="tab:red", label="activation A")
    ax.axhline(model.a_thr * 1e7, color="tab:red", ls=":", label="threshold")
    ax.axhline(model.a50 * 1e7, color="0.4", ls="--", label="half brightness")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("A (x1e-7)")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    qx = ax.twinx()
    qx.plot(t, trace[:, 1], "s-", color="tab:cyan", ms=3)
    qx.set_ylabel("memory trace Q (uA)", color="tab:cyan")
    b = 1 / (1 + np.exp(-model.sig_slope * (trace[-1, 0] - model.a50)))
    ax.set_title(f"(b) Dynaphos dynamics at {current:.0f} uA\nbrightness b = {b:.2f}", fontsize=9)

    percept(axs[2], iso.image.cpu().numpy(), "(c) short lambda = 100 um: the Dynaphos blob")
    percept(axs[3], state.image.cpu().numpy(), f"(d) with the axon map (lambda = {lam:.0f} um)")
    fig.tight_layout()
    path = OUT / "dynaphos_axonmap_anatomy.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def streaks(implant, model) -> Path:
    names, lambdas = ["A1", "C5", "F10"], [100.0, 500.0, 1500.0]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for r, name in enumerate(names):
        for c, lam in enumerate(lambdas):
            state, _ = run(model, [implant.names.index(name)], 60.0, lam)
            img = state.image.cpu().numpy()
            ax = axes[r, c]
            ax.imshow(img, cmap="inferno", vmin=0, vmax=1, extent=EXTENT)
            ax.text(-15, -15, f"elongation {elongation(img):.2f}", color="w", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"lambda = {lam:.0f} um")
            if c == 0:
                ax.set_ylabel(f"electrode {name}")
    fig.suptitle(f"60 uA, {N_FRAMES} frames: the axon map stretches the Dynaphos blob")
    fig.tight_layout()
    path = OUT / "dynaphos_axonmap_streaks.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def lambda_current(implant, model) -> Path:
    e = [implant.names.index("A1")]
    currents, lambdas = [60.0, 150.0, 300.0], [100.0, 500.0, 1500.0]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for r, current in enumerate(currents):
        for c, lam in enumerate(lambdas):
            state, _ = run(model, e, current, lam)
            img = state.image.cpu().numpy()
            rho = float(state.aux["sigma"][e[0]])
            ax = axes[r, c]
            ax.imshow(img, cmap="inferno", vmin=0, vmax=1, extent=EXTENT)
            ax.set_xlim(-16, 0)
            ax.set_ylim(-2, 14)
            ax.text(-15.5, 12.5, f"lambda/rho = {lam / rho:.1f}", color="w", fontsize=8)
            ax.text(-15.5, -1.3, f"elongation {elongation(img):.2f}", color="w", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"lambda = {lam:.0f} um")
            if c == 0:
                ax.set_ylabel(f"{current:.0f} uA (rho = {rho:.0f} um)")
    fig.suptitle("Electrode A1: current grows rho, lambda grows the streak")
    fig.tight_layout()
    path = OUT / "dynaphos_axonmap_lambda_current.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def samples(implant, model, n=12, seed=0) -> Path:
    """Random stimulations as the dataset generator draws them (sentionaut-world)."""
    rng = np.random.default_rng(seed)
    cfg = Config(model=MODEL, implant="argusii", **WINDOW)
    ranges = ActionRanges()
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.4), dpi=100)
    for ax in axes.ravel():
        action, rec = sample_action(cfg, implant.n_electrodes, rng, ranges, torch.device("cpu"))
        state = None
        for _ in range(N_FRAMES):  # hold the sampled action so A can cross threshold
            state = model.step(state, action)
        on = np.flatnonzero(rec["amp"])
        label = ", ".join(f"{implant.names[i]} {rec['amp'][i]:.0f}uA" for i in on)
        ax.imshow(state.image.cpu().numpy(), cmap="inferno", vmin=0, vmax=1, extent=EXTENT)
        ax.set_title(f"{label}\nlambda={rec['axlambda']:.0f} um", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Random samples (dataset action sampler, {N_FRAMES} frames each)")
    fig.tight_layout()
    path = OUT / "dynaphos_axonmap_samples.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def backend_check(implant) -> None:
    if not HAS_JAX:
        return
    _, _, torch_model = build("dynaphos_axonmap")
    _, _, jax_model = build("dynaphos_axonmap_jax")
    zone = [implant.names.index(n) for n in ("A1", "B2", "C5")]
    a, _ = run(torch_model, zone, 200.0, 1000.0, 8)
    b, _ = run(jax_model, zone, 200.0, 1000.0, 8)
    print(f"torch vs JAX, 8 frames: max |diff| = {float((a.image - b.image).abs().max()):.1e}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    implant, topo, model = build()
    for path in (
        anatomy(implant, topo, model),
        streaks(implant, model),
        lambda_current(implant, model),
        samples(implant, model),
    ):
        print(f"wrote {path}")
    backend_check(implant)


if __name__ == "__main__":
    main()
