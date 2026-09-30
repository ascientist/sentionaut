"""Dynaphos x axon map: how lambda turns a Dynaphos blob into an axonal streak.

Renders docs/assets/demos/dynaphos_axonmap_streaks.png. Uses the JAX backend
when JAX is installed (``uv sync --extra jax``), else the torch one; both run on
GPU when available (torch: CUDA/MPS, JAX: ``--extra jax-cuda``).

    uv run python examples/dynaphos_axonmap_streaks.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from sentionaut.core.base import Action  # noqa: E402
from sentionaut.core.config import Config  # noqa: E402
from sentionaut.core.registry import build_components  # noqa: E402

OUT = Path("docs/assets/demos/dynaphos_axonmap_streaks.png")
MODEL = "dynaphos_axonmap_jax" if importlib.util.find_spec("jax") else "dynaphos_axonmap"
ELECTRODES = ["A1", "C5", "F10"]  # inferotemporal corner, centre, superonasal corner
LAMBDAS = [100.0, 500.0, 1500.0]
CURRENT_UA, N_FRAMES = 60.0, 5  # 60 uA needs a few 20 ms frames to cross A_thr


def main() -> None:
    cfg = Config(model=MODEL, implant="argusii", xrange=(-16, 16), yrange=(-16, 16), xystep=0.5)
    implant, _, model = build_components(cfg)
    fig, axes = plt.subplots(len(ELECTRODES), len(LAMBDAS), figsize=(9, 9))
    for r, name in enumerate(ELECTRODES):
        for c, lam in enumerate(LAMBDAS):
            amp = torch.zeros(implant.n_electrodes)
            amp[implant.names.index(name)] = CURRENT_UA
            state = None
            for _ in range(N_FRAMES):
                state = model.step(state, Action(amp=amp, axlambda=lam))
            ax = axes[r, c]
            ax.imshow(state.image.cpu(), cmap="inferno", vmin=0, vmax=1, extent=(-16, 16, -16, 16))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"λ = {lam:.0f} µm")
            if c == 0:
                ax.set_ylabel(f"electrode {name}")
    fig.suptitle(f"Dynaphos × axon map ({MODEL}), {CURRENT_UA:.0f} µA, {N_FRAMES} frames")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
