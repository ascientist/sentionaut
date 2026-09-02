# Sentionaut

A modular, GPU-native PyTorch framework of swappable prosthetic-vision
components that reimplement [pulse2percept](https://pulse2percept.readthedocs.io)'s
retinal **Biphasic Axon Map** plus the cortical **Scoreboard** and **Dynaphos**
models, expose a shared world-model interface `f(s_t, a_t) -> s_{t+1}`, and feed
a single conditioned learned world model evaluated against per-model specialists.

The three analytical models run differentiably on GPU (Apple **MPS** locally,
CUDA on cluster) and are parity-tested against pulse2percept.

**Docs:** [docs/index.md](docs/index.md)

## Components

Three axes are independently swappable via a `Config`:

- **Implant** (electrode geometry + pose): retinal `argusii`, `alphaims`,
  `alphaams`, `prima`, and a configurable dense `grid`; cortical `orion`,
  `cortivis`, `icvp`, `neuralink`.
- **Topography** (visual-field map): retinal Jansonius axon map, cortical
  `Polimeni2006Map` (dva <-> cortex, split hemispheres, cortical magnification),
  plus an optional `neuropythy` MRI-derived map (lazy, falls back to Polimeni).
- **PerceptModel**: retinal `axonmap` (`BiphasicAxonMapTorch`), cortical
  `scoreboard` (`ScoreboardTorch`), cortical `dynaphos` (`DynaphosTorch`).

```python
import torch
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components
from sentionaut.core.base import Action

cfg = Config(model="axonmap", implant="argusii", xrange=(-8, 8), yrange=(-8, 8), xystep=0.5)
implant, topo, model = build_components(cfg)        # picks MPS if available
amp = torch.zeros(implant.n_electrodes); amp[20] = 2.0
percept = model.forward(Action(amp=amp,
                               freq=torch.full_like(amp, 30.0),
                               phase_dur=torch.full_like(amp, 0.45)))
```

### Action / State schema

- `Action`: per-electrode `amp` / `freq` / `phase_dur` / `delay`, model-level
  spatial params (`rho`, `axlambda`), and an implant `Pose` (translation +
  rotation). Unused fields are ignored per model.
- `State`: percept image `(H, W)` plus optional temporal channels in `aux`
  (Dynaphos threads activation `A` and charge trace `Q` across `step`; Axon Map
  and Scoreboard carry a fading brightness field via `FadingTemporal`).

### Amplitude units

| model | `Action.amp` units | typical range |
| --- | --- | --- |
| axonmap | × threshold (unitless) | 0.5–3.0 |
| scoreboard | µA | 50–300 |
| dynaphos | µA | 50–300 |

Dataset generation and the Streamlit demo use these bands; parity tests keep
their own fixed values.

## Documentation

Browse on GitHub: [docs/index.md](docs/index.md).

```bash
make docs         # build → site/
make docs-serve   # preview on localhost:8000
```

A Zensical site is also pushed to the `gh-pages` branch. GitHub Pages is not
enabled on this private repo, so
[ascientist.github.io/sentionaut](https://ascientist.github.io/sentionaut/)
404s until an admin sets **Settings → Pages → Deploy from a branch →
`gh-pages` / root**. Private Pages also need GitHub Pro.

## Setup

```bash
make setup        # uv sync --extra dev
```

Everything runs through `uv` / `uv run`.

## Parity with pulse2percept

`uv run pytest -m "not slow"` checks numerical parity against pulse2percept
0.9.0 on small subsampled grids. Measured max-abs errors:

| model | error vs pulse2percept |
| --- | --- |
| Biphasic Axon Map | ~6e-8 (effectively exact) |
| Cortical Scoreboard | ~3e-5 (peaks ~30) |
| Cortical Dynaphos | ~2e-7 per frame |

## Animations (local deliverable)

```bash
make animate MODEL=all OUTDIR=artifacts
```

Renders one dual-panel (percept | tissue-geometry) clip per physics model on
MPS: the Axon Map sweeps `rho`/`axlambda` and translates the array; the cortical
Scoreboard and Dynaphos sweep the implant to expose cortical-magnification growth
(Dynaphos also shows temporal charge buildup). Outputs go to `artifacts/`.

## Interactive demo

```bash
make demo         # streamlit run src/sentionaut/demo_app.py
```

Pick implant / model / grid and sweep action params; left panel renders the
percept, right panel the tissue schematic.

## World-model dataset

```bash
make world WORLD_DATASET=data/world.h5 EPISODES=256 SEQ_LEN=16
```

Produces a combined multi-config HDF5 of `(config_id, episode_id, s_t, a_t,
s_{t+1})` transitions. Dynaphos rows include rasterized `aux_t` (A/Q maps);
axonmap/scoreboard aux channels are zero-padded. Metadata records `dt_ms` and
per-config `percept_scale` for normalization. Use `--silent-tail` for zero-drive
fade steps after each pulse.

## Learned unified world model + ablation

`UnifiedWorldModel` is a ViT-style **transformer** (no convolutions): a linear
patch embedding + 2D sinusoidal positional embeddings, a multi-head
self-attention encoder, and a linear patch-unembedding head. It is conditioned on
percept-model id, implant id, and topography params via prepended conditioning
tokens **and** FiLM. A `mode` switch collapses the categorical conditioning to
train per-model **specialists** (the ablation baseline) from the identical
architecture, or **shared_trunk** (shared encoder, per-model output heads).
Input is 3-channel (percept + Dynaphos A/Q maps; aux zero-padded otherwise).
Val-only eval holds out entire `config_id`s. Defaults are small and configurable (`dim`, `depth`, `heads`,
`patch_size`) so the smoke tests run on the M1.

```bash
make train  WORLD_DATASET=data/world.h5
make ablate WORLD_DATASET=data/world.h5
```

Locally only the wiring is proven (single-batch training step + single-batch
inference, see `tests/test_learned_smoke.py`); full training/ablation runs on the
cluster.

## Scaling / cluster

Every size knob is config-driven (grid range/step, electrode count, episodes,
batch size, epochs, device) with small local defaults. SLURM launchers for the
Digital Research Alliance of Canada live in `scripts/` (`gen_dataset.sh`,
`train_unified.sh`, `ablate.sh`), all using `uv run`.

## Layout

```
src/sentionaut/
  core/        interfaces, config, registry, device
  topography/  axon_map (retinal), cortical (Polimeni2006 torch port)
  implants/    electrode geometries as tensors
  models/      effects, fading, axonmap, scoreboard, dynaphos
  learned/     dataset, model (UnifiedWorldModel), metrics, train + ablation
  calibrate.py subject rho/axlambda grid search + JSON sidecar
  world.py     WorldModel f(s_t, a_t) -> s_{t+1}
  generate.py  multi-config dataset generation
  animate.py   per-model animations
  demo_app.py  Streamlit demo
```
