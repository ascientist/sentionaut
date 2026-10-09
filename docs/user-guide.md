# User guide

[Home](index.md) · [Getting started](getting-started.md) · [User guide](user-guide.md) · [Nomenclature](nomenclature.md) · [Models](models/index.md) · [References](references.md)

Everything past the [minimal example](getting-started.md): the action and state
schema, the command-line workflows, and how to scale runs to a cluster. Every
command runs through `uv` / `uv run`. The `make` targets are thin wrappers.

## Components

Three axes swap independently via a `Config`:

- **Implant** (electrode geometry + pose): retinal `argusii`, `alphaims`,
  `alphaams`, `prima`, and a configurable dense `grid`; cortical `orion`,
  `cortivis`, `icvp`, `neuralink`.
- **Topography** (visual-field map): retinal Jansonius axon map, cortical
  `Polimeni2006Map` (dva ↔ cortex, split hemispheres, cortical magnification),
  plus an optional `neuropythy` MRI-derived map (lazy, falls back to Polimeni).
- **PerceptModel**: retinal `axonmap` (`BiphasicAxonMapTorch`), cortical
  `scoreboard` (`ScoreboardTorch`), cortical `dynaphos` (`DynaphosTorch`).
  See the [model catalog](models/index.md).

`build_components(cfg)` returns `(implant, topography, model)` on the best
available device: CUDA, then Apple MPS, then CPU. Override with
`SENTIONAUT_DEVICE`.

## Action and state schema

- `Action`: per-electrode `amp` / `freq` / `phase_dur` / `delay`, model-level
  spatial params (`rho`, `axlambda`), and an implant `Pose` (translation +
  rotation). Each model ignores the fields it does not use.
- `State`: percept image `(H, W)` plus optional temporal channels in `aux`.
  Dynaphos threads activation `A` and charge trace `Q` across `step`; Axon Map
  and Scoreboard carry a fading brightness field via `FadingTemporal`.

`WorldModel` wraps any percept model as `f(s_t, a_t) → s_{t+1}`.

### Amplitude units

| model | `Action.amp` units | typical range |
| --- | --- | --- |
| axonmap | × threshold (unitless) | 0.5–3.0 |
| scoreboard | µA | 50–300 |
| dynaphos | µA | 50–300 |

Dataset generation and the Streamlit demo use these bands. Parity tests keep
their own fixed values.

## Common tasks

| Task | Command | Output |
| --- | --- | --- |
| Install | `make setup` | `.venv` with dev extras |
| Interactive demo | `make demo` | Streamlit app |
| Model animations | `make animate MODEL=all OUTDIR=artifacts` | GIFs in `artifacts/` |
| Documentation demos | `make demos` | GIFs + figures in `docs/assets/demos/` |
| World-model dataset | `make world WORLD_DATASET=data/world.h5` | HDF5 transitions |
| Train learned world model | `make train WORLD_DATASET=data/world.h5` | loss + eval JSON |
| Specialist ablation | `make ablate WORLD_DATASET=data/world.h5` | per-mode eval JSON |
| NeuroStim tutorials | `make neurostim` / `neurostim-canonical` / `neurostim-percept` | figures in `docs/assets/neurostim/` |
| Tests | `make test` | fast parity + smoke tests |
| Docs | `make docs` / `make docs-serve` | `site/` / `localhost:8000` |

### Interactive demo

```bash
make demo         # streamlit run src/sentionaut/demo_app.py
```

Pick implant, model and grid, then sweep the action parameters. The left panel
renders the percept, the right panel the tissue schematic.

### Animations

```bash
make animate MODEL=all OUTDIR=artifacts
```

Renders one dual-panel (percept | tissue geometry) clip per physics model. The
Axon Map sweeps `rho` / `axlambda` and translates the array. Scoreboard and
Dynaphos sweep the implant to expose cortical-magnification growth, and Dynaphos
also shows temporal charge buildup.

`make demos` renders the fixed-zone stimulation sequences shown on each model
page. [How to read them](models/demos.md).

### World-model dataset

```bash
make world WORLD_DATASET=data/world.h5 EPISODES=256 SEQ_LEN=16
```

Produces a combined multi-config HDF5 of `(config_id, episode_id, s_t, a_t,
s_{t+1})` transitions. Dynaphos rows include rasterized `aux_t` (A/Q maps);
axonmap and scoreboard aux channels are zero-padded. Metadata records `dt_ms`
and per-config `percept_scale` for normalization. Use `--silent-tail` for
zero-drive fade steps after each pulse.

### Learned world model and ablation

```bash
make train  WORLD_DATASET=data/world.h5
make ablate WORLD_DATASET=data/world.h5
```

`UnifiedWorldModel` is a conditioned ViT-style transformer; the ablation trains
per-model specialists from the identical architecture. Architecture, modes and
evaluation split: [World model](models/world-model.md). The distilled retinal
specialist: [Axon-map world model](models/axonmap-world.md).

## Parity with pulse2percept

`uv run pytest -m "not slow"` checks numerical parity against pulse2percept
0.9.0 on small subsampled grids. Measured max-abs errors:

| model | error vs pulse2percept |
| --- | --- |
| Biphasic Axon Map | ~6e-8 (effectively exact) |
| Cortical Scoreboard | ~3e-5 (peaks ~30) |
| Cortical Dynaphos | ~2e-7 per frame |

## Scaling to a cluster

Every size knob is config-driven (grid range/step, electrode count, episodes,
batch size, epochs, device) with small local defaults. SLURM launchers live in
`scripts/` (`gen_dataset.sh`, `train_unified.sh`, `ablate.sh`, and the `mila_*`
scripts), all using `uv run`.

## Package layout

```
src/sentionaut/
  core/        interfaces, config, registry, device
  topography/  axon_map (retinal), cortical (Polimeni2006 torch port)
  implants/    electrode geometries as tensors
  models/      effects, fading, axonmap, scoreboard, dynaphos
  learned/     dataset, UnifiedWorldModel, AxonMapWorld, metrics, train + ablation
  neurostim/   closed-loop control environments and baselines
  calibrate.py subject rho/axlambda grid search + JSON sidecar
  world.py     WorldModel f(s_t, a_t) -> s_{t+1}
  generate.py  multi-config dataset generation
  animate.py   per-model animations
  demo_app.py  Streamlit demo
```
