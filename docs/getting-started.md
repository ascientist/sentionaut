# Getting started

[Home](index.md) · [Getting started](getting-started.md) · [User guide](user-guide.md) · [Nomenclature](nomenclature.md) · [Models](models/index.md) · [References](references.md)

## Install

```bash
make setup        # uv sync --extra dev
```

Everything runs through `uv` / `uv run`.

## Minimal example

```python
import torch
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components
from sentionaut.core.base import Action

cfg = Config(model="axonmap", implant="argusii", xrange=(-8, 8), yrange=(-8, 8), xystep=0.5)
implant, topo, model = build_components(cfg)
amp = torch.zeros(implant.n_electrodes)
amp[20] = 2.0
percept = model.forward(Action(
    amp=amp,
    freq=torch.full_like(amp, 30.0),
    phase_dur=torch.full_like(amp, 0.45),
))
```

Swap `model` / `implant` in `Config` to try scoreboard or dynaphos. See the
[model catalog](models/index.md) for modality tags and paper references.

## Docs site

The documentation is published at
[ascientist.github.io/sentionaut](https://ascientist.github.io/sentionaut/) and
rebuilt on every push to `main`. The same pages read on GitHub under
[`docs/`](https://github.com/ascientist/sentionaut/tree/main/docs). Local preview:

```bash
make docs         # uv run zensical build → site/
make docs-serve   # live preview on localhost:8000
```

## Next

- [User guide](user-guide.md) — schema, workflows, parity, cluster runs
- [Stimulation demos](models/demos.md) — what each model predicts, step by step
- [NeuroStim formulation](neurostim-formulation.md) — the closed-loop control problem
- [Nomenclature](nomenclature.md) — how models are classified
- [Models](models/index.md) — one page per model
- [References](references.md) — full citations
