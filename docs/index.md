# Sentionaut

Sentionaut is a modular, GPU-native PyTorch framework for **prosthetic vision**.
It reimplements pulse2percept's retinal and cortical phosphene models as
differentiable Torch ports, wraps them in a shared world-model interface
`f(s_t, a_t) → s_{t+1}`, and trains a learned transformer on the resulting
transitions.

Where [pulse2percept](https://pulse2percept.readthedocs.io) is the CPU reference
simulator, Sentionaut is the differentiable, world-model-facing layer on top of
the same physics.

## Components

Three axes swap independently via a `Config`:

- **Implant** — electrode geometry and pose (retinal: Argus II, Alpha IMS/AMS,
  PRIMA, grid; cortical: Orion, Cortivis, ICVP, Neuralink)
- **Topography** — visual-field ↔ tissue map (Jansonius axon map; Polimeni 2006;
  optional Neuropythy MRI)
- **Percept model** — phosphene physics (`axonmap`, `scoreboard`, `dynaphos`)

Implants and topography are supporting geometry. The percept models (and the
learned world model) are classified by **modality** — see
[Nomenclature](nomenclature.md).

## Models at a glance

| Model | Modality | Tissue | Paper |
| --- | --- | --- | --- |
| [Axon map](models/axonmap.md) | `brain2vision` | retinal | Granley & Beyeler 2021 |
| [Scoreboard](models/scoreboard.md) | `brain2vision` | cortical | Beyeler et al. 2019 |
| [Dynaphos](models/dynaphos.md) | `brain2vision` | cortical | van der Grinten et al. 2024 |
| [World model](models/world-model.md) | `brain2vision` | multi | learned (no paper) |

All current models map neural stimulation to a predicted visual percept
(`brain2vision`). Browse the [catalog](models/index.md) for details and
citations.
