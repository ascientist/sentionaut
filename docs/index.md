# Sentionaut

[Home](index.md) · [Getting started](getting-started.md) · [Nomenclature](nomenclature.md) · [Models](models/index.md) · [References](references.md)

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

## Closed-loop control

The [canonical formulation](neurostim-formulation.md) states the problem
precisely: adaptive tracking over a population of random, slowly drifting,
partially observed dynamical systems. It covers the patient, drift and
fast neural state, and the image and patient generalisation axes.
[Baselines](neurostim-baselines.md) runs 14 methods, from LQG to diffusion
policies, on four regimes that each favour a different family.
[NeuroStim](neurostim.md) is a toy constrained POMDP for closed-loop
neurostimulation: 3 latents, 4 overlapping electrodes, an unknown per-patient
recruitment matrix, adaptation and safety constraints, plus baselines from
bandit to PPO-Lagrangian and supervised DAgger imitation.
[NeuroStim-Percept](neurostim-percept.md) turns the target into an MNIST
image. It learns stimulation through a learned world model with a
reconstruction loss, with no RL.

All current models map neural stimulation to a predicted visual percept
(`brain2vision`). Browse the [catalog](models/index.md) for details and
citations.
