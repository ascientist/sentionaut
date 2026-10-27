# Models

[Home](../index.md) · [Getting started](../getting-started.md) · [User guide](../user-guide.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

Catalog of Sentionaut models. Every entry is a **`stim2vision`** forward model: electrode
stimulation in, predicted visual percept out. See
[Nomenclature](../nomenclature.md) for the tagging rules.

| ID | Class | Modality | Tissue | Paper |
| --- | --- | --- | --- | --- |
| [axonmap](axonmap.md) | `BiphasicAxonMapTorch` | `stim2vision` | retinal | Granley & Beyeler 2021 |
| [axonmap-world](axonmap-world.md) | `AxonMapWorld` | `stim2vision` | retinal | distilled from axon map |
| [scoreboard](scoreboard.md) | `ScoreboardTorch` | `stim2vision` | cortical | Beyeler et al. 2019 |
| [dynaphos](dynaphos.md) | `DynaphosTorch` | `stim2vision` | cortical | van der Grinten et al. 2024 |
| [world-model](world-model.md) | `UnifiedWorldModel` | `stim2vision` | multi | learned (no paper) |

Torch ports are parity-tested against pulse2percept 0.9.0. Full citations live
on [References](../references.md).

New to these models? Start with [how to read the stimulation demos](demos.md):
each physics model page then walks through a short clip of the same
stimulation sequence, so you can compare how the three models respond.
