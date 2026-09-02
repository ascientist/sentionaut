# Models

[Home](../index.md) · [Getting started](../getting-started.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

Catalog of Sentionaut models. Every entry is **`brain2vision`**: electrode
stimulation in, predicted visual percept out. See
[Nomenclature](../nomenclature.md) for the tagging rules.

| ID | Class | Modality | Tissue | Paper |
| --- | --- | --- | --- | --- |
| [axonmap](axonmap.md) | `BiphasicAxonMapTorch` | `brain2vision` | retinal | Granley & Beyeler 2021 |
| [scoreboard](scoreboard.md) | `ScoreboardTorch` | `brain2vision` | cortical | Beyeler et al. 2019 |
| [dynaphos](dynaphos.md) | `DynaphosTorch` | `brain2vision` | cortical | van der Grinten et al. 2024 |
| [world-model](world-model.md) | `UnifiedWorldModel` | `brain2vision` | multi | learned (no paper) |

Torch ports are parity-tested against pulse2percept 0.9.0. Full citations live
on [References](../references.md).
