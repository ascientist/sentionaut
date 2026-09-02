# Nomenclature

[Home](index.md) · [Getting started](getting-started.md) · [Nomenclature](nomenclature.md) · [Models](models/index.md) · [References](references.md)

Models are tagged with an ML-style **`source2target`** label built from a fixed
modality vocabulary. The goal is a short name that says what goes in and what
comes out — the same pattern as `brain2text` or `image2text`.

## Vocabulary

| Modality | Role |
| --- | --- |
| `vision` | visual percept / phosphene appearance |
| `image` | pixel grid as data (when distinct from percept) |
| `video` | temporal visual stream |
| `sound` | auditory signal |
| `language` | text / speech tokens |
| `brain` | neural tissue interface (retina or cortex stimulation, neural readout) |

Only these tokens appear in modality tags. Tissue (retinal vs cortical) is a
separate axis, not part of the tag.

## Direction

| Tag | Meaning |
| --- | --- |
| `brain2vision` | neural stimulation → predicted visual percept |
| `vision2brain` | visual input → stimulation / neural encoding |

Examples of the same pattern outside this repo: `brain2text`, `image2text`,
`speech2text`.

## Current models

Every Sentionaut percept model (and the learned world model) is **`brain2vision`**:
an `Action` of electrode drive produces a `State.image` phosphene map. Retinal
vs cortical differs by implant and topography, not by modality tag.

There are no `vision2brain` (encoding) models in-tree yet. When they appear,
they use the same vocabulary.

## What is not tagged

Implants and topography maps are supporting geometry. They do not get a
`source2target` label; they are described under components on the home page.
