# Axon map

[Home](../index.md) · [Getting started](../getting-started.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

**Modality:** `brain2vision` · **Tissue:** retinal

## Paper

Jacob Granley, Michael Beyeler.
*A computational model of phosphene appearance for epiretinal prostheses.*
IEEE EMBC 2021.
[doi:10.1109/EMBC46164.2021.9629663](https://doi.org/10.1109/EMBC46164.2021.9629663)

Spatial axon trajectories follow Beyeler et al. 2019
([doi:10.1038/s41598-019-45416-4](https://doi.org/10.1038/s41598-019-45416-4)).

## What it does

Predicts phosphene appearance for epiretinal implants by activating nerve-fiber
bundles (streaks) and scaling brightness, size, and streak length from biphasic
pulse amplitude, frequency, and phase duration. `Action.amp` is in units of
threshold (typically 0.5–3.0).

## Usage

```python
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

cfg = Config(model="axonmap", implant="argusii")
implant, topo, model = build_components(cfg)
```

Source: `src/sentionaut/models/axonmap.py`
