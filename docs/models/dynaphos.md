# Dynaphos

[Home](../index.md) · [Getting started](../getting-started.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

**Modality:** `brain2vision` · **Tissue:** cortical

## Paper

Maureen van der Grinten, Jaap de Ruyter van Steveninck, Antonio Lozano, et al.
*Towards biologically plausible phosphene simulation for the differentiable
optimization of visual cortical prostheses.*
eLife 13, e85812 (2024).
[doi:10.7554/eLife.85812](https://doi.org/10.7554/eLife.85812)

Visuotopic map from Polimeni et al. 2006
([doi:10.1016/j.visres.2006.03.006](https://doi.org/10.1016/j.visres.2006.03.006)).

## What it does

Spatiotemporal cortical phosphene model with per-electrode charge and
activation traces. Predicts brightness that builds and fades over time;
optional co-stimulation leak between nearby electrodes. `Action.amp` is in µA
(typically 50–300).

## Demo

![Dynaphos stimulation sequence on a fixed Orion array](../assets/demos/dynaphos_sequence.gif)

Same fixed Orion zone and stimulation sequence as the
[Scoreboard demo](scoreboard.md#demo): single electrodes, pairs with the
centre electrode, an amplitude ramp (100, 200, 300 µA), then a repeated pulse
train. Here the activation trace integrates over time, so phosphenes take a few
frames to appear and stay visible during part of each rest until activation
drops below threshold. The memory trace also accumulates with every pulse,
which slowly lowers the effective current of repeated stimulation.
Regenerate with `make demos`.

## Usage

```python
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

cfg = Config(model="dynaphos", implant="orion")
implant, topo, model = build_components(cfg)
```

Source: `src/sentionaut/models/dynaphos.py`
