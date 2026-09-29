# Scoreboard

[Home](../index.md) · [Getting started](../getting-started.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

**Modality:** `brain2vision` · **Tissue:** cortical

## Paper

Michael Beyeler, Devyani Nanduri, James D. Weiland, Ariel Rokem, Geoffrey M. Boynton, Ione Fine.
*A model of ganglion axon pathways accounts for percepts elicited by retinal implants.*
Scientific Reports 9, 9199 (2019).
[doi:10.1038/s41598-019-45416-4](https://doi.org/10.1038/s41598-019-45416-4)

Cortical electrode ↔ visual-field mapping uses Polimeni et al. 2006
([doi:10.1016/j.visres.2006.03.006](https://doi.org/10.1016/j.visres.2006.03.006)).

## What it does

Places a Gaussian blob at each active cortical electrode's visual-field
location (the "scoreboard" baseline: no axon streaks). Cortical magnification
makes peripheral phosphenes larger. `Action.amp` is in µA (typically 50–300).

## Demo

![Scoreboard stimulation sequence on a fixed Orion array](../assets/demos/scoreboard_sequence.gif)

The Orion array stays in place while four neighbouring electrodes (red)
receive a stimulation sequence: each electrode alone, then pairs with the
centre electrode, then all four on an amplitude ramp (100, 200, 300 µA), and
finally a repeated pulse train. Electrodes being stimulated are circled in
yellow. The percept panel is zoomed onto the zone's visual-field location,
because near the fovea cortical magnification makes each phosphene a fraction
of a degree wide. Regenerate with `make demos`.

## Usage

```python
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

cfg = Config(model="scoreboard", implant="orion")
implant, topo, model = build_components(cfg)
```

Source: `src/sentionaut/models/scoreboard.py`
