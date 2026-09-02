# Scoreboard

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

## Usage

```python
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

cfg = Config(model="scoreboard", implant="orion")
implant, topo, model = build_components(cfg)
```

Source: `src/sentionaut/models/scoreboard.py`
