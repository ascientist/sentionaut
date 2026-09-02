# Dynaphos

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

## Usage

```python
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

cfg = Config(model="dynaphos", implant="orion")
implant, topo, model = build_components(cfg)
```

Source: `src/sentionaut/models/dynaphos.py`
