# World model

**Modality:** `brain2vision` · **Tissue:** multi (conditioned)

## Paper

No original paper. Learned model trained on transitions from the three
analytical teachers ([axonmap](axonmap.md), [scoreboard](scoreboard.md),
[dynaphos](dynaphos.md)).

## What it does

ViT-style transformer that predicts the next percept `s_{t+1}` from current
state `s_t` and stimulation action `a_t`. Conditioned on percept-model id,
implant id, and topography params (shared, specialist, or shared-trunk modes).
Input is 3-channel (percept + Dynaphos A/Q maps; aux zero-padded otherwise).

## Usage

```python
from sentionaut.learned.model import UnifiedWorldModel

model = UnifiedWorldModel(dim=128, depth=4, heads=4, patch_size=8)
# train with: make train WORLD_DATASET=data/world.h5
```

Source: `src/sentionaut/learned/model.py`
