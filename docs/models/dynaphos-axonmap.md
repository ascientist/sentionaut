# Dynaphos × axon map

[Home](../index.md) · [Getting started](../getting-started.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

**Modality:** `brain2vision` · **Tissue:** retinal (epiretinal)

Two backends with identical physics, both GPU-capable:

| ID | Class | Backend | GPU |
| --- | --- | --- | --- |
| `dynaphos_axonmap` | `DynaphosAxonMapTorch` | PyTorch | CUDA / MPS via `get_device` |
| `dynaphos_axonmap_jax` | `DynaphosAxonMapJax` | JAX (`jit`, `lax.scan`) | CUDA with `--extra jax-cuda` |

## Papers

- Temporal dynamics, brightness, threshold, current spread:
  van der Grinten et al. 2024, eLife 13, e85812
  ([doi:10.7554/eLife.85812](https://doi.org/10.7554/eLife.85812)).
- Axon layout: Beyeler et al. 2019, Sci Rep 9, 9199
  ([doi:10.1038/s41598-019-45416-4](https://doi.org/10.1038/s41598-019-45416-4)),
  Jansonius axon trajectories as built by pulse2percept.

## What it does

It runs [Dynaphos](dynaphos.md) (charge trace, leaky activation, sigmoid
brightness, detection threshold, current-dependent size) on a retinal
implant. The phosphene is rendered through the [axon map](axonmap.md), so it
follows the nerve-fibre bundles instead of being a round blob. `Action.amp`
is in µA (typically 50–300).

## Why the two models combine this way

Both papers use the same spatial primitive, a Gaussian of activated tissue
around the electrode. They only differ in the space where the Gaussian lives.

**Dynaphos** (Eqs 5–6) estimates the diameter of activated tissue from the
current and draws the phosphene as a Gaussian with $2\sigma = P$:

$$ D = 2\sqrt{I/K}\ \text{mm}, \qquad P = \frac{D}{M}\ \text{dva}, \qquad \sigma = \frac{P}{2} $$

In tissue coordinates that is a Gaussian with standard deviation
$D/2 = \sqrt{I/K}$. The cortical magnification $M$ only converts mm of cortex
to degrees.

**Axon map** (Beyeler 2019, Eq 9) keeps the Gaussian in retinal µm. The
retina → visual field transform of the axon topography places it, and each
axon segment is weighted by its path distance to the soma:

$$ I_\text{axon}(q) = \exp\!\Big(-\frac{\lVert q - x_e\rVert^2}{2\rho^2}\Big)\,
   \exp\!\Big(-\frac{d_\text{soma}(q)^2}{2\lambda^2}\Big) $$

**The hybrid** keeps Dynaphos Eqs 7–13 unchanged. It replaces the
"$\div M$, isotropic Gaussian in dva" step with the axon-map kernel, and uses
the Dynaphos current spread as a per-electrode, amplitude-dependent $\rho$:

$$ \rho_e = \frac{D_e}{2} = \sqrt{I_e / K}\quad(\text{in µm}) $$

$$ I(p) = \min\Big(1,\ \max_{q \in \text{axon}(p)} \sum_{e:\,A_e \ge A_\text{thr}}
   b_e\, \exp\!\Big(-\frac{\lVert q - x_e\rVert^2}{2\rho_e^2}\Big)\,
   \exp\!\Big(-\frac{d_\text{soma}(q)^2}{2\lambda^2}\Big)\Big),
   \qquad b_e = \frac{1}{1 + e^{-k(A_e - A_{50})}} $$

As in Dynaphos, size uses only the instantaneous current (not the trace $Q$),
and an electrode that stops keeps its last size while $A$ decays.

### The default $K$ is plausible for retina

$K = 675$ µA/mm² is the cortical estimate (Tehovnik et al. 2006) used by
Dynaphos. It gives $\rho \approx 270$–$670$ µm for 50–300 µA. That is the
same order as the per-patient $\rho$ fitted for Argus II in Beyeler 2019
(144–437 µm). There is no retinal fit of $K$ yet. Treat it as a parameter
(`params["excitability"]`).

### Checks in the test suite

- $A$ and $Q$ follow Dynaphos Eqs 7–13 exactly, frame by frame.
- $\rho_e = 1000\sqrt{I/K}$ µm.
- With $F_\text{size}=F_\text{streak}=1$ and $F_\text{bright}=b$, the render
  equals the pulse2percept-parity-tested `BiphasicAxonMapTorch` kernel.
- As $\lambda \to 0$ it reduces to an isotropic Dynaphos Gaussian at each
  pixel's retinal location.
- Off-fovea, a larger $\lambda$ elongates the phosphene (anisotropy
  ≈1.0 → >1.4). A larger $\lambda$ never removes light.
- JAX matches torch to ~1e-5 over multi-frame sequences with pose and
  co-stimulation. `lax.scan` rollouts match step-by-step loops. `jax.grad` and
  `jax.vmap` work.

## Axon streaks versus λ

![Dynaphos x axon map streaks for three Argus II electrodes and three lambdas](../assets/demos/dynaphos_axonmap_streaks.png)

60 µA for five 20 ms frames on three Argus II electrodes:

- **Corners (A1, F10).** Axons there follow the arcuate bundles, so a large
  $\lambda$ turns the Dynaphos blob into a curved streak. The streak points
  away from the optic disc.
- **Centre (C5).** Near the fovea the streak barely shows. The axons are
  short, and $\rho$ is already comparable to $\lambda$.
- **Larger current means a larger $\rho$**, which washes out the streak (the
  ratio $\lambda/\rho$ sets the elongation).

The few dark pixels on the horizontal meridian are the raphe discontinuity
of the axon map (also present in pulse2percept).

Regenerate with `uv run python examples/dynaphos_axonmap_streaks.py`.

## Usage

PyTorch (differentiable end to end with torch autograd):

```python
import torch
from sentionaut.core.base import Action
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

cfg = Config(model="dynaphos_axonmap", implant="argusii", axlambda=1000.0)
implant, topo, model = build_components(cfg)          # CUDA / MPS if available
amp = torch.zeros(implant.n_electrodes); amp[24] = 150.0   # uA on C5
state = None
for _ in range(5):
    state = model.step(state, Action(amp=amp.to(topo.coords.device)))
state.image            # (H, W) percept
state.aux              # per-electrode A, Q, sigma (= rho_e, microns)
```

JAX through the same framework interface (`uv sync --extra jax`, add
`--extra jax-cuda` for NVIDIA GPUs):

```python
cfg = Config(model="dynaphos_axonmap_jax", implant="argusii")
implant, topo, model = build_components(cfg)
frames = model.predict_sequence(Action(amp=amp), n_steps=10)   # one fused lax.scan
```

Pure functional JAX core, for `jax.grad` / `jax.vmap` / custom loops:

```python
import jax, jax.numpy as jnp
from sentionaut.models import dynaphos_axonmap_jax as dj

p, geom, shape = model.params, model.geom, topo.grid_shape
n = implant.n_electrodes
elec_xy = jnp.asarray(implant.electrode_coords().cpu().numpy())
amps = jnp.zeros((10, n)).at[:, 24].set(150.0)                 # (T, E) stimulation plan
state, frames = dj.rollout(p, geom, dj.init_state(n), amps,
                           jnp.full((10, n), p.freq), jnp.full((10, n), p.p_dur),
                           elec_xy, 500.0, grid_shape=shape)
```

Notes:

- Epiretinal implants only (PRIMA is rejected, as for the axon map).
- `Action.axlambda` overrides λ per call. `Action.rho` is ignored because ρ
  follows from the current.
- The torch ↔ JAX adapter copies tensors via DLPack (zero-copy on a shared
  CUDA device). Torch autograd does not cross it, so differentiate the JAX
  core with `jax.grad` instead.
- In the learned world model both backends share one model id.

Sources: `src/sentionaut/models/dynaphos_axonmap.py`,
`src/sentionaut/models/dynaphos_axonmap_jax.py`
