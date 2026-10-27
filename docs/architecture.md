# Architecture: encoders and forward models

[Home](index.md) · [Getting started](getting-started.md) · [User guide](user-guide.md) · [Nomenclature](nomenclature.md) · [Models](models/index.md) · [References](references.md)

A prosthesis has two halves. An **encoder** turns a signal from the world
(an image, a sound) into electrode stimulation. A **forward model** turns
stimulation into what the person perceives. Sentionaut keeps the two apart,
so any encoder can be scored against any forward model.

```text
              encoder                         forward model
 signal ──► <modality>2stim ──► stimulation ──► stim2vision ──► percept
 (image,    (vision2stim,        amp (B, E)     (axon map,       (B, H, W)
  sound)     sound2stim)                         Dynaphos,
                 ▲                               world model)
                 │                                   │
                 └──── loss(percept, target) ◄───────┘
                       backpropagated through the forward model
```

A third kind of model, a **decoder**, reads *recorded* neural activity back
into stimulus content (`brain2vision` in the usual sense, as in POYO+).
There are none in-tree; the name is reserved for them (see
[Nomenclature](nomenclature.md)).

## Encoders: a signal in some modality → stimulation

Source: `src/sentionaut/encoders/`.

Every encoder is a `StimulusEncoder`. It takes one frame of its modality
and returns amplitudes `(B, E)` in `[0, a_max]`, in the units of the forward
model it drives: multiples of threshold for the axon map, µA for Scoreboard
and Dynaphos. Amplitudes below `a_min` are returned as exact zeros, because
that is how the forward models know an electrode is off.
`encode_sequence` applies the encoder frame by frame to `(B, T, ...)`.

| Encoder | Tag | Learned | What it does |
| --- | --- | --- | --- |
| `PhospheneMapEncoder` | `vision2stim` | no | Samples the image at each electrode's phosphene centre and uses the value as the amplitude. This is the strategy of current devices; it ignores streaks, phosphene size and electrode interactions. |
| `ClosedLoopVisionEncoder` | `vision2stim` | yes | A network that sees the target image and the percept from the previous step, so it can correct fading and charge build-up over a pulse train. Trained through a forward model (below). |
| `BandEnvelopeEncoder` | `sound2stim` | no | Cuts audio into 20 ms frames, pools each frame's spectrum into one log-spaced band per electrode and log-compresses the band energy into an amplitude: the continuous interleaved sampling (CIS) idea of cochlear implants. |

`probe_centroids(render, n_electrodes, amp)` finds each electrode's
phosphene centre from any forward model by stimulating electrodes one at a
time. That is how a `PhospheneMapEncoder` is fitted to an implant.

```python
import torch
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components
from sentionaut.encoders import PhospheneMapEncoder, probe_centroids

cfg = Config(model="axonmap", implant="argusii", xrange=(-15, 15), yrange=(-10, 10), xystep=1.0)
implant, topo, model = build_components(cfg)
centres = probe_centroids(lambda a: model.spatial_forward_batch(a), implant.n_electrodes, 2.0)
encoder = PhospheneMapEncoder(centres, topo.grid_shape, a_max=3.0, a_min=0.5)

image = torch.rand(1, *topo.grid_shape)   # an image on the percept grid
amp = encoder(image)                      # (1, 60) in multiples of threshold
percept = model.spatial_forward_batch(amp)
```

**Sound.** `BandEnvelopeEncoder` is ready for a cochlear forward model, but
Sentionaut has none yet: every forward model in-tree is visual. Driving a
visual implant with it is sensory substitution, and its `order` argument
says which electrode gets which band (for example, low pitch at the bottom
of the visual field).

## Forward models: stimulation → percept

Source: `src/sentionaut/models/` (analytical) and `src/sentionaut/learned/`
(learned). All share the world-model contract `f(s_t, a_t) → s_{t+1}`.

| Forward model | Kind | Tag | State | Batched, differentiable step |
| --- | --- | --- | --- | --- |
| [Axon map](models/axonmap.md) `BiphasicAxonMapTorch` | analytical, retinal | `stim2vision` | fading brightness | `step_batch(brightness, amp)` |
| [Scoreboard](models/scoreboard.md) `ScoreboardTorch` | analytical, cortical | `stim2vision` | fading brightness | – |
| [Dynaphos](models/dynaphos.md) `DynaphosTorch` | analytical, cortical | `stim2vision` | per-electrode activation, charge, size | `step_batch(state, amp)` |
| [Axon-map world model](models/axonmap-world.md) `AxonMapWorld` | learned from the axon map | `stim2vision` | fading brightness | yes |
| `TokenWorldModel` ([imagination](imagination.md)) | learned online from any teacher | `stim2vision` | percept + per-electrode memory | yes |
| [World model](models/world-model.md) `UnifiedWorldModel` | learned, several teachers | `stim2vision` | percept + aux maps | yes |

The analytical models are torch ports of pulse2percept 0.9.0 and match it
numerically (see the parity tests). The batched steps take amplitudes
`(B, E)` and are tested against a loop over the single-sample path. The axon
map's `spatial_forward_batch(..., cutoff=5.0)` keeps only the axon segments
within five Gaussian widths of each electrode: it is about 3× faster and
differs from the exact sum by less than `3e-6` of the peak.

**Non-smooth points.** Both teachers have steps without gradients. The axon
map drops electrodes at zero amplitude, as pulse2percept does, so brightness
jumps from 0 to `a3 · freq` when an electrode turns on. Dynaphos renders
nothing until activation crosses `a_thr`. A gradient through the teacher
cannot see those steps; a learned world model fitted to the teacher's
outputs smooths them. Whether that helps is measured on the
[imagination](imagination.md) page.

## Training an encoder through a forward model

The encoder is trained by backpropagating `loss(percept, target)` through a
differentiable forward model. Sentionaut supports both choices of forward
model, and `sentionaut.learned.imagine` compares them:

- **through the teacher**: the analytical model itself. Exact, but each step
  costs whatever the teacher costs, forward and backward;
- **through a learned world model** fitted online to the teacher's own
  transitions (the SuperMimic loop): collect, fit, imagine.

```bash
uv run sentionaut-imagine --arm wm-bptt --preset axonmap --budget-s 1800
uv run sentionaut-imagine --arm teacher-bptt --preset dynaphos --budget-s 1800
```

## Where patient data enters

A forward model is a hypothesis about one patient. Its free parameters
(`rho` and `axlambda` for the axon map; thresholds and excitability for
Dynaphos) are fitted to what patients report.
[Patient data](patient-data.md) lists the datasets that pulse2percept's
models were built from and fits the axon map to the phosphene drawings of
four Argus patients.
