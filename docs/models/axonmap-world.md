# Axon-map world model

[Home](../index.md) · [Getting started](../getting-started.md) · [User guide](../user-guide.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

**Modality:** `brain2vision` · **Tissue:** retinal

Specialist student of the [axon map](axonmap.md). The shared
[world model](world-model.md) is a different network: it learns three teachers
at once, and it has to invent the fade. This page is the axon-map student, why
it is the smaller thing to call, and what that costs on a GPU and off one.

## What you call today

An axon-map percept is three steps.

1. Build the axon topography once. That step is pulse2percept on the CPU. It
   grows a polyline per pixel and caches a tensor of shape
   `(pixels, axon samples, 2)`.
2. Reduce that tensor on the device you asked for. For every active electrode,
   every pixel, and every axon sample, the model evaluates a Gaussian current
   spread and a streak weight, sums electrodes, and takes the max along the
   axon. That field is the spatial drive.
3. Step a leaky brightness ODE. With time step `dt` and time constant `tau`,

   `B_next = B + dt * (drive - B) / tau`,

   then values at or below the percept threshold become zero. The ODE does not
   depend on the axon polylines. Brightness `B` is already the whole state.

Step 1 is a cache miss measured in seconds to minutes, and it is not batched
across patients. Step 2 allocates an intermediate proportional to
`electrodes × pixels × axon samples`. Step 3 is a few elementwise ops.

## What the student keeps and what it learns

The student learns the drive. It does not learn the fade. `step(state, action)`
still means "brightness in, stimulation in, next brightness out", the same
`Action` and the same `State.image` as `BiphasicAxonMapTorch`.

```python
from sentionaut.implants.registry import build_implant
from sentionaut.core.config import Config
from sentionaut.learned.axon_world import AxonMapWorld

student = AxonMapWorld(grid_shape=(32, 32), n_electrodes=60)
student.bind(build_implant(Config(implant="argusii"), device))
state = student.step(None, action)
```

`bind` stores the implant's `(electrodes, 2)` coordinates. Inference does not
import pulse2percept, does not open an axon pickle, and does not care which
eye or cache key produced the teacher. Subject `rho` and `axlambda` stay
fields of `Action`, so one checkpoint covers the range of those scalars seen
in the distillation data.

Training reads transitions written by the axon-map teacher
(`sentionaut-world --model axonmap`) and minimizes mean squared error on
`B_next` after the exact fade, unrolled for a few steps. The extra steps stop
a one-step fit from hiding a drive that only looks right for a single `dt`.

## Why this shape, and what we do not copy

A world model here is `f(B_t, action_t) → B_{t+1}` (Ha and Schmidhuber, 2018).
`B` is already a sufficient state of the teacher, so there is no recurrent
latent (DreamerV3) and no video context (Genie). The percept image is the
product, so the loss stays on that image rather than on a JEPA embedding
(V-JEPA 2). The field is smooth and low-entropy, so a diffusion sampler
(DIAMOND, Cosmos) would add a loop we would then have to run on CPU.

The pieces that do transfer:

- Patch tokens with parameter-free 2D sin/cos positions (Dosovitskiy et al.,
  2020). The grid size changes with `xystep`; a learned position table would
  not.
- Cross-attention from patches to a set of electrode tokens
  `(x, y, amplitude, frequency, phase duration)` (Perceiver, Jaegle et al.,
  2021). That matches the teacher, which is a sum over a variable set of
  electrodes. A padded `max_electrodes × 3` vector never sees where an
  electrode sits, so a new layout needs a new embedding row. Coordinates do
  not.
- Adaptive layer norm on `rho` and `axlambda` (DiT, Peebles and Xie, 2023).
  Those two scalars rescale the whole spatial kernel, so they modulate every
  block instead of sitting in one extra token.

## Simpler to use

The analytical call site is "construct an implant, build a topography, build a
model, then `step`". The topography build is the part that fails when
pulse2percept is missing, when the cache directory is not where the process
expects, or when the grid key `(xrange, yrange, xystep, axlambda, eye)` has
never been computed.

The student call site is "load a checkpoint, `bind` an implant, `step`". The
implant is a list of names and an `(E, 2)` tensor. Electrode positions still
matter, because the streaks depend on them. The axon polylines do not, because
those were the teacher's private implementation of the drive.

The same `step` is what a policy calls inside a rollout. Swapping the teacher
for the student does not change the loop, only what has to be installed and
what has to be cached before the first call.

## On a GPU

The teacher's GPU kernel is a broadcast over electrodes, pixels, and axon
samples. It is fast when that tensor fits and the device is busy moving it.
It stops being cheap when you batch patients: each patient brings their own
`(pixels, samples, 2)` topography, and the intermediate grows in the batch
dimension until the allocation fails. The topography itself is still built on
the CPU, one grid at a time, before any of that GPU work starts.

The student is a short stack of cross-attention over `patches × electrodes`
at a small width (`dim=64`, two blocks in the default). That cost does not
grow with axon samples. A batch of patients is a batch dimension on the same
module, as long as they share a grid and an electrode count, which this
Argus II distillation does. `rho` and `axlambda` are per-example inputs, so
the batch can mix subjects without a new topography.

Use the teacher when you need a percept the student has not been shown,
including a grid or an implant outside the distillation set. Use the student
when you will call `step` many times on the grids it was trained on: policy
search, long rollouts, and sweeps of pulse parameters.

## Off a GPU

On CPU the teacher is limited by memory bandwidth on the axon tensor, and a
cache miss still runs the polyline build before the first frame. The student
is a handful of GEMMs in ordinary PyTorch. No ONNX export is required for
that. `model.to("cpu")` and `step` are the whole runtime path.

That is the practical split. Interactive demos, tests, and a laptop rollout
call the student. A one-off percept at a new resolution, or the data
generation job that teaches the student, calls the axon map on a GPU.

A local CPU timing of twenty forwards, after the teacher topography was
built, is below. Building that topography on a cache miss took 35.2 s. The
forwards themselves are both under a millisecond: on a 5×5 grid the axon
tensor is small, and the teacher forward (0.367 ms) is faster than the
student (0.941 ms). The student is the one you can call without that build
and without holding the axon tensor. This is a smoke measurement, not a
cluster result. Larger grids are timed on the GPU runs, and those numbers
stay in `timing.json` on scratch.

| | grid | calls | mean forward |
| --- | --- | --- | --- |
| axon map `spatial_forward` | 5×5 | 20 | 0.367 ms |
| `AxonMapWorld.spatial_drive` | 5×5 | 20 | 0.941 ms |

## Ceiling

The student cannot be more accurate than the teacher it was distilled from.
A layout far from the Argus II coordinates in the training rollouts needs
those coordinates passed to `bind`, and it needs teacher rollouts that cover
the new geometry if the set encoder has not seen similar arrangements. This
is not a claim of zero-shot clinical fidelity.

Source: `src/sentionaut/learned/axon_world.py`
