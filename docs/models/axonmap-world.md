# Axon-map world model

[Home](../index.md) · [Getting started](../getting-started.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

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
latent (DreamerV3) and no video context (Genie). The
[video world model](axonmap-video.md) is the Genie-style counterpart that
learns the fade instead of being given it. The percept image is the
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

The grid decides which one is cheaper. On a toy grid the axon tensor is a few
kilobytes and the teacher's broadcast wins outright; the student only pays off
once there are enough pixels for the reduction over axon samples to dominate.
[Results](#results) times both at the 97×97 grid this distillation actually
uses.

## Results

Every number and figure below comes from one CPU-only reproduction of the
pipeline: a 4-core x86-64 container, 15 GB RAM, no GPU, torch 2.12.1. Rerun it
with `make axon-world`, or the three commands that target expands to.

**Teacher and data.** `axonmap` on Argus II (60 electrodes), `(-12, 12)` dva in
both axes at `xystep` 0.25, so a 97×97 percept grid and 74 axon samples per
pixel. `dt` 20 ms, fade `tau` 100 ms. 512 episodes of 4 stimulated steps plus 2
silent fade steps is 3072 transitions and a 465 MB HDF5. Each episode draws 1–3
active electrodes and fresh `rho` in [150, 300] and `axlambda` in [400, 700].
The last 20% of episodes are held out, which is 1230 training windows and 306
held-out windows of K = 4. Loss and every MSE here are in percept units divided
by the teacher's own 99th percentile (0.219), where the held-out teacher frames
reach 2.50.

Two students, same data and schedule, 50 epochs of Adam at 1e-3. The default is
`dim` 64 / `depth` 2; the wide one is `dim` 128 / `depth` 4, which is the same
architecture with more capacity and not a different formulation.

### Samples

Held-out episodes, teacher on top, student below it, absolute difference under
that. The student is free-running: it sees the first teacher frame and then only
its own previous prediction, for six steps, which is two more than the K = 4 it
was trained on. Columns t=5 and t=6 have no active electrode, so they are the
analytical fade the student inherits rather than learns.

![Default student on held-out episodes](../assets/axon-world/axon_world_samples.png)

The default student puts the phosphenes in the right place and fades them at the
right rate, but it blurs the axonal streak into the blob it starts from,
undershoots the brightest pixel, and shows 4×4 blocking from the patch decoder.
The wide student recovers the streak direction and most of the peak:

![Wide student on held-out episodes](../assets/axon-world/axon_world_wide_samples.png)

### Training and validation

![Default student loss and validation](../assets/axon-world/axon_world_validation.png)

![Wide student loss and validation](../assets/axon-world/axon_world_wide_validation.png)

Training and held-out curves sit on top of each other for the default student,
and it is still descending at epoch 50, so that run is capacity- and
schedule-limited rather than overfit. The wide student opens a small gap after
epoch 35, which is where early stopping would start to matter. In both, error
grows from the first unrolled step to the third and then falls on the fourth:
the drift is real, and the fourth step is the silent one the exact fade handles.

| held-out metric | default | wide |
| --- | --- | --- |
| parameters | 106,704 | 814,224 |
| checkpoint on disk | 1.3 MB | 9.9 MB |
| K-step MSE | 0.00497 | 0.00311 |
| RMSE | 0.0705 | 0.0558 |
| MAE | 0.0197 | 0.0130 |
| RMSE / teacher range | 2.8% | 2.2% |
| Pearson r vs teacher | 0.910 | 0.947 |
| mean peak-brightness shortfall | 0.285 | 0.103 |
| worst peak-brightness error | 1.29 | 0.881 |

The training loop's own final-epoch `val_mse` (0.00484 and 0.00311) agrees with
the MSE column; it averages per batch instead of per pixel. Peak brightness is
the weak spot in both: the student is biased low on the single brightest pixel,
by 11% of the teacher's maximum even in the wide run.

### Hardware

Teacher generation, all 3072 transitions:

| | time | note |
| --- | --- | --- |
| Jansonius topography, cold cache | 1.35 s | regrows the bundles and writes the pickle |
| Jansonius topography, warm cache | 0.65 s | pickle read |
| teacher steps | 24.1 s | 125 transitions/s |
| HDF5 write | 0.47 s | 465 MB, 32-row slabs |

The axon tensor the student never allocates is 5.6 MB here (97×97 pixels × 74
samples × 2 coordinates, fp32).

Student training, 50 epochs:

| | default | wide |
| --- | --- | --- |
| batch size | 256 | 64 |
| loader workers | 3 | 2 |
| wall clock | 8 m 49 s | 24 m 58 s |
| data loading | 14.9 s | 6.6 s |
| forward + backward | 461.6 s | 1368.5 s |
| per-epoch validation | 46.2 s | 116.1 s |
| throughput | 129 windows/s | 45 windows/s |

Batch size is not free on CPU: the wide student at batch 256 was OOM-killed at
15 GB, because the unrolled loss keeps `K × depth` attention maps of
`batch × heads × 625 patches × 60 electrodes` alive for the backward pass.
Batch 64 fits. On the GPU runs reported in the pull request the same shapes were
under 1 GB, and data loading rather than the step was the first bottleneck.

Spatial cost per call, mean of 30 after a warmup, same CPU, same 97×97 grid:

| batch | teacher `spatial_forward` | student `spatial_drive` (default) | student (wide) |
| --- | --- | --- | --- |
| 1 | 15.6 ms | 1.49 ms | 3.49 ms |
| 8 | no batch dimension | 4.79 ms (0.60 ms/sample) | 14.6 ms (1.82 ms/sample) |
| 64 | no batch dimension | 43.5 ms (0.68 ms/sample) | 137 ms (2.14 ms/sample)  |

The teacher has no batch dimension, so the honest comparison is per sample: at
this grid the default student is 10× faster than the teacher on a single call
and 23× faster batched, before counting the topography build and the 5.6 MB it
does not have to hold.

Raw numbers, including the per-step error breakdown and the full config, are in
[`axon_world_report.json`](../assets/axon-world/axon_world_report.json) and
[`axon_world_wide_report.json`](../assets/axon-world/axon_world_wide_report.json).

## Ceiling

The student cannot be more accurate than the teacher it was distilled from.
A layout far from the Argus II coordinates in the training rollouts needs
those coordinates passed to `bind`, and it needs teacher rollouts that cover
the new geometry if the set encoder has not seen similar arrangements. This
is not a claim of zero-shot clinical fidelity.

Two limits are visible in the measured samples rather than argued from the
design. The patch decoder writes each `patch_size × patch_size` block from one
token, so a low-capacity run leaves 4×4 blocking; smaller patches or an
overlapping decode would remove it at a cost in tokens. And both students
undershoot the brightest pixel, because MSE over a field that is mostly zero
pays little for the peak; a peak-weighted or log-brightness loss is the lever
there.

Source: `src/sentionaut/learned/axon_world.py`,
report: `src/sentionaut/learned/axon_report.py`
