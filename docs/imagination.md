# Imagination: training an encoder through a learned world model

[Home](index.md) · [Getting started](getting-started.md) · [User guide](user-guide.md) · [Nomenclature](nomenclature.md) · [Architecture](architecture.md) · [Models](models/index.md) · [References](references.md)

This page asks the SuperMimic question for prostheses. SuperMimic (Liao et
al., CoRL 2026) trains a humanoid controller by backpropagating through a
learned world model fitted online to a rigid-body simulator, and reaches the
quality of a 128-GPU PPO run on one GPU. Here the controller is a **vision
encoder** (`vision2stim`), and the simulator is one of the differentiable
percept models: the retinal **axon map** or the cortical **Dynaphos**
(`stim2vision`).

There is one difference that matters. MuJoCo gives no useful gradients, so
SuperMimic had to learn a model to get them. The Sentionaut ports are
differentiable, so the encoder can also be trained by backpropagating
through the teacher itself. That control decides whether the learned model
earns its place.

Source: `src/sentionaut/learned/imagine.py` · CLI: `sentionaut-imagine` ·
Report: `scripts/imagine_report.py`

## Task

An encoder sees an MNIST digit drawn on the percept grid and the percept
from the previous step, and sets the amplitude of every electrode for 10
steps of 20 ms. It is scored on the percept at steps 3–10:

```text
loss = mean_t ‖percept_t / scale − target‖²  +  0.02 · mean charge / a_max
```

`scale` is the 99th percentile of the teacher's percepts under random
stimulation, so the two teachers are on the same footing.

| | Axon map | Dynaphos |
| --- | --- | --- |
| Implant | Argus II, 60 electrodes (retina) | Orion, 60 electrodes (V1, one hemisphere) |
| Percept grid | 31 × 21, 1° step, x ±15°, y ±10° | 26 × 26, 0.2° step, x −4°…1°, y ±2.5° |
| Amplitude | 0–3 × threshold; below 0.5 the electrode is off | 0–300 µA; below 30 µA the electrode is off |
| Non-smooth step | brightness jumps to `a3·freq` when an electrode turns on | nothing renders until activation crosses `a_thr` |
| Teacher cost, 32 episodes × 10 steps, one CPU thread | 9.3 s forward (sparse, 5σ); world model 0.3 s | 0.1 s forward |

Electrodes below the floor amplitude are switched off by the implant (the
environment), as SuperMimic's simulator executes the policy's raw action. A
gradient through the teacher sees neither the gate nor the on-step; a
learned model fitted to the teacher's outputs sees a smoothed version of
both.

## Arms

All arms use the same encoder (a two-layer MLP on target and current
percept), the same Adam learning rate and batch of 64 episodes, the same
device, and a 30-minute training budget. Evaluation runs on the teacher, on
held-out digits, outside the clock.

| Arm | Gradient from | Teacher used for |
| --- | --- | --- |
| **Backprop through teacher** | the analytical model (checkpointed per step) | every update |
| **Backprop through world model** (SuperMimic loop) | a `TokenWorldModel` fitted online | collecting episodes only |
| REINFORCE in teacher | score function, per-step baseline | every update |
| REINFORCE in world model | score function on imagined episodes | collecting episodes only |

The world-model loop follows SuperMimic's three phases, repeated:

1. **Collect** 64 teacher episodes with the current encoder plus action noise
   (5% of `a_max`) into a replay buffer (512 episodes on the first cycle).
2. **Fit** the world model for 40 steps on 8-step autoregressive windows
   seeded at logged states (Dynaphos: the per-electrode activation, charge
   and size).
3. **Imagine**: 20 encoder updates by backpropagating the episode loss
   through the frozen world model, from the blank percept.

**The world model is not SuperMimic's MLP.** `TokenWorldModel` treats each
electrode as a token with its own recurrent memory (a shared GRU cell driven
by that electrode's amplitude), because Dynaphos's state is per-electrode
charge and activation. Image patches carry the previous percept and query
the electrode tokens by cross-attention, with a learned null key so a patch
can receive nothing; the output is a residual update of the percept.
About 94 k parameters.

Differences from SuperMimic, stated plainly: episodes are 10 steps, so the
imagined horizon is the whole episode and there is no critic; rollouts start
from the blank percept, not from replay states; the scale is one CPU thread,
not an H100.

## Results

The full runs are GPU jobs on the Mila cluster (below); their figures and
tables come from `scripts/imagine_report.py` once the jobs finish.

### CPU pilot (axon map, one seed, one CPU thread, 30 minutes per arm)

| Arm | Held-out loss | Percept MSE | Charge | Teacher steps |
| --- | ---: | ---: | ---: | ---: |
| Backprop through teacher | 0.043 | 0.041 | 0.108 | 14,720 |
| Backprop through world model | 0.061 | 0.061 | 0.000 | 16,000 |
| REINFORCE in teacher | 0.070 | 0.066 | 0.180 | 42,240 |
| REINFORCE in world model | 0.097 | 0.092 | 0.291 | 18,560 |

One axon-map step costs about 0.3 s on one CPU thread for 32 episodes (sparse
path), against about 0.03 s for the world model. Even so, backpropagating
through the teacher won this pilot. The world-model arm failed in an
instructive way:

- after the first cycle the encoder learned that switching every electrode
  off beats switching them all on (loss 0.058 against 0.139);
- its action noise (5% of `a_max`, 0.15× threshold) never crossed the 0.5×
  threshold gate, so every later teacher episode was dark;
- the world model learned "nothing happens" (fit error 10⁻⁴) and gave no
  gradient back towards stimulation.

SuperMimic adds action noise to widen the range the model is fitted on.
With a gate, the noise has to cross it. The loop now replaces half of each
collected batch with random stimulation (30% of electrodes, amplitudes above
the gate), the way the axon-map distillation data are generated. The pilot
above predates that fix and is kept as the record of the failure.

## Running on the Mila cluster

The experiment, the benchmark and the drawing fits run as Slurm jobs on the
`main` partition with one L40S GPU each. `main` caps a job at 2 GPUs, 8 CPUs
and 48 GB for 5 days, and its jobs are not preempted by other `main` jobs, so a
timed run is not killed half-way. Every task asks for the same GPU type, so the
30-minute budgets are comparable across arms. From a login node, in
`$HOME/sentionaut` on this branch:

```bash
sbatch scripts/mila_imagine.sh        # 24 tasks: 2 teachers x 4 arms x 3 seeds, 8 at a time
sbatch scripts/mila_bench_forward.sh  # throughput: pulse2percept (cython, jax) vs ports vs world model
sbatch scripts/mila_fit_drawings.sh   # axon map fitted to the Beyeler 2019 drawings
```

`scripts/mila_prepare.sh` runs first in every job. Under a file lock, it
installs the environment on the GPU node and builds the MNIST and axon-map
caches on `$SCRATCH`, so array tasks do not race on them. Results land in
`$SCRATCH/sentionaut/{imagine/runs,bench,p2p}`. Then:

```bash
uv run python scripts/imagine_report.py --runs $SCRATCH/sentionaut/imagine/runs
```

