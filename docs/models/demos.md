# Stimulation demos: how to read them

[Home](../index.md) · [Getting started](../getting-started.md) · [User guide](../user-guide.md) · [Nomenclature](../nomenclature.md) · [Models](index.md) · [References](../references.md)

Each physics model page has a short demo clip. All three clips answer the same
question:

> **If we keep the implant still and stimulate one small spot, what does the
> patient see, and how does it change over time?**

This page explains the shared setup, the stimulation sequence, and how to read
the figures. The model pages then walk through what is specific to each model:
[Axon map](axonmap.md#demo-walkthrough),
[Scoreboard](scoreboard.md#demo-walkthrough),
[Dynaphos](dynaphos.md#demo-walkthrough).

## 1. The setup

- **The implant never moves.** Its pose is fixed for the whole clip, so any
  change in the percept comes from the stimulation, not from geometry.
- **One zone of four electrodes.** The electrode closest to the centre of the
  array plus its three nearest neighbours (Argus II for the retinal model,
  Orion for the cortical models). They are drawn in **red** in the tissue panel.
- **Model parameters are fixed.** For the axon map, $\rho = 200\,\mu m$ and
  $\lambda = 500\,\mu m$. For the Scoreboard, $\rho = 1000\,\mu m$ on the cortex.
- **Time.** One frame is one call to `model.step`, which advances
  $\Delta t = 20$ ms of simulated time. The GIFs play at 12 frames per second,
  so they run about 4 times slower than real time.

## 2. The stimulation sequence

The same schedule is used for all three models (only the amplitude unit
changes). Every step is a pulse followed by a rest, so you can watch each
response appear and fade.

| Stage | Electrodes | Amplitude (retinal / cortical) | Frames |
| --- | --- | --- | --- |
| Singles | each zone electrode alone, 4 steps | 2 × threshold / 200 µA | 5 on + 4 rest each |
| Pairs | centre + one neighbour, 3 steps | 2 × threshold / 200 µA | 5 on + 4 rest each |
| Ramp | all four together, 3 steps | 1, 2, 3 × threshold / 100, 200, 300 µA | 5 on + 4 rest each |
| Pulse train | all four together, 4 pulses | 2 × threshold / 200 µA | 3 on + 3 rest each |

That is 114 frames, or 2.28 s of simulated time. The model state is carried
from one frame to the next for the whole clip, so what happened earlier (for
example, leftover brightness, or charge build-up in Dynaphos) affects what
comes later.

## 3. How to read the figures

**The clip.** Left: the predicted percept in degrees of visual angle (dva).
Right: the tissue with the electrode array. Red dots are the zone; yellow
circles mark the electrodes being stimulated in the current frame. The title
says which stage you are in, which electrodes are on, and at what amplitude.
The colour scale is the same for the whole clip, so a darker frame really is a
dimmer percept.

**The key frames.** Six snapshots, numbered (1) to (6): the end of the first
single pulse, the first pair, the three ramp levels, and the last rest frame
before the pulse train.

**The timeline.** The black curve is the brightest pixel of the percept at
each frame. Coloured bands show when stimulation is on (blue: singles, green:
pairs, orange: ramp, purple: pulse train); white gaps are rests. The numbered
dots are the key frames, so you can match each snapshot to its moment in time.

## 4. One idea shared by Axon map and Scoreboard: fading

Both models compute a spatial "drive" from the stimulation, then pass it
through the same leaky integrator (`FadingTemporal`, time constant
$\tau = 100$ ms):

$$
B_{t+1} = B_t + \frac{\Delta t}{\tau}\,\big(\text{drive}_t - B_t\big)
       = B_t + 0.2\,\big(\text{drive}_t - B_t\big).
$$

Each frame, the brightness closes 20% of the gap to the drive. Two practical
consequences you can see in every timeline:

- A 5-frame pulse only reaches $1 - 0.8^5 \approx 67\%$ of its steady
  brightness, so the percept is still rising when the pulse stops.
- A 4-frame rest leaves $0.8^4 \approx 41\%$ of the brightness, so the percept
  never returns to black between steps. Each step starts from what the
  previous one left behind.

Dynaphos uses a different, richer temporal model; see its page.

## 5. Side-by-side summary

| Question | Axon map | Scoreboard | Dynaphos |
| --- | --- | --- | --- |
| Shape of one phosphene | streak along the axon bundle | round blob | small round, sharp-edged blob |
| What does more current do? | mostly a **bigger** phosphene | a **brighter** phosphene, same size | a **bigger** phosphene; brightness saturates |
| How fast does it appear and fade? | 20% per frame (100 ms time constant) | same as Axon map | appears in 1–2 frames, lingers about 200 ms after the pulse |
| Does repeated stimulation weaken it? | no (no memory) | no (no memory) | **yes**: the memory trace lowers the effective current |

## 6. Try it yourself

Regenerate all clips and figures (written to `docs/assets/demos/`):

```bash
make demos
```

Or step a model yourself with the same schedule:

```python
import torch
from sentionaut.animate import sequence_schedule, stimulation_zone
from sentionaut.core.base import Action, Pose
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

cfg = Config(model="scoreboard", implant="orion", xrange=(-6, 6), yrange=(-6, 6), xystep=0.2)
implant, topo, model = build_components(cfg)
zone = stimulation_zone(implant.electrode_coords().cpu().numpy(), size=4)

state = None
for step in sequence_schedule(zone, levels=(100.0, 200.0, 300.0)):
    amp = torch.zeros(implant.n_electrodes)
    amp[list(step.electrodes)] = step.amp
    state = model.step(state, Action(amp=amp, rho=1000.0, pose=Pose()))
    names = [implant.names[e] for e in step.electrodes]  # e.g. ['66'], as in the figures
    print(step.stage, names, float(state.image.max()))
```

Change `on`/`rest` in `sequence_schedule` to see the effect of longer pulses
or longer rests: with a long enough rest the percept goes fully dark, and with
a long enough pulse it reaches its steady brightness.

Source: `src/sentionaut/animate.py` (`sequence_schedule`, `stimulation_zone`,
`animate_sequence`).
