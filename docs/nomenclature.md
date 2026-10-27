# Nomenclature

[Home](index.md) · [Getting started](getting-started.md) · [User guide](user-guide.md) · [Nomenclature](nomenclature.md) · [Models](models/index.md) · [References](references.md)

Models are tagged with an ML-style **`source2target`** label built from a fixed
modality vocabulary, the same pattern as `image2text` or `speech2text`. The
label says what goes in and what comes out.

## Vocabulary

| Token | Meaning |
| --- | --- |
| `vision` | visual content: an image, a video frame, or a visual percept |
| `sound` | an audio signal or an auditory percept |
| `language` | text or speech tokens |
| `stim` | electrical stimulation written into tissue: electrode amplitudes and pulse parameters |
| `brain` | neural activity *recorded* from tissue (spikes, multi-unit activity, calcium imaging) |

`stim` and `brain` are kept apart on purpose. They point in opposite
directions: `stim` is what the implant writes, `brain` is what a recording
reads. In the wider literature `brain2X` always means decoding `X` from
recorded activity (`brain2text`, Brain2Image), so Sentionaut keeps that sense.

Tissue (retinal or cortical) is a separate axis, not part of the tag.

## The three directions

| Tag | Role | Example | In Sentionaut |
| --- | --- | --- | --- |
| `vision2stim`, `sound2stim` | **encoder**: signal → stimulation | the processor of a retinal or cochlear implant | `PhospheneMapEncoder`, `ClosedLoopVisionEncoder`, `BandEnvelopeEncoder` |
| `stim2vision` | **forward model** (generator): stimulation → percept | pulse2percept, Dynaphos | every percept model and learned world model |
| `brain2vision` | **decoder**: recorded activity → visual content | POYO+, image reconstruction from V1 | none yet (NeuroStim-Percept's toy readout `D` is one) |

See [Architecture](architecture.md) for how they connect.

## Current models

Every percept model (axon map, Scoreboard, Dynaphos) and every learned world
model is **`stim2vision`**: an `Action` of electrode amplitudes produces a
percept. Retinal and cortical models differ by implant and topography, not
by tag.

Earlier versions of these docs tagged the forward models `brain2vision`. That
collided with the standard decoding sense and is retired for them.

## What is not tagged

Implants and topography maps are supporting geometry. They do not get a
`source2target` label; they are described under components in the
[user guide](user-guide.md).
