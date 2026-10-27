"""The encoder interface shared by every input modality."""

from __future__ import annotations

import torch
import torch.nn as nn


class StimulusEncoder(nn.Module):
    """``<modality>2stim``: a signal frame -> amplitudes ``(B, E)`` in ``[0, a_max]``.

    ``modality`` names the input (``"vision"``, ``"sound"``). Amplitude units
    follow the forward model the encoder drives: multiples of threshold for the
    axon map, µA for Scoreboard and Dynaphos. ``a_min`` is the smallest
    amplitude worth delivering; encoders return exact zeros below it, which is
    how a forward model knows an electrode is off.
    """

    modality: str = ""

    def __init__(self, n_electrodes: int, a_max: float, a_min: float = 0.0):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.a_max = a_max
        self.a_min = a_min

    def level_to_amp(self, level: torch.Tensor, on: float = 0.0) -> torch.Tensor:
        """Map a level in ``[0, 1]`` to ``[a_min, a_max]``; levels ``<= on`` give 0."""
        amp = self.a_min + (self.a_max - self.a_min) * level.clamp(0.0, 1.0)
        return amp * (level > on).to(amp.dtype)

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, ...)`` frame by frame -> ``(B, T, E)``."""
        return torch.stack([self(frames[:, t]) for t in range(frames.shape[1])], 1)
