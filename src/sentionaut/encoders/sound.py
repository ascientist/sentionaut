"""``sound2stim`` encoders: an audio waveform -> electrode amplitudes per frame."""

from __future__ import annotations

import math

import torch

from .base import StimulusEncoder


class BandEnvelopeEncoder(StimulusEncoder):
    """Band envelopes -> one electrode per band (the CIS idea of cochlear implants).

    The waveform is cut into frames of ``frame_ms`` (one frame per stimulation
    step, 20 ms by default). Each frame's spectrum is pooled into
    ``n_electrodes`` log-spaced bands between ``f_lo`` and ``f_hi``; each band's
    energy is log-compressed over ``dynamic_range_db`` into a level in
    ``[0, 1]``, and the level sets the amplitude of that band's electrode.

    ``order`` lists the electrode driven by each band, lowest band first. For a
    cochlear array that is apex to base. For a visual implant it is a sensory
    substitution choice (for example, electrodes sorted bottom to top), and no
    auditory forward model exists in Sentionaut yet.
    """

    modality = "sound"

    def __init__(
        self,
        n_electrodes: int,
        a_max: float,
        a_min: float = 0.0,
        sample_rate: int = 16000,
        frame_ms: float = 20.0,
        f_lo: float = 200.0,
        f_hi: float = 7000.0,
        dynamic_range_db: float = 40.0,
        order: list[int] | None = None,
    ):
        super().__init__(n_electrodes, a_max, a_min)
        self.sample_rate = sample_rate
        self.frame = int(round(sample_rate * frame_ms / 1000.0))
        self.dynamic_range_db = dynamic_range_db
        freqs = torch.fft.rfftfreq(self.frame, 1.0 / sample_rate)
        edges = torch.logspace(math.log10(f_lo), math.log10(f_hi), n_electrodes + 1)
        bands = ((freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])).float()
        empty = bands.sum(1) == 0  # narrow low bands may miss every FFT bin
        nearest = (freqs[None] - edges[:-1, None].add(edges[1:, None]).div(2)[:, :1]).abs()
        bands[empty] = (nearest[empty] == nearest[empty].min(1, keepdim=True).values).float()
        self.register_buffer("bands", bands / bands.sum(1, keepdim=True))
        self.register_buffer("window", torch.hann_window(self.frame))
        order = list(range(n_electrodes)) if order is None else list(order)
        if sorted(order) != list(range(n_electrodes)):
            raise ValueError("order must be a permutation of the electrodes")
        self.register_buffer("order", torch.tensor(order))

    def frames(self, wave: torch.Tensor) -> torch.Tensor:
        """``(B, N)`` samples -> ``(B, T, frame)`` non-overlapping windowed frames."""
        n = wave.shape[-1] // self.frame
        return wave[..., : n * self.frame].reshape(*wave.shape[:-1], n, self.frame) * self.window

    def levels(self, frames: torch.Tensor) -> torch.Tensor:
        """``(..., frame)`` -> band levels ``(..., E)`` in ``[0, 1]`` (0 dB = full scale)."""
        power = torch.fft.rfft(frames, dim=-1).abs() ** 2 / (self.frame / 2) ** 2
        db = 10.0 * torch.log10((power @ self.bands.T).clamp(min=1e-12))
        return (db / self.dynamic_range_db + 1.0).clamp(0.0, 1.0)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """One frame ``(B, frame)`` -> amplitudes ``(B, E)``."""
        level = self.levels(frame * self.window)
        amp = self.level_to_amp(level)
        out = torch.zeros_like(amp)
        return out.index_copy(1, self.order, amp)

    def encode_wave(self, wave: torch.Tensor) -> torch.Tensor:
        """A whole waveform ``(B, N)`` -> amplitudes ``(B, T, E)``, one row per frame."""
        amp = self.level_to_amp(self.levels(self.frames(wave)))
        out = torch.zeros_like(amp)
        return out.index_copy(2, self.order, amp)
