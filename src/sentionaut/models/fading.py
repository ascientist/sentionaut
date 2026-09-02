"""Discrete FadingTemporal step matching pulse2percept ``FadingTemporal``."""

from __future__ import annotations

import torch


class FadingTemporalTorch:
    """Leaky integrator: dB/dt = -(A + B) / tau (Euler, ms units)."""

    def __init__(self, tau_ms: float = 100.0, thresh_percept: float = 0.0):
        self.tau_ms = tau_ms
        self.thresh_percept = thresh_percept

    def step(self, B: torch.Tensor, drive: torch.Tensor, dt_ms: float) -> torch.Tensor:
        B_new = B + dt_ms * (-(drive + B) / self.tau_ms)
        return torch.where(B_new > self.thresh_percept, B_new, torch.zeros_like(B_new))

    @staticmethod
    def initial_brightness(
        shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(shape, device=device, dtype=dtype)
