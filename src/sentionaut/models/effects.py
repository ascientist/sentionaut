"""Torch ports of the Granley 2021 biphasic effect models (Eqs 3-6).

Coefficients are taken verbatim from pulse2percept 0.9.0
(``models/granley2021.py``); see DEBRIEF.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EffectParams:
    a0: float = 2.095
    a1: float = 0.054326
    a2: float = 0.1492147
    a3: float = 0.0163851
    a4: float = 0.0
    a5: float = 1.0812
    a6: float = -0.35338
    a7: float = 0.54
    a8: float = 0.21
    a9: float = 1.56
    min_rho: float = 10.0
    min_lambda: float = 10.0


DEFAULTS = EffectParams()


def scale_threshold(pdur: torch.Tensor, p: EffectParams = DEFAULTS) -> torch.Tensor:
    """Eq 3: amplitude scaling factor a_tilde = a1 + a0 * pdur."""
    return p.a1 + p.a0 * pdur


def f_bright(
    freq: torch.Tensor, amp: torch.Tensor, pdur: torch.Tensor, p: EffectParams = DEFAULTS
) -> torch.Tensor:
    """Eq 4: F_bright = a2 * (amp * a_tilde) + a3 * freq + a4."""
    return p.a2 * (amp * scale_threshold(pdur, p)) + p.a3 * freq + p.a4


def f_size(
    freq: torch.Tensor,
    amp: torch.Tensor,
    pdur: torch.Tensor,
    rho: float | torch.Tensor,
    p: EffectParams = DEFAULTS,
) -> torch.Tensor:
    """Eq 5: F_size = max(a5 * amp * a_tilde + a6, min_rho^2 / rho^2)."""
    min_f = p.min_rho**2 / (rho**2)
    val = p.a5 * amp * scale_threshold(pdur, p) + p.a6
    return torch.clamp(val, min=float(min_f) if not torch.is_tensor(min_f) else min_f)


def f_streak(
    freq: torch.Tensor,
    amp: torch.Tensor,
    pdur: torch.Tensor,
    axlambda: float | torch.Tensor,
    p: EffectParams = DEFAULTS,
) -> torch.Tensor:
    """Eq 6: F_streak = max(a9 - a7 * pdur^a8, min_lambda^2 / axlambda^2)."""
    min_f = p.min_lambda**2 / (axlambda**2)
    val = p.a9 - p.a7 * pdur**p.a8
    return torch.clamp(val, min=float(min_f) if not torch.is_tensor(min_f) else min_f)
