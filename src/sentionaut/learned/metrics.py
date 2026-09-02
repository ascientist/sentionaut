"""Evaluation metrics for learned vs analytical world models."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

from ..core.base import Action, State
from ..world import WorldModel


def mse(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    if isinstance(a, torch.Tensor):
        return float(torch.mean((a - b) ** 2).item())
    return float(np.mean((a - b) ** 2))


def max_abs(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    if isinstance(a, torch.Tensor):
        return float(torch.max(torch.abs(a - b)).item())
    return float(np.max(np.abs(a - b)))


def ssim(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    x = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    y = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    dr = max(float(x.max() - x.min()), float(y.max() - y.min()), 1e-8)
    side = min(x.shape[-2], x.shape[-1])
    win = min(7, side if side % 2 else side - 1)
    win = max(3, win)
    return float(structural_similarity(x, y, data_range=dr, win_size=win))


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mse": mse(pred, target),
        "max_abs": max_abs(pred, target),
        "ssim": ssim(pred[0], target[0])
        if pred.shape[0] == 1
        else ssim(pred.mean(0), target.mean(0)),
    }


def analytical_rollout_mse(
    wm: WorldModel,
    s_t: torch.Tensor,
    action_vecs: list[torch.Tensor],
    s_tp1: torch.Tensor,
    *,
    n_electrodes: int,
) -> dict[str, float]:
    """One-step analytical rollout vs stored ``s_tp1``."""
    amp = action_vecs[0][:n_electrodes]
    freq = action_vecs[0][n_electrodes : 2 * n_electrodes]
    pdur = action_vecs[0][2 * n_electrodes : 3 * n_electrodes]
    rho = float(action_vecs[0][-2])
    axl = float(action_vecs[0][-1])
    act = Action(
        amp=amp,
        freq=freq,
        phase_dur=pdur,
        rho=rho,
        axlambda=axl if axl > 0 else None,
    )
    state = State(image=s_t.squeeze(0))
    pred = wm.step(state, act).image
    return batch_metrics(pred.unsqueeze(0), s_tp1)
