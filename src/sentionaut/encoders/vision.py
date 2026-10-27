"""``vision2stim`` encoders: an image on the percept grid -> electrode amplitudes."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .base import StimulusEncoder


@torch.no_grad()
def probe_centroids(
    render: Callable[[torch.Tensor], torch.Tensor], n_electrodes: int, amp: float
) -> torch.Tensor:
    """Phosphene centre of each electrode, in grid pixels ``(E, 2)`` as ``(row, col)``.

    ``render(amp (B, E)) -> percepts (B, H, W)`` is any forward model. Each
    electrode is driven alone at ``amp`` and the brightness-weighted centroid
    of its percept is returned. Electrodes that produce nothing get ``nan``.
    """
    img = render(torch.eye(n_electrodes) * amp)
    h, w = img.shape[-2:]
    kw = {"dtype": img.dtype, "device": img.device}
    rr, cc = torch.meshgrid(torch.arange(h, **kw), torch.arange(w, **kw), indexing="ij")
    total = img.sum((1, 2), keepdim=True)
    wts = img / total.clamp(min=1e-12)
    cent = torch.stack([(wts * rr).sum((1, 2)), (wts * cc).sum((1, 2))], -1)
    return torch.where(total.view(-1, 1) > 0, cent, torch.full_like(cent, float("nan")))


class PhospheneMapEncoder(StimulusEncoder):
    """The standard baseline: drive each electrode by the image at its phosphene.

    The image is sampled (bilinearly) at each electrode's phosphene centre and
    the value becomes the amplitude; pixels at or below ``on`` leave the
    electrode off. No learning. This is the "scoreboard" strategy of current
    devices, which ignores streaks, phosphene size and electrode interactions.
    """

    modality = "vision"

    def __init__(
        self,
        centroids: torch.Tensor,
        grid: tuple[int, int],
        a_max: float,
        a_min: float = 0.0,
        on: float = 0.3,
    ):
        super().__init__(centroids.shape[0], a_max, a_min)
        h, w = grid
        c = centroids.clone()
        valid = torch.isfinite(c).all(-1)
        c[~valid] = 0.0
        # grid_sample coordinates: x = col, y = row, in [-1, 1]
        xy = torch.stack([c[:, 1] / max(w - 1, 1), c[:, 0] / max(h - 1, 1)], -1) * 2 - 1
        self.register_buffer("xy", xy)
        self.register_buffer("valid", valid)
        self.on = on

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grid = self.xy.view(1, 1, -1, 2).expand(image.shape[0], -1, -1, -1)
        level = nn.functional.grid_sample(
            image[:, None], grid, mode="bilinear", align_corners=True
        )[:, 0, 0]
        return self.level_to_amp(level, self.on) * self.valid


class ClosedLoopVisionEncoder(StimulusEncoder):
    """Learned encoder: (target image, current percept) -> amplitudes.

    It sees the percept the forward model produced at the previous step, so it
    can correct fading and charge build-up over a pulse train. It is trained by
    backpropagating a percept loss through a differentiable forward model:
    the analytical teacher, or a learned world model (``sentionaut.learned.imagine``).
    """

    modality = "vision"

    def __init__(self, grid: tuple[int, int], n_electrodes: int, a_max: float, hidden: int = 512):
        super().__init__(n_electrodes, a_max)
        n_pix = grid[0] * grid[1]
        self.net = nn.Sequential(
            nn.Linear(2 * n_pix, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_electrodes),
        )

    def logits(self, target: torch.Tensor, percept: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([target.flatten(1), percept.flatten(1)], -1))

    def forward(self, target: torch.Tensor, percept: torch.Tensor | None = None) -> torch.Tensor:
        if percept is None:
            percept = torch.zeros_like(target)
        return self.a_max * torch.sigmoid(self.logits(target, percept))
