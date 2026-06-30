"""``ScoreboardTorch``: cortical scoreboard (Gaussian patches in cortex coords).

I(p) = sum_regions sum_e amp_e * exp(-||c(p) - c_e||^2 / (2 rho^2)), with current
restricted to the matching cortical hemisphere, displayed on the dva grid.
"""

from __future__ import annotations

import torch

from ..core.base import Action, Implant, PerceptModel
from ..topography.cortical import CorticalTopography


class ScoreboardTorch(PerceptModel):
    def __init__(self, rho: float = 200.0, thresh_percept: float = 0.0):
        super().__init__()
        self.rho = rho
        self.thresh_percept = thresh_percept

    def build(self, implant: Implant, topography: CorticalTopography) -> "ScoreboardTorch":
        self.implant = implant
        self.topography = topography
        self._built = True
        return self

    def forward(self, action: Action) -> torch.Tensor:
        topo = self.topography
        device = topo.grid_x.device
        action = action.to(device)
        amp = action.amp
        active = amp != 0
        if active.sum() == 0:
            return torch.zeros(topo.grid_shape, device=device, dtype=topo.grid_x.dtype)

        amp_a = amp[active]
        elec_xy = self.implant.electrode_coords(action.pose)[active]  # (E, 2)
        rho = action.rho if action.rho is not None else self.rho
        rho = torch.as_tensor(rho, device=device, dtype=topo.grid_x.dtype)
        two_rho2 = 2.0 * rho**2
        boundary = topo.boundary

        n_pixels = topo.grid_x.numel()
        total = torch.zeros(n_pixels, device=device, dtype=topo.grid_x.dtype)
        elec_left = elec_xy[:, 0] < boundary  # (E,)

        for region in topo.regions:
            cxy = topo.cortex_xy[region]  # (P, 2)
            valid = torch.isfinite(cxy).all(dim=-1)
            cx = torch.nan_to_num(cxy[:, 0], nan=0.0)
            cy = torch.nan_to_num(cxy[:, 1], nan=0.0)
            pix_left = cx < boundary
            for e in range(elec_xy.shape[0]):
                same = pix_left == elec_left[e]
                dx = cx - elec_xy[e, 0]
                dy = cy - elec_xy[e, 1]
                d2 = dx * dx + dy * dy
                contrib = amp_a[e] * torch.exp(-d2 / two_rho2)
                total = total + contrib * same * valid

        total = torch.where(total > self.thresh_percept, total, torch.zeros_like(total))
        return total.reshape(topo.grid_shape)
