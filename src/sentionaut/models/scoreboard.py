"""``ScoreboardTorch``: cortical scoreboard (Gaussian patches in cortex coords).

I(p) = sum_regions sum_e amp_e * exp(-||c(p) - c_e||^2 / (2 rho^2)), with current
restricted to the matching cortical hemisphere, displayed on the dva grid.
Temporal state uses ``FadingTemporal`` on the spatial drive field.
"""

from __future__ import annotations

import torch

from ..core.base import Action, Implant, PerceptModel, State
from ..topography.cortical import CorticalTopography
from .fading import FadingTemporalTorch


class ScoreboardTorch(PerceptModel):
    def __init__(
        self,
        rho: float = 200.0,
        thresh_percept: float = 0.0,
        dt_ms: float = 20.0,
        fade_tau_ms: float = 100.0,
        max_percept: float | None = None,
    ):
        super().__init__()
        self.rho = rho
        self.thresh_percept = thresh_percept
        self.dt_ms = dt_ms
        self.fade_tau_ms = fade_tau_ms
        self.max_percept = max_percept
        self.fading = FadingTemporalTorch(tau_ms=fade_tau_ms, thresh_percept=thresh_percept)

    def build(self, implant: Implant, topography: CorticalTopography) -> "ScoreboardTorch":
        self.implant = implant
        self.topography = topography
        self._built = True
        return self

    def spatial_forward(self, action: Action) -> torch.Tensor:
        topo = self.topography
        device = topo.grid_x.device
        action = action.to(device)
        amp = action.amp
        active = amp != 0
        if active.sum() == 0:
            return torch.zeros(topo.grid_shape, device=device, dtype=topo.grid_x.dtype)

        amp_a = amp[active]
        elec_xy = self.implant.electrode_coords(action.pose)[active]
        rho = action.rho if action.rho is not None else self.rho
        rho = torch.as_tensor(rho, device=device, dtype=topo.grid_x.dtype)
        two_rho2 = 2.0 * rho**2
        boundary = topo.boundary

        n_pixels = topo.grid_x.numel()
        total = torch.zeros(n_pixels, device=device, dtype=topo.grid_x.dtype)
        elec_left = elec_xy[:, 0] < boundary

        for region in topo.regions:
            cxy = topo.cortex_xy[region]
            valid = torch.isfinite(cxy).all(dim=-1)
            cx = torch.nan_to_num(cxy[:, 0], nan=0.0)
            cy = torch.nan_to_num(cxy[:, 1], nan=0.0)
            pix_left = cx < boundary
            cortex = torch.stack([cx, cy], dim=-1)
            d2 = torch.cdist(cortex, elec_xy, p=2).pow(2)
            same = (pix_left[:, None] == elec_left[None, :]) & valid[:, None]
            contrib = (amp_a[None, :] * torch.exp(-d2 / two_rho2) * same).sum(dim=-1)
            total = total + contrib

        total = torch.where(total > self.thresh_percept, total, torch.zeros_like(total))
        out = total.reshape(topo.grid_shape)
        if self.max_percept is not None:
            out = torch.clamp(out, max=self.max_percept)
        return out

    def forward(self, action: Action) -> torch.Tensor:
        return self.spatial_forward(action)

    def step(self, state: State | None, action: Action) -> State:
        topo = self.topography
        device = topo.grid_x.device
        drive = self.spatial_forward(action)
        if state is None:
            B = FadingTemporalTorch.initial_brightness(
                topo.grid_shape, device, topo.grid_x.dtype
            )
        else:
            B = state.image
        B_new = self.fading.step(B, -drive, self.dt_ms)
        return State(image=B_new)

    def predict_sequence(
        self, action: Action, n_steps: int, dt_ms: float | None = None
    ) -> torch.Tensor:
        dt = self.dt_ms if dt_ms is None else dt_ms
        state = None
        frames = []
        zero = Action(amp=torch.zeros_like(action.amp), rho=action.rho, pose=action.pose)
        for step_i in range(n_steps):
            act = action if step_i == 0 else zero
            drive = self.spatial_forward(act)
            if state is None:
                B = FadingTemporalTorch.initial_brightness(
                    self.topography.grid_shape, drive.device, drive.dtype
                )
            else:
                B = state.image
            B_new = self.fading.step(B, -drive, dt)
            state = State(image=B_new)
            frames.append(B_new)
        return torch.stack(frames, dim=0)
