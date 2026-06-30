"""``BiphasicAxonMapTorch``: GPU reduction parity-faithful to pulse2percept.

I(p) = max_q [ sum_e F_bright_e * exp(-||q - x_e||^2 / (2 rho^2 F_size_e))
               * sens(q)^(1/F_streak_e) ]
with sens(q) = exp(-d_soma(q)^2 / (2 axlambda^2)), thresholded by thresh_percept.
"""

from __future__ import annotations

import torch

from ..core.base import Action, Implant, PerceptModel
from ..topography.axon_map import AxonMapTopography
from . import effects


class BiphasicAxonMapTorch(PerceptModel):
    def __init__(
        self,
        rho: float = 200.0,
        axlambda: float = 500.0,
        thresh_percept: float = 0.0,
        effect_params: effects.EffectParams = effects.DEFAULTS,
    ):
        super().__init__()
        self.rho = rho
        self.axlambda = axlambda
        self.thresh_percept = thresh_percept
        self.effects = effect_params

    def build(self, implant: Implant, topography: AxonMapTopography) -> "BiphasicAxonMapTorch":
        self.implant = implant
        self.topography = topography
        self._built = True
        return self

    def forward(self, action: Action) -> torch.Tensor:
        topo = self.topography
        device = topo.coords.device
        action = action.to(device)
        amp = action.amp
        active = amp != 0
        if active.sum() == 0:
            return torch.zeros(topo.grid_shape, device=device, dtype=topo.coords.dtype)

        freq = action.freq[active]
        pdur = action.phase_dur[active]
        amp_a = amp[active]
        elec_xy = self.implant.electrode_coords(action.pose)[active]  # (E, 2)

        rho = action.rho if action.rho is not None else self.rho
        axlambda = action.axlambda if action.axlambda is not None else self.axlambda
        rho = torch.as_tensor(rho, device=device, dtype=topo.coords.dtype)
        axlambda = torch.as_tensor(axlambda, device=device, dtype=topo.coords.dtype)

        fb = effects.f_bright(freq, amp_a, pdur, self.effects)
        fs = effects.f_size(freq, amp_a, pdur, rho, self.effects)
        fst = effects.f_streak(freq, amp_a, pdur, axlambda, self.effects)

        coords = topo.coords  # (P, L, 2)
        d_soma = topo.d_soma  # (P, L)
        mask = topo.mask  # (P, L)
        sens = torch.exp(-(d_soma**2) / (2.0 * axlambda**2))  # (P, L)

        acc = torch.zeros_like(d_soma)
        two_rho2 = 2.0 * rho**2
        for e in range(elec_xy.shape[0]):
            dx = coords[..., 0] - elec_xy[e, 0]
            dy = coords[..., 1] - elec_xy[e, 1]
            d2 = dx * dx + dy * dy
            spatial = torch.exp(-d2 / (two_rho2 * fs[e]))
            streak = sens.pow(1.0 / fst[e]) * mask
            acc = acc + fb[e] * spatial * streak

        intensity = acc.max(dim=1).values  # (P,)
        intensity = torch.where(
            intensity > self.thresh_percept, intensity, torch.zeros_like(intensity)
        )
        return intensity.reshape(topo.grid_shape)
