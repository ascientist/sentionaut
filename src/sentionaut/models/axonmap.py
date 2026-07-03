"""``BiphasicAxonMapTorch``: GPU reduction parity-faithful to pulse2percept.

I(p) = max_q [ sum_e F_bright_e * exp(-||q - x_e||^2 / (2 rho^2 F_size_e))
               * sens(q)^(1/F_streak_e) ]
with sens(q) = exp(-d_soma(q)^2 / (2 axlambda^2)), thresholded by thresh_percept.
Temporal state uses ``FadingTemporal`` on the spatial drive field.
"""

from __future__ import annotations

import torch

from ..core.base import Action, Implant, PerceptModel, State
from ..topography.axon_map import AxonMapTopography
from . import effects
from .fading import FadingTemporalTorch


class BiphasicAxonMapTorch(PerceptModel):
    def __init__(
        self,
        rho: float = 200.0,
        axlambda: float = 500.0,
        thresh_percept: float = 0.0,
        effect_params: effects.EffectParams = effects.DEFAULTS,
        dt_ms: float = 20.0,
        fade_tau_ms: float = 100.0,
        max_percept: float | None = None,
    ):
        super().__init__()
        self.rho = rho
        self.axlambda = axlambda
        self.thresh_percept = thresh_percept
        self.effects = effect_params
        self.dt_ms = dt_ms
        self.fade_tau_ms = fade_tau_ms
        self.max_percept = max_percept
        self.fading = FadingTemporalTorch(tau_ms=fade_tau_ms, thresh_percept=thresh_percept)

    def build(self, implant: Implant, topography: AxonMapTopography) -> "BiphasicAxonMapTorch":
        self.implant = implant
        self.topography = topography
        self._built = True
        return self

    def spatial_forward(self, action: Action) -> torch.Tensor:
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
        elec_xy = self.implant.electrode_coords(action.pose)[active]

        rho = action.rho if action.rho is not None else self.rho
        axlambda = action.axlambda if action.axlambda is not None else self.axlambda
        rho = torch.as_tensor(rho, device=device, dtype=topo.coords.dtype)
        axlambda = torch.as_tensor(axlambda, device=device, dtype=topo.coords.dtype)

        fb = effects.f_bright(freq, amp_a, pdur, self.effects)
        fs = effects.f_size(freq, amp_a, pdur, rho, self.effects)
        fst = effects.f_streak(freq, amp_a, pdur, axlambda, self.effects)

        coords = topo.coords
        d_soma = topo.d_soma
        mask = topo.mask
        sens = torch.exp(-(d_soma**2) / (2.0 * axlambda**2))

        elec = elec_xy[:, None, None, :]
        d2 = ((coords[None, ...] - elec) ** 2).sum(dim=-1)
        two_rho2 = 2.0 * rho**2
        spatial = torch.exp(-d2 / (two_rho2 * fs[:, None, None]))
        streak = sens.pow(1.0 / fst[:, None, None]) * mask[None, ...]
        acc = (fb[:, None, None] * spatial * streak).sum(dim=0)
        intensity = acc.max(dim=1).values
        intensity = torch.where(
            intensity > self.thresh_percept, intensity, torch.zeros_like(intensity)
        )
        out = intensity.reshape(topo.grid_shape)
        if self.max_percept is not None:
            out = torch.clamp(out, max=self.max_percept)
        return out

    def forward(self, action: Action) -> torch.Tensor:
        return self.spatial_forward(action)

    def step(self, state: State | None, action: Action) -> State:
        topo = self.topography
        device = topo.coords.device
        drive = self.spatial_forward(action)
        if state is None:
            B = FadingTemporalTorch.initial_brightness(
                topo.grid_shape, device, topo.coords.dtype
            )
        else:
            B = state.image
        # Spatial drive is positive brightness; FadingTemporal expects cathodic (negative) A.
        B_new = self.fading.step(B, -drive, self.dt_ms)
        return State(image=B_new)

    def predict_sequence(
        self, action: Action, n_steps: int, dt_ms: float | None = None
    ) -> torch.Tensor:
        dt = self.dt_ms if dt_ms is None else dt_ms
        state = None
        frames = []
        zero = Action(
            amp=torch.zeros_like(action.amp),
            freq=action.freq,
            phase_dur=action.phase_dur,
            rho=action.rho,
            axlambda=action.axlambda,
            pose=action.pose,
        )
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
