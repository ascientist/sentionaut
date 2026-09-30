"""``DynaphosAxonMapTorch``: Dynaphos dynamics rendered through the retinal axon map.

Dynaphos (van der Grinten 2024) and the axon map (Beyeler 2019, Granley 2021)
share one spatial primitive: a Gaussian of activated tissue around each
electrode. They differ in where that Gaussian lives:

* Dynaphos (Eqs 5-6) spreads current over a disc of diameter
  ``D = 2 sqrt(I / K)`` mm and draws the phosphene as a Gaussian with
  ``2 sigma = P = D / M`` dva. In tissue coordinates that is a Gaussian of
  standard deviation ``D / 2``; the cortical magnification ``M`` only converts
  mm of cortex to dva.
* The axon map (Beyeler 2019 Eq 9) keeps the Gaussian in retinal microns and
  lets the retina -> visual-field transform of the axon topography place it,
  weighting each axon segment by ``exp(-d_soma^2 / (2 lambda^2))``.

This model keeps Dynaphos Eqs 7-13 (effective current, activation, memory trace,
sigmoid brightness, detection threshold) unchanged and replaces the ``/ M``
isotropic render with the axon-map kernel, using the Dynaphos current spread as
a per-electrode, amplitude-dependent ``rho``:

    rho_e = 1000 * D_e / 2 = 1000 * sqrt(I_e / K)                    [um]
    I(p)  = max_{q in axon(p)} sum_{e: A_e >= A_thr}
              b_e * exp(-||q - x_e||^2 / (2 rho_e^2)) * exp(-d_soma(q)^2 / (2 lambda^2))

with ``b_e = sigmoid(slope * (A_e - A50))`` and the percept clamped to [0, 1] as
in Dynaphos. ``Action.amp`` is in uA. ``Action.axlambda`` overrides ``lambda``;
``Action.rho`` is ignored because ``rho`` follows from the current.
"""

from __future__ import annotations

import torch

from ..core.base import Action, Implant, State
from ..topography.axon_map import AxonMapTopography
from .dynaphos import DynaphosTorch


class DynaphosAxonMapTorch(DynaphosTorch):
    def __init__(
        self,
        params: dict | None = None,
        axlambda: float = 500.0,
        min_rho: float = 10.0,
        costim_enabled: bool = False,
        costim_kappa: float = 1.0,
        max_percept: float | None = None,
    ):
        super().__init__(
            params=params,
            costim_enabled=costim_enabled,
            costim_kappa=costim_kappa,
            max_percept=max_percept,
        )
        self.axlambda = axlambda
        # Same floor as the Granley 2021 effect models (min_rho, microns).
        self.min_rho = min_rho

    def build(self, implant: Implant, topography: AxonMapTopography) -> "DynaphosAxonMapTorch":
        self.implant = implant
        self.topography = topography
        elec_xy = implant.electrode_coords().to(topography.coords.device)
        d2 = torch.cdist(elec_xy, elec_xy, p=2).pow(2)
        d2.fill_diagonal_(float("inf"))
        self._elec_d2 = d2
        self._built = True
        return self

    def _apply_costim(self, amp: torch.Tensor) -> torch.Tensor:
        # Vectorised form of DynaphosTorch._apply_costim (no host sync per pair):
        # leaked_i = amp_i + kappa * sum_{j != i, active} amp_j / d_ij^2 for active i.
        if not self.costim_enabled:
            return amp
        active = (amp > 0).to(amp.dtype)
        inv_d2 = 1.0 / torch.clamp(self._elec_d2.to(amp.device, amp.dtype), min=1e-12)
        return amp + active * self.costim_kappa * (inv_d2 @ (amp * active))

    def initial_state(self, device: torch.device | None = None) -> State:
        topo = self.topography
        device = device or topo.coords.device
        dtype = topo.coords.dtype
        z = torch.zeros(self.implant.n_electrodes, device=device, dtype=dtype)
        image = torch.zeros(topo.grid_shape, device=device, dtype=dtype)
        return State(image=image, aux={"A": z.clone(), "Q": z.clone(), "sigma": z.clone()})

    def _axlambda(self, action: Action, device, dtype) -> torch.Tensor:
        lam = action.axlambda if action.axlambda is not None else self.axlambda
        return torch.as_tensor(lam, device=device, dtype=dtype)

    def step(self, state: State | None, action: Action) -> State:
        topo = self.topography
        device = topo.coords.device
        action = action.to(device)
        if state is None:
            state = self.initial_state(device)
        sigma = state.aux["sigma"]

        amp, freq, p_dur = self._stim_inputs(action, device)
        A, Q, D, brightness = self._dynamics(state.aux["A"], state.aux["Q"], amp, freq, p_dur)
        # Tissue-space Gaussian SD: D/2 mm -> microns. Size uses the instantaneous
        # current only (not Q), and persists while the electrode is off, as in Dynaphos.
        rho = torch.clamp(500.0 * D, min=self.min_rho)
        sigma = torch.where(amp > 0, rho, sigma)

        elec_xy = self.implant.electrode_coords(action.pose).to(device)
        axlambda = self._axlambda(action, device, topo.coords.dtype)
        image = self._render_axon(A, sigma, brightness, elec_xy, axlambda)
        if self.max_percept is not None:
            image = torch.clamp(image, max=self.max_percept)
        return State(image=image, aux={"A": A, "Q": Q, "sigma": sigma})

    def _render_axon(self, A, sigma, brightness, elec_xy, axlambda) -> torch.Tensor:
        topo = self.topography
        active = A >= self.a_thr
        if not active.any():
            return torch.zeros(topo.grid_shape, device=topo.coords.device, dtype=topo.coords.dtype)
        idx = torch.nonzero(active, as_tuple=False).flatten()
        xy = elec_xy[idx]
        s = sigma[idx]
        b = brightness[idx]

        P, L, _ = topo.coords.shape
        q = topo.coords.reshape(P * L, 2)
        d2 = torch.cdist(q, xy, p=2).pow(2)  # (P*L, E_active)
        spread = torch.exp(-d2 / (2.0 * s[None, :] ** 2)) @ b  # sum_e b_e * Gaussian_e
        sens = torch.exp(-(topo.d_soma**2) / (2.0 * axlambda**2)) * topo.mask
        intensity = (spread.reshape(P, L) * sens).max(dim=1).values
        return torch.clamp(intensity.reshape(topo.grid_shape), 0.0, 1.0)

    def rasterize_aux(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize per-electrode A and Q through the axon map (learned aux channels)."""
        A = state.aux["A"]
        Q = state.aux["Q"]
        sigma = state.aux["sigma"]
        device = self.topography.coords.device
        elec_xy = self.implant.electrode_coords().to(device)
        lam = torch.as_tensor(self.axlambda, device=device, dtype=A.dtype)
        b_a = torch.sigmoid(self.sig_slope * (A - self.a50))
        b_q = torch.clamp(Q / (Q.max() + 1e-12), 0.0, 1.0) if Q.max() > 0 else Q
        return (
            self._render_axon(A, sigma, b_a, elec_xy, lam),
            self._render_axon(A, sigma, b_q, elec_xy, lam),
        )
