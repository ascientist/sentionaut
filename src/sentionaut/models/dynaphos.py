"""``DynaphosTorch``: cortical Dynaphos model (van der Grinten 2024 / p2p port).

Gaussian phosphenes in dva whose size derives from cortical magnification +
stimulation current, plus a temporal leaky integrator with charge accumulation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ..core.base import Action, Implant, PerceptModel, State
from ..topography.cortical import CorticalTopography

_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def _load_dynaphos_params() -> dict:
    with open(_PARAMS_PATH) as fh:
        return yaml.safe_load(fh)["dynaphos"]


class DynaphosTorch(PerceptModel):
    def __init__(
        self,
        params: dict | None = None,
        costim_enabled: bool = False,
        costim_kappa: float = 1.0,
        max_percept: float | None = None,
    ):
        super().__init__()
        p = params or _load_dynaphos_params()
        self.dt = float(p["dt"])
        self.tau_act = float(p["tau_act"])
        self.rheobase = float(p["rheobase"])
        self.tau_trace = float(p["tau_trace"])
        self.kappa_trace = float(p["kappa_trace"])
        self.excitability = float(p["excitability"])
        self.sig_slope = float(p["sig_slope"])
        self.a50 = float(p["a50"])
        self.a_thr = float(p["a_thr"])
        self.freq = float(p["freq"])
        self.p_dur = float(p["p_dur"])
        self.costim_enabled = costim_enabled
        self.costim_kappa = costim_kappa
        self.max_percept = max_percept

    def build(self, implant: Implant, topography: CorticalTopography) -> "DynaphosTorch":
        self.implant = implant
        self.topography = topography
        device = topography.grid_x.device
        dtype = topography.grid_x.dtype
        elec_xy = implant.electrode_coords()
        ex = elec_xy[:, 0].detach().cpu().to(torch.float64)
        ey = elec_xy[:, 1].detach().cpu().to(torch.float64)
        px, py = topography.polimeni.v1_to_dva(ex, ey)
        r = torch.hypot(px, py)
        M = topography.polimeni.magnification(r)
        self.ploc = torch.stack([px, py], dim=-1).to(device=device, dtype=dtype)
        self.M = M.to(device=device, dtype=dtype)
        self.elec_left = (elec_xy[:, 0] < topography.boundary).to(device)
        self.xRange = topography.grid_x[0, :].contiguous()
        self.yRange = topography.grid_y[:, 0].contiguous()
        # ponytail: O(E²) pair scan at build; fine for E ≤ 512 demo scale.
        n = elec_xy.shape[0]
        d2 = torch.cdist(elec_xy, elec_xy, p=2).pow(2)
        d2.fill_diagonal_(float("inf"))
        self._elec_d2 = d2
        self._built = True
        return self

    def _phosphene_geometry(self, pose):
        if pose is None or (pose.dx == 0.0 and pose.dy == 0.0 and pose.rot == 0.0):
            return self.ploc, self.M, self.elec_left
        topo = self.topography
        device = topo.grid_x.device
        dtype = topo.grid_x.dtype
        elec_xy = self.implant.electrode_coords(pose)
        ex = elec_xy[:, 0].detach().cpu().to(torch.float64)
        ey = elec_xy[:, 1].detach().cpu().to(torch.float64)
        px, py = topo.polimeni.v1_to_dva(ex, ey)
        r = torch.hypot(px, py)
        M = topo.polimeni.magnification(r)
        ploc = torch.stack([px, py], dim=-1).to(device=device, dtype=dtype)
        elec_left = (elec_xy[:, 0] < topo.boundary).to(device)
        return ploc, M.to(device=device, dtype=dtype), elec_left

    def initial_state(self, device: torch.device | None = None) -> State:
        topo = self.topography
        device = device or topo.grid_x.device
        n = self.implant.n_electrodes
        z = torch.zeros(n, device=device, dtype=topo.grid_x.dtype)
        image = torch.zeros(topo.grid_shape, device=device, dtype=topo.grid_x.dtype)
        return State(image=image, aux={"A": z.clone(), "Q": z.clone(), "sigma": z.clone()})

    def _apply_costim(self, amp: torch.Tensor) -> torch.Tensor:
        if not self.costim_enabled:
            return amp
        active = amp > 0
        if active.sum() < 2:
            return amp
        I = amp.clone()
        d2 = self._elec_d2.to(I.device)
        for i in torch.nonzero(active, as_tuple=False).flatten().tolist():
            for j in torch.nonzero(active, as_tuple=False).flatten().tolist():
                if i == j:
                    continue
                dist2 = max(float(d2[i, j]), 1e-12)
                I[i] = I[i] + self.costim_kappa * amp[j] / dist2
        return I

    def step(self, state: State | None, action: Action) -> State:
        topo = self.topography
        device = topo.grid_x.device
        action = action.to(device)
        if state is None:
            state = self.initial_state(device)
        A = state.aux["A"]
        Q = state.aux["Q"]
        sigma = state.aux["sigma"]

        amp = action.amp.to(device)
        amp = self._apply_costim(amp)
        n = amp.shape[0]
        if action.freq is None:
            freq = torch.full((n,), self.freq, device=device, dtype=amp.dtype)
        else:
            freq = action.freq.to(device)
        if action.phase_dur is None:
            p_dur = torch.full((n,), self.p_dur, device=device, dtype=amp.dtype)
        else:
            p_dur = action.phase_dur.to(device)

        ploc, M, elec_left = self._phosphene_geometry(action.pose)
        I0 = self.rheobase
        K = self.excitability
        Ieff = torch.clamp((amp - I0 - Q) * freq * (p_dur / 1000.0), min=0.0)
        Q = Q + ((-Q / (self.tau_trace / 1000.0)) + Ieff * self.kappa_trace) * (self.dt / 1000.0)
        D = 2.0 * torch.sqrt(torch.clamp(amp, min=0.0) / K)
        P = D / M
        sigma = torch.where(amp > 0, torch.clamp(P / 2.0, min=1e-22), sigma)
        A = A + ((-A / (self.tau_act / 1000.0)) + Ieff * 1e-6) * (self.dt / 1000.0)
        brightness = torch.sigmoid(self.sig_slope * (A - self.a50))

        image = self._render(A, sigma, brightness, ploc, elec_left)
        if self.max_percept is not None:
            image = torch.clamp(image, max=self.max_percept)
        return State(image=image, aux={"A": A, "Q": Q, "sigma": sigma})

    def _render(self, A, sigma, brightness, ploc, elec_left) -> torch.Tensor:
        topo = self.topography
        H, W = topo.grid_shape
        device = topo.grid_x.device
        dtype = topo.grid_x.dtype
        active = A >= self.a_thr
        if not active.any():
            return torch.zeros(H, W, device=device, dtype=dtype)

        idx = torch.nonzero(active, as_tuple=False).flatten()
        x0 = ploc[idx, 0]
        y0 = ploc[idx, 1]
        s = sigma[idx]
        b = brightness[idx]
        el = elec_left[idx]

        gx = torch.exp(-((self.xRange[None, :] - x0[:, None]) ** 2) / (2.0 * s[:, None] ** 2))
        if topo.polimeni.split_map:
            left_mask = self.xRange[None, :] <= 0
            cutoff = torch.where(el[:, None], left_mask, ~left_mask)
            gx = torch.where(cutoff, torch.zeros_like(gx), gx)
        gy = torch.exp(-((self.yRange[:, None] - y0[None, :]) ** 2) / (2.0 * s[None, :] ** 2))
        gauss = gy.T[:, :, None] * gx[:, None, :]
        bright = (gauss * b[:, None, None]).sum(dim=0)
        return torch.clamp(bright, 0.0, 1.0)

    def rasterize_aux(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize per-electrode A and Q to (H, W) maps for learned aux channels."""
        A = state.aux["A"]
        Q = state.aux["Q"]
        sigma = state.aux["sigma"]
        ploc = self.ploc
        elec_left = self.elec_left
        b_a = torch.sigmoid(self.sig_slope * (A - self.a50))
        b_q = torch.clamp(Q / (Q.max() + 1e-12), 0.0, 1.0) if Q.max() > 0 else Q
        a_map = self._render(A, sigma, b_a, ploc, elec_left)
        q_map = self._render(
            torch.where(A >= self.a_thr, A, torch.zeros_like(A)),
            sigma,
            b_q,
            ploc,
            elec_left,
        )
        return a_map, q_map

    def forward(self, action: Action) -> torch.Tensor:
        return self.step(None, action).image

    def predict_sequence(self, action: Action, n_steps: int) -> torch.Tensor:
        state = self.initial_state()
        frames = []
        for _ in range(n_steps):
            state = self.step(state, action)
            frames.append(state.image)
        return torch.stack(frames, dim=0)
