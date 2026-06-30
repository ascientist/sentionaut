"""``DynaphosTorch``: cortical Dynaphos model (van der Grinten 2024 / p2p port).

Gaussian phosphenes in dva whose size derives from cortical magnification +
stimulation current, plus a temporal leaky integrator with charge accumulation.
This is the one model whose ``step`` truly threads temporal state across time
(activation ``A`` and memory trace ``Q`` per electrode).
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
    def __init__(self, params: dict | None = None):
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

    def build(self, implant: Implant, topography: CorticalTopography) -> "DynaphosTorch":
        self.implant = implant
        self.topography = topography
        device = topography.grid_x.device
        dtype = topography.grid_x.dtype
        elec_xy = implant.electrode_coords()  # (N, 2) cortex microns
        # Phosphene location in dva + eccentricity-dependent magnification.
        # The complex-log inverse map runs in float64 on CPU (MPS lacks float64).
        ex = elec_xy[:, 0].detach().cpu().to(torch.float64)
        ey = elec_xy[:, 1].detach().cpu().to(torch.float64)
        px, py = topography.polimeni.v1_to_dva(ex, ey)
        r = torch.hypot(px, py)
        M = topography.polimeni.magnification(r)  # mm/dva
        self.ploc = torch.stack([px, py], dim=-1).to(device=device, dtype=dtype)
        self.M = M.to(device=device, dtype=dtype)
        self.elec_left = (elec_xy[:, 0] < topography.boundary).to(device)
        self.xRange = topography.grid_x[0, :].contiguous()  # (W,)
        self.yRange = topography.grid_y[:, 0].contiguous()  # (H,)
        self._built = True
        return self

    def _phosphene_geometry(self, pose):
        """Return (ploc, M, elec_left) for a given pose (cached for identity)."""
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
        freq = self.freq if action.freq is None else float(action.freq.flatten()[0])
        p_dur = self.p_dur if action.phase_dur is None else float(action.phase_dur.flatten()[0])

        ploc, M, elec_left = self._phosphene_geometry(action.pose)
        I0 = self.rheobase
        K = self.excitability
        Ieff = torch.clamp((amp - I0 - Q) * freq * (p_dur / 1000.0), min=0.0)
        Q = Q + ((-Q / (self.tau_trace / 1000.0)) + Ieff * self.kappa_trace) * (self.dt / 1000.0)
        D = 2.0 * torch.sqrt(torch.clamp(amp, min=0.0) / K)  # mm
        P = D / M  # dva
        sigma = torch.where(amp > 0, torch.clamp(P / 2.0, min=1e-22), sigma)
        A = A + ((-A / (self.tau_act / 1000.0)) + Ieff * 1e-6) * (self.dt / 1000.0)
        brightness = torch.sigmoid(self.sig_slope * (A - self.a50))

        image = self._render(A, sigma, brightness, ploc, elec_left)
        return State(image=image, aux={"A": A, "Q": Q, "sigma": sigma})

    def _render(self, A, sigma, brightness, ploc, elec_left) -> torch.Tensor:
        topo = self.topography
        H, W = topo.grid_shape
        device = topo.grid_x.device
        dtype = topo.grid_x.dtype
        bright = torch.zeros(H, W, device=device, dtype=dtype)
        active = A >= self.a_thr
        x0 = ploc[:, 0]
        y0 = ploc[:, 1]
        for e in torch.nonzero(active, as_tuple=False).flatten().tolist():
            s = sigma[e]
            if s <= 0:
                continue
            gx = torch.exp(-((self.xRange - x0[e]) ** 2) / (2.0 * s**2))  # (W,)
            if topo.polimeni.split_map:
                cutoff = self.xRange <= 0 if bool(elec_left[e]) else self.xRange > 0
                gx = torch.where(cutoff, torch.zeros_like(gx), gx)
            gy = torch.exp(-((self.yRange - y0[e]) ** 2) / (2.0 * s**2))  # (H,)
            gauss = torch.outer(gy, gx)
            bright = bright + gauss * brightness[e]
        return torch.clamp(bright, 0.0, 1.0)

    def forward(self, action: Action) -> torch.Tensor:
        return self.step(None, action).image

    def predict_sequence(self, action: Action, n_steps: int) -> torch.Tensor:
        """Run ``n_steps`` of constant stimulation, returning ``(n_steps, H, W)``."""
        state = self.initial_state()
        frames = []
        for _ in range(n_steps):
            state = self.step(state, action)
            frames.append(state.image)
        return torch.stack(frames, dim=0)
