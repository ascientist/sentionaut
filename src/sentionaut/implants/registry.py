"""Retinal + cortical electrode geometries loaded from pulse2percept as tensors."""

from __future__ import annotations

import numpy as np
import torch

from ..core.base import Implant, Pose

_RETINAL = {"argusii": "ArgusII", "alphaims": "AlphaIMS"}
_CORTICAL = {"orion": "Orion", "cortivis": "Cortivis", "icvp": "ICVP"}


def _load_p2p_implant(name: str):
    key = name.lower()
    if key in _RETINAL:
        import pulse2percept.implants as imp

        return getattr(imp, _RETINAL[key])()
    if key in _CORTICAL:
        import pulse2percept.implants.cortex as cimp

        return getattr(cimp, _CORTICAL[key])()
    known = sorted(set(_RETINAL) | set(_CORTICAL))
    raise ValueError(f"Unknown implant '{name}'. Known: {known}.")


class TensorImplant(Implant):
    """Implant whose electrode coordinates live as a cached ``(N, 2)`` tensor."""

    def __init__(self, name: str, names: list[str], coords: torch.Tensor):
        self.name = name
        self.names = names
        self._coords = coords  # (N, 2), microns, base (pose-free) coordinates

    @property
    def device(self) -> torch.device:
        return self._coords.device

    @property
    def n_electrodes(self) -> int:
        return self._coords.shape[0]

    def electrode_coords(self, pose: Pose | None = None) -> torch.Tensor:
        if pose is None or (pose.dx == 0.0 and pose.dy == 0.0 and pose.rot == 0.0):
            return self._coords
        c = float(np.cos(pose.rot))
        s = float(np.sin(pose.rot))
        rot = torch.tensor([[c, -s], [s, c]], dtype=self._coords.dtype, device=self._coords.device)
        out = self._coords @ rot.T
        offset = torch.tensor(
            [pose.dx, pose.dy], dtype=self._coords.dtype, device=self._coords.device
        )
        return out + offset

    def to(self, device: torch.device) -> "TensorImplant":
        return TensorImplant(self.name, self.names, self._coords.to(device))


def build_implant(name: str, device: torch.device) -> TensorImplant:
    p2p = _load_p2p_implant(name)
    names = list(p2p.electrodes.keys())
    coords = np.array([[p2p.electrodes[e].x, p2p.electrodes[e].y] for e in names], dtype=np.float32)
    return TensorImplant(name.lower(), names, torch.from_numpy(coords).to(device))
