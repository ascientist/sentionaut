"""Retinal + cortical electrode geometries loaded from pulse2percept as tensors.

Retinal: ``argusii`` (primary epiretinal), ``alphaims``/``alphaams`` (epiretinal),
``prima`` (subretinal photovoltaic), and a configurable dense ``grid``
(``ElectrodeGrid``). Cortical: ``orion``/``cortivis``/``icvp`` plus ``neuralink``
(an ``EnsembleImplant`` of ``LinearEdgeThread``s placed via the cortical map).
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.base import Implant, Pose
from ..core.config import Config

_RETINAL_SIMPLE = {
    "argusii": "ArgusII",
    "alphaims": "AlphaIMS",
    "alphaams": "AlphaAMS",
    "prima": "PRIMA",
}
_CORTICAL_SIMPLE = {"orion": "Orion", "cortivis": "Cortivis", "icvp": "ICVP"}


def _coords_from_p2p(p2p_obj) -> tuple[list[str], np.ndarray]:
    names = list(p2p_obj.electrodes.keys())
    coords = np.array(
        [[p2p_obj.electrodes[e].x, p2p_obj.electrodes[e].y] for e in names], dtype=np.float32
    )
    return names, coords


def _build_p2p_implant(config: Config):
    name = config.implant.lower()
    if name in _RETINAL_SIMPLE:
        import pulse2percept.implants as imp

        return getattr(imp, _RETINAL_SIMPLE[name])()
    if name in _CORTICAL_SIMPLE:
        import pulse2percept.implants.cortex as cimp

        return getattr(cimp, _CORTICAL_SIMPLE[name])()
    if name == "grid":
        from pulse2percept.implants import ElectrodeGrid

        return ElectrodeGrid(
            tuple(config.implant_grid_shape), config.implant_grid_spacing, type="rect"
        )
    if name == "neuralink":
        from pulse2percept.implants.cortex import LinearEdgeThread, Neuralink
        from pulse2percept.topography import Polimeni2006Map

        vfmap = Polimeni2006Map(regions=["v1"])
        return Neuralink.from_cortical_map(
            LinearEdgeThread,
            vfmap,
            xrange=tuple(config.neuralink_xrange),
            yrange=tuple(config.neuralink_yrange),
            xystep=config.neuralink_xystep,
            region="v1",
        )
    known = sorted(set(_RETINAL_SIMPLE) | set(_CORTICAL_SIMPLE) | {"grid", "neuralink"})
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


def build_implant(config: Config, device: torch.device) -> TensorImplant:
    p2p = _build_p2p_implant(config)
    names, coords = _coords_from_p2p(p2p)
    return TensorImplant(config.implant.lower(), names, torch.from_numpy(coords).to(device))
