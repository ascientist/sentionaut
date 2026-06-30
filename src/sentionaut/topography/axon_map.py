"""Retinal axon-map topography: raw per-pixel axon coords + d_soma as tensors.

The heavy Jansonius axon growth is done once on CPU by pulse2percept's
``AxonMapSpatial`` (reusing its ``axons.pickle``). We then extract the RAW
per-pixel axon-point coordinates and recover ``d_soma`` from the cached
sensitivity (rather than keeping the axlambda-baked sensitivity), so that ``rho``
and ``axlambda`` remain differentiable runtime inputs to the torch model.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

from ..core.base import Topography
from ..core.config import Config

_CACHE_DIR = Path("data/axon_cache")


def _cache_key(config: Config) -> str:
    payload = (
        config.xrange,
        config.yrange,
        config.xystep,
        config.axlambda,
        config.eye,
    )
    return hashlib.md5(repr(payload).encode()).hexdigest()[:16]


class AxonMapTopography(Topography):
    def __init__(
        self,
        grid_shape: tuple[int, int],
        coords: torch.Tensor,
        d_soma: torch.Tensor,
        mask: torch.Tensor,
        axlambda_build: float,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
    ):
        self.grid_shape = grid_shape
        # coords: (P, L, 2) microns (retinal); d_soma/mask: (P, L)
        self.coords = coords
        self.d_soma = d_soma
        self.mask = mask
        self.axlambda_build = axlambda_build
        # dva grid coordinates for plotting / tissue overlays
        self.grid_x = grid_x
        self.grid_y = grid_y

    def to(self, device: torch.device) -> "AxonMapTopography":
        return AxonMapTopography(
            self.grid_shape,
            self.coords.to(device),
            self.d_soma.to(device),
            self.mask.to(device),
            self.axlambda_build,
            self.grid_x.to(device),
            self.grid_y.to(device),
        )

    @classmethod
    def build(cls, config: Config, device: torch.device) -> "AxonMapTopography":
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _CACHE_DIR / f"axonmap_{_cache_key(config)}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                blob = pickle.load(fh)
        else:
            blob = cls._extract(config)
            with open(cache_path, "wb") as fh:
                pickle.dump(blob, fh)
        topo = cls(
            grid_shape=blob["grid_shape"],
            coords=torch.from_numpy(blob["coords"]),
            d_soma=torch.from_numpy(blob["d_soma"]),
            mask=torch.from_numpy(blob["mask"]),
            axlambda_build=blob["axlambda_build"],
            grid_x=torch.from_numpy(blob["grid_x"]),
            grid_y=torch.from_numpy(blob["grid_y"]),
        )
        return topo.to(device)

    @staticmethod
    def _extract(config: Config) -> dict:
        from pulse2percept.models import BiphasicAxonMapModel

        model = BiphasicAxonMapModel(
            xrange=config.xrange,
            yrange=config.yrange,
            xystep=config.xystep,
            axlambda=config.axlambda,
            eye=config.eye,
        )
        model.build()
        spatial = model.spatial
        contrib = np.asarray(spatial.axon_contrib, dtype=np.float32)  # (total_seg, 3)
        idx_start = np.asarray(spatial.axon_idx_start, dtype=np.int64)
        idx_end = np.asarray(spatial.axon_idx_end, dtype=np.int64)
        grid_shape = tuple(spatial.grid.x.shape)
        n_pixels = idx_start.shape[0]
        lengths = idx_end - idx_start
        max_len = int(lengths.max()) if n_pixels else 0

        coords = np.zeros((n_pixels, max_len, 2), dtype=np.float32)
        d_soma = np.zeros((n_pixels, max_len), dtype=np.float32)
        mask = np.zeros((n_pixels, max_len), dtype=np.float32)

        axlambda = float(config.axlambda)
        # sensitivity = exp(-d_soma^2 / (2 axlambda^2)) -> recover d_soma.
        for p in range(n_pixels):
            s, e = int(idx_start[p]), int(idx_end[p])
            seg = contrib[s:e]
            n = seg.shape[0]
            if n == 0:
                continue
            coords[p, :n, :] = seg[:, :2]
            sens = np.clip(seg[:, 2], 1e-12, 1.0)
            d_soma[p, :n] = np.sqrt(np.maximum(-2.0 * axlambda**2 * np.log(sens), 0.0))
            mask[p, :n] = 1.0

        return {
            "grid_shape": grid_shape,
            "coords": coords,
            "d_soma": d_soma,
            "mask": mask,
            "axlambda_build": axlambda,
            "grid_x": np.asarray(spatial.grid.x, dtype=np.float32),
            "grid_y": np.asarray(spatial.grid.y, dtype=np.float32),
        }
