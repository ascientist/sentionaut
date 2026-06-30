"""Optional NeuropythyMap-backed cortical topography (lazy, with Polimeni fallback).

pulse2percept's ``NeuropythyMap`` builds a subject-specific dva<->cortex mapping
from MRI-derived surfaces, which requires the optional ``neuropythy`` package
*and* downloadable FreeSurfer/template data. Neither is part of the default
install, so this wrapper is fully lazy: it attempts to build a NeuropythyMap and,
on any failure (missing package or data), transparently falls back to the
analytical :class:`~sentionaut.topography.cortical.CorticalTopography`
(Polimeni2006Map). The returned object always satisfies the ``Topography``
interface and works with the cortical Scoreboard/Dynaphos models.

See DEBRIEF.md for the rationale behind keeping this optional/lazy.
"""

from __future__ import annotations

import warnings

import torch

from ..core.config import Config
from .cortical import CorticalTopography


def neuropythy_available() -> bool:
    try:
        import neuropythy  # noqa: F401
    except Exception:
        return False
    return True


class NeuropythyTopography:
    """Factory that returns a NeuropythyMap-backed cortical topography or a fallback.

    The product is a :class:`CorticalTopography` whose per-region ``cortex_xy``
    grid coordinates come from a NeuropythyMap when one can be built; the
    PolimeniTorch helper is retained for the magnification / inverse map that the
    Dynaphos model needs (documented approximation in the neuropythy path).
    """

    @staticmethod
    def build(config: Config, device: torch.device) -> CorticalTopography:
        base = CorticalTopography.build(config, device)
        base.source = "polimeni"  # type: ignore[attr-defined]
        if not neuropythy_available():
            warnings.warn(
                "neuropythy is not installed; falling back to Polimeni2006Map "
                "for the cortical topography. Install the 'neuropythy' extra and "
                "provide subject data to enable the MRI-derived map.",
                stacklevel=2,
            )
            return base
        try:
            from pulse2percept.topography import NeuropythyMap

            vfmap = NeuropythyMap(regions=list(config.regions))
            cortex_xy = {}
            from .cortical import _build_dva_grid

            gx, gy = _build_dva_grid(config)
            xr = gx.ravel().to(torch.float64).numpy()
            yr = gy.ravel().to(torch.float64).numpy()
            for region in config.regions:
                mapped = vfmap.from_dva()[region](xr, yr)
                xc, yc = mapped[0], mapped[1]
                cortex_xy[region] = torch.tensor([xc, yc], dtype=torch.float32).T.to(device)
            base.cortex_xy = cortex_xy
            base.source = "neuropythy"  # type: ignore[attr-defined]
        except Exception as exc:  # missing data / API drift -> analytical fallback
            warnings.warn(
                f"NeuropythyMap unavailable ({type(exc).__name__}: {exc}); "
                "falling back to Polimeni2006Map.",
                stacklevel=2,
            )
        return base
