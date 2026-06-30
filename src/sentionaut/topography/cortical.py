"""Cortical topography: torch port of pulse2percept's ``Polimeni2006Map``.

Implements the dva<->cortex complex-log mapping for regions v1/v2/v3, the
split-hemisphere handling, and the cortical-magnification factor used by the
cortical percept models. The (one-time) mapping is computed in float64 on CPU
for numerical fidelity, then cached as float32 tensors on the active device.
Parity with pulse2percept's own map is checked in the test suite.
"""

from __future__ import annotations

import math

import torch

from ..core.base import Topography
from ..core.config import Config

# Map parameter presets. Scoreboard uses the library defaults; Dynaphos overrides
# them with the values from the Dynaphos paper (see DEBRIEF.md).
SCOREBOARD_MAP = dict(k=15.0, a=0.69, b=80.0, alpha1=1.0, alpha2=0.333, alpha3=0.25)
DYNAPHOS_MAP = dict(k=17.3, a=0.75, b=120.0, alpha1=0.95, alpha2=0.333, alpha3=0.25)
LEFT_OFFSET = -20000.0


def _cart2pol(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.atan2(y, x)
    rho = torch.hypot(x, y)
    return theta, rho


def _pol2cart(theta: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return rho * torch.cos(theta), rho * torch.sin(theta)


class PolimeniTorch:
    """Differentiable torch implementation of the Polimeni2006 mapping."""

    def __init__(self, k, a, b, alpha1, alpha2, alpha3, left_offset=LEFT_OFFSET):
        self.k = float(k)
        self.a = float(a)
        self.b = float(b)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.alpha3 = float(alpha3)
        self.left_offset = float(left_offset)
        self.split_map = True

    def _invert_left_pol(self, theta, radius, inverted=None):
        if inverted is None:
            inverted = (theta > (math.pi / 2)) | (theta < -(math.pi / 2))
        theta = torch.where(inverted, math.pi - theta, theta)
        theta = torch.where(theta > math.pi, theta - 2 * math.pi, theta)
        theta = torch.where(theta <= -math.pi, theta + 2 * math.pi, theta)
        theta = -theta
        return theta, radius, inverted

    def _invert_left_cart(self, x, y, inverted=None, boundary=0.0):
        if inverted is None:
            inverted = x < boundary
            x = torch.where(inverted, -x + self.left_offset, x)
            return x, y, inverted
        x = torch.where(inverted, -x + self.left_offset, x)
        return x, y, inverted

    def _add_nans(self, x, y, theta, radius, allow_zero=True):
        idx = (theta <= -math.pi / 2) | (theta >= math.pi / 2) | (radius < 0) | (radius > 90)
        close0 = torch.isclose(theta, torch.zeros_like(theta), atol=1e-6)
        if not allow_zero:
            idx = idx | close0
        else:
            idx = idx | (close0 & (radius == 0))
        nan = torch.full_like(x, float("nan"))
        return torch.where(idx, nan, x), torch.where(idx, nan, y)

    def _jitter(self, x, axis_only_x=True):
        close = torch.isclose(x, torch.zeros_like(x), rtol=0, atol=1e-7)
        sign = math.copysign(1e-3, float(x.mean()))
        return torch.where(close, x + sign, x)

    def _wlog(self, z):
        ratio = (z + self.a) / (z + self.b)
        w = self.k * torch.log(ratio) - self.k * math.log(self.a / self.b)
        return w

    def dva_to_region(self, region, x, y):
        region = region.lower()
        x = x.clone()
        y = y.clone()
        if region in ("v1", "v2", "v3"):
            x = self._jitter(x)
        if region in ("v2", "v3"):
            y = self._jitter(y)
        theta, radius = _cart2pol(x, y)
        theta, radius, inverted = self._invert_left_pol(theta, radius)
        if region == "v1":
            thetaR = self.alpha1 * theta
            z = torch.polar(radius, thetaR)
        elif region == "v2":
            phi1 = math.pi / 2 * (1 - self.alpha1)
            phi2 = math.pi / 2 * (1 - self.alpha2)
            thetaR = self.alpha2 * theta + torch.sign(theta) * (phi2 + phi1)
            z = -torch.conj(torch.polar(radius, thetaR))
        elif region == "v3":
            phi1 = math.pi / 2 * (1 - self.alpha1)
            phi2 = math.pi / 2 * (1 - self.alpha2)
            thetaR = self.alpha3 * theta + torch.sign(theta) * (math.pi - phi1 - phi2)
            z = torch.polar(radius, thetaR)
        else:
            raise ValueError(region)
        w = self._wlog(z)
        xc, yc = torch.real(w), torch.imag(w)
        allow_zero = region == "v1"
        xc, yc = self._add_nans(xc, yc, theta, radius, allow_zero=allow_zero)
        xc = xc * 1000.0
        yc = yc * 1000.0
        xc, yc, _ = self._invert_left_cart(xc, yc, ~inverted)
        return xc, yc

    def v1_to_dva(self, x, y):
        x, y, inverted = self._invert_left_cart(x, y, boundary=self.left_offset / 2)
        x = x / 1000.0
        y = y / 1000.0
        w = torch.complex(x, y)
        ew = torch.exp(w / self.k)
        z = (self.a - self.a * ew) / (self.a / self.b * ew - 1)
        t1, t2 = torch.real(z), torch.imag(z)
        r = torch.sqrt(t1**2 + t2**2)
        thetav1 = torch.atan2(t2, t1)
        theta = thetav1 / self.alpha1
        theta, r, _ = self._invert_left_pol(theta, r, ~inverted)
        return _pol2cart(theta, r)

    def magnification(self, r: torch.Tensor) -> torch.Tensor:
        """Cortical magnification factor M (mm/dva) at eccentricity ``r`` (dva)."""
        return self.k * (self.b - self.a) / ((r + self.a) * (r + self.b))


def _build_dva_grid(config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    x0, x1 = config.xrange
    y0, y1 = config.yrange
    step = config.xystep
    nx = int(round(abs(x1 - x0) / step) + 1) if x1 != x0 else 1
    ny = int(round(abs(y1 - y0) / step) + 1) if y1 != y0 else 1
    xflat = torch.linspace(x0, x1, nx, dtype=torch.float64)
    yflat = torch.linspace(y0, y1, ny, dtype=torch.float64)
    gx, gy = torch.meshgrid(xflat, torch.flip(yflat, [0]), indexing="xy")
    return gx, gy


class CorticalTopography(Topography):
    def __init__(
        self,
        grid_shape: tuple[int, int],
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        cortex_xy: dict[str, torch.Tensor],
        polimeni: PolimeniTorch,
        regions: tuple[str, ...],
    ):
        self.grid_shape = grid_shape
        self.grid_x = grid_x  # (H, W) dva
        self.grid_y = grid_y
        self.cortex_xy = cortex_xy  # region -> (P, 2) cortex microns
        self.polimeni = polimeni
        self.regions = regions

    @property
    def boundary(self) -> float:
        return self.polimeni.left_offset / 2

    def to(self, device: torch.device) -> "CorticalTopography":
        return CorticalTopography(
            self.grid_shape,
            self.grid_x.to(device),
            self.grid_y.to(device),
            {k: v.to(device) for k, v in self.cortex_xy.items()},
            self.polimeni,
            self.regions,
        )

    @classmethod
    def build(cls, config: Config, device: torch.device) -> "CorticalTopography":
        params = DYNAPHOS_MAP if config.model == "dynaphos" else SCOREBOARD_MAP
        polimeni = PolimeniTorch(**params)
        gx, gy = _build_dva_grid(config)
        grid_shape = tuple(gx.shape)
        cortex_xy = {}
        for region in config.regions:
            xc, yc = polimeni.dva_to_region(region, gx.ravel(), gy.ravel())
            cortex_xy[region] = torch.stack([xc, yc], dim=-1).to(torch.float32)
        topo = cls(
            grid_shape=grid_shape,
            grid_x=gx.to(torch.float32),
            grid_y=gy.to(torch.float32),
            cortex_xy=cortex_xy,
            polimeni=polimeni,
            regions=tuple(config.regions),
        )
        return topo.to(device)
