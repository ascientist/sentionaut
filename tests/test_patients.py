"""Shape descriptors used to fit the axon map to phosphene drawings."""

from __future__ import annotations

import math

import torch

from sentionaut.patients import moments


def _ellipse(theta: float, a: float = 6.0, b: float = 2.0):
    ys = torch.linspace(10, -10, 81)
    xs = torch.linspace(-10, 10, 81)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    c, s = math.cos(theta), math.sin(theta)
    u, v = c * xx + s * yy, -s * xx + c * yy
    img = ((u / a) ** 2 + (v / b) ** 2 <= 1).float()[None]
    return img, xx, yy


def test_moments_area_orientation_elongation():
    for theta in (0.0, 0.5, -1.0):
        img, xx, yy = _ellipse(theta)
        m = moments(img, xx, yy)
        assert abs(float(m["area"]) - math.pi * 6 * 2) / (math.pi * 12) < 0.05
        assert abs(math.sin(2 * (float(m["theta"]) - theta))) < 0.05
        assert abs(float(m["elong"]) - (1 - 2 / 6)) < 0.05
        assert abs(float(m["cx"])) < 0.1 and abs(float(m["cy"])) < 0.1
