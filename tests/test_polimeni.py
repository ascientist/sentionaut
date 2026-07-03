"""Polimeni2006Map round-trip vs pulse2percept."""

from __future__ import annotations

import numpy as np
import torch

from sentionaut.topography.cortical import SCOREBOARD_MAP, PolimeniTorch


def test_polimeni_dva_v1_roundtrip():
    from pulse2percept.topography import Polimeni2006Map

    p2p = Polimeni2006Map()
    ours = PolimeniTorch(**SCOREBOARD_MAP)

    xs = np.linspace(-3, 3, 7)
    ys = np.linspace(-3, 3, 7)
    max_err = 0.0
    for x in xs:
        for y in ys:
            cx, cy = p2p.dva_to_v1(np.array([x]), np.array([y]))
            rx, ry = p2p.v1_to_dva(cx, cy)
            tx, ty = ours.v1_to_dva(
                torch.tensor(cx, dtype=torch.float64),
                torch.tensor(cy, dtype=torch.float64),
            )
            ox, oy = ours.dva_to_region("v1", torch.tensor(x), torch.tensor(y))
            max_err = max(
                max_err,
                abs(float(rx[0]) - x),
                abs(float(ry[0]) - y),
                abs(float(tx[0]) - x),
                abs(float(ty[0]) - y),
                abs(float(ox) - float(cx[0])),
                abs(float(oy) - float(cy[0])),
            )
    assert max_err < 0.05
