"""Synthetic subject calibration recovery."""

from __future__ import annotations

import numpy as np

from sentionaut.calibrate import calibrate_subject
from sentionaut.core.config import Config


def test_calibrate_recovers_rho_axlambda():
    cfg = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    true_rho, true_axl = 220.0, 550.0
    targets = []
    for elec in ("C5", "D5", "E5"):
        from sentionaut.calibrate import _render_percept

        percept = _render_percept(cfg, true_rho, true_axl, [{"electrode": elec}])[0]
        targets.append({"electrode": elec, "target": percept.tolist()})
    cal = calibrate_subject(
        cfg,
        targets,
        rho_range=(180.0, 260.0),
        axlambda_range=(500.0, 600.0),
        n_grid=9,
    )
    assert abs(cal.rho - true_rho) / true_rho < 0.15
    assert abs(cal.axlambda - true_axl) / true_axl < 0.15
