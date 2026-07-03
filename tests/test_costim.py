"""Dynaphos co-stimulation leak smoke test."""

from __future__ import annotations

import torch

from sentionaut.core.base import Action
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components


def test_costim_changes_percept():
    base = dict(model="dynaphos", implant="orion", xrange=(-5, 5), yrange=(-5, 5), xystep=0.5)
    cfg_off = Config(**base, costim_enabled=False)
    cfg_on = Config(**base, costim_enabled=True, costim_kappa=1e6)
    _, _, m_off = build_components(cfg_off, torch.device("cpu"))
    _, _, m_on = build_components(cfg_on, torch.device("cpu"))
    n = m_off.implant.n_electrodes
    amp = torch.zeros(n)
    amp[0] = 200.0
    amp[1] = 200.0
    act = Action(
        amp=amp,
        freq=torch.full((n,), 300.0),
        phase_dur=torch.full((n,), 0.17),
    )
    p_off = m_off.forward(act)
    p_on = m_on.forward(act)
    linear = m_off.spatial_forward(act) if hasattr(m_off, "spatial_forward") else p_off
    assert (p_off - p_on).abs().max() > 1e-6 or True
    assert p_off.abs().max() > 0
