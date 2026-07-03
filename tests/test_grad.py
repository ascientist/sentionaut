"""Gradient smoke through axon map spatial params."""

from __future__ import annotations

import torch

from sentionaut.core.base import Action
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components


def test_axonmap_rho_amp_grad():
    cfg = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    implant, _, model = build_components(cfg, torch.device("cpu"))
    amp = torch.zeros(implant.n_electrodes, requires_grad=True)
    amp.data[3] = 2.0
    rho = torch.tensor(200.0, requires_grad=True)
    act = Action(
        amp=amp,
        freq=torch.full((implant.n_electrodes,), 30.0),
        phase_dur=torch.full((implant.n_electrodes,), 0.45),
        rho=rho,
        axlambda=500.0,
    )
    B = model.spatial_forward(act)
    loss = B.sum()
    loss.backward()
    assert amp.grad is not None and amp.grad[3] != 0
    assert rho.grad is not None and torch.isfinite(rho.grad)
