"""Neuralink cortical path: parity (scoreboard) + smoke (dynaphos), subsampled."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sentionaut.core.base import Action
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def _p2p_neuralink():
    from pulse2percept.implants.cortex import LinearEdgeThread, Neuralink
    from pulse2percept.topography import Polimeni2006Map

    vfmap = Polimeni2006Map(regions=["v1"])
    return Neuralink.from_cortical_map(
        LinearEdgeThread, vfmap, xrange=(-3, 3), yrange=(-3, 3), xystep=2.0, region="v1"
    )


def test_neuralink_scoreboard_parity():
    xr, yr, step, rho = (-5, 5), (-5, 5), 0.5, 1500.0
    from pulse2percept.models.cortex import ScoreboardModel

    sb = ScoreboardModel(xrange=xr, yrange=yr, xystep=step, rho=rho, regions=["v1"])
    sb.build()
    imp = _p2p_neuralink()
    names = list(imp.electrodes.keys())
    sel = {names[0]: 200.0, names[100]: 150.0, names[300]: 180.0}
    imp.stim = sel
    ref = np.asarray(sb.predict_percept(imp).data)[..., 0]

    cfg = Config(
        model="scoreboard",
        implant="neuralink",
        xrange=xr,
        yrange=yr,
        xystep=step,
        rho=rho,
        regions=("v1",),
    )
    implant, _, model = build_components(cfg, torch.device("cpu"))
    amp = torch.zeros(implant.n_electrodes)
    for name, val in sel.items():
        amp[implant.names.index(name)] = val
    out = model.forward(Action(amp=amp)).numpy()

    assert ref.shape == out.shape
    assert np.abs(ref - out).max() < 1e-2


@pytest.mark.parametrize("device", DEVICES)
def test_neuralink_dynaphos_smoke(device):
    cfg = Config(
        model="dynaphos",
        implant="neuralink",
        xrange=(-4, 4),
        yrange=(-4, 4),
        xystep=1.0,
        regions=("v1",),
    )
    implant, _, model = build_components(cfg, torch.device(device))
    amp = torch.zeros(implant.n_electrodes, device=device)
    amp[0] = 300.0
    seq = model.predict_sequence(Action(amp=amp), 4)
    assert seq.shape[0] == 4
    assert seq.shape[1:] == model.topography.grid_shape
    assert seq.device.type == device
    assert torch.isfinite(seq).all()
