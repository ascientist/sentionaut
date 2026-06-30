"""Shape/dtype/device contracts of the world-model ``step`` (incl. MPS)."""

from __future__ import annotations

import pytest
import torch

from sentionaut.core.base import Action
from sentionaut.core.config import Config
from sentionaut.world import WorldModel

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def _action(n, device, cortical):
    amp = torch.zeros(n, device=device)
    amp[0] = 200.0 if cortical else 2.0
    if cortical:
        return Action(amp=amp)
    return Action(
        amp=amp,
        freq=torch.full((n,), 30.0, device=device),
        phase_dur=torch.full((n,), 0.45, device=device),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "model,implant,cortical",
    [("axonmap", "argusii", False), ("scoreboard", "orion", True), ("dynaphos", "orion", True)],
)
def test_step_contract(device, model, implant, cortical):
    cfg = Config(model=model, implant=implant, xrange=(-4, 4), yrange=(-4, 4), xystep=1.0)
    wm = WorldModel.from_config(cfg, torch.device(device))
    H, W = wm.grid_shape
    state = wm.initial_state(torch.device(device))
    action = _action(wm.model.implant.n_electrodes, torch.device(device), cortical)
    nxt = wm.step(state, action)
    assert nxt.image.shape == (H, W)
    assert nxt.image.dtype == torch.float32
    assert nxt.image.device.type == device
    assert torch.isfinite(nxt.image).all()


def test_dynaphos_threads_state():
    cfg = Config(model="dynaphos", implant="orion", xrange=(-4, 4), yrange=(-4, 4), xystep=1.0)
    wm = WorldModel.from_config(cfg, torch.device("cpu"))
    n = wm.model.implant.n_electrodes
    amp = torch.zeros(n)
    amp[0] = 250.0
    action = Action(amp=amp)
    s1 = wm.step(None, action)
    s2 = wm.step(s1, action)
    # Charge/activation accumulate, so the second step is at least as bright.
    assert float(s2.image.max()) >= float(s1.image.max()) - 1e-6
    assert "Q" in s2.aux and "A" in s2.aux
