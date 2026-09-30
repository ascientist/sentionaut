"""Shape, exact fade, and one Adam step for AxonMapWorld. No axon topography."""

from __future__ import annotations

import torch

from sentionaut.core.base import Action, State
from sentionaut.implants.registry import TensorImplant
from sentionaut.learned.axon_world import AxonMapWorld


def _model(h=8, w=8, n=3, **kwargs):
    model = AxonMapWorld((h, w), n, dim=32, depth=1, heads=4, patch_size=4, **kwargs)
    coords = torch.tensor([[0.0, 0.0], [200.0, -100.0], [-150.0, 80.0]])
    model.bind(TensorImplant("fake", ["a", "b", "c"], coords[:n]))
    return model


def _action(n=3, amp=1.0):
    return Action(
        amp=torch.full((n,), amp),
        freq=torch.full((n,), 40.0),
        phase_dur=torch.full((n,), 0.2),
        rho=220.0,
        axlambda=480.0,
    )


def test_step_shape_without_topography():
    model = _model()
    state = model.step(None, _action())
    assert state.image.shape == (8, 8)
    assert torch.isfinite(state.image).all()


def test_step_matches_exact_fade():
    model = _model()
    brightness = torch.rand(8, 8)
    drive = torch.rand(8, 8)
    model.spatial_drive = lambda *args, **kwargs: drive
    out = model.step(State(image=brightness), _action())
    expected = model.fading.step(brightness, -drive, model.dt_ms)
    assert torch.allclose(out.image, expected)


def test_silent_action_is_finite():
    model = _model()
    out = model.step(State(image=torch.ones(8, 8)), _action(amp=0.0))
    assert torch.isfinite(out.image).all()


def test_adam_step_on_synthetic_batch():
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    brightness = torch.rand(2, 8, 8)
    amp = torch.rand(2, 3)
    freq = torch.full((2, 3), 30.0)
    pdur = torch.full((2, 3), 0.2)
    rho = torch.tensor([200.0, 250.0])
    axl = torch.tensor([400.0, 600.0])
    opt.zero_grad()
    pred = model.predict_next(brightness, amp, freq, pdur, rho, axl)
    loss = torch.nn.functional.mse_loss(pred, torch.rand_like(pred))
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert pred.shape == brightness.shape
