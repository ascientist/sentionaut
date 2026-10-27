"""Batched teacher steps equal a loop of the single-sample paths."""

from __future__ import annotations

import torch

from sentionaut.core.base import Action, State
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components


def test_axonmap_batch_matches_loop():
    cfg = Config(model="axonmap", implant="argusii", xrange=(-6, 6), yrange=(-6, 6), xystep=1.0)
    implant, _, model = build_components(cfg, device=torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    amp = torch.rand(4, implant.n_electrodes, generator=g) * 3.0
    amp[amp < 1.5] = 0.0  # mixed on/off electrodes
    amp[3] = 0.0  # one all-off sample
    batch = model.spatial_forward_batch(amp, freq=20.0, phase_dur=0.45, chunk=7)
    for b in range(4):
        n = implant.n_electrodes
        ref = model.spatial_forward(
            Action(amp=amp[b], freq=torch.full((n,), 20.0), phase_dur=torch.full((n,), 0.45))
        )
        assert torch.allclose(batch[b], ref, atol=1e-5), b


def test_axonmap_batch_is_differentiable():
    cfg = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    implant, _, model = build_components(cfg, device=torch.device("cpu"))
    amp = torch.full((2, implant.n_electrodes), 2.0, requires_grad=True)
    model.step_batch(None, amp).sum().backward()
    assert amp.grad is not None and amp.grad.abs().sum() > 0


def test_dynaphos_batch_matches_loop():
    cfg = Config(model="dynaphos", implant="orion", xrange=(-6, 6), yrange=(-6, 6), xystep=0.5)
    implant, _, model = build_components(cfg, device=torch.device("cpu"))
    g = torch.Generator().manual_seed(1)
    n = implant.n_electrodes
    amps = torch.rand(5, 2, n, generator=g) * 200.0
    amps[amps < 100.0] = 0.0
    state, imgs = None, []
    for t in range(5):
        state, img = model.step_batch(state, amps[t])
        imgs.append(img)
    for b in range(2):
        s = None
        for t in range(5):
            s = model.step(s, Action(amp=amps[t, b]))
            assert torch.allclose(imgs[t][b], s.image, atol=1e-5), (b, t)
            assert torch.allclose(state["A"][b], s.aux["A"]) if t == 4 else True
    assert isinstance(s, State)
