"""FadingTemporal step parity vs pulse2percept."""

from __future__ import annotations

import numpy as np
import torch

from sentionaut.models.fading import FadingTemporalTorch


def test_fading_step_vs_p2p():
    from pulse2percept.models import FadingTemporal
    from pulse2percept.stimuli import Stimulus

    tau, dt = 100.0, 20.0
    ft_p2p = FadingTemporal(tau=tau, dt=dt)
    ft_p2p.build()
    ft_torch = FadingTemporalTorch(tau_ms=tau)

    times = np.arange(0, 220, dt, dtype=np.float32)
    data = np.zeros((1, len(times)), dtype=np.float32)
    data[0, :3] = -1.0
    stim = Stimulus(data, electrodes=["e0"], time=times)
    ref = ft_p2p._predict_temporal(stim, times)[0]

    B = torch.zeros(1)
    trace = []
    for t in range(len(times)):
        A = torch.tensor(data[0, t])
        B = ft_torch.step(B, A, dt)
        trace.append(float(B.item()))
    assert np.max(np.abs(ref - np.array(trace))) < 1e-5


def test_axonmap_fade_persists():
    from sentionaut.core.base import Action
    from sentionaut.core.config import Config
    from sentionaut.core.registry import build_components

    cfg = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    implant, _, model = build_components(cfg, torch.device("cpu"))
    amp = torch.zeros(implant.n_electrodes)
    amp[5] = 2.0
    act = Action(
        amp=amp,
        freq=torch.full((implant.n_electrodes,), 30.0),
        phase_dur=torch.full((implant.n_electrodes,), 0.45),
    )
    state = model.step(None, act)
    peak = state.image.max().item()
    state = model.step(state, Action(amp=torch.zeros_like(amp)))
    assert state.image.max().item() > 0
    assert state.image.max().item() < peak
