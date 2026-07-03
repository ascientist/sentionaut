"""Numerical parity vs pulse2percept 0.9.0 on small subsampled configs."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from sentionaut.core.base import Action, Pose
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components

CPU = torch.device("cpu")


def test_axonmap_parity():
    xr, yr, step, rho, axl = (-8, 8), (-8, 8), 0.5, 200.0, 500.0
    from pulse2percept.implants import ArgusII
    from pulse2percept.models import BiphasicAxonMapModel
    from pulse2percept.stimuli import BiphasicPulseTrain

    m = BiphasicAxonMapModel(xrange=xr, yrange=yr, xystep=step, rho=rho, axlambda=axl)
    m.build()
    imp = ArgusII()
    sel = {"C5": (2.0, 20.0, 0.45), "E7": (1.5, 30.0, 0.45)}
    imp.stim = {
        e: BiphasicPulseTrain(
            freq=f,
            amp=a,
            phase_dur=pd,
            interphase_dur=0,
            delay_dur=0,
            stim_dur=200.0,
            cathodic_first=True,
        )
        for e, (a, f, pd) in sel.items()
    }
    ref = np.asarray(m.predict_percept(imp).data)[..., 0]

    cfg = Config(
        model="axonmap", implant="argusii", xrange=xr, yrange=yr, xystep=step, rho=rho, axlambda=axl
    )
    implant, _, model = build_components(cfg, CPU)
    amp = torch.zeros(implant.n_electrodes)
    freq = torch.zeros(implant.n_electrodes)
    pdur = torch.zeros(implant.n_electrodes)
    for e, (a, f, pd) in sel.items():
        i = implant.names.index(e)
        amp[i], freq[i], pdur[i] = a, f, pd
    out = model.forward(Action(amp=amp, freq=freq, phase_dur=pdur)).numpy()

    assert np.abs(ref - out).max() < 1e-3


def test_scoreboard_parity():
    xr, yr, step, rho = (-5, 5), (-5, 5), 0.5, 1000.0
    from pulse2percept.implants.cortex import Orion
    from pulse2percept.models.cortex import ScoreboardModel

    sb = ScoreboardModel(xrange=xr, yrange=yr, xystep=step, rho=rho, regions=["v1"])
    sb.build()
    imp = Orion()
    names = list(imp.electrodes.keys())
    imp.stim = {names[0]: 100.0, names[10]: 80.0}
    ref = np.asarray(sb.predict_percept(imp).data)[..., 0]

    cfg = Config(
        model="scoreboard",
        implant="orion",
        xrange=xr,
        yrange=yr,
        xystep=step,
        rho=rho,
        regions=("v1",),
    )
    implant, _, model = build_components(cfg, CPU)
    amp = torch.zeros(implant.n_electrodes)
    amp[0], amp[10] = 100.0, 80.0
    out = model.forward(Action(amp=amp)).numpy()

    # Scoreboard percept peaks ~30, so absolute tolerance is scaled up accordingly.
    assert np.abs(ref - out).max() < 1e-2


def test_dynaphos_parity():
    xr, yr, step = (-5, 5), (-5, 5), 0.25
    from pulse2percept.implants.cortex import Orion
    from pulse2percept.models.cortex import DynaphosModel
    from pulse2percept.stimuli import BiphasicPulseTrain

    dm = DynaphosModel(xrange=xr, yrange=yr, xystep=step, regions=["v1"])
    dm.build()
    imp = Orion()
    names = list(imp.electrodes.keys())
    freq, pdur, amp_val = 300.0, 0.170, 200.0
    imp.stim = {
        names[0]: BiphasicPulseTrain(
            freq=freq,
            amp=amp_val,
            phase_dur=pdur,
            interphase_dur=0,
            delay_dur=0,
            stim_dur=200.0,
            cathodic_first=True,
        )
    }
    p = dm.predict_percept(imp)
    ref = np.asarray(p.data)  # (H, W, T); frame 0 is the zero baseline.
    n_frames = ref.shape[2]

    cfg = Config(
        model="dynaphos", implant="orion", xrange=xr, yrange=yr, xystep=step, regions=("v1",)
    )
    implant, _, model = build_components(cfg, CPU)
    amp = torch.zeros(implant.n_electrodes)
    amp[0] = amp_val
    act = Action(
        amp=amp,
        freq=torch.full((implant.n_electrodes,), freq),
        phase_dur=torch.full((implant.n_electrodes,), pdur),
    )
    seq = model.predict_sequence(act, n_frames - 1).numpy()  # (T-1, H, W)
    ref_frames = ref[..., 1:].transpose(2, 0, 1)  # align (H, W, T) -> (T, H, W)

    assert np.abs(ref_frames - seq).max() < 1e-3


def test_scoreboard_temporal_fade():
    cfg = Config(
        model="scoreboard", implant="orion", xrange=(-5, 5), yrange=(-5, 5),
        xystep=0.5, rho=1000.0, regions=("v1",),
    )
    implant, _, model = build_components(cfg, CPU)
    amp = torch.zeros(implant.n_electrodes)
    amp[0] = 100.0
    act = Action(amp=amp)
    state = model.step(None, act)
    peak = state.image.max().item()
    state = model.step(state, Action(amp=torch.zeros_like(amp)))
    assert state.image.max().item() > 0
    assert state.image.max().item() < peak


@pytest.mark.slow
def test_mps_parity_smoke():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cfg = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=1.0)
    _, _, cpu_m = build_components(cfg, CPU)
    _, _, mps_m = build_components(cfg, torch.device("mps"))
    amp = torch.zeros(cpu_m.implant.n_electrodes)
    amp[5] = 2.0
    act_cpu = Action(
        amp=amp,
        freq=torch.full_like(amp, 30.0),
        phase_dur=torch.full_like(amp, 0.45),
    )
    act_mps = Action(
        amp=amp.to("mps"),
        freq=torch.full_like(amp, 30.0).to("mps"),
        phase_dur=torch.full_like(amp, 0.45).to("mps"),
    )
    c = cpu_m.forward(act_cpu).numpy()
    g = mps_m.forward(act_mps).detach().cpu().numpy()
    assert np.abs(c - g).max() < 1e-3


@pytest.mark.slow
def test_cpu_gpu_speed_benchmark():
    cfg = Config(model="axonmap", implant="argusii", xrange=(-20, 20), yrange=(-20, 20), xystep=0.2)
    results = {}
    for dev in ("cpu", "mps"):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        implant, _, model = build_components(cfg, torch.device(dev))
        amp = torch.zeros(implant.n_electrodes, device=dev)
        amp[20] = 2.0
        act = Action(
            amp=amp,
            freq=torch.full((implant.n_electrodes,), 30.0, device=dev),
            phase_dur=torch.full((implant.n_electrodes,), 0.45, device=dev),
            pose=Pose(),
        )
        model.forward(act)  # warmup
        t0 = time.perf_counter()
        for _ in range(20):
            model.forward(act)
        results[dev] = time.perf_counter() - t0
    assert results
