"""AxonVideoWorld: causal in time, step() equals the batched forward, stream data round-trips."""

from __future__ import annotations

import h5py
import numpy as np
import torch

from sentionaut.core.base import Action
from sentionaut.core.config import Config
from sentionaut.generate import StreamSchedule, build_configs, generate_world_dataset
from sentionaut.implants.registry import TensorImplant
from sentionaut.learned.axon_video import (
    AxonVideoWorld,
    fade_constant,
    horizon_errors,
    load_video,
    train_video,
)


def _model(context=4):
    torch.manual_seed(0)
    model = AxonVideoWorld((8, 8), 3, dim=32, depth=2, heads=4, patch_size=4, context=context)
    coords = torch.tensor([[0.0, 0.0], [200.0, -100.0], [-150.0, 80.0]])
    model.bind(TensorImplant("fake", ["a", "b", "c"], coords))
    # Zero-initialised output would make every test pass trivially.
    torch.nn.init.normal_(model.patch_unembed.weight, std=0.1)
    return model.eval()


def _inputs(b=2, t=5, n=3):
    g = torch.Generator().manual_seed(1)
    return (
        torch.rand(b, t, 8, 8, generator=g),
        torch.rand(b, t, n, generator=g),
        torch.full((b, t, n), 40.0),
        torch.full((b, t, n), 0.2),
        torch.full((b, t), 220.0),
        torch.full((b, t), 480.0),
    )


def test_future_frames_do_not_leak():
    model = _model(context=8)
    frames, *acts = _inputs()
    base = model(frames, *acts)
    frames2 = frames.clone()
    frames2[:, 3:] = torch.rand_like(frames2[:, 3:])
    acts2 = [a.clone() for a in acts]
    acts2[0][:, 3:] = 0.0
    other = model(frames2, *acts2)
    assert torch.allclose(base[:, :3], other[:, :3], atol=1e-5)
    assert not torch.allclose(base[:, 3:], other[:, 3:])


def test_step_matches_windowed_forward():
    model = _model(context=3)
    frames, amp, freq, pdur, rho, axl = _inputs(b=1, t=5)
    state = None
    image = torch.zeros(8, 8)
    history = []
    with torch.no_grad():
        for t in range(5):
            action = Action(
                amp=amp[0, t], freq=freq[0, t], phase_dur=pdur[0, t], rho=220.0, axlambda=480.0
            )
            history.append(image)
            state = model.step(state, action)
            lo = max(0, t + 1 - model.context)
            ctx = torch.stack(history[lo:], dim=0)[None]
            ref = model(
                ctx,
                amp[:, lo : t + 1],
                freq[:, lo : t + 1],
                pdur[:, lo : t + 1],
                rho[:, lo : t + 1],
                axl[:, lo : t + 1],
            )[0, -1]
            assert torch.allclose(state.image, ref, atol=1e-5)
            assert state.aux["frames"].shape[0] == min(t + 1, model.context)
            image = state.image


def test_all_silent_frames_are_finite():
    model = _model()
    frames, amp, *rest = _inputs()
    out = model(frames, torch.zeros_like(amp), *rest)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_stream_round_trip(tmp_path):
    base = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    dataset = tmp_path / "stream.h5"
    generate_world_dataset(
        dataset,
        build_configs(["axonmap"], base),
        episodes=5,
        sequence_length=8,
        silent_tail=3,
        device=torch.device("cpu"),
        seed=0,
        stream=StreamSchedule(hold_prob=0.6, silent_prob=0.3),
    )
    with h5py.File(dataset, "r") as h5:
        g = h5["world"]
        ep, amp, rho = g["episode_id"][:], g["amp"][:], g["rho"][:]
    for e in np.unique(ep):
        assert np.unique(rho[ep == e]).size == 1
    held = [
        np.array_equal(amp[i], amp[i - 1]) and amp[i].any()
        for i in range(1, len(amp))
        if ep[i] == ep[i - 1]
    ]
    assert any(held)

    ckpt = tmp_path / "video.pt"
    timing = train_video(
        dataset,
        context=4,
        dim=32,
        depth=1,
        epochs=2,
        batch_size=4,
        device=torch.device("cpu"),
        ckpt_path=ckpt,
    )
    assert len(timing["val_history"]) == 2 and np.isfinite(timing["val_history"]).all()

    model = load_video(ckpt)
    assert model.context == 4
    horizon = horizon_errors(model, dataset, torch.device("cpu"))
    assert len(horizon["mse_by_frame"]) == 11
    fade = fade_constant(model, dataset, torch.device("cpu"), min_signal=0.0)
    assert fade["n_silent_steps"] > 0
    assert abs(fade["teacher_tau_ms_median"] - 100.0) < 1.0
