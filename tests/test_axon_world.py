"""Shape, exact fade, and one Adam step for AxonMapWorld. No axon topography.

``test_report_round_trip`` is the end-to-end check: generate, distill, reload the
checkpoint without knowing its hyperparameters, and score the held-out episodes.
"""

from __future__ import annotations

import json

import torch

from sentionaut.core.base import Action, State
from sentionaut.core.config import Config
from sentionaut.generate import build_configs, generate_world_dataset
from sentionaut.implants.registry import TensorImplant
from sentionaut.learned.axon_report import (
    bench_forward,
    holdout_metrics,
    load_student,
    plot_samples,
    plot_validation,
)
from sentionaut.learned.axon_world import AxonMapWorld, distill_hdf5


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


def test_report_round_trip(tmp_path):
    base = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    dataset = tmp_path / "tiny.h5"
    generate_world_dataset(
        dataset,
        build_configs(["axonmap"], base),
        episodes=4,
        sequence_length=1,
        silent_tail=1,
        device=torch.device("cpu"),
        seed=0,
    )
    ckpt = tmp_path / "student.pt"
    timing = distill_hdf5(
        dataset, epochs=2, batch_size=2, rollout_k=2, device=torch.device("cpu"), ckpt_path=ckpt
    )
    assert len(timing["val_history"]) == 2

    student = load_student(ckpt)
    assert student.grid_shape == timing["model"].grid_shape

    metrics = holdout_metrics(student, dataset, rollout_k=2)
    assert metrics["n_val_windows"] > 0
    assert 0.0 <= metrics["mse"] < float("inf")
    assert len(metrics["mse_per_step"]) == 2

    assert plot_samples(student, dataset, tmp_path / "samples.png", n_episodes=1).stat().st_size > 0
    assert plot_validation(timing, metrics, tmp_path / "curves.png").stat().st_size > 0

    hardware = bench_forward(dataset, student, calls=1)
    assert hardware["teacher_spatial_forward_ms"] > 0
    assert hardware["student_spatial_drive_ms"]["1"] > 0
    json.dumps(hardware)
