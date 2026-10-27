"""Smoke tests for training an encoder through a teacher or a learned world model."""

from __future__ import annotations

import json

import pytest
import torch

from sentionaut.learned.imagine import (
    Replay,
    RunConfig,
    TaskSpec,
    Teacher,
    TokenWorldModel,
    fit_world_model,
    run,
    teacher_rollout,
    world_model_error,
)


@pytest.fixture(scope="module")
def setup():
    teacher = Teacher(TaskSpec.preset("dynaphos"))
    g = torch.Generator().manual_seed(0)
    targets = {
        "train": torch.rand(32, *teacher.grid, generator=g),
        "test": torch.rand(8, *teacher.grid, generator=g),
    }
    return teacher, targets


def test_gate_turns_small_amplitudes_off(setup):
    teacher, _ = setup
    amp = torch.tensor([[0.0, 10.0, 40.0, 500.0]])
    out = teacher.execute(amp)
    assert out.tolist() == [[0.0, 0.0, 40.0, 300.0]]


def test_world_model_fit_reduces_loss(setup):
    teacher, targets = setup
    from sentionaut.learned.imagine import Encoder

    enc = Encoder(teacher.grid, teacher.n_elec, teacher.spec.a_max)
    buf = Replay(256)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        imgs, phys, acts = teacher_rollout(teacher, enc, targets["train"], noise=0.1, g=g)
    buf.add(imgs, phys, acts.clamp(0, teacher.spec.a_max))
    wm = TokenWorldModel(teacher.grid, teacher.electrode_xy(), teacher.spec.a_max, teacher.n_phys)
    windows = buf.windows(32, 4, torch.Generator().manual_seed(2))

    @torch.no_grad()
    def held_loss():
        imgs, phys0, acts = windows
        img, mem = imgs[:, 0], wm.init_memory(phys0)
        loss = 0.0
        for j in range(4):
            img, mem = wm(img, mem, acts[:, j])
            loss += float(((img - imgs[:, j + 1]) ** 2).mean())
        return loss / 4

    before = held_loss()
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    fit_world_model(wm, opt, buf, 60, 16, 4, g)
    assert held_loss() < 0.8 * before


@pytest.mark.parametrize("arm", ["teacher-bptt", "wm-bptt", "reinforce-sim", "reinforce-wm"])
def test_arms_run_and_log(setup, tmp_path, arm):
    teacher, targets = setup
    rc = RunConfig(
        arm=arm,
        preset="dynaphos",
        budget_s=1.0,
        batch=8,
        eval_every_s=0.5,
        eval_n=4,
        warmup_episodes=16,
        collect=8,
        wm_steps=2,
        wm_warmup_steps=2,
        policy_steps=2,
        threads=1,
        out=str(tmp_path),
    )
    r = run(rc, teacher=teacher, targets=targets)
    log = r["log"]
    assert log[0]["teacher_steps"] == 0 and log[-1]["final"]
    assert log[-1]["teacher_steps"] > 0
    saved = json.loads((tmp_path / f"dynaphos_{arm}_s0.json").read_text())
    assert saved["config"]["arm"] == arm


def test_untrained_world_model_error_is_one(setup):
    teacher, targets = setup
    from sentionaut.learned.imagine import Encoder

    enc = Encoder(teacher.grid, teacher.n_elec, teacher.spec.a_max)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        imgs, phys, acts = teacher_rollout(teacher, enc, targets["train"][:8], explore=0.5, g=g)
    wm = TokenWorldModel(teacher.grid, teacher.electrode_xy(), teacher.spec.a_max, teacher.n_phys)
    # The output layer starts at zero, so the model predicts the blank start throughout.
    err = world_model_error(wm, imgs, phys, acts.clamp(0, teacher.spec.a_max))
    assert err == pytest.approx(1.0)


@pytest.mark.parametrize("max_error", [0.0, float("inf")])
def test_imagination_waits_for_an_accurate_model(setup, tmp_path, max_error):
    teacher, targets = setup
    rc = RunConfig(
        preset="dynaphos",
        budget_s=1.0,
        batch=8,
        eval_n=4,
        warmup_episodes=16,
        collect=8,
        wm_steps=2,
        wm_warmup_steps=2,
        wm_max_error=max_error,
        policy_steps=2,
        threads=1,
        out=str(tmp_path),
    )
    last = run(rc, teacher=teacher, targets=targets)["log"][-1]
    assert last["wm_err"] > 0
    if max_error == 0.0:
        assert last["imagined_steps"] == 0 and last["skipped_cycles"] > 0
    else:
        assert last["imagined_steps"] > 0 and last["skipped_cycles"] == 0


def test_baselines(setup, tmp_path):
    teacher, targets = setup
    r = run(RunConfig(arm="baselines", preset="dynaphos", out=str(tmp_path)), teacher, targets)
    b = r["baselines"]
    assert b["zero"]["charge"] == 0.0
    assert b["phosphene_map"]["loss"] > 0
