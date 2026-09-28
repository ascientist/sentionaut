"""NeuroStim-Percept: differentiable brain, world model and policy (no MNIST download)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sentionaut.neurostim.percept import (  # noqa: E402
    AutoEncoder,
    Brain,
    BrainConfig,
    Classifier,
    collect,
    evaluate,
    exploration_actions,
    project_charge,
    train_policy,
    train_world_model,
    true_simulator,
    wm_simulator,
)

CFG = BrainConfig(horizon=6, hold_from=3)


@pytest.fixture(scope="module")
def parts():
    torch.manual_seed(0)
    ae = AutoEncoder(CFG.d).eval().requires_grad_(False)
    clf = Classifier().eval().requires_grad_(False)
    return Brain(CFG), ae, clf, torch.rand(32, 784), torch.randint(0, 10, (32,))


def test_rollout_shapes_and_gradient(parts):
    brain, *_ = parts
    B = brain.sample_patients(3, seed=0)
    a = torch.zeros(3, CFG.horizon, CFG.m, requires_grad=True)
    xs, hs = brain.rollout(B, a, seed=0)
    assert xs.shape == (3, CFG.horizon + 1, CFG.d) and hs.shape == (3, CFG.horizon + 1, CFG.m)
    xs[:, -1].sum().backward()
    assert a.grad.abs().sum() > 0  # stimulation → state is differentiable


def test_adaptation_and_charge_projection(parts):
    brain, *_ = parts
    a = project_charge(torch.ones(2, CFG.horizon, CFG.m), CFG.q_max)
    assert torch.allclose(a.abs().sum(-1), torch.full((2, CFG.horizon), CFG.q_max))
    _, hs = brain.rollout(brain.sample_patients(2, 0), a, noise=False)
    assert torch.all(hs[:, 1:].diff(dim=1) > 0)  # sustained stimulation keeps adapting


def test_patients_differ_and_probe_is_shared(parts):
    brain, ae, *_ = parts
    B = brain.sample_patients(2, seed=1)
    assert not torch.allclose(B[0], B[1])
    ctx = brain.calibrate(ae, B, seed=0)
    assert ctx.shape == (2, CFG.m * CFG.d)
    a = exploration_actions(CFG, 16, seed=0)
    assert a.shape == (16, CFG.horizon, CFG.m)
    assert torch.all(a.abs().sum(-1) <= CFG.q_max + 1e-5)


@pytest.mark.parametrize("random_patients", [False, True])
def test_world_model_and_policy_smoke(parts, random_patients):
    brain, ae, clf, images, labels = parts
    n = 16
    B = brain.sample_patients(n if random_patients else 1, seed=2).expand(n, -1, -1)
    data = collect(brain, ae, B, exploration_actions(CFG, n, 0), seed=0, calibrate=random_patients)
    n_ctx = data["ctx"].shape[1] if random_patients else 0
    wm = train_world_model(data, ae, CFG, n_ctx=n_ctx, steps=3, batch=8, verbose=False)

    def patient(k, step):
        Bk = brain.sample_patients(k, seed=step) if random_patients else B[:1].expand(k, -1, -1)
        return Bk, brain.calibrate(ae, Bk, seed=step) if random_patients else None

    for sim in (wm_simulator(wm, brain, ae), true_simulator(brain, ae)):
        pi = train_policy(
            sim, ae, images, CFG, patient, n_ctx=n_ctx, steps=2, batch=8, verbose=False
        )
        Bt, ctx = patient(len(images), 99)
        r = evaluate(pi, brain, ae, clf, images, labels, Bt, ctx)
        assert r["percepts"].shape == (len(images), CFG.horizon, 784)
        assert 0 <= r["accuracy"] <= 1 and r["charge"] <= CFG.q_max + 1e-5
