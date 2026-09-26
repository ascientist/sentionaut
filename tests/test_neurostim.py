"""NeuroStim toy CMDP: API contract, each difficulty axis, and baseline ordering."""

from __future__ import annotations

import numpy as np
import pytest

from sentionaut.neurostim import PRESETS, make
from sentionaut.neurostim.baselines import (
    OracleGreedy,
    ZeroPolicy,
    default_baselines,
    evaluate,
    rollout,
)

QUIET = dict(process_noise=0.0, obs_noise=0.0, x0_std=0.0, cubic=0.0)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_api_contract(name):
    env = make(name)
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    for key in ("x", "h", "B", "A", "pending"):
        assert key in info
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    assert isinstance(r, float) and not term and not trunc
    assert info["cost"] in (0.0, 1.0)
    assert set(info["cost_terms"]) == {"charge", "adaptation"}


def test_seed_determinism_and_new_patient_per_seed():
    env = make("NeuroStim-v0")
    o1, i1 = env.reset(seed=7)
    o2, i2 = env.reset(seed=7)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(i1["B"], i2["B"])
    _, i3 = env.reset(seed=8)
    assert not np.allclose(i1["B"], i3["B"])


def test_episode_truncates_at_horizon():
    env = make("NeuroStim-v0", horizon=5)
    env.reset(seed=0)
    flags = [env.step(np.zeros(4))[3] for _ in range(5)]
    assert flags == [False] * 4 + [True]


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_delay_shifts_the_response(delay):
    env = make("NeuroStim-Easy-v0", delay=delay, **QUIET)
    env.reset(seed=0)
    xs = [env.step(np.eye(4)[0])[4]["x"].copy()]
    xs += [env.step(np.zeros(4))[4]["x"].copy() for _ in range(delay + 1)]
    moved = [np.linalg.norm(x) > 1e-9 for x in xs]
    assert moved.index(True) == delay


def test_adaptation_attenuates_repeated_stimulation():
    a = np.array([0.0, 0.0, 1.0, 0.0])
    gains = {}
    for adaptation in (False, True):
        env = make("NeuroStim-v0", randomize_B=False, adaptation=adaptation, **QUIET)
        env.reset(seed=0)
        env.A = np.zeros((3, 3))  # isolate the per-step stimulation effect
        first = env.step(a)[4]["x"].copy()
        for _ in range(30):
            last = env.step(a)[4]["x"].copy()
        gains[adaptation] = np.linalg.norm(last) / np.linalg.norm(first)
    assert gains[False] == pytest.approx(1.0)
    assert gains[True] < 0.6


def test_constraint_costs_fire():
    env = make("NeuroStim-v0", adapt_rate=0.5)
    env.reset(seed=0)
    info = env.step(np.ones(4))[4]  # charge 4 > q_max = 2, h = 0.5 < h_max
    assert info["cost_terms"]["charge"] == pytest.approx(2.0)
    info = env.step(np.ones(4))[4]  # h = 0.975 > h_max
    assert info["cost_terms"]["adaptation"] > 0
    assert info["cost"] == 1.0


def test_actions_are_clipped():
    env = make("NeuroStim-v0")
    env.reset(seed=0)
    info = env.step(np.full(4, 50.0))[4]
    assert info["charge"] == pytest.approx(4.0)


def test_all_baselines_run():
    env = make("NeuroStim-Hard-v0", horizon=20)
    for policy in default_baselines():
        traj = rollout(env, policy, seed=0)
        assert traj.actions.shape == (20, 4)
        assert np.all(np.abs(traj.actions) <= 1.0)


def test_oracle_beats_doing_nothing():
    env = make("NeuroStim-v0", horizon=60)
    zero = evaluate(env, ZeroPolicy(), episodes=5)
    oracle = evaluate(env, OracleGreedy(), episodes=5)
    assert oracle["return"] > zero["return"] + 50
    assert oracle["ss_error"] < 0.5 * zero["ss_error"]


def test_ppo_smoke():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from sentionaut.neurostim.ppo import PPOConfig, train_ppo

    cfg = PPOConfig(total_steps=512, n_envs=2, n_steps=128, epochs=1, cost_limit=5.0)
    policy, log = train_ppo(lambda: make("NeuroStim-v0", horizon=64), cfg, verbose=False)
    traj = rollout(make("NeuroStim-v0", horizon=16), policy, seed=0)
    assert traj.actions.shape == (16, 4)
    assert log and "lambda" in log[-1]
