"""Canonical random-LDS formulation: oracle optimality, the ladder, and the ARX realisation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from sentionaut.neurostim.canonical import (  # noqa: E402
    AdaptiveController,
    OracleController,
    Population,
    PopulationController,
    ZeroController,
    evaluate,
    level,
    lq_tracker,
    run_episode,
)


def test_lq_tracker_hits_reachable_target_noise_free():
    cfg = level("L0 known patient", process_noise=0.0, obs_noise=0.0, horizon=150)
    pop = Population(cfg)
    patient = pop.sample(np.random.default_rng(0))
    z = pop.targets[0]
    x_ss, u_ss, K = lq_tracker(patient.A, patient.B, pop.D, np.zeros(cfg.n), z, cfg.lam_u, 10.0)
    np.testing.assert_allclose(patient.A @ x_ss + patient.B @ u_ss, x_ss, atol=1e-10)
    assert np.max(np.abs(np.linalg.eigvals(patient.A - patient.B @ K))) < 1  # stabilising
    err, _ = run_episode(pop, patient, OracleController(), z, np.random.default_rng(0))
    assert err[-20:].mean() < 1e-3


def test_population_equals_oracle_when_patients_are_identical():
    cfg = level("L0 known patient", horizon=100)
    pop_r = evaluate(cfg, PopulationController, n_patients=3)
    orc_r = evaluate(cfg, OracleController, n_patients=3)
    assert pop_r["ss_error"] == pytest.approx(orc_r["ss_error"])


def test_patients_differ_but_share_targets():
    pop = Population(level("L1 random patients"))
    a, b = pop.sample(np.random.default_rng(1)), pop.sample(np.random.default_rng(2))
    assert not np.allclose(a.B, b.B)
    assert Population(level("L2 + partial observation")).targets.shape == pop.targets.shape


@pytest.mark.parametrize("name", ["L1 random patients", "L2 + partial observation"])
def test_adaptive_beats_population_on_random_patients(name):
    cfg = level(name, horizon=250)
    zero = evaluate(cfg, ZeroController, n_patients=6)["ss_median"]
    popc = evaluate(cfg, PopulationController, n_patients=6)["ss_median"]
    adapt = evaluate(cfg, AdaptiveController, n_patients=6)["ss_median"]
    oracle = evaluate(cfg, OracleController, n_patients=6)["ss_median"]
    assert oracle <= adapt < min(zero, popc)


def test_arx_lag_matches_observability_index():
    ctrl = AdaptiveController()
    for observe, lags in [("full", 1), ("percept", 2)]:
        pop = Population(level("L1 random patients", observe=observe))
        ctrl.reset(pop, pop.targets[0], {})
        assert ctrl.L == lags  # ceil(n / dim o): 6/6 -> 1, 6/3 -> 2
