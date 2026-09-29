"""Baseline suite on the canonical problem: dynamics, classical controllers, learned smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from sentionaut.neurostim.canonical import (  # noqa: E402
    OracleController,
    Population,
    ZeroController,
    evaluate,
    recruit,
    regime,
    step_jacobians,
    step_mean,
)
from sentionaut.neurostim.canonical_control import (  # noqa: E402
    DeePCController,
    HInfController,
    ILQRController,
    KoopmanController,
    MPPIController,
    PIController,
    hinf_gain,
)


@pytest.mark.parametrize("R", ["A", "C", "D"])
def test_jacobians_match_finite_differences(R):
    cfg = regime(R)
    pop = Population(cfg)
    pat = pop.sample(np.random.default_rng(0))
    rng = np.random.default_rng(1)
    x, u = rng.normal(size=cfg.n), rng.normal(size=cfg.m)
    fx, fu = step_jacobians(cfg, pat.A, pat.B, x, u)
    h = 1e-6
    fx_num = np.stack(
        [
            (
                step_mean(cfg, pat.A, pat.B, x + h * e, u)
                - step_mean(cfg, pat.A, pat.B, x - h * e, u)
            )
            / (2 * h)
            for e in np.eye(cfg.n)
        ],
        1,
    )
    fu_num = np.stack(
        [
            (
                step_mean(cfg, pat.A, pat.B, x, u + h * e)
                - step_mean(cfg, pat.A, pat.B, x, u - h * e)
            )
            / (2 * h)
            for e in np.eye(cfg.m)
        ],
        1,
    )
    np.testing.assert_allclose(fx, fx_num, atol=1e-6)
    np.testing.assert_allclose(fu, fu_num, atol=1e-6)


def test_even_recruitment_is_polarity_insensitive():
    cfg = regime("D")
    u = np.array([0.3, -1.2, 0.7, 2.0])
    np.testing.assert_allclose(recruit(cfg, u), recruit(cfg, -u))


def test_hinf_gain_tends_to_lqr_and_stabilises():
    from scipy.linalg import solve_discrete_are

    pop = Population(regime("A"))
    A, B, c = pop.A_bar, pop.B_bar, pop.cfg
    Q, R = pop.D.T @ pop.D + 1e-6 * np.eye(c.n), c.lam_u * np.eye(c.m)
    P = solve_discrete_are(A, B, Q, R)
    K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    np.testing.assert_allclose(hinf_gain(A, B, np.eye(c.n), Q, R, 1e4), K_lqr, atol=1e-4)
    ctrl = HInfController()
    ctrl.reset(pop, pop.targets[0], {})
    assert np.max(np.abs(np.linalg.eigvals(A - B @ ctrl.K))) < 1


def test_nmpc_matches_lqg_on_known_linear_patient():
    cfg = regime("A", horizon=100)
    lqg = evaluate(cfg, OracleController, n_patients=3)["ss_median"]
    ilqr = evaluate(cfg, ILQRController, n_patients=3)["ss_median"]
    assert ilqr < 2 * lqg


@pytest.mark.parametrize(
    "make", [PIController, HInfController, DeePCController, KoopmanController, MPPIController]
)
def test_classical_controllers_run_on_every_regime(make):
    for R in "ABCD":
        r = evaluate(regime(R, horizon=90), make, n_patients=1)
        assert np.isfinite(r["ss_median"])


def test_mppi_handles_multimodal_recruitment():
    cfg = regime("D", horizon=100)
    zero = evaluate(cfg, ZeroController, n_patients=3)["ss_median"]
    mppi = evaluate(cfg, MPPIController, n_patients=3)["ss_median"]
    assert mppi < 0.2 * zero


def test_learned_methods_smoke():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from sentionaut.neurostim.canonical_learned import (
        JEPAConfig,
        SACConfig,
        TDMPCConfig,
        collect_demos,
        train_bc,
        train_diffusion,
        train_jepa,
        train_sac,
        train_tdmpc,
    )

    cfg = regime("D", horizon=40)
    demos = collect_demos(cfg, 4, seed=0, n_envs=4)
    assert demos[0].shape[:2] == (4, 40) and demos[1].shape == (4, 40, cfg.m)
    ctrls = [
        train_sac(
            cfg, SACConfig(steps=640, start=320, n_envs=16, updates_per_step=1), verbose=False
        ),
        train_tdmpc(
            cfg, TDMPCConfig(steps=640, start=320, n_envs=16, updates_per_step=1), verbose=False
        ),
        train_jepa(
            cfg, JEPAConfig(episodes=100, steps=5, plan_horizon=3, probe_steps=5), verbose=False
        ),
        train_bc(cfg, demos, steps=5, verbose=False),
        train_diffusion(cfg, demos, steps=5, verbose=False),
    ]
    for ctrl in ctrls:
        r = evaluate(cfg, lambda c=ctrl: c, n_patients=1)
        assert np.isfinite(r["ss_error"])


def test_vicreg_terms_detect_collapse():
    torch = pytest.importorskip("torch")
    from sentionaut.neurostim.canonical_learned import collapse_stats, vicreg_terms

    iso = torch.randn(512, 16)
    collapsed = torch.randn(512, 1).expand(512, 16) * 0.01  # one tiny shared direction
    var_iso, cov_iso = vicreg_terms(iso)
    var_col, cov_col = vicreg_terms(collapsed)
    assert var_iso < 0.1 < var_col
    assert collapse_stats(iso)["eff_rank"] > 14 and collapse_stats(collapsed)["eff_rank"] < 2


def test_training_seeds_never_reuse_evaluation_patients():
    """evaluate() draws held-out patients from seeds 1000..1000+N; every training
    stream (SAC 10k+, TD-MPC 20k+, JEPA 30k+, demos 50k+) must stay clear of them."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "src/sentionaut/neurostim/canonical_learned.py").read_text()
    tut = (root / "examples/neurostim_baselines_tutorial.py").read_text()
    offsets = [int(x.replace("_", "")) for x in re.findall(r"seed=(\d[\d_]*) \+ ", src)]
    offsets.append(int(re.search(r"DEMO_SEED = (\d[\d_]*)", tut).group(1).replace("_", "")))
    eval_seeds = set(range(1000, 1000 + 200))
    for off in offsets:
        assert not eval_seeds & set(range(off, off + 100)), off  # 100 training seeds


def test_sigreg_penalises_collapse_scale_and_tails_not_gaussians():
    torch = pytest.importorskip("torch")
    from sentionaut.neurostim.canonical_learned import sigreg

    torch.manual_seed(0)
    gauss = sigreg(torch.randn(512, 16))
    assert gauss < 2
    assert sigreg(torch.randn(512, 1).expand(512, 16) * 0.01) > 20 * gauss  # collapse
    assert sigreg(3 * torch.randn(512, 16)) > 20 * gauss  # wrong scale
    assert sigreg(torch.distributions.StudentT(2.0).sample((512, 16))) > 10 * gauss  # tails
