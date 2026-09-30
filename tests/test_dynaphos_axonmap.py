"""Dynaphos x axon-map hybrid: physics, torch <-> JAX parity, gradients, framework wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sentionaut.core.base import Action
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components
from sentionaut.models.dynaphos import DynaphosTorch
from sentionaut.world import WorldModel

GRID = dict(xrange=(-6, 6), yrange=(-6, 6), xystep=1.0)
ZONE = ["C5", "C6", "D5"]  # neighbouring Argus II electrodes at the array centre


def _build(model="dynaphos_axonmap", device="cpu", **kw):
    cfg = Config(model=model, implant="argusii", **{**GRID, **kw})
    return build_components(cfg, torch.device(device))


ZONE_IDX = [24, 25, 34]  # C5, C6, D5 (checked in test_zone_indices)


def _action(n, amp_uA=200.0, electrodes=ZONE_IDX, device="cpu", **kw):
    amp = torch.zeros(n, device=device)
    amp[electrodes] = amp_uA
    return Action(amp=amp, **kw)


def _second_moment_ratio(img: torch.Tensor) -> float:
    """Major/minor axis ratio of a percept's intensity-weighted covariance."""
    img = img.detach().cpu().double()
    H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H).double(), torch.arange(W).double(), indexing="ij")
    w = img / img.sum()
    mx, my = (w * xx).sum(), (w * yy).sum()
    cov = torch.stack(
        [
            torch.stack([(w * (xx - mx) ** 2).sum(), (w * (xx - mx) * (yy - my)).sum()]),
            torch.stack([(w * (xx - mx) * (yy - my)).sum(), (w * (yy - my) ** 2).sum()]),
        ]
    )
    ev = torch.linalg.eigvalsh(cov)
    return float(torch.sqrt(ev[1] / ev[0]))


def test_zone_indices():
    implant, _, _ = _build()
    assert [implant.names.index(n) for n in ZONE] == ZONE_IDX


def test_temporal_dynamics_match_dynaphos_equations():
    """A, Q follow Dynaphos Eqs 7-13 exactly; only the spatial render changes."""
    implant, _, model = _build()
    n = implant.n_electrodes
    seq = [200.0] * 5 + [0.0] * 3 + [120.0] * 4
    p = model
    A = torch.zeros(n)
    Q = torch.zeros(n)
    state = None
    for a in seq:
        act = _action(n, a)
        state = model.step(state, act)
        amp = act.amp
        Ieff = torch.clamp((amp - p.rheobase - Q) * p.freq * (p.p_dur / 1000.0), min=0.0)
        Q = Q + (-Q / (p.tau_trace / 1000.0) + Ieff * p.kappa_trace) * (p.dt / 1000.0)
        A = A + (-A / (p.tau_act / 1000.0) + Ieff * 1e-6) * (p.dt / 1000.0)
        torch.testing.assert_close(state.aux["A"], A)
        torch.testing.assert_close(state.aux["Q"], Q)


def test_rho_is_dynaphos_current_spread():
    """rho_e = D_e / 2 in microns with D = 2 sqrt(I / K) (Dynaphos Eq 6)."""
    implant, _, model = _build()
    for current in (50.0, 100.0, 300.0):
        s = model.step(None, _action(implant.n_electrodes, current))
        expected = 1000.0 * np.sqrt(current / model.excitability)
        assert float(s.aux["sigma"][ZONE_IDX[0]]) == pytest.approx(expected, rel=1e-5)


def _idx(implant, names):
    return [implant.names.index(n) for n in names]


def test_phosphene_elongates_along_axons():
    """Off-fovea, a larger lambda stretches the phosphene along the arcuate bundle."""
    implant, _, model = _build(xrange=(-16, 16), yrange=(-16, 16), xystep=1.0)
    n = implant.n_electrodes
    images = {}
    for lam in (100.0, 1500.0):
        state = None
        for _ in range(5):  # 60 uA needs a few frames to cross A_thr
            state = model.step(state, _action(n, 60.0, _idx(implant, ["A1"]), axlambda=lam))
        images[lam] = state.image
    short, long = images[100.0], images[1500.0]
    assert short.max() > 0
    assert _second_moment_ratio(short) < 1.15  # ~isotropic Dynaphos blob
    assert _second_moment_ratio(long) > 1.4  # streak
    # The streak only adds axon segments further from the soma, so it can only grow.
    assert (long >= short - 1e-6).all()


def test_render_equals_biphasic_axonmap_kernel():
    """With F_size = F_streak = 1, F_bright = b and rho = sqrt(I/K), the render is
    exactly the pulse2percept-parity-tested BiphasicAxonMapTorch kernel."""
    from sentionaut.models import effects
    from sentionaut.models.axonmap import BiphasicAxonMapTorch

    implant, topo, model = _build()
    n = implant.n_electrodes
    zone = _idx(implant, ZONE)
    current, lam = 150.0, 800.0
    act = _action(n, current, zone, axlambda=lam)
    s = model.step(None, act)
    b = float(torch.sigmoid(model.sig_slope * (s.aux["A"][zone[0]] - model.a50)))
    rho = float(s.aux["sigma"][zone[0]])
    # f_bright = a2 * amp * (a1 + a0 * pdur) -> b ; f_size = a6 = 1 ; f_streak = a9 = 1.
    ep = effects.EffectParams(
        a0=0.0, a1=1.0, a2=b / current, a3=0.0, a5=0.0, a6=1.0, a7=0.0, a9=1.0
    )
    ref_model = BiphasicAxonMapTorch(rho=rho, axlambda=lam, effect_params=ep).build(implant, topo)
    ref = ref_model.spatial_forward(
        Action(amp=act.amp, freq=torch.ones(n), phase_dur=torch.ones(n))
    )
    torch.testing.assert_close(s.image, torch.clamp(ref, 0, 1), atol=1e-5, rtol=1e-4)


def test_small_lambda_reduces_to_isotropic_gaussian():
    """lambda -> 0 keeps only each pixel's soma segment: the image becomes the
    isotropic Dynaphos Gaussian in retinal space (the axon step is the only change)."""
    implant, topo, model = _build()
    topo.d_soma = topo.d_soma.clone()
    topo.d_soma[:, 0] = 0.0  # pin segment 0 onto the soma (it is within ~25 um)
    n = implant.n_electrodes
    s = model.step(None, _action(n, 150.0, _idx(implant, ZONE), axlambda=1e-3))
    b = torch.sigmoid(model.sig_slope * (s.aux["A"] - model.a50))
    on = s.aux["A"] >= model.a_thr
    xy = implant.electrode_coords()[on]
    d2 = torch.cdist(topo.coords[:, 0, :], xy).pow(2)
    ref = (torch.exp(-d2 / (2 * s.aux["sigma"][on] ** 2)) * b[on]).sum(1)
    ref = torch.clamp(ref, 0, 1).reshape(topo.grid_shape)
    assert ref.max() > 0.3
    torch.testing.assert_close(s.image, ref, atol=1e-5, rtol=1e-4)


def test_vectorised_costim_matches_dynaphos_loop():
    implant, _, model = _build(costim_enabled=True, costim_kappa=5e4)
    amp = torch.zeros(implant.n_electrodes)
    amp[ZONE_IDX] = torch.tensor([100.0, 150.0, 200.0])
    torch.testing.assert_close(model._apply_costim(amp), DynaphosTorch._apply_costim(model, amp))


def test_gradients_are_finite():
    implant, _, model = _build()
    amp = torch.zeros(implant.n_electrodes)
    amp[ZONE_IDX] = 200.0
    amp.requires_grad_(True)
    lam = torch.tensor(500.0, requires_grad=True)
    state = None
    for _ in range(3):
        state = model.step(state, Action(amp=amp, axlambda=lam))
    # Use a pre-saturation brightness target so the sigmoid does not zero the grads.
    state.aux["A"].sum().backward(retain_graph=True)
    assert torch.isfinite(amp.grad).all() and (amp.grad[ZONE_IDX] > 0).all()
    amp.grad = None
    state.image.sum().backward()
    assert torch.isfinite(amp.grad).all()
    assert lam.grad is not None and torch.isfinite(lam.grad) and lam.grad > 0


def test_dynaphos_off_electrode_grads_are_finite():
    """Regression: sqrt(clamp(amp, 0)) gave NaN grads for every silent electrode."""
    cfg = Config(model="dynaphos", implant="orion", xrange=(-4, 4), yrange=(-4, 4), xystep=1.0)
    implant, _, model = build_components(cfg, torch.device("cpu"))
    amp = torch.zeros(implant.n_electrodes)
    amp[0] = 250.0
    amp.requires_grad_(True)
    model.step(None, Action(amp=amp)).image.sum().backward()
    assert torch.isfinite(amp.grad).all()


def test_world_model_and_aux():
    cfg = Config(model="dynaphos_axonmap", implant="argusii", **GRID)
    wm = WorldModel.from_config(cfg, torch.device("cpu"))
    n = wm.model.implant.n_electrodes
    states = wm.rollout([_action(n, 200.0)] * 3 + [_action(n, 0.0)] * 2)
    assert all(s.image.shape == wm.grid_shape for s in states)
    assert float(states[-1].image.max()) > 0  # phosphene outlives the pulse
    a_map, q_map = wm.model.rasterize_aux(states[-1])
    assert a_map.shape == q_map.shape == wm.grid_shape


def test_prima_rejected():
    with pytest.raises(ValueError):
        Config(model="dynaphos_axonmap", implant="prima")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_cuda_matches_cpu():
    _, _, cpu = _build(device="cpu")
    _, _, gpu = _build(device="cuda")
    n = cpu.implant.n_electrodes
    s_cpu = s_gpu = None
    for _ in range(4):
        s_cpu = cpu.step(s_cpu, _action(n))
        s_gpu = gpu.step(s_gpu, _action(n, device="cuda"))
    assert s_gpu.image.device.type == "cuda"
    torch.testing.assert_close(s_gpu.image.cpu(), s_cpu.image, atol=1e-5, rtol=1e-4)
