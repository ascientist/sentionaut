"""JAX backend of the Dynaphos x axon-map hybrid: parity with torch, scan, grad, vmap."""

from __future__ import annotations

import pytest
import torch
from test_dynaphos_axonmap import GRID, ZONE_IDX, _action, _build

from sentionaut.core.base import Pose
from sentionaut.core.config import Config
from sentionaut.world import WorldModel

jax = pytest.importorskip("jax", reason="JAX backend needs `uv sync --extra jax`")
jnp = jax.numpy


def _sequence(n):
    pose = Pose(dx=150.0, dy=-80.0, rot=0.05)
    amps = [200.0] * 4 + [0.0] * 2 + [90.0, 300.0]
    return [
        _action(
            n,
            a,
            freq=torch.full((n,), 60.0),
            phase_dur=torch.full((n,), 0.45),
            axlambda=800.0,
            pose=pose,
        )
        for a in amps
    ]


@pytest.mark.parametrize("costim", [False, True])
def test_jax_matches_torch(costim):
    kw = dict(costim_enabled=costim, costim_kappa=5e4)
    _, _, tm = _build("dynaphos_axonmap", **kw)
    _, _, jm = _build("dynaphos_axonmap_jax", **kw)
    ts = js = None
    for act in _sequence(tm.implant.n_electrodes):
        ts = tm.step(ts, act)
        js = jm.step(js, act)
        torch.testing.assert_close(js.image, ts.image, atol=2e-5, rtol=1e-4)
        for k in ("A", "Q", "sigma"):
            torch.testing.assert_close(js.aux[k], ts.aux[k], atol=1e-9, rtol=1e-5)
    assert float(ts.image.max()) > 0


def test_jax_scan_rollout_matches_steps():
    from sentionaut.models import dynaphos_axonmap_jax as dj

    _, topo, jm = _build("dynaphos_axonmap_jax")
    n = jm.implant.n_electrodes
    act = _action(n, 180.0)
    frames = jm.predict_sequence(act, 6)
    state = None
    for t in range(6):
        state = jm.step(state, act)
        torch.testing.assert_close(frames[t], state.image, atol=1e-6, rtol=1e-5)
    assert frames.shape == (6, *topo.grid_shape)
    assert isinstance(jm.geom, dj.AxonGeometry)


def test_jax_grad_and_vmap():
    from sentionaut.models import dynaphos_axonmap_jax as dj

    _, topo, jm = _build("dynaphos_axonmap_jax")
    n = jm.implant.n_electrodes
    p, geom, shape = jm.params, jm.geom, topo.grid_shape
    elec_xy = jnp.asarray(jm.implant.electrode_coords().numpy())
    freq = jnp.full((n,), p.freq)
    pdur = jnp.full((n,), p.p_dur)
    amp0 = jnp.zeros(n).at[jnp.array(ZONE_IDX)].set(200.0)

    def activation(amp):
        amps = jnp.broadcast_to(amp, (3, n))
        state, _ = dj.rollout(
            p, geom, dj.init_state(n), amps, jnp.broadcast_to(freq, (3, n)),
            jnp.broadcast_to(pdur, (3, n)), elec_xy, 500.0, grid_shape=shape,
        )  # fmt: skip
        return state.A.sum()

    g = jax.grad(activation)(amp0)
    assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all(g[jnp.array(ZONE_IDX)] > 0))

    def image(amp):
        _, img = dj.step(p, geom, dj.init_state(n), amp, freq, pdur, elec_xy, 500.0, shape)
        return img

    batch = jnp.stack([amp0, amp0 * 0.5, jnp.zeros(n)])
    imgs = jax.vmap(image)(batch)
    assert imgs.shape == (3, *shape)
    assert float(imgs[2].max()) == 0.0


def test_jax_world_model_contract():
    cfg = Config(model="dynaphos_axonmap_jax", implant="argusii", **GRID)
    wm = WorldModel.from_config(cfg, torch.device("cpu"))
    n = wm.model.implant.n_electrodes
    s = wm.step(wm.initial_state(), _action(n))
    assert s.image.shape == wm.grid_shape and s.image.dtype == torch.float32
    assert torch.isfinite(s.image).all()
    a_map, _ = wm.model.rasterize_aux(s)
    assert a_map.shape == wm.grid_shape


@pytest.mark.skipif(
    not torch.cuda.is_available() or jax.default_backend() != "gpu",
    reason="needs CUDA torch and a CUDA JAX backend",
)
def test_jax_gpu_matches_torch_gpu():
    _, _, tm = _build("dynaphos_axonmap", device="cuda")
    _, _, jm = _build("dynaphos_axonmap_jax", device="cuda")
    n = tm.implant.n_electrodes
    ts = js = None
    for _ in range(4):
        ts = tm.step(ts, _action(n, device="cuda"))
        js = jm.step(js, _action(n, device="cuda"))
    assert js.image.device.type == "cuda"
    torch.testing.assert_close(js.image, ts.image, atol=2e-5, rtol=1e-4)
