"""JAX port of ``DynaphosAxonMapTorch`` (Dynaphos dynamics + retinal axon map).

Same equations as ``models/dynaphos_axonmap.py`` (see its docstring). Two layers:

* A pure functional core (``init_state`` / ``step`` / ``rollout``) over plain
  arrays. It is ``jit``-compiled with fixed shapes (inactive electrodes are
  masked rather than dropped), so it runs unchanged on CPU, CUDA or TPU,
  composes with ``jax.grad`` / ``jax.vmap``, and ``rollout`` fuses a whole
  stimulation sequence into one ``lax.scan``.
* ``DynaphosAxonMapJax``: a ``PerceptModel`` adapter so the framework (registry,
  ``WorldModel``, dataset generation) can drive the JAX core with torch
  ``Action`` / ``State`` objects. Tensors cross via DLPack (zero-copy on a
  shared GPU) with a host fallback. Torch autograd does not flow through the
  adapter; differentiate the functional core with ``jax.grad`` instead.

JAX picks its own backend: install the ``jax-cuda`` extra to run on NVIDIA GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ..core.base import Action, Implant, PerceptModel, State
from ..topography.axon_map import AxonMapTopography
from .dynaphos import _load_dynaphos_params


@dataclass(frozen=True)
class DynaphosAxonMapParams:
    """Scalar model constants. Frozen (hashable) so ``jit`` treats them as static."""

    dt: float
    tau_act: float
    rheobase: float
    tau_trace: float
    kappa_trace: float
    excitability: float
    sig_slope: float
    a50: float
    a_thr: float
    freq: float
    p_dur: float
    axlambda: float = 500.0
    min_rho: float = 10.0
    costim_enabled: bool = False
    costim_kappa: float = 1.0
    max_percept: float | None = None

    @classmethod
    def from_dynaphos(cls, params: dict | None = None, **overrides) -> "DynaphosAxonMapParams":
        p = dict(params or _load_dynaphos_params())
        names = {f.name for f in fields(cls)}
        kw = {k: float(v) for k, v in p.items() if k in names}
        kw.update(overrides)
        return cls(**kw)


class AxonGeometry(NamedTuple):
    """Array pytree of the retinal axon map and electrode pair distances."""

    coords: jax.Array  # (P, L, 2) axon-segment positions, retinal microns
    d_soma: jax.Array  # (P, L) path distance of each segment to its soma, microns
    mask: jax.Array  # (P, L) 1 for real segments, 0 for padding
    inv_elec_d2: jax.Array  # (E, E) 1 / ||x_i - x_j||^2, zero on the diagonal


class DynaphosState(NamedTuple):
    A: jax.Array  # (E,) tissue activation
    Q: jax.Array  # (E,) memory trace, uA
    sigma: jax.Array  # (E,) last current-spread SD, retinal microns


def make_geometry(coords, d_soma, mask, elec_xy) -> AxonGeometry:
    elec_xy = jnp.asarray(elec_xy, dtype=jnp.float32)
    d2 = jnp.sum((elec_xy[:, None, :] - elec_xy[None, :, :]) ** 2, axis=-1)
    n = elec_xy.shape[0]
    inv = jnp.where(jnp.eye(n, dtype=bool), 0.0, 1.0 / jnp.maximum(d2, 1e-12))
    return AxonGeometry(
        coords=jnp.asarray(coords, dtype=jnp.float32),
        d_soma=jnp.asarray(d_soma, dtype=jnp.float32),
        mask=jnp.asarray(mask, dtype=jnp.float32),
        inv_elec_d2=inv,
    )


def init_state(n_electrodes: int, dtype=jnp.float32) -> DynaphosState:
    z = jnp.zeros((n_electrodes,), dtype=dtype)
    return DynaphosState(A=z, Q=z, sigma=z)


def apply_costim(params: DynaphosAxonMapParams, geom: AxonGeometry, amp: jax.Array) -> jax.Array:
    if not params.costim_enabled:
        return amp
    active = (amp > 0).astype(amp.dtype)
    return amp + active * params.costim_kappa * (geom.inv_elec_d2 @ (amp * active))


def dynamics(params: DynaphosAxonMapParams, A, Q, amp, freq, p_dur):
    """Dynaphos Eqs 6-13: returns (A, Q, D, brightness); ``D`` in mm."""
    p = params
    Ieff = jnp.maximum((amp - p.rheobase - Q) * freq * (p_dur / 1000.0), 0.0)
    Q = Q + ((-Q / (p.tau_trace / 1000.0)) + Ieff * p.kappa_trace) * (p.dt / 1000.0)
    on = amp > 0
    D = jnp.where(on, 2.0 * jnp.sqrt(jnp.where(on, amp, 1.0) / p.excitability), 0.0)
    A = A + ((-A / (p.tau_act / 1000.0)) + Ieff * 1e-6) * (p.dt / 1000.0)
    brightness = jax.nn.sigmoid(p.sig_slope * (A - p.a50))
    return A, Q, D, brightness


def render(params, geom: AxonGeometry, A, sigma, brightness, elec_xy, axlambda) -> jax.Array:
    """Axon-map render of the supra-threshold phosphenes: returns ``(P,)``."""
    active = A >= params.a_thr
    w = jnp.where(active, brightness, 0.0)
    s = jnp.where(active, sigma, 1.0)  # safe SD for masked electrodes
    diff = geom.coords[:, :, None, :] - elec_xy[None, None, :, :]
    d2 = jnp.sum(diff**2, axis=-1)  # (P, L, E)
    spread = jnp.exp(-d2 / (2.0 * s**2)) @ w  # (P, L)
    sens = jnp.exp(-(geom.d_soma**2) / (2.0 * axlambda**2)) * geom.mask
    intensity = jnp.max(spread * sens, axis=1)
    intensity = jnp.clip(intensity, 0.0, 1.0)
    if params.max_percept is not None:
        intensity = jnp.minimum(intensity, params.max_percept)
    return intensity


def _step(params, geom, state: DynaphosState, amp, freq, p_dur, elec_xy, axlambda):
    amp = apply_costim(params, geom, amp)
    A, Q, D, brightness = dynamics(params, state.A, state.Q, amp, freq, p_dur)
    rho = jnp.maximum(500.0 * D, params.min_rho)  # D/2 mm -> microns
    sigma = jnp.where(amp > 0, rho, state.sigma)
    image = render(params, geom, A, sigma, brightness, elec_xy, axlambda)
    return DynaphosState(A=A, Q=Q, sigma=sigma), image


@partial(jax.jit, static_argnames=("params", "grid_shape"))
def step(params, geom, state, amp, freq, p_dur, elec_xy, axlambda, grid_shape):
    """One frame. Returns ``(new_state, image (H, W))``."""
    new_state, image = _step(params, geom, state, amp, freq, p_dur, elec_xy, axlambda)
    return new_state, image.reshape(grid_shape)


@partial(jax.jit, static_argnames=("params", "grid_shape"))
def rollout(params, geom, state, amps, freqs, p_durs, elec_xy, axlambda, grid_shape):
    """Scan ``T`` frames of per-electrode inputs ``(T, E)``. Returns ``(state, (T, H, W))``."""

    def body(s, xs):
        amp, freq, p_dur = xs
        s, image = _step(params, geom, s, amp, freq, p_dur, elec_xy, axlambda)
        return s, image.reshape(grid_shape)

    return jax.lax.scan(body, state, (amps, freqs, p_durs))


def _to_jax(t: torch.Tensor) -> jax.Array:
    t = t.detach().contiguous()
    if t.device.type == "cuda":
        try:
            return jnp.from_dlpack(t)
        except Exception:  # JAX without a CUDA backend: go through the host.
            pass
    return jnp.asarray(t.cpu().numpy())


def _to_torch(x: jax.Array, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        try:
            return torch.from_dlpack(x).to(device)
        except Exception:
            pass
    return torch.from_numpy(np.asarray(x).copy()).to(device)


class DynaphosAxonMapJax(PerceptModel):
    """Framework adapter: torch ``Action``/``State`` in, JAX core underneath."""

    def __init__(
        self,
        params: dict | None = None,
        axlambda: float = 500.0,
        min_rho: float = 10.0,
        costim_enabled: bool = False,
        costim_kappa: float = 1.0,
        max_percept: float | None = None,
    ):
        super().__init__()
        self.params = DynaphosAxonMapParams.from_dynaphos(
            params,
            axlambda=float(axlambda),
            min_rho=float(min_rho),
            costim_enabled=bool(costim_enabled),
            costim_kappa=float(costim_kappa),
            max_percept=max_percept,
        )

    # Same scalar attributes as DynaphosAxonMapTorch, so callers can swap backends.
    axlambda = property(lambda self: self.params.axlambda)
    dt = property(lambda self: self.params.dt)
    a_thr = property(lambda self: self.params.a_thr)
    a50 = property(lambda self: self.params.a50)
    sig_slope = property(lambda self: self.params.sig_slope)
    excitability = property(lambda self: self.params.excitability)
    rheobase = property(lambda self: self.params.rheobase)

    def build(self, implant: Implant, topography: AxonMapTopography) -> "DynaphosAxonMapJax":
        self.implant = implant
        self.topography = topography
        self.geom = make_geometry(
            _to_jax(topography.coords),
            _to_jax(topography.d_soma),
            _to_jax(topography.mask),
            _to_jax(implant.electrode_coords()),
        )
        self._built = True
        return self

    @property
    def device(self) -> torch.device:
        return self.topography.coords.device

    def initial_state(self, device: torch.device | None = None) -> State:
        device = device or self.device
        z = torch.zeros(self.implant.n_electrodes, device=device, dtype=torch.float32)
        image = torch.zeros(self.topography.grid_shape, device=device, dtype=torch.float32)
        return State(image=image, aux={"A": z.clone(), "Q": z.clone(), "sigma": z.clone()})

    def _inputs(self, action: Action):
        n = self.implant.n_electrodes
        amp = _to_jax(action.amp.float())
        freq = (
            jnp.full((n,), self.params.freq, jnp.float32)
            if action.freq is None
            else _to_jax(action.freq.float())
        )
        p_dur = (
            jnp.full((n,), self.params.p_dur, jnp.float32)
            if action.phase_dur is None
            else _to_jax(action.phase_dur.float())
        )
        elec_xy = _to_jax(self.implant.electrode_coords(action.pose).float())
        lam = self.params.axlambda if action.axlambda is None else action.axlambda
        lam = _to_jax(lam.float()) if torch.is_tensor(lam) else jnp.float32(lam)
        return amp, freq, p_dur, elec_xy, lam

    def step(self, state: State | None, action: Action) -> State:
        device = self.device
        if state is None:
            state = self.initial_state(device)
        js = DynaphosState(*(_to_jax(state.aux[k].float()) for k in ("A", "Q", "sigma")))
        amp, freq, p_dur, elec_xy, lam = self._inputs(action)
        js, image = step(
            self.params, self.geom, js, amp, freq, p_dur, elec_xy, lam,
            grid_shape=tuple(self.topography.grid_shape),
        )  # fmt: skip
        aux = {k: _to_torch(getattr(js, k), device) for k in ("A", "Q", "sigma")}
        return State(image=_to_torch(image, device), aux=aux)

    def forward(self, action: Action) -> torch.Tensor:
        return self.step(None, action).image

    def predict_sequence(self, action: Action, n_steps: int) -> torch.Tensor:
        """Hold ``action`` for ``n_steps`` frames in one fused ``lax.scan``."""
        amp, freq, p_dur, elec_xy, lam = self._inputs(action)
        tile = lambda x: jnp.broadcast_to(x, (n_steps,) + x.shape)  # noqa: E731
        _, frames = rollout(
            self.params, self.geom, init_state(self.implant.n_electrodes),
            tile(amp), tile(freq), tile(p_dur), elec_xy, lam,
            grid_shape=tuple(self.topography.grid_shape),
        )  # fmt: skip
        return _to_torch(frames, self.device)

    def rasterize_aux(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize per-electrode A and Q through the axon map (learned aux channels)."""
        A, Q, sigma = (_to_jax(state.aux[k].float()) for k in ("A", "Q", "sigma"))
        p = self.params
        elec_xy = _to_jax(self.implant.electrode_coords().float())
        lam = jnp.float32(p.axlambda)
        b_a = jax.nn.sigmoid(p.sig_slope * (A - p.a50))
        qmax = jnp.max(Q)
        b_q = jnp.where(qmax > 0, jnp.clip(Q / (qmax + 1e-12), 0.0, 1.0), Q)
        shape = self.topography.grid_shape
        maps = [render(p, self.geom, A, sigma, b, elec_xy, lam).reshape(shape) for b in (b_a, b_q)]
        return tuple(_to_torch(m, self.device) for m in maps)
