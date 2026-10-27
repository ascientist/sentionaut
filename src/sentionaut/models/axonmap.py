"""``BiphasicAxonMapTorch``: GPU reduction parity-faithful to pulse2percept.

I(p) = max_q [ sum_e F_bright_e * exp(-||q - x_e||^2 / (2 rho^2 F_size_e))
               * sens(q)^(1/F_streak_e) ]
with sens(q) = exp(-d_soma(q)^2 / (2 axlambda^2)), thresholded by thresh_percept.
Temporal state uses ``FadingTemporal`` on the spatial drive field.

Docs: https://ascientist.github.io/sentionaut/models/axonmap/
"""

from __future__ import annotations

import torch

from ..core.base import Action, Implant, PerceptModel, Pose, State
from ..topography.axon_map import AxonMapTopography
from . import effects
from .fading import FadingTemporalTorch


class BiphasicAxonMapTorch(PerceptModel):
    def __init__(
        self,
        rho: float = 200.0,
        axlambda: float = 500.0,
        thresh_percept: float = 0.0,
        effect_params: effects.EffectParams = effects.DEFAULTS,
        dt_ms: float = 20.0,
        fade_tau_ms: float = 100.0,
        max_percept: float | None = None,
    ):
        super().__init__()
        self.rho = rho
        self.axlambda = axlambda
        self.thresh_percept = thresh_percept
        self.effects = effect_params
        self.dt_ms = dt_ms
        self.fade_tau_ms = fade_tau_ms
        self.max_percept = max_percept
        self.fading = FadingTemporalTorch(tau_ms=fade_tau_ms, thresh_percept=thresh_percept)

    def build(self, implant: Implant, topography: AxonMapTopography) -> "BiphasicAxonMapTorch":
        self.implant = implant
        self.topography = topography
        self._built = True
        return self

    def spatial_forward(self, action: Action) -> torch.Tensor:
        topo = self.topography
        device = topo.coords.device
        action = action.to(device)
        amp = action.amp
        active = amp != 0
        if active.sum() == 0:
            return torch.zeros(topo.grid_shape, device=device, dtype=topo.coords.dtype)

        freq = action.freq[active]
        pdur = action.phase_dur[active]
        amp_a = amp[active]
        elec_xy = self.implant.electrode_coords(action.pose)[active]

        rho = action.rho if action.rho is not None else self.rho
        axlambda = action.axlambda if action.axlambda is not None else self.axlambda
        rho = torch.as_tensor(rho, device=device, dtype=topo.coords.dtype)
        axlambda = torch.as_tensor(axlambda, device=device, dtype=topo.coords.dtype)

        fb = effects.f_bright(freq, amp_a, pdur, self.effects)
        fs = effects.f_size(freq, amp_a, pdur, rho, self.effects)
        fst = effects.f_streak(freq, amp_a, pdur, axlambda, self.effects)

        coords = topo.coords
        d_soma = topo.d_soma
        mask = topo.mask
        sens = torch.exp(-(d_soma**2) / (2.0 * axlambda**2))

        elec = elec_xy[:, None, None, :]
        d2 = ((coords[None, ...] - elec) ** 2).sum(dim=-1)
        two_rho2 = 2.0 * rho**2
        spatial = torch.exp(-d2 / (two_rho2 * fs[:, None, None]))
        streak = sens.pow(1.0 / fst[:, None, None]) * mask[None, ...]
        acc = (fb[:, None, None] * spatial * streak).sum(dim=0)
        intensity = acc.max(dim=1).values
        intensity = torch.where(
            intensity > self.thresh_percept, intensity, torch.zeros_like(intensity)
        )
        out = intensity.reshape(topo.grid_shape)
        if self.max_percept is not None:
            out = torch.clamp(out, max=self.max_percept)
        return out

    def spatial_forward_batch(
        self,
        amp: torch.Tensor,
        freq: float | torch.Tensor = 20.0,
        phase_dur: float | torch.Tensor = 0.45,
        rho: float | torch.Tensor | None = None,
        axlambda: float | torch.Tensor | None = None,
        chunk: int = 8,
        cutoff: float | None = None,
        pose: Pose | None = None,
    ) -> torch.Tensor:
        """Batched, differentiable drive: ``amp (B, E)`` -> ``(B, H, W)``.

        Same arithmetic as ``spatial_forward``. Electrodes with ``amp == 0`` are
        dropped (pulse2percept skips them), so brightness jumps from 0 to
        ``a3 * freq`` when an electrode turns on; that step has no gradient.
        ``freq`` and ``phase_dur`` are scalars or ``(B, E)``. The implant pose
        is ``pose`` (default: as built).

        ``cutoff`` (in Gaussian widths) keeps, per electrode, only the axon
        segments within ``cutoff * rho * sqrt(F_size_max)``; the dropped terms
        are below ``exp(-cutoff**2 / 2)`` of the electrode's peak. ``None`` is
        exact. It needs scalar ``freq`` and ``phase_dur``, the build ``rho`` and
        no ``pose``.
        """
        if cutoff is not None:
            if pose is not None:
                raise ValueError("cutoff does not support a pose")
            return self._spatial_forward_sparse(amp, freq, phase_dur, axlambda, cutoff)
        topo = self.topography
        coords, d_soma, mask = topo.coords, topo.d_soma, topo.mask
        dtype, device = coords.dtype, coords.device
        amp = amp.to(device=device, dtype=dtype)
        n_b, n_e = amp.shape
        freq = torch.as_tensor(freq, device=device, dtype=dtype).expand(n_b, n_e)
        pdur = torch.as_tensor(phase_dur, device=device, dtype=dtype).expand(n_b, n_e)
        rho = torch.as_tensor(self.rho if rho is None else rho, device=device, dtype=dtype)
        axl = torch.as_tensor(
            self.axlambda if axlambda is None else axlambda, device=device, dtype=dtype
        )
        shared_pdur = torch.as_tensor(phase_dur).numel() == 1
        on = (amp > 0).to(dtype)
        fb = effects.f_bright(freq, amp, pdur, self.effects) * on
        fs = effects.f_size(freq, amp, pdur, rho, self.effects)
        fst = effects.f_streak(freq, amp, pdur, axl, self.effects)
        sens = torch.exp(-(d_soma**2) / (2.0 * axl**2))
        elec = self.implant.electrode_coords(pose).to(device=device, dtype=dtype)
        acc = torch.zeros(n_b, *d_soma.shape, device=device, dtype=dtype)
        for s in range(0, n_e, chunk):
            e = slice(s, s + chunk)
            d2 = ((coords[None] - elec[e, None, None, :]) ** 2).sum(-1)  # (c, P, S)
            spatial = torch.exp(-d2[None] / (2.0 * rho**2 * fs[:, e, None, None]))
            if shared_pdur:  # the streak does not depend on amplitude: share it over the batch
                streak = (sens[None] ** (1.0 / fst[0, e, None, None]) * mask)[None]
            else:
                streak = sens[None, None] ** (1.0 / fst[:, e, None, None]) * mask
            acc = acc + (fb[:, e, None, None] * spatial * streak).sum(1)
        intensity = acc.max(dim=-1).values
        intensity = torch.where(
            intensity > self.thresh_percept, intensity, torch.zeros_like(intensity)
        )
        out = intensity.reshape(n_b, *topo.grid_shape)
        if self.max_percept is not None:
            out = torch.clamp(out, max=self.max_percept)
        return out

    def _sparse_index(self, cutoff: float, max_amp: float = 3.0, pdur: float = 0.45):
        """Per-electrode flat (pixel, segment) indices within the cutoff radius, padded."""
        key = (cutoff, max_amp, pdur, float(self.rho))
        if getattr(self, "_sparse_cache", (None,))[0] == key:
            return self._sparse_cache[1]
        topo = self.topography
        coords, mask = topo.coords, topo.mask
        p = self.effects
        fs_max = max(p.a5 * max_amp * (p.a1 + p.a0 * pdur) + p.a6, p.min_rho**2 / self.rho**2)
        r2 = (cutoff * self.rho) ** 2 * fs_max
        elec = self.implant.electrode_coords().to(coords)
        flat = coords.reshape(-1, 2)
        keep = [torch.nonzero(((flat - xy) ** 2).sum(-1) <= r2).flatten() for xy in elec]
        k = max(1, max(len(i) for i in keep))
        pad = flat.shape[0]  # dummy slot past the end
        idx = torch.full((len(keep), k), pad, dtype=torch.long, device=coords.device)
        for j, i in enumerate(keep):
            idx[j, : len(i)] = i
        valid = idx < pad
        safe = idx.clamp(max=pad - 1)
        d2 = ((flat[safe] - elec[:, None, :]) ** 2).sum(-1)
        dsoma = topo.d_soma.reshape(-1)[safe]
        m = mask.reshape(-1)[safe] * valid
        cache = {"idx": idx, "d2": d2, "dsoma": dsoma, "mask": m, "fs_max": fs_max}
        self._sparse_cache = (key, cache)
        return cache

    def _spatial_forward_sparse(self, amp, freq, phase_dur, axlambda, cutoff):
        topo = self.topography
        dtype, device = topo.coords.dtype, topo.coords.device
        c = self._sparse_index(cutoff, pdur=float(phase_dur))
        amp = amp.to(device=device, dtype=dtype)
        n_b, n_e = amp.shape
        freq = torch.as_tensor(freq, device=device, dtype=dtype)
        pdur = torch.as_tensor(phase_dur, device=device, dtype=dtype)
        axl = torch.as_tensor(
            self.axlambda if axlambda is None else axlambda, device=device, dtype=dtype
        )
        fb = effects.f_bright(freq, amp, pdur, self.effects) * (amp > 0).to(dtype)
        fs = effects.f_size(freq, amp, pdur, self.rho, self.effects)
        fst = effects.f_streak(freq, amp[:1, :1], pdur, axl, self.effects).reshape(-1)[0]
        streak = torch.exp(-(c["dsoma"] ** 2) / (2.0 * axl**2)) ** (1.0 / fst) * c["mask"]
        spatial = torch.exp(-c["d2"][None] / (2.0 * self.rho**2 * fs[..., None]))
        contrib = fb[..., None] * spatial * streak[None]  # (B, E, K)
        n_ps = topo.coords.shape[0] * topo.coords.shape[1]
        acc = torch.zeros(n_b, n_ps + 1, device=device, dtype=dtype)
        acc = acc.scatter_add(1, c["idx"].reshape(1, -1).expand(n_b, -1), contrib.reshape(n_b, -1))
        acc = acc[:, :n_ps].view(n_b, *topo.coords.shape[:2])
        intensity = acc.max(dim=-1).values
        intensity = torch.where(
            intensity > self.thresh_percept, intensity, torch.zeros_like(intensity)
        )
        out = intensity.reshape(n_b, *topo.grid_shape)
        if self.max_percept is not None:
            out = torch.clamp(out, max=self.max_percept)
        return out

    def step_batch(self, brightness: torch.Tensor | None, amp: torch.Tensor, **kw) -> torch.Tensor:
        """One batched fade step: ``(B, H, W)`` brightness and ``(B, E)`` amp -> next."""
        drive = self.spatial_forward_batch(amp, **kw)
        if brightness is None:
            brightness = torch.zeros_like(drive)
        return self.fading.step(brightness, -drive, self.dt_ms)

    def forward(self, action: Action) -> torch.Tensor:
        return self.spatial_forward(action)

    def step(self, state: State | None, action: Action) -> State:
        topo = self.topography
        device = topo.coords.device
        drive = self.spatial_forward(action)
        if state is None:
            B = FadingTemporalTorch.initial_brightness(topo.grid_shape, device, topo.coords.dtype)
        else:
            B = state.image
        # Spatial drive is positive brightness; FadingTemporal expects cathodic (negative) A.
        B_new = self.fading.step(B, -drive, self.dt_ms)
        return State(image=B_new)

    def predict_sequence(
        self, action: Action, n_steps: int, dt_ms: float | None = None
    ) -> torch.Tensor:
        dt = self.dt_ms if dt_ms is None else dt_ms
        state = None
        frames = []
        zero = Action(
            amp=torch.zeros_like(action.amp),
            freq=action.freq,
            phase_dur=action.phase_dur,
            rho=action.rho,
            axlambda=action.axlambda,
            pose=action.pose,
        )
        for step_i in range(n_steps):
            act = action if step_i == 0 else zero
            drive = self.spatial_forward(act)
            if state is None:
                B = FadingTemporalTorch.initial_brightness(
                    self.topography.grid_shape, drive.device, drive.dtype
                )
            else:
                B = state.image
            B_new = self.fading.step(B, -drive, dt)
            state = State(image=B_new)
            frames.append(B_new)
        return torch.stack(frames, dim=0)
