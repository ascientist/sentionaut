# Sentionaut implementation debrief

Running log of non-obvious decisions, sourced parameter values, deferrals, and
delivery outcome. Workspace root: `/Users/jacoblavoie/git_repo/retinawm`; the
Python project lives in the nested `retinawm/` directory and the package is
`sentionaut` (renamed from `retinawm`).

## Environment / versions

- `pulse2percept==0.9.0` kept as-is. Verified during `uv sync` that 0.9.0 ships
  `pulse2percept.models.cortex.ScoreboardModel`, `...cortex.DynaphosModel`, and
  `pulse2percept.topography.Polimeni2006Map`. No version bump required.
- `torch==2.12.1` installed via uv. MPS is available on this M1; device helper
  selects `mps`, falls back to `cpu`.
- Python 3.11, managed entirely through `uv` / `uv run`.

## Sourced parameter values (reconciled against pulse2percept 0.9.0 source)

All effect coefficients and map constants are taken directly from the installed
pulse2percept 0.9.0 source so the torch ports are parity-faithful by construction
(not guessed):

- Biphasic Axon Map effect models (Granley 2021), from
  `pulse2percept/models/granley2021.py`:
  - Bright: `a0=2.095, a1=0.054326, a2=0.1492147, a3=0.0163851, a4=0`.
    `F_bright = a2*(amp*(a1+a0*pdur)) + a3*freq + a4`.
  - Size: `a5=1.0812, a6=-0.35338, min_rho=10`.
    `F_size = max(a5*amp*(a1+a0*pdur) + a6, min_rho^2/rho^2)`.
  - Streak: `a7=0.54, a8=0.21, a9=1.56, min_lambda=10`.
    `F_streak = max(a9 - a7*pdur^a8, min_lambda^2/axlambda^2)`.
  - Reduction (from the jax reference `predict_one_point_jax`):
    `I(p)=max_q sum_e F_bright_e * exp(-||q-x_e||^2/(2 rho^2 F_size_e)) * sens(q)^(1/F_streak_e)`
    with `sens(q)=exp(-d_soma(q)^2/(2 axlambda^2))`, then thresholded by
    `thresh_percept`.
- Polimeni2006Map (cortical), from `pulse2percept/topography/cortex.py`:
  - Scoreboard/default uses `k=15, a=0.69, b=80, alpha1=1, alpha2=0.333, alpha3=0.25`.
  - Dynaphos overrides the map to `a=0.75, k=17.3, b=120, alpha1=0.95`
    (`pulse2percept/models/cortex/dynaphos.py`).
  - `left_offset=-20000`, `split_map=True`, hemisphere boundary `left_offset/2`.
- Dynaphos constants, from `DynaphosModel.get_default_params`:
  `dt=20, tau_act=111.111111, rheobase=23.9, tau_trace=1.96765520573e6,
  kappa_trace=13.95528162, excitability=675, sig_slope=19152642.500946816,
  a50=1.057631326853325e-07, a_thr=9.141886000943878e-08, freq=300, p_dur=0.170`.
  These are mirrored in `src/sentionaut/config/params.yaml`.

## Implementation decisions

- Axon map topography stores RAW per-pixel axon-point coords + recovered
  `d_soma` (not the axlambda-baked sensitivity), so `rho`/`axlambda` are runtime
  differentiable inputs. `d_soma` is recovered from the cached sensitivity via
  `d_soma = sqrt(-2*axlambda_build^2*ln(sensitivity))`. Caveat: the *set* of
  retained segments is fixed at build time (trimmed by `min_ax_sensitivity` at
  the build axlambda), so making axlambda much larger than the build value at
  runtime cannot resurrect pruned segments. Parity tests use the same axlambda
  as the build, so this is exact for the tested regime. To revisit: rebuild the
  topography with a larger build-time axlambda / smaller `min_ax_sensitivity`.
- Cortical topography is a faithful torch re-implementation of `dva_to_v1/v2/v3`
  (complex-log mapping). Computed in float64 on CPU at build time for numerical
  fidelity, then cached as tensors moved to the active device. Parity validated
  against pulse2percept's own `Polimeni2006Map` in tests.

## Parity tolerances (see tests/test_parity.py)

Measured on small subsampled grids (CPU, float32), max-abs error vs
pulse2percept 0.9.0 `predict_percept`:

- Axon map (ArgusII, 0.5 dva step): ~6e-8 (effectively exact).
- Cortical Scoreboard (Orion, 0.5 dva step, rho=1000): ~3e-5.
- Cortical Dynaphos (Orion, 0.25 dva step): ~2e-7 per frame after aligning the
  leading baseline frame (see note). Cortical map port vs p2p Polimeni2006Map:
  ~0.01 micron (float32 rounding); v1_to_dva inverse ~1e-6 dva.

Test thresholds are set conservatively at `max_abs < 1e-3` (relative to percept
peaks of order 0.1-1.0 for retinal/Dynaphos and ~30 for scoreboard, so
scoreboard uses `< 1e-2`). These leave generous headroom over the measured
errors while still catching real regressions.

Dynaphos frame alignment: pulse2percept's `t_percept` includes `t=0`, whose
frame is the pre-integration zero baseline. `DynaphosTorch.predict_sequence`
performs one integration step before emitting its first frame, so torch frame
`i` corresponds to p2p frame `i+1`. The parity test compares against
`ref[..., 1:]` accordingly.

## Deferrals / stubs

- NeuropythyMap topography: NOT used. It requires the optional `neuropythy`
  package plus subject-specific MRI-derived templates, which is out of scope for
  an analytical, self-contained GPU port. Only the analytical Polimeni2006Map is
  ported. To revisit: add a topography adapter that wraps a prebuilt
  `pulse2percept.topography.NeuropythyMap` if neuropythy + data are installed.
- Neuralink cortical implant: NOT included in the implant registry. The plan
  explicitly lists Orion/Cortivis/ICVP (cortical) + ArgusII/AlphaIMS (retinal);
  Neuralink is left out to match the plan scope. It can be added trivially via
  `pulse2percept.implants.cortex.Neuralink` in the registry if needed.
- Learned world model is only smoke-proven locally (single-batch train step +
  single-batch inference), per the plan. Real training/ablation is cluster-side.

## GitHub delivery

- SUCCESS. `gh` was authenticated as `ascientist` (ssh git protocol). Created a
  new PRIVATE repo and pushed `main`:
  - URL: https://github.com/ascientist/sentionaut
  - Visibility: PRIVATE, default branch `main`, single commit pushed.
- No interactive prompts were hit. Caches/datasets (`data/`, `*.h5`,
  `axons.pickle`, `.venv`) are gitignored; the three rendered animations under
  `artifacts/` are tracked as the local deliverable.

## Final local validation

- `uv run pytest -m "not slow"`: 15 passed, 1 deselected (the slow CPU/MPS
  benchmark).
- `ruff check` + `ruff format --check`: clean.
- Three animations rendered on MPS to `artifacts/{axonmap,scoreboard,dynaphos}`
  (gif + mp4).
- End-to-end scale dry-run (generate multi-config dataset -> train 1 epoch ->
  ablate shared vs specialist) executed on CPU without error.

## Notes for resuming the cluster work

- Large dataset generation / training / ablation run via `scripts/*.sh` (adapt
  the `--account`, module, and venv placeholder lines for your DRAC allocation).
- The DEBRIEF in the repo (`retinawm/DEBRIEF.md`) is a committed copy of this
  file; this outer copy at the workspace root is the canonical working log.
