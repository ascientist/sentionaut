"""Canonical formulation tutorial: control of a random, drifting linear dynamical system.

    θ ~ P_Θ (patient)   φ_{t+1} = ρφ_t + η_t (drift)   r ~ P_R (image), z* = E(r)
    x_{t+1} = A_θ x_t + (B_θ + φ_t) u_t + w_t,   o_t = C x_t + v_t,   z_t = D x_t
    min_π  E[ Σ_t ‖D x_t − z*‖² + λ‖u_t‖² ]      π sees (z*, o_≤t, u_<t) only

Four experiments, each isolating one source of difficulty (see
https://ascientist.github.io/sentionaut/neurostim-formulation/):

1. The benchmark ladder L0 → L3: known patient, random patients, partial
   observation, drift.
2. Inter-patient variability σ_B: when does "one controller for everybody"
   stop working?
3. The cost of learning: error over time while the adaptive controller
   identifies a new patient.
4. Drift vs memory: the forgetting factor trades tracking lag against noise.

Run: ``uv run python examples/neurostim_canonical_tutorial.py`` (~6 min),
``--quick`` for a smoke run.
"""

# %%
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sentionaut.neurostim.canonical import (  # noqa: E402
    LEVELS,
    AdaptiveController,
    OracleController,
    PopulationController,
    ZeroController,
    evaluate,
    level,
)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--outdir", type=Path, default=Path("docs/assets/neurostim"))
parser.add_argument("--quick", action="store_true")
args, _ = parser.parse_known_args()
args.outdir.mkdir(parents=True, exist_ok=True)
N = 6 if args.quick else 30  # held-out patients per configuration
CONTROLLERS = {
    "zero": ZeroController,
    "population LQG": PopulationController,
    "adaptive (RLS + CE)": AdaptiveController,
    "oracle LQG *": OracleController,
}
COLORS = {"population LQG": "C3", "adaptive (RLS + CE)": "C0", "oracle LQG *": "k"}


def save(fig, name):
    fig.savefig(args.outdir / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.outdir / name}")


def quick(cfg):
    return replace(cfg, horizon=min(cfg.horizon, 200)) if args.quick else cfg


# %% [1] The ladder --------------------------------------------------------
print("\n[1] Benchmark ladder (steady-state percept error ‖Dx − z*‖², last half)")
rows = []
for name in LEVELS:
    cfg = quick(level(name))
    for cname, make in CONTROLLERS.items():
        r = evaluate(cfg, make, n_patients=N)
        rows.append((name, cname, r))
        print(
            f"  {name:26s} {cname:20s} median {r['ss_median']:.4f}  mean {r['ss_error']:8.4f}"
            f"  fail {r['fail_rate']:.2f}"
        )
lines = [
    "| level | controller | median error | mean error | failure rate |",
    "| --- | --- | ---: | ---: | ---: |",
]
for name, cname, r in rows:
    lines.append(
        f"| {name} | {cname} | {r['ss_median']:.4f} | {r['ss_error']:.4f} | {r['fail_rate']:.2f} |"
    )
table = "\n".join(lines)
(args.outdir / "results_canonical.md").write_text(table + "\n")
print("\n" + table)

# %% [2] How different must patients be before personalisation pays? -------
print("\n[2] Inter-patient variability sweep (L1)")
sigmas = [0.0, 0.1, 0.2, 0.35, 0.5, 1.0]
sweep = {c: [] for c in COLORS}
for s in sigmas:
    cfg = quick(level("L1 random patients", sigma_B=s))
    for cname in COLORS:
        sweep[cname].append(evaluate(cfg, CONTROLLERS[cname], n_patients=N)["ss_median"])
    print(f"  σ_B = {s:4.2f}  " + "  ".join(f"{c} {sweep[c][-1]:.4f}" for c in COLORS))
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for cname, ys in sweep.items():
    ax.plot(sigmas, ys, "o-", color=COLORS[cname], label=cname)
ax.axhline(
    evaluate(quick(level("L1 random patients")), ZeroController, n_patients=N)["ss_median"],
    color="0.6",
    ls="--",
    label="no stimulation",
)
ax.set_yscale("log")
ax.set_xlabel(r"inter-patient spread $\sigma_B$ (relative to $\bar B$)")
ax.set_ylabel("median steady-state error")
ax.legend(fontsize=8)
ax.set_title("One controller for everybody vs identify each patient", fontsize=9)
save(fig, "c1_patient_spread.png")

# %% [3] The cost of learning ------------------------------------------------
print("\n[3] Error over time on new patients (L1): the identification transient")
cfg = quick(level("L1 random patients"))
fig, ax = plt.subplots(figsize=(6, 3.6))
for cname in COLORS:
    curve = evaluate(cfg, CONTROLLERS[cname], n_patients=N)["error_curve"]
    ax.plot(curve, color=COLORS[cname], label=cname, lw=1.2)
ax.axvline(30, color="0.5", ls=":", lw=1, label="adaptive: probing ends")
ax.set_yscale("log")
ax.set_xlabel("time step")
ax.set_ylabel("median ‖Dx − z*‖² across patients")
ax.legend(fontsize=8)
save(fig, "c2_learning_curve.png")

# %% [4] Drift: memory vs lag ----------------------------------------------
print("\n[4] Drift x forgetting factor (full observation, adaptive controller)")
drifts = [0.003, 0.01, 0.03]
forgets = [0.9, 0.95, 0.98, 0.99, 0.995, 1.0]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for d in drifts:
    cfg = quick(level("L3 + drift", observe="full", drift=d))
    ys = [
        evaluate(cfg, lambda f=f: AdaptiveController(forget=f), n_patients=N)["ss_median"]
        for f in forgets
    ]
    best = forgets[int(np.argmin(ys))]
    print(
        f"  drift {d:.3f}: "
        + "  ".join(f"λ={f}:{y:.4f}" for f, y in zip(forgets, ys))
        + f"  best λ={best}"
    )
    ax.plot([1 / (1 - f) if f < 1 else 2000 for f in forgets], ys, "o-", label=f"drift η = {d}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"memory $1/(1-\lambda_f)$ steps  (rightmost: no forgetting)")
ax.set_ylabel("median steady-state error")
ax.legend(fontsize=8)
ax.set_title("Faster drift needs shorter memory", fontsize=9)
save(fig, "c3_drift_memory.png")
print("\n* = privileged (knows θ and φ_t); the optimum for a known linear-Gaussian system.")
