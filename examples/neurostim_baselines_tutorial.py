"""Baseline suite on the canonical random-LDS problem: the "ARIMA test".

Four regimes (``canonical.REGIMES``), each built so that one family of
methods *should* win or tie:

    A  LTI + Gaussian, one known patient        -> LQG is optimal
    B  unknown LTI patient (random θ)           -> online identification is hard to beat
    C  smooth nonlinear + drift                 -> NMPC / Koopman / MPPI
    D  partial obs + heterogeneity + multimodal -> where learned / generative methods get their chance

Every method sees the same held-out patients, targets and noise. Privileged
methods (``*``) read the true state and model; they are ceilings, not
candidates. See docs/neurostim-baselines.md.

Run: ``uv run python examples/neurostim_baselines_tutorial.py`` (~1.5 h on 4
CPU cores, mostly training the learned methods), ``--quick`` for a smoke run.
"""

# %%
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentionaut.neurostim.canonical import (  # noqa: E402
    AdaptiveController,
    OracleController,
    Population,
    PopulationController,
    ZeroController,
    evaluate,
    regime,
    run_episode,
)
from sentionaut.neurostim.canonical_control import (  # noqa: E402
    DeePCController,
    HInfController,
    ILQRController,
    KoopmanController,
    MPPIController,
    PIController,
)
from sentionaut.neurostim.canonical_learned import (  # noqa: E402
    JEPAConfig,
    SACConfig,
    TDMPCConfig,
    chunk_dataset,
    collect_demos,
    train_bc,
    train_diffusion,
    train_jepa,
    train_sac,
    train_tdmpc,
)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--outdir", type=Path, default=Path("docs/assets/neurostim"))
parser.add_argument("--quick", action="store_true")
parser.add_argument("--regimes", default="ABCD")
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="training seed of the learned methods and demonstrations (evaluation patients stay fixed)",
)
args, _ = parser.parse_known_args()
args.outdir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
Q = args.quick
N_PATIENTS = 4 if Q else 20
# Training-data seeds must never coincide with the evaluation-patient seeds
# (evaluate() uses 1000, 1001, ...): a shared seed means a shared patient.
DEMO_SEED = 50_000

# Display name, family, constructor. Learned methods are trained per regime below.
CLASSICAL = [
    ("zero", "reference", ZeroController),
    ("Oracle LQG *", "ceiling", OracleController),
    ("iLQR-NMPC *", "ceiling", ILQRController),
    ("MPPI *", "sampling", MPPIController),
    ("LQG (nominal)", "classical", PopulationController),
    ("PI", "classical", PIController),
    ("H∞ robust", "robust", HInfController),
    ("Adaptive MPC (RLS)", "adaptive", AdaptiveController),
    ("DeePC", "data-driven", DeePCController),
    ("Koopman-MPC", "learned dynamics", KoopmanController),
]
LEARNED = ["SAC", "TD-MPC2-style", "JEPA-style + MPPI", "Diffusion policy", "BC (MSE)"]
FAMILY = {
    "SAC": "model-free RL",
    "TD-MPC2-style": "model-based RL",
    "JEPA-style + MPPI": "representation",
    "Diffusion policy": "generative",
    "BC (MSE)": "imitation (control)",
}


def budgets():
    if Q:
        return (
            SACConfig(steps=4000, start=1000),
            TDMPCConfig(steps=4000, start=1000),
            JEPAConfig(episodes=200, steps=200, probe_steps=200),
            64,
            300,
            300,
        )
    sd = args.seed
    return SACConfig(seed=sd), TDMPCConfig(seed=sd), JEPAConfig(seed=sd), 256, 5000, 15000


def save(fig, name):
    fig.savefig(args.outdir / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.outdir / name}")


# %% [1] Run everything ----------------------------------------------------
results: dict[str, dict[str, dict]] = {}
demos_by_regime, policies_D = {}, {}
for R in args.regimes:
    cfg = regime(R)
    if Q:
        cfg = regime(R, horizon=80)
    results[R] = {}
    print(f"\n=== regime {R} ===", flush=True)
    for name, _fam, make in CLASSICAL:
        t = time.time()
        r = evaluate(cfg, make, n_patients=N_PATIENTS)
        results[R][name] = r
        print(
            f"  {name:22s} median {r['ss_median']:.4f}  fail {r['fail_rate']:.2f}  ({time.time() - t:.0f}s)",
            flush=True,
        )
    sac_c, td_c, jepa_c, n_demo, bc_steps, diff_steps = budgets()
    t = time.time()
    demos = collect_demos(cfg, n_demo, seed=DEMO_SEED + args.seed)
    demos_by_regime[R] = demos
    print(
        f"  demonstrations from the MPPI teacher: {demos[0].shape[0]} episodes ({time.time() - t:.0f}s)"
    )
    trainers = {
        "SAC": lambda: train_sac(cfg, sac_c, verbose=False),
        "TD-MPC2-style": lambda: train_tdmpc(cfg, td_c, verbose=False),
        "JEPA-style + MPPI": lambda: train_jepa(cfg, jepa_c, verbose=False),
        "Diffusion policy": lambda: train_diffusion(
            cfg, demos, steps=diff_steps, verbose=False, seed=args.seed
        ),
        "BC (MSE)": lambda: train_bc(cfg, demos, steps=bc_steps, verbose=False, seed=args.seed),
    }
    for name, train in trainers.items():
        t = time.time()
        ctrl = train()
        if R == "D":
            policies_D[name] = ctrl
        r = evaluate(cfg, lambda c=ctrl: c, n_patients=N_PATIENTS)
        results[R][name] = r
        print(
            f"  {name:22s} median {r['ss_median']:.4f}  fail {r['fail_rate']:.2f}  ({time.time() - t:.0f}s incl. training)",
            flush=True,
        )

# %% [2] Results table ------------------------------------------------------
methods = [n for n, _, _ in CLASSICAL] + LEARNED
family = {n: f for n, f, _ in CLASSICAL} | FAMILY
regs = list(results)
lines = [
    "| method | family | " + " | ".join(f"regime {R}" for R in regs) + " |",
    "| --- | --- | " + " | ".join("---:" for _ in regs) + " |",
]
for mth in methods:
    cells = []
    for R in regs:
        r = results[R][mth]
        best = min(results[R][k]["ss_median"] for k in methods if k != "zero")
        mark = "**" if r["ss_median"] <= 1.1 * best else ""
        fail = f" ({r['fail_rate']:.0%} fail)" if r["fail_rate"] > 0 else ""
        cells.append(f"{mark}{r['ss_median']:.3f}{mark}{fail}")
    lines.append(f"| {mth} | {family[mth]} | " + " | ".join(cells) + " |")
table = "\n".join(lines)
(args.outdir / "results_baselines.md").write_text(table + "\n")
(args.outdir / "results_baselines.json").write_text(
    json.dumps(
        {
            R: {k: {kk: vv for kk, vv in v.items() if kk != "error_curve"} for k, v in d.items()}
            for R, d in results.items()
        },
        indent=1,
    )
)
print("\n" + table)

# %% [3] Heatmap: error relative to the best privileged controller ----------
ceil = {
    R: min(results[R][k]["ss_median"] for k in ("Oracle LQG *", "iLQR-NMPC *", "MPPI *"))
    for R in regs
}
M = np.array([[np.log10(results[R][mth]["ss_median"] / ceil[R]) for R in regs] for mth in methods])
fig, ax = plt.subplots(figsize=(2.2 + 1.3 * len(regs), 0.42 * len(methods) + 1))
im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(regs)), [f"{R}" for R in regs])
ax.set_yticks(range(len(methods)), methods)
for i in range(len(methods)):
    for j in range(len(regs)):
        ax.text(j, i, f"{10 ** M[i, j]:.1f}×", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="log10(error / best privileged)")
ax.set_title("Median error relative to the best privileged controller", fontsize=9)
save(fig, "b1_heatmap.png")

# %% [4] Why regime D is multimodal, and what BC does with it --------------
if "D" in demos_by_regime and policies_D:
    S, A = demos_by_regime["D"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    late = A[:, 40:].reshape(-1, A.shape[-1]).numpy()
    axes[0].hist(late[:, 0], bins=60, color="0.3")
    axes[0].set_title("teacher's electrode-1 stimulation (t > 40)", fontsize=9)
    axes[0].set_xlabel("u₁")
    # One held-out state: many diffusion samples vs the single BC output.
    feats, _ = chunk_dataset(S[-1:], A[-1:], 8)
    f = feats[60:61]
    diff = policies_D["Diffusion policy"].act_fn
    samples = diff(f.expand(400, -1)).reshape(400, -1, A.shape[-1])[:, 0].numpy()
    bc = policies_D["BC (MSE)"].act_fn(f).detach().reshape(-1, A.shape[-1])[0].numpy()
    axes[1].scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.4, label="diffusion samples")
    axes[1].scatter([bc[0]], [bc[1]], c="r", marker="x", s=120, label="BC (MSE) output")
    axes[1].set_xlabel("u₁")
    axes[1].set_ylabel("u₂")
    axes[1].legend(fontsize=8)
    axes[1].set_title("same history: action distribution", fontsize=9)
    cfg = regime("D")
    pop = Population(cfg)
    for name, make in [
        ("iLQR-NMPC *", ILQRController),
        ("MPPI *", MPPIController),
        ("Adaptive MPC (RLS)", AdaptiveController),
        ("Diffusion policy", lambda: policies_D["Diffusion policy"]),
        ("BC (MSE)", lambda: policies_D["BC (MSE)"]),
    ]:
        curves = []
        for i in range(8 if not Q else 2):
            rng = np.random.default_rng(1000 + i)
            pat = pop.sample(rng)
            z = pop.targets[rng.integers(len(pop.targets))]
            curves.append(run_episode(pop, pat, make(), z, rng)[0])
        axes[2].plot(np.median(curves, 0), label=name, lw=1)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("time step")
    axes[2].set_ylabel("median ‖Dx − z*‖²")
    axes[2].legend(fontsize=7)
    axes[2].set_title("regime D over time", fontsize=9)
    save(fig, "b2_multimodal.png")

print(
    "\n* = privileged (true state and model). Bold = within 10% of the best non-reference method."
)
