"""NeuroStim tutorial: the toy closed-loop neurostimulation CMDP and its baselines.

Run as a script (``uv run python examples/neurostim_tutorial.py``) or cell by
cell (``# %%`` markers work in VS Code / Jupyter). Figures and a results table
are written to ``--outdir`` (default ``docs/assets/neurostim``).

    x_{t+1} = A x_t - alpha x_t^3 + B diag(1/(1+lambda h_t)) tanh(a_{t-delta}) + w_t
    h_{t+1} = rho h_t + kappa |a_t|
    o_t     = C x_t + eps_t
    r_t     = -||x_{t+1} - x*||^2 - w_E ||a_t||^2 - w_D ||a_t - a_{t-1}||^2
    s.t.      sum_i |a_t^i| <= q_max,   max_i h_t^i <= h_max
"""

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentionaut.neurostim import PRESETS, make  # noqa: E402
from sentionaut.neurostim.baselines import (  # noqa: E402
    NominalPI,
    OnlineSysID,
    OracleGreedy,
    SingleElectrodeBandit,
    ZeroPolicy,
    default_baselines,
    evaluate,
    rollout,
)
from sentionaut.neurostim.imitation import DAggerConfig, train_dagger  # noqa: E402
from sentionaut.neurostim.ppo import PPOConfig, train_ppo  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--outdir", type=Path, default=Path("docs/assets/neurostim"))
parser.add_argument("--episodes", type=int, default=30, help="held-out patients per eval")
parser.add_argument("--quick", action="store_true", help="tiny training budgets (smoke run)")
args, _ = parser.parse_known_args()
args.outdir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
LATENT = ["$x_1$", "$x_2$", "$x_3$"]
ELEC = ["E1", "E2", "E3", "E4"]


def save(fig, name):
    path = args.outdir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# %% [1] One patient, probed open loop -----------------------------------
# Canonical env with the design-note B (not a random patient) and no noise, so
# the only thing moving x is the stimulation we choose.
print("\n[1] Probing electrodes one at a time (open loop)")
env = make("NeuroStim-v0", randomize_B=False, process_noise=0.0, obs_noise=0.0, x0_std=0.0)
obs, info = env.reset(seed=0)
print("  B (recruitment, x stim_gain) =\n", np.round(info["B"], 2))
print("  A (intrinsic dynamics)       =\n", np.round(info["A"], 2))

fig, axes = plt.subplots(2, 4, figsize=(13, 5), sharex=True, sharey="row")
T_on, T = 40, 80
for e in range(4):
    env.reset(seed=0)
    xs, hs = [], []
    for t in range(T):
        a = np.eye(4)[e] * 0.8 if t < T_on else np.zeros(4)
        info = env.step(a)[4]
        xs.append(info["x"]), hs.append(info["h"][e])
    xs = np.array(xs)
    for k in range(3):
        axes[0, e].plot(xs[:, k], label=LATENT[k])
    axes[0, e].axvspan(0, T_on, color="0.9")
    axes[0, e].set_title(f"stimulate {ELEC[e]} at 0.8")
    axes[1, e].plot(hs, color="k")
    axes[1, e].axhline(env.cfg.h_max, ls="--", color="r", lw=1)
    axes[1, e].set_xlabel("step")
axes[0, 0].set_ylabel("latent state")
axes[1, 0].set_ylabel("adaptation $h$")
axes[0, 0].legend(fontsize=8)
fig.suptitle(
    "Each electrode moves all latents (overlapping B); sustained stimulation builds "
    "adaptation, so the response sags while stimulation is still on"
)
save(fig, "01_probe.png")

# Adaptation in one number: drive delivered by E3 after 40 steps vs step 1.
c = env.cfg
h_ss = c.adapt_rate / (1 - c.adapt_decay) * 0.8
print(f"  steady-state h at |a|=0.8: {h_ss:.2f} -> gain 1/(1+lambda h) = {1 / (1 + 2 * h_ss):.2f}")

# %% [2] Baselines across the difficulty ladder -------------------------
print(f"\n[2] Baselines on {len(PRESETS)} presets, {args.episodes} held-out patients each")
rows = []
for name in PRESETS:
    env = make(name)
    for policy in default_baselines():
        m = evaluate(env, policy, episodes=args.episodes)
        rows.append((name, policy.name + (" *" if policy.privileged else ""), m))
        print(
            f"  {name:18s} {policy.name:16s} return {m['return']:8.1f} ± {m['return_se']:5.1f}"
            f"  ss_err {m['ss_error']:6.3f}  viol {m['violation_rate']:5.3f}  charge {m['charge']:.2f}"
        )

# %% [3] What each controller does on one held-out patient ---------------
print("\n[3] One held-out patient on NeuroStim-v0")
env = make("NeuroStim-v0")
policies = [ZeroPolicy(), SingleElectrodeBandit(), NominalPI(), OnlineSysID(), OracleGreedy()]
fig, axes = plt.subplots(3, len(policies), figsize=(16, 7.5), sharex=True, sharey="row")
for j, policy in enumerate(policies):
    tr = rollout(env, policy, seed=1003)
    for k in range(3):
        (ln,) = axes[0, j].plot(tr.x[:, k], lw=1.2, label=LATENT[k])
        axes[0, j].axhline(tr.target[k], color=ln.get_color(), ls=":", lw=1)
    axes[1, j].imshow(
        tr.actions.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
    )
    axes[1, j].set_yticks(range(4), ELEC)
    axes[2, j].plot(tr.h, lw=1)
    axes[2, j].axhline(env.cfg.h_max, ls="--", color="r", lw=1)
    viol = tr.costs.mean()
    axes[0, j].set_title(
        f"{policy.name}\nreturn {tr.rewards.sum():.0f}, viol {viol:.0%}", fontsize=9
    )
    axes[2, j].set_xlabel("step")
axes[0, 0].set_ylabel("latent x (dotted = x*)")
axes[1, 0].set_ylabel("stimulation a")
axes[2, 0].set_ylabel("adaptation h")
axes[0, 0].legend(fontsize=8)
save(fig, "02_controllers.png")

# %% [4] Reinforcement learning ----------------------------------------
# (a) PPO vs PPO-Lagrangian on a single known-population patient: does the
#     learner respect the constraints when they are not in the reward?
# (b) "Universal" PPO trained across random patients: can a history-conditioned
#     policy personalise without an explicit identification step?
# (c) Same, but handed the true B: if (c) works and (b) does not, the
#     bottleneck is identifying the patient, not controlling one.
print("\n[4] PPO baselines")
steps_fixed = 20_000 if args.quick else 300_000
steps_random = 20_000 if args.quick else 800_000
runs = {
    "PPO (fixed patient)": (dict(randomize_B=False), PPOConfig(total_steps=steps_fixed)),
    "PPO-Lagrangian (fixed patient)": (
        dict(randomize_B=False),
        PPOConfig(total_steps=steps_fixed, cost_limit=5.0),
    ),
    "PPO (random patients)": ({}, PPOConfig(total_steps=steps_random, history=16)),
    # Diagnostic: same task, but the policy is *given* the true B_t.
    "PPO + true B (random patients) *": (
        {},
        PPOConfig(total_steps=steps_random, privileged_B=True),
    ),
}
# Non-RL references on the fixed patient (random-patient ones are in [2]).
fixed_env = make("NeuroStim-v0", randomize_B=False)
for ref in (NominalPI(), OnlineSysID(), OracleGreedy()):
    name = ref.name + (" *" if ref.privileged else "")
    rows.append(("v0, fixed B", name, evaluate(fixed_env, ref, episodes=args.episodes)))

logs = {}
for label, (kw, pcfg) in runs.items():
    print(f"  training {label}")
    policy, logs[label] = train_ppo(lambda kw=kw: make("NeuroStim-v0", **kw), pcfg, log_every=50)
    m = evaluate(make("NeuroStim-v0", **kw), policy, episodes=args.episodes)
    rows.append(("v0, fixed B" if kw else "NeuroStim-v0", label, m))
    print(
        f"    eval: return {m['return']:.1f}  ss_err {m['ss_error']:.3f}  viol {m['violation_rate']:.3f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for label, log in logs.items():
    s = np.array([e["step"] for e in log])
    axes[0].plot(s, [e["return"] for e in log], label=label)
    axes[1].plot(s, [e["cost"] for e in log], label=label)
axes[0].set_ylabel("episode return")
axes[1].set_ylabel("constraint-violating steps / episode")
axes[1].axhline(5.0, ls="--", color="r", lw=1, label="budget $d$ = 5")
for ax in axes:
    ax.set_xlabel("environment steps")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}k")
axes[1].legend(fontsize=8)
save(fig, "03_ppo_learning.png")

# %% [5] Supervised imitation (DAgger), no RL ----------------------------
# The student sees the same (o, a_prev) history as PPO. It regresses the
# privileged oracle's action at every step it visits: a dense supervised
# target instead of a scalar return.
# (a) fixed patient: can plain regression recover the oracle?
# (b) random patients: can it identify each patient implicitly?
# (c) random patients plus a fixed 8-step calibration sweep (one +-pulse per
#     electrode) whose responses stay in the input: supervised inference from
#     a designed experiment.
print("\n[5] Supervised imitation (DAgger from the oracle)")
iters = 2 if args.quick else 8
dagger_runs = {
    "DAgger (fixed patient)": (dict(randomize_B=False), DAggerConfig(iterations=iters)),
    "DAgger (random patients)": ({}, DAggerConfig(iterations=iters, episodes_per_iter=120)),
    "DAgger + calibration (random patients)": (
        {},
        DAggerConfig(iterations=iters, episodes_per_iter=120, calibration=True),
    ),
}
if args.quick:
    for _, dcfg in dagger_runs.values():
        dcfg.episodes_per_iter, dcfg.epochs = 4, 2
dagger_logs = {}
for label, (kw, dcfg) in dagger_runs.items():
    print(f"  training {label}")
    policy, dagger_logs[label] = train_dagger(lambda kw=kw: make("NeuroStim-v0", **kw), dcfg)
    m = evaluate(make("NeuroStim-v0", **kw), policy, episodes=args.episodes)
    rows.append(("v0, fixed B" if kw else "NeuroStim-v0", label, m))
    print(
        f"    eval: return {m['return']:.1f}  ss_err {m['ss_error']:.3f}  viol {m['violation_rate']:.3f}"
    )

ref = {(setting, name): m["return"] for setting, name, m in rows}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
panels = [
    ("fixed patient", "v0, fixed B", ["DAgger (fixed patient)"], "PPO-Lagrangian (fixed patient)"),
    (
        "random patients",
        "NeuroStim-v0",
        ["DAgger (random patients)", "DAgger + calibration (random patients)"],
        "PPO (random patients)",
    ),
]
for ax, (title, setting, labels, ppo_label) in zip(axes, panels):
    for label in labels:
        log = dagger_logs[label]
        ax.plot([e["iteration"] for e in log], [e["return"] for e in log], "o-", label=label)
    for name, style in [("oracle greedy *", "k--"), ("online sysID", "k:"), (ppo_label, "r-.")]:
        if (setting, name) in ref:
            ax.axhline(ref[(setting, name)], ls=style[1:], color=style[0], lw=1, label=name)
    ax.axhline(ref.get((setting, "zero"), -406.0), color="0.6", lw=1, label="no stimulation")
    ax.set_title(title)
    ax.set_xlabel("DAgger iteration (0 = behaviour cloning)")
    ax.legend(fontsize=7)
axes[0].set_ylabel("validation return")
save(fig, "04_dagger.png")

# %% [6] Results table --------------------------------------------------
lines = [
    "| setting | policy | return | steady-state error | violation rate | charge |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for setting, name, m in rows:
    lines.append(
        f"| {setting} | {name} | {m['return']:.1f} ± {m['return_se']:.1f} | {m['ss_error']:.3f} "
        f"| {m['violation_rate']:.3f} | {m['charge']:.2f} |"
    )
table = "\n".join(lines)
(args.outdir / "results.md").write_text(table + "\n")
print("\n" + table)
print("\n* = privileged (reads the latent state); not a deployable policy.")
