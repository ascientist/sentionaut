"""NeuroStim-Percept tutorial: learn stimulation for a target image through a world model.

    target y ─► π ─► a_{1:T} ─► brain: x_{t+1} = A x - αx³ + B g(a, h) + w ─► percept D(x_t)
                          └──► world model W (GRU, learned from stimulation → percept data)

The brain is the NeuroStim dynamics lifted to the 16-D code space of a frozen
MNIST autoencoder, with 24 electrodes and adaptation. Its percept is the
decoded image ``D(x_t)``. The policy is trained **only through the learned
world model**, by backprop on ``Σ_{t ≥ 10} ||D(ẑ_t) − y||²`` (no RL). It is
scored on the true brain: pixel MSE, and the accuracy of a frozen classifier
on the percept.

Run: ``uv run python examples/neurostim_percept_tutorial.py`` (about 45 min on
4 CPU cores; ``--quick`` for a smoke run). Downloads MNIST to ``data/mnist``.

Docs: https://ascientist.github.io/sentionaut/neurostim-percept/
"""

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from sentionaut.neurostim.percept import (  # noqa: E402
    Brain,
    BrainConfig,
    collect,
    evaluate,
    exploration_actions,
    load_mnist,
    project_charge,
    train_autoencoder,
    train_classifier,
    train_policy,
    train_world_model,
    true_simulator,
    wm_simulator,
)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--outdir", type=Path, default=Path("docs/assets/neurostim"))
parser.add_argument("--quick", action="store_true", help="tiny budgets (smoke run)")
args, _ = parser.parse_known_args()
args.outdir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
Q = args.quick
N_EPISODES = 400 if Q else 6000  # exploration episodes, fixed patient
N_EPISODES_RAND = 800 if Q else 20000  # random patients: one new patient per episode
WM_STEPS = 100 if Q else 2000
PI_STEPS = 50 if Q else 1000
WM_STEPS_RAND = 100 if Q else 3000  # one model for all patients: needs longer
PI_STEPS_RAND = 50 if Q else 2000
N_TEST = 200 if Q else 1000
rows = []


def save(fig, name):
    path = args.outdir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def report(setting, name, r):
    rows.append((setting, name, r))
    print(
        f"  {setting:15s} {name:34s} mse {r['mse']:.4f}  acc {r['accuracy']:.3f}  "
        f"charge {r['charge']:.2f}  h>h_max {r['h_violation']:.3f}"
    )


def img(ax, x, title=None):
    ax.imshow(x.reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


# %% [1] The percept readout: a frozen MNIST autoencoder ------------------
# D (decoder) is the brain's "neural state → percept" map; its 16-D code
# space is the latent neural state x. E(y) gives each target image a target
# neural state. The classifier asks "is the percept recognisable?".
print("\n[1] Vision side: MNIST autoencoder (percept readout) + classifier")
mnist = load_mnist()
train_x = mnist["train_x"]
test_x, test_y = mnist["test_x"][:N_TEST], mnist["test_y"][:N_TEST]
ae = train_autoencoder(train_x, epochs=1 if Q else 10)
clf = train_classifier(train_x, mnist["train_y"], epochs=1 if Q else 3)
with torch.no_grad():
    ceiling = ae.decode(ae.encode(test_x))
    ceil = {
        "mse": float((ceiling - test_x).pow(2).mean()),
        "accuracy": float((clf(ceiling).argmax(-1) == test_y).float().mean()),
        "charge": 0.0,
        "h_violation": 0.0,
    }
report("any", "ceiling D(E(y)) (latent reached exactly)", ceil)

cfg = BrainConfig()
brain = Brain(cfg)
B_fixed = brain.sample_patients(1, seed=123)


def fixed_patient(n, step):
    return B_fixed.expand(n, -1, -1), None


# %% [2] What stimulation does to the percept -----------------------------
# Hold one random stimulation pattern for T steps: the percept forms within a
# few steps. Adaptation h keeps building (the drive shrinks as 1 / (1 + λh)),
# but at this charge level it only reaches ~0.3 within 30 steps.
print("\n[2] Holding one stimulation pattern: the percept forms; adaptation builds")
a_hold = project_charge(torch.rand(1, 1, cfg.m).expand(1, cfg.horizon, cfg.m) * 2 - 1, cfg.q_max)
with torch.no_grad():
    xs, hs = brain.rollout(B_fixed, a_hold, seed=0, noise=False)
    frames = ae.decode(xs[0])
steps = [0, 2, 5, 10, 20, 30]
fig, axes = plt.subplots(1, len(steps) + 1, figsize=(12, 2.2))
for ax, t in zip(axes, steps):
    img(ax, frames[t], f"t = {t}")
axes[-1].plot(hs[0].mean(-1).numpy(), color="k")
axes[-1].axhline(cfg.h_max, ls="--", color="r", lw=1)
axes[-1].set_title("mean adaptation h", fontsize=8)
save(fig, "p1_hold.png")

# %% [3] Fixed patient -----------------------------------------------------
print("\n[3] Fixed patient")
B_test = B_fixed.expand(N_TEST, -1, -1)
report(
    "fixed patient", "no stimulation", evaluate(None, brain, ae, clf, test_x, test_y, B_test, None)
)

# (a) Explore: stimulate the true brain with varied patterns, record percepts.
print("  collecting exploration data and fitting the world model")
data = collect(
    brain,
    ae,
    B_fixed.expand(N_EPISODES, -1, -1),
    exploration_actions(cfg, N_EPISODES, 0),
    10,
    False,
)
wm = train_world_model(data, ae, cfg, steps=WM_STEPS)

# World-model check on unseen stimulation: predicted vs true percepts.
a_val = exploration_actions(cfg, 4, seed=77)
with torch.no_grad():
    true_frames = ae.decode(brain.rollout(B_fixed.expand(4, -1, -1), a_val, seed=3)[0][:, 1:])
    rest = ae.decode(cfg.x0_std * torch.randn(4, cfg.d))
    pred_frames = ae.decode(wm(ae.encode(rest), a_val))
ts = [0, 4, 9, 19, 29]
fig, axes = plt.subplots(4, 2 * len(ts), figsize=(14, 5.6))
for i in range(4):
    for j, t in enumerate(ts):
        img(axes[i, j], true_frames[i, t], f"true t={t + 1}" if i == 0 else None)
        img(axes[i, len(ts) + j], pred_frames[i, t], f"W t={t + 1}" if i == 0 else None)
fig.suptitle("Unseen stimulation sequences: true brain (left) vs world model open-loop (right)")
save(fig, "p2_world_model.png")

# (b) Learn stimulation through the world model only.
print("  training the policy through the world model (round 1)")
pi_wm1 = train_policy(
    wm_simulator(wm, brain, ae), ae, train_x, cfg, fixed_patient, steps=PI_STEPS, verbose=False
)
r_wm1 = evaluate(pi_wm1, brain, ae, clf, test_x, test_y, B_test, None)
report("fixed patient", "policy via world model (round 1)", r_wm1)

# (c) Round 2: stimulate with the round-1 policy, add those episodes, refit.
#     The policy's own action distribution is where the model must be right.
print("  round 2: on-policy data, refit world model and policy")
idx = torch.randint(0, len(train_x), (N_EPISODES,))
with torch.no_grad():
    a_on = pi_wm1(ae.encode(train_x[idx]))
a_on = project_charge((a_on + 0.1 * torch.randn_like(a_on)).clamp(-1, 1), cfg.q_max)
data2 = collect(brain, ae, B_fixed.expand(N_EPISODES, -1, -1), a_on, 20, False)
data = {k: torch.cat([data[k], data2[k]]) if data[k] is not None else None for k in data}
wm = train_world_model(data, ae, cfg, steps=WM_STEPS // 2, wm=wm)
pi_wm2 = train_policy(
    wm_simulator(wm, brain, ae), ae, train_x, cfg, fixed_patient, steps=PI_STEPS, verbose=False
)
r_wm2 = evaluate(pi_wm2, brain, ae, clf, test_x, test_y, B_test, None)
report("fixed patient", "policy via world model (round 2)", r_wm2)

# (d) Privileged reference: the same policy trained through the true brain.
print("  privileged reference: backprop through the true brain")
pi_true = train_policy(
    true_simulator(brain, ae), ae, train_x, cfg, fixed_patient, steps=PI_STEPS, verbose=False
)
r_true = evaluate(pi_true, brain, ae, clf, test_x, test_y, B_test, None)
report("fixed patient", "policy via true brain *", r_true)

# %% [4] Random patients ---------------------------------------------------
# Every episode is a new B. Before stimulating, a calibration sweep pulses each
# of the 24 electrodes once from rest and records the encoded percept change:
# 24 x 16 numbers, a noisy view of B. That record conditions both W and π
# through a hypernetwork (the patient multiplies the action, so the context
# emits the matrix that the action is multiplied by).
print("\n[4] Random patients (one new B per episode, calibration probe as context)")
B_rand_test = brain.sample_patients(N_TEST, seed=999)
ctx_test = brain.calibrate(ae, B_rand_test, seed=5)
n_ctx = ctx_test.shape[1]


def random_patient(n, step):
    B = brain.sample_patients(n, seed=10_000 + step)
    return B, brain.calibrate(ae, B, seed=step)


report(
    "random patients",
    "no stimulation",
    evaluate(None, brain, ae, clf, test_x, test_y, B_rand_test, None),
)
report(
    "random patients",
    "fixed-patient policy (no personalisation)",
    evaluate(pi_wm2, brain, ae, clf, test_x, test_y, B_rand_test, None),
)
del data, data2  # fixed-patient episodes are no longer needed
print("  collecting multi-patient data and fitting the context world model")
B_data = brain.sample_patients(N_EPISODES_RAND, seed=7)
data_r = collect(brain, ae, B_data, exploration_actions(cfg, N_EPISODES_RAND, 1), 30, True)
wm_ctx = train_world_model(data_r, ae, cfg, n_ctx=n_ctx, steps=WM_STEPS_RAND)
pi_ctx = train_policy(
    wm_simulator(wm_ctx, brain, ae),
    ae,
    train_x,
    cfg,
    random_patient,
    n_ctx=n_ctx,
    steps=PI_STEPS_RAND,
    verbose=False,
)
r_ctx = evaluate(pi_ctx, brain, ae, clf, test_x, test_y, B_rand_test, ctx_test)
report("random patients", "world model + calibration context", r_ctx)
pi_ctx_true = train_policy(
    true_simulator(brain, ae),
    ae,
    train_x,
    cfg,
    random_patient,
    n_ctx=n_ctx,
    steps=PI_STEPS_RAND,
    verbose=False,
)
r_ctx_true = evaluate(pi_ctx_true, brain, ae, clf, test_x, test_y, B_rand_test, ctx_test)
report("random patients", "true brain + calibration context *", r_ctx_true)

# %% [5] What the patient sees ---------------------------------------------
cols = [
    ("target", test_x),
    ("ceiling\nD(E(y))", ceiling),
    ("fixed: world\nmodel (r2)", r_wm2["percepts"][:, cfg.hold_from - 1 :].mean(1)),
    ("fixed: true\nbrain *", r_true["percepts"][:, cfg.hold_from - 1 :].mean(1)),
    ("random: world\nmodel + ctx", r_ctx["percepts"][:, cfg.hold_from - 1 :].mean(1)),
    ("random: true\nbrain + ctx *", r_ctx_true["percepts"][:, cfg.hold_from - 1 :].mean(1)),
]
n_show = 8
fig, axes = plt.subplots(n_show, len(cols), figsize=(1.5 * len(cols), 1.5 * n_show))
for j, (title, ims) in enumerate(cols):
    for i in range(n_show):
        img(axes[i, j], ims[i], title if i == 0 else None)
fig.suptitle("Percepts on the true brain (mean over the hold window t ≥ 10)", fontsize=9)
save(fig, "p3_reconstructions.png")

fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for j, t in enumerate(range(0, cfg.horizon, 3)):
    img(axes[0, j], r_wm2["percepts"][0, t], f"t={t + 1}")
    axes[1, j].bar(range(cfg.m), r_wm2["actions"][0, t].numpy(), color="k")
    axes[1, j].set_ylim(-1, 1)
    axes[1, j].set_xticks([]), axes[1, j].set_yticks([])
fig.suptitle(
    "Fixed patient, world-model policy: percept (top) and 24-electrode stimulation (bottom)"
)
save(fig, "p4_sequence.png")

# %% [6] Results table ------------------------------------------------------
lines = [
    "| setting | policy | pixel MSE | classifier accuracy | charge | h > h_max |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for setting, name, r in rows:
    lines.append(
        f"| {setting} | {name} | {r['mse']:.4f} | {r['accuracy']:.3f} | {r['charge']:.2f} "
        f"| {r['h_violation']:.3f} |"
    )
table = "\n".join(lines)
(args.outdir / "results_percept.md").write_text(table + "\n")
print("\n" + table)
print("\n* = privileged (gradients through the true brain); a reference, not a method.")
