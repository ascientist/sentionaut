"""JEPA collapse diagnostics on the canonical problem.

For each named variant, train the JEPA-style model and report:

- collapse: per-dimension std (min / mean) and effective rank of the latent,
  for encoder outputs and for 5-step predictions, over training;
- information: held-out linear-probe MSE of the percept from the latent, and
  from the 5-step *predicted* latent (what the planner actually uses);
- control: median steady-state error and failure rate on held-out patients.

Reference point: a linear probe on the raw observation history reaches the
observation-noise floor (~0.005 in regime A).

    uv run python examples/neurostim_jepa_diagnostics.py --regime A --variants old,vicreg
    sbatch --array=0-15 scripts/slurm_neurostim_jepa.sh   # variants x regimes on a cluster
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from sentionaut.neurostim.canonical import evaluate, regime
from sentionaut.neurostim.canonical_learned import JEPAConfig, collect_random, mlp, train_jepa


def mlp_probe_mse(x, y, steps=3000, seed=0):
    """Nonlinear probe: is the percept in the representation at all, linearly or not?"""
    torch.manual_seed(seed)
    n = len(x) // 2
    net = mlp(x.shape[1], y.shape[1], hidden=128)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        b = torch.randint(0, n, (256,))
        loss = F.mse_loss(net(x[b]), y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.mse_loss(net(x[n:]), y[n:]).item()


VARIANTS = {
    # Control: no regulariser, stop-gradient targets. Nothing prevents collapse, so
    # the diagnostics must show it here, or they cannot be trusted elsewhere.
    "noreg": dict(reg="none"),
    # LeJEPA: SIGReg on every embedding, no stop-gradient, lambda = 0.05.
    "lejepa": dict(reg="sigreg", stop_grad=False),
    # The configuration before this fix: weak weights, only the first encoding regularised.
    "old": dict(
        pred_w=1.0,
        var_w=1.0,
        cov_w=0.04,
        reg_predictions=False,
        reg_all_encodings=False,
        probe="linear",
    ),
    # New regularisation, but the linear read-out the planner used before.
    "vicreg-linprobe": dict(probe="linear"),
    # VICReg 25/25/1 on every encoding and every prediction (the new default).
    "vicreg": dict(),
    "vicreg-ema": dict(ema=0.99),
    "vicreg-L32": dict(latent=32),
    "vicreg-long": dict(steps=20000),
    # Planner variants. Even with the TRUE model, value-free sampling planning fails
    # here (short horizons excite hidden modes): these test how far planning alone goes.
    "vicreg-H20k4": dict(plan_horizon=20, plan_knots=4, rollout=10),
    "vicreg-H30k6": dict(plan_horizon=30, plan_knots=6, rollout=10),
}

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--regime", default="A")
parser.add_argument("--variants", default=",".join(VARIANTS))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--patients", type=int, default=10)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--out", type=Path, default=Path("results/neurostim_jepa"))
args = parser.parse_args()
torch.set_num_threads(4)
args.out.mkdir(parents=True, exist_ok=True)
cfg = regime(args.regime)

# Held-out data (different seed from training) for the information metrics.
sim, S, A, Obs = collect_random(cfg, 200, seed=99_999)
perc = Obs @ (sim.D.T if cfg.observe == "full" else torch.eye(Obs.shape[-1]))

for name in args.variants.split(","):
    kw = dict(VARIANTS[name], seed=args.seed)
    if args.quick:
        kw.update(episodes=200, steps=300, log_every=100)
    jc = JEPAConfig(**kw)
    t = time.time()
    log: list = []
    ctrl = train_jepa(cfg, jc, verbose=True, log=log)
    with torch.no_grad():
        hist = ctrl.obs_only(S)
        probe, enc, pred = ctrl.probe, ctrl.enc, ctrl.pred

        z_now = enc(hist[:, 50])
        probe_now = F.mse_loss(probe(z_now), perc[:, 49]).item()
        z = z_now
        for k in range(5):
            z = pred(torch.cat([z, A[:, 50 + k]], -1))
        probe_pred5 = F.mse_loss(probe(z), perc[:, 54]).item()
    with torch.no_grad():  # held-out episodes, all time steps
        Hall = hist[:, 1:].reshape(-1, hist.shape[-1])
        Zall = enc(Hall)
    Pall = perc.reshape(-1, perc.shape[-1])
    mlp_steps = 200 if args.quick else 3000
    mlp_latent = mlp_probe_mse(Zall, Pall, mlp_steps)
    mlp_input = mlp_probe_mse(Hall, Pall, mlp_steps)
    r = evaluate(cfg, lambda c=ctrl: c, n_patients=2 if args.quick else args.patients)
    result = {
        "variant": name,
        "regime": args.regime,
        "seed": args.seed,
        "config": asdict(jc),
        "train_log": log,
        "heldout_probe_mse": probe_now,
        "heldout_probe_mse_pred5": probe_pred5,
        "mlp_probe_mse_latent": mlp_latent,
        "mlp_probe_mse_input": mlp_input,
        "percept_var": float(perc.var(dim=(0, 1)).mean()),
        "ss_median": r["ss_median"],
        "fail_rate": r["fail_rate"],
        "seconds": time.time() - t,
    }
    print(
        f"== {name} [{args.regime}, seed {args.seed}]  probe {probe_now:.4f}  probe(pred5) "
        f"{probe_pred5:.4f}  mlp-probe latent {mlp_latent:.4f} (raw input {mlp_input:.4f})  "
        f"control median {r['ss_median']:.4f}  fail {r['fail_rate']:.2f}  "
        f"({result['seconds']:.0f}s)",
        flush=True,
    )
    (args.out / f"{args.regime}_{name}_seed{args.seed}.json").write_text(
        json.dumps(result, indent=1)
    )
