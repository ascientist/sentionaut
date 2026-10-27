"""Figures and tables for the imagination experiment (``sentionaut.learned.imagine``).

Reads the run JSONs in ``--runs`` and writes, per preset:

- ``<preset>_curves.png``: held-out loss against training wall-clock and against
  teacher steps, one line per arm (median over seeds, seeds thin);
- ``results.md``: final numbers, and time and teacher steps to reach the
  phosphene-map baseline and the best final loss within 10%.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["teacher-bptt", "wm-bptt", "reinforce-sim", "reinforce-wm"]
LABEL = {
    "teacher-bptt": "Backprop through teacher",
    "wm-bptt": "Backprop through world model",
    "reinforce-sim": "REINFORCE in teacher",
    "reinforce-wm": "REINFORCE in world model",
}
COLOR = {
    "teacher-bptt": "#2a78d6",
    "wm-bptt": "#eb6834",
    "reinforce-sim": "#1baf7a",
    "reinforce-wm": "#eda100",
}
STYLE = {"teacher-bptt": "-", "wm-bptt": "-", "reinforce-sim": "--", "reinforce-wm": "--"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def first_reach(log, key, level):
    for r in log:
        if not r["final"] and r["loss"] <= level:
            return r[key]
    return None


def fmt_t(v):
    return "not reached" if v is None else f"{v:,.0f} s"


def fmt_n(v):
    return "not reached" if v is None else f"{v:,.0f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="data/imagine")
    p.add_argument("--out", default="docs/assets/imagination")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(lambda: defaultdict(list))
    baselines = {}
    for f in sorted(Path(a.runs).glob("*.json")):
        d = json.loads(f.read_text())
        if "baselines" in d:
            baselines[f.name.split("_")[0]] = d["baselines"]
            continue
        c = d["config"]
        runs[c["preset"]][c["arm"]].append(d)
    md = ["# Imagination results", ""]
    for preset, arms in runs.items():
        base = baselines.get(preset, {})
        pm = base.get("phosphene_map", {}).get("loss")
        finals = {arm: min(r["log"][-1]["loss"] for r in rs) for arm, rs in arms.items()}
        best = min(finals.values())
        devices = sorted({r.get("device", "cpu") for rs in arms.values() for r in rs})
        where = "one CPU thread" if devices == ["cpu"] else ", ".join(devices)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, key, xlabel in (
            (axes[0], "t", f"Training wall-clock (s, {where})"),
            (axes[1], "teacher_steps", "Teacher steps (episodes × 10)"),
        ):
            for arm in ARMS:
                for i, r in enumerate(arms.get(arm, [])):
                    pts = [(e[key], e["loss"]) for e in r["log"] if not e["final"] and e[key] > 0]
                    if not pts:
                        continue
                    xs, ys = zip(*pts)
                    ax.plot(
                        xs,
                        ys,
                        STYLE[arm],
                        color=COLOR[arm],
                        lw=2 if i == 0 else 1,
                        alpha=1 if i == 0 else 0.5,
                        label=LABEL[arm] if i == 0 else None,
                    )
                    if i == 0:
                        ax.annotate(
                            LABEL[arm].split(" ")[0] + (" (WM)" if "wm" in arm else ""),
                            (xs[-1], ys[-1]),
                            xytext=(4, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color=INK,
                            va="center",
                        )
            for name, v, ls in (
                ("phosphene map", pm, ":"),
                ("no stimulation", base.get("zero", {}).get("loss"), "-."),
            ):
                if v is not None:
                    ax.axhline(v, color=MUTED, lw=1, ls=ls)
                    ax.annotate(
                        name,
                        (1, v),
                        xycoords=("axes fraction", "data"),
                        xytext=(-4, 3),
                        textcoords="offset points",
                        ha="right",
                        fontsize=8,
                        color=MUTED,
                    )
            ax.set_xscale("log")
            ax.set_xlabel(xlabel, color=INK)
            ax.grid(True, color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("Held-out loss on the teacher (lower is better)", color=INK)
        axes[0].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle(
            f"{preset}: training an encoder (64 held-out digits; final point uses 256)", color=INK
        )
        fig.tight_layout()
        fig.savefig(out / f"{preset}_curves.png", dpi=130)
        plt.close(fig)

        md += [f"## {preset}", "", f"![{preset} curves]({preset}_curves.png)", ""]
        if base:
            md += [
                "| Reference (no learning) | Loss | Percept MSE | Charge |",
                "| --- | ---: | ---: | ---: |",
            ]
            for k, v in base.items():
                md.append(f"| {k} | {v['loss']:.4f} | {v['mse']:.4f} | {v['charge']:.3f} |")
            md.append("")
        md += [
            "| Arm | Seed | Final loss | Percept MSE | Charge | Wall-clock | Teacher steps | "
            "Time to phosphene map | Steps to phosphene map | Time to best+10% | Steps to best+10% |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm in ARMS:
            for r in arms.get(arm, []):
                f = r["log"][-1]
                lvl = best * 1.1
                md.append(
                    f"| {LABEL[arm]} | {r['config']['seed']} | {f['loss']:.4f} | {f['mse']:.4f} | "
                    f"{f['charge']:.3f} | {f['t']:,.0f} s | {f['teacher_steps']:,} | "
                    f"{fmt_t(first_reach(r['log'], 't', pm) if pm else None)} | "
                    f"{fmt_n(first_reach(r['log'], 'teacher_steps', pm) if pm else None)} | "
                    f"{fmt_t(first_reach(r['log'], 't', lvl))} | "
                    f"{fmt_n(first_reach(r['log'], 'teacher_steps', lvl))} |"
                )
        md.append("")
    (out / "results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
