"""Aggregate multi-seed baseline runs: median error, mean ± s.d. across seeds.

    uv run python scripts/aggregate_neurostim_seeds.py results/neurostim_baselines

Reads every ``*/results_baselines.json`` below the given directory (one per
seed and regime, as written by ``scripts/slurm_neurostim_baselines.sh``) and
prints a Markdown table: rows are methods, columns are regimes.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(root: str) -> None:
    runs = defaultdict(lambda: defaultdict(list))  # regime -> method -> [median per seed]
    for f in sorted(Path(root).glob("*/results_baselines.json")):
        for regime, methods in json.loads(f.read_text()).items():
            for method, r in methods.items():
                runs[regime][method].append(r["ss_median"])
    if not runs:
        sys.exit(f"no results_baselines.json under {root}")
    regimes = sorted(runs)
    methods = list(next(iter(runs.values())))
    print("| method | " + " | ".join(f"regime {r}" for r in regimes) + " |")
    print("| --- | " + " | ".join("---:" for _ in regimes) + " |")
    for m in methods:
        cells = []
        for r in regimes:
            v = np.array(runs[r].get(m, []))
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f} (n={len(v)})")
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/neurostim_baselines")
