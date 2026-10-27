"""Throughput of axon-map forward models on one grid, against batch size.

Engines (pick with --engines):

- ``p2p-cython``: pulse2percept 0.9.0 ``BiphasicAxonMapModel``, default engine
  (third party). One stimulus per ``predict_percept`` call, as its API takes.
- ``p2p-jax``: the same model with ``engine="jax"`` (third party, jitted).
  Needs jax; run this script from an environment that has it.
- ``torch-exact``, ``torch-sparse``: this repo's ``spatial_forward_batch``,
  exact, and with ``cutoff=5``.
- ``wm``: the learned ``TokenWorldModel`` from ``sentionaut.learned.imagine``
  (one step, untrained weights: the cost does not depend on them).
- ``extra``: any ``module:function`` given by ``--extra``, called as
  ``f(amp)`` with ``amp`` a float32 numpy array ``(B, 60)``; it must return
  ``(B, H, W)``. Use it to time another implementation on the same stimuli.

Prints one JSON line per (engine, batch, mode) with samples per second.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np

XR, YR, STEP = (-15.0, 15.0), (-10.0, 10.0), 1.0


def stimuli(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.5, 3.0, size=(n, 60)).astype(np.float32)
    amp[rng.random((n, 60)) > 0.3] = 0.0
    return amp


def timeit(fn, reps: int, min_time: float = 1.0) -> float:
    fn()  # warm-up (jit, caches)
    t0, n = time.perf_counter(), 0
    while n < reps or time.perf_counter() - t0 < min_time:
        fn()
        n += 1
    return (time.perf_counter() - t0) / n


def p2p_engine(engine: str, threads: int):
    from pulse2percept.implants import ArgusII
    from pulse2percept.models import BiphasicAxonMapModel
    from pulse2percept.stimuli import BiphasicPulseTrain

    model = BiphasicAxonMapModel(
        xrange=XR, yrange=YR, xystep=STEP, engine=engine, n_threads=threads
    )
    model.build()
    implant = ArgusII()
    names = list(implant.electrode_names)

    def run(amp: np.ndarray):
        out = []
        for row in amp:
            implant.stim = {
                names[i]: BiphasicPulseTrain(20, float(row[i]), 0.45) for i in np.nonzero(row)[0]
            }
            out.append(model.predict_percept(implant))
        return out

    return run, None


def torch_engine(kind: str, threads: int):
    import torch

    from sentionaut.core.config import Config
    from sentionaut.core.registry import build_components

    torch.set_num_threads(threads)
    cfg = Config(model="axonmap", implant="argusii", xrange=XR, yrange=YR, xystep=STEP)
    implant, topo, model = build_components(cfg, device=torch.device("cpu"))
    if kind == "wm":
        from sentionaut.learned.imagine import TokenWorldModel

        xy = torch.rand(implant.n_electrodes, 2) * 2 - 1
        wm = TokenWorldModel(tuple(topo.grid_shape), xy, a_max=3.0)

        def fwd(amp):
            img = torch.zeros(amp.shape[0], *topo.grid_shape)
            return wm(img, wm.init_memory(torch.zeros(amp.shape[0], 60, 0)), amp)[0]

    else:
        cut = 5.0 if kind == "torch-sparse" else None

        def fwd(amp):
            return model.spatial_forward_batch(amp, cutoff=cut)

    def run(amp: np.ndarray):
        with torch.no_grad():
            return fwd(torch.from_numpy(amp))

    def run_grad(amp: np.ndarray):
        a = torch.from_numpy(amp).requires_grad_(True)
        fwd(a).sum().backward()

    return run, run_grad


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--engines", nargs="+", default=["p2p-cython", "torch-exact", "torch-sparse", "wm"]
    )
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--extra", default=None, help="module:function to time as engine 'extra'")
    p.add_argument("--max-p2p-batch", type=int, default=32)
    a = p.parse_args()
    for eng in a.engines:
        if eng.startswith("p2p-"):
            run, run_grad = p2p_engine(eng.split("-", 1)[1], a.threads)
        elif eng == "extra":
            mod, fn = a.extra.split(":")
            run, run_grad = getattr(importlib.import_module(mod), fn), None
        else:
            run, run_grad = torch_engine(eng, a.threads)
        for b in a.batches:
            if eng.startswith("p2p-") and b > a.max_p2p_batch:
                continue  # a per-sample loop: cost is linear in b
            amp = stimuli(b)
            modes = [("forward", run)] + ([("forward+backward", run_grad)] if run_grad else [])
            for mode, fn in modes:
                sec = timeit(lambda: fn(amp), reps=3)
                print(
                    json.dumps(
                        {
                            "engine": eng,
                            "batch": b,
                            "mode": mode,
                            "threads": a.threads,
                            "sec_per_call": sec,
                            "samples_per_s": b / sec,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
