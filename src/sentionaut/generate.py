"""Multi-config world-model dataset generation using the GPU percept models.

Samples model-appropriate actions, rolls out ``f(s_t, a_t) -> s_{t+1}`` for each
config, and records ``(config_id, s_t, a_t, s_{t+1})`` transitions in a combined
HDF5 covering all requested models. All configs in a run share the same percept
grid so states stack into fixed-shape arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import h5py
import numpy as np
import torch

from .core.base import Action, Pose
from .core.config import CORTICAL_MODELS, Config
from .core.device import get_device
from .core.registry import build_components

# Default implant per model family.
DEFAULT_IMPLANT = {
    "axonmap": "argusii",
    "scoreboard": "orion",
    "dynaphos": "orion",
}


@dataclass
class ActionRanges:
    # Retinal (threshold-factor amplitude) vs cortical (microamp current).
    amp_retinal: tuple[float, float] = (0.5, 3.0)
    amp_cortical: tuple[float, float] = (50.0, 300.0)
    freq: tuple[float, float] = (10.0, 100.0)
    phase_dur: tuple[float, float] = (0.1, 0.5)
    rho_retinal: tuple[float, float] = (150.0, 300.0)
    axlambda: tuple[float, float] = (400.0, 700.0)
    rho_cortical: tuple[float, float] = (800.0, 1200.0)
    max_active: int = 3


def _uniform(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def sample_action(
    config: Config,
    n_electrodes: int,
    rng: np.random.Generator,
    ranges: ActionRanges,
    device: torch.device,
) -> tuple[Action, dict]:
    """Sample a random action for ``config`` and return it plus a record dict."""
    cortical = config.model in CORTICAL_MODELS
    n_active = int(rng.integers(1, ranges.max_active + 1))
    idx = rng.choice(n_electrodes, size=n_active, replace=False)

    amp = np.zeros(n_electrodes, dtype=np.float32)
    freq = np.zeros(n_electrodes, dtype=np.float32)
    pdur = np.zeros(n_electrodes, dtype=np.float32)
    amp_range = ranges.amp_cortical if cortical else ranges.amp_retinal
    for e in idx:
        amp[e] = _uniform(rng, *amp_range)
        freq[e] = _uniform(rng, *ranges.freq)
        pdur[e] = _uniform(rng, *ranges.phase_dur)

    rho = _uniform(rng, *(ranges.rho_cortical if cortical else ranges.rho_retinal))
    axlambda = _uniform(rng, *ranges.axlambda)

    action = Action(
        amp=torch.from_numpy(amp).to(device),
        freq=torch.from_numpy(freq).to(device),
        phase_dur=torch.from_numpy(pdur).to(device),
        rho=rho,
        axlambda=axlambda if not cortical else None,
        pose=Pose(),
    )
    record = {"amp": amp, "freq": freq, "pdur": pdur, "rho": rho, "axlambda": axlambda}
    return action, record


def build_configs(models: list[str], base: Config) -> list[Config]:
    configs = []
    for m in models:
        configs.append(
            Config(
                model=m,
                implant=DEFAULT_IMPLANT[m],
                xrange=base.xrange,
                yrange=base.yrange,
                xystep=base.xystep,
                regions=base.regions,
                device=base.device,
            )
        )
    return configs


def generate_world_dataset(
    output_path: Path,
    configs: list[Config],
    episodes: int,
    sequence_length: int,
    *,
    device: torch.device | None = None,
    seed: int | None = None,
    ranges: ActionRanges | None = None,
    compression: str | None = None,
) -> Path:
    """Generate a combined multi-config transition dataset."""
    if not configs:
        raise click.BadParameter("at least one config required")
    device = device or get_device(configs[0].device)
    ranges = ranges or ActionRanges()
    rng = np.random.default_rng(seed)

    built = []
    grid_shape = None
    max_elec = 0
    for cfg in configs:
        implant, topo, model = build_components(cfg, device)
        from .world import WorldModel

        wm = WorldModel(model, cfg)
        if grid_shape is None:
            grid_shape = wm.grid_shape
        elif grid_shape != wm.grid_shape:
            raise ValueError("All configs must share the same percept grid shape.")
        max_elec = max(max_elec, implant.n_electrodes)
        built.append((cfg, implant, wm))

    H, W = grid_shape
    n_total = len(built) * episodes * sequence_length

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["config_table"] = json.dumps([c.to_dict() for c, _, _ in built])
        meta.attrs["grid_shape"] = (H, W)
        meta.attrs["max_electrodes"] = max_elec
        meta.attrs["action_features"] = json.dumps(["amp", "freq", "phase_dur"])

        g = h5.create_group("world")
        ckw = dict(compression=compression) if compression else {}
        s_t = g.create_dataset("s_t", shape=(n_total, H, W), dtype=np.float32, **ckw)
        s_tp1 = g.create_dataset("s_tp1", shape=(n_total, H, W), dtype=np.float32, **ckw)
        amp_d = g.create_dataset("amp", shape=(n_total, max_elec), dtype=np.float32, **ckw)
        freq_d = g.create_dataset("freq", shape=(n_total, max_elec), dtype=np.float32, **ckw)
        pdur_d = g.create_dataset("phase_dur", shape=(n_total, max_elec), dtype=np.float32, **ckw)
        rho_d = g.create_dataset("rho", shape=(n_total,), dtype=np.float32)
        axl_d = g.create_dataset("axlambda", shape=(n_total,), dtype=np.float32)
        cfg_id = g.create_dataset("config_id", shape=(n_total,), dtype=np.int32)

        i = 0
        for cfg_idx, (cfg, implant, wm) in enumerate(built):
            n_e = implant.n_electrodes
            for _ in range(episodes):
                state = wm.initial_state(device)
                prev = torch.zeros(H, W, device=device)
                for _ in range(sequence_length):
                    action, rec = sample_action(cfg, n_e, rng, ranges, device)
                    state = wm.step(state, action)
                    frame = state.image.detach()
                    s_t[i] = prev.cpu().numpy()
                    s_tp1[i] = frame.cpu().numpy()
                    amp_d[i, :n_e] = rec["amp"]
                    freq_d[i, :n_e] = rec["freq"]
                    pdur_d[i, :n_e] = rec["pdur"]
                    rho_d[i] = rec["rho"]
                    axl_d[i] = rec["axlambda"]
                    cfg_id[i] = cfg_idx
                    prev = frame
                    i += 1
    return output_path


def _pair(ctx, param, value):
    return (float(value[0]), float(value[1]))


@click.command()
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--model",
    "models",
    multiple=True,
    default=("axonmap", "scoreboard", "dynaphos"),
    show_default=True,
    help="Percept model(s) to include (repeatable).",
)
@click.option("--episodes", type=int, default=8, show_default=True)
@click.option("--sequence-length", type=int, default=4, show_default=True)
@click.option(
    "--xrange", nargs=2, type=float, callback=_pair, default=(-5.0, 5.0), show_default=True
)
@click.option(
    "--yrange", nargs=2, type=float, callback=_pair, default=(-5.0, 5.0), show_default=True
)
@click.option("--xystep", type=float, default=0.5, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--compression", type=str, default=None)
def world_cli(
    output_path,
    models,
    episodes,
    sequence_length,
    xrange,
    yrange,
    xystep,
    device,
    seed,
    compression,
):
    base = Config(
        model=list(models)[0],
        implant=DEFAULT_IMPLANT[list(models)[0]],
        xrange=xrange,
        yrange=yrange,
        xystep=xystep,
        device=device,
    )
    configs = build_configs(list(models), base)
    path = generate_world_dataset(
        output_path,
        configs,
        episodes,
        sequence_length,
        seed=seed,
        compression=compression,
    )
    click.echo(str(path))


@click.command()
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--model", "model", type=str, default="axonmap", show_default=True)
@click.option("--samples", "samples", type=int, default=64, show_default=True)
@click.option(
    "--xrange", nargs=2, type=float, callback=_pair, default=(-8.0, 8.0), show_default=True
)
@click.option(
    "--yrange", nargs=2, type=float, callback=_pair, default=(-8.0, 8.0), show_default=True
)
@click.option("--xystep", type=float, default=0.5, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
def cli(output_path, model, samples, xrange, yrange, xystep, device, seed):
    """Single-model dataset (one transition per sample from a zero baseline)."""
    base = Config(
        model=model,
        implant=DEFAULT_IMPLANT[model],
        xrange=xrange,
        yrange=yrange,
        xystep=xystep,
        device=device,
    )
    path = generate_world_dataset(
        output_path,
        build_configs([model], base),
        episodes=samples,
        sequence_length=1,
        seed=seed,
    )
    click.echo(str(path))


if __name__ == "__main__":  # pragma: no cover
    world_cli()
