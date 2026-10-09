"""Multi-config world-model dataset generation using the GPU percept models."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import click
import h5py
import numpy as np
import torch

from .core.base import Action, Pose
from .core.config import CORTICAL_MODELS, Config
from .core.device import get_device
from .core.registry import build_components

DEFAULT_IMPLANT = {
    "axonmap": "argusii",
    "scoreboard": "orion",
    "dynaphos": "orion",
}


@dataclass
class ActionRanges:
    amp_retinal: tuple[float, float] = (0.5, 3.0)
    amp_cortical: tuple[float, float] = (50.0, 300.0)
    freq: tuple[float, float] = (10.0, 100.0)
    phase_dur: tuple[float, float] = (0.1, 0.5)
    rho_retinal: tuple[float, float] = (150.0, 300.0)
    axlambda: tuple[float, float] = (400.0, 700.0)
    rho_cortical: tuple[float, float] = (800.0, 1200.0)
    max_active: int = 3


@dataclass
class StreamSchedule:
    """Stimulation as a video: holds let brightness rise, gaps let it fade."""

    hold_prob: float = 0.5
    silent_prob: float = 0.3


def _uniform(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def sample_action(
    config: Config,
    n_electrodes: int,
    rng: np.random.Generator,
    ranges: ActionRanges,
    device: torch.device,
    *,
    silent: bool = False,
) -> tuple[Action, dict]:
    cortical = config.model in CORTICAL_MODELS
    if silent:
        amp = np.zeros(n_electrodes, dtype=np.float32)
        freq = np.zeros(n_electrodes, dtype=np.float32)
        pdur = np.zeros(n_electrodes, dtype=np.float32)
        rho = config.rho
        axlambda = config.axlambda
    else:
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
                dt_ms=base.dt_ms,
                fade_tau_ms=base.fade_tau_ms,
                costim_enabled=base.costim_enabled,
                costim_kappa=base.costim_kappa,
            )
        )
    return configs


def _aux_maps(wm, state) -> tuple[np.ndarray, np.ndarray]:
    from .models.dynaphos import DynaphosTorch

    H, W = wm.grid_shape
    if isinstance(wm.model, DynaphosTorch):
        a_map, q_map = wm.model.rasterize_aux(state)
        return a_map.detach().cpu().numpy(), q_map.detach().cpu().numpy()
    return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)


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
    dt_ms: float | None = None,
    silent_tail: int = 0,
    timing_path: Path | None = None,
    write_slab: int = 32,
    stream: StreamSchedule | None = None,
) -> Path:
    if not configs:
        raise click.BadParameter("at least one config required")
    device = device or get_device(configs[0].device)
    ranges = ranges or ActionRanges()
    rng = np.random.default_rng(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_build = time.perf_counter()
    built = []
    grid_shape = None
    max_elec = 0
    percept_scales: dict[int, float] = {}
    for cfg in configs:
        if dt_ms is not None:
            cfg = Config(**{**cfg.to_dict(), "dt_ms": dt_ms})
        implant, topo, model = build_components(cfg, device)
        from .world import WorldModel

        wm = WorldModel(model, cfg)
        if grid_shape is None:
            grid_shape = wm.grid_shape
        elif grid_shape != wm.grid_shape:
            raise ValueError("All configs must share the same percept grid shape.")
        max_elec = max(max_elec, implant.n_electrodes)
        built.append((cfg, implant, wm))
        print(f"percept device: {wm._device()}", flush=True)
    t_build = time.perf_counter() - t_build

    H, W = grid_shape
    steps_per_ep = sequence_length + silent_tail
    n_total = len(built) * episodes * steps_per_ep

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["config_table"] = json.dumps([c.to_dict() for c, _, _ in built])
        meta.attrs["grid_shape"] = (H, W)
        meta.attrs["max_electrodes"] = max_elec
        meta.attrs["action_features"] = json.dumps(["amp", "freq", "phase_dur"])
        meta.attrs["dt_ms"] = float(built[0][0].dt_ms)
        meta.attrs["costim_enabled"] = bool(built[0][0].costim_enabled)
        meta.attrs["stream"] = json.dumps(stream.__dict__ if stream else None)

        g = h5.create_group("world")
        ckw = dict(compression=compression) if compression else {}
        s_t = g.create_dataset("s_t", shape=(n_total, H, W), dtype=np.float32, **ckw)
        s_tp1 = g.create_dataset("s_tp1", shape=(n_total, H, W), dtype=np.float32, **ckw)
        aux_t = g.create_dataset("aux_t", shape=(n_total, 2, H, W), dtype=np.float32, **ckw)
        amp_d = g.create_dataset("amp", shape=(n_total, max_elec), dtype=np.float32, **ckw)
        freq_d = g.create_dataset("freq", shape=(n_total, max_elec), dtype=np.float32, **ckw)
        pdur_d = g.create_dataset("phase_dur", shape=(n_total, max_elec), dtype=np.float32, **ckw)
        rho_d = g.create_dataset("rho", shape=(n_total,), dtype=np.float32)
        axl_d = g.create_dataset("axlambda", shape=(n_total,), dtype=np.float32)
        cfg_id = g.create_dataset("config_id", shape=(n_total,), dtype=np.int32)
        episode_id = g.create_dataset("episode_id", shape=(n_total,), dtype=np.int32)
        step_in_episode = g.create_dataset("step_in_episode", shape=(n_total,), dtype=np.int32)

        i = 0
        ep_global = 0
        t_step = 0.0
        t_io = 0.0
        n_steps = 0
        pending: list[dict] = []
        slab = max(1, int(write_slab))

        def flush() -> None:
            nonlocal i
            if not pending:
                return
            n = len(pending)
            sl = slice(i, i + n)
            s_t[sl] = np.stack([row["s_t"] for row in pending])
            s_tp1[sl] = np.stack([row["s_tp1"] for row in pending])
            aux_t[sl] = np.stack([row["aux"] for row in pending])
            amp_d[sl] = np.stack([row["amp"] for row in pending])
            freq_d[sl] = np.stack([row["freq"] for row in pending])
            pdur_d[sl] = np.stack([row["pdur"] for row in pending])
            rho_d[sl] = np.array([row["rho"] for row in pending], dtype=np.float32)
            axl_d[sl] = np.array([row["axlambda"] for row in pending], dtype=np.float32)
            cfg_id[sl] = np.array([row["cfg"] for row in pending], dtype=np.int32)
            episode_id[sl] = np.array([row["ep"] for row in pending], dtype=np.int32)
            step_in_episode[sl] = np.array([row["step"] for row in pending], dtype=np.int32)
            i += n
            pending.clear()

        for cfg_idx, (cfg, implant, wm) in enumerate(built):
            n_e = implant.n_electrodes
            scale_samples = []
            for _ in range(min(episodes, 4)):
                action, _ = sample_action(cfg, n_e, rng, ranges, device)
                t0 = time.perf_counter()
                frame = wm.step(wm.initial_state(device), action).image.detach()
                _sync_device(device)
                t_step += time.perf_counter() - t0
                scale_samples.append(float(np.percentile(frame.cpu().numpy(), 99)))
            percept_scales[cfg_idx] = max(scale_samples) if scale_samples else 1.0

            for _ in range(episodes):
                state = wm.initial_state(device)
                subject = sample_action(cfg, n_e, rng, ranges, device)[1] if stream else None
                held = None
                for step in range(steps_per_ep):
                    silent = step >= sequence_length
                    if stream and not silent:
                        silent = rng.random() < stream.silent_prob
                    if (
                        stream
                        and not silent
                        and held is not None
                        and rng.random() < stream.hold_prob
                    ):
                        action, rec = held
                    else:
                        action, rec = sample_action(cfg, n_e, rng, ranges, device, silent=silent)
                    if subject is not None:
                        # One retina per video: rho and axlambda are the subject, not the frame.
                        axl = subject["axlambda"] if action.axlambda is not None else None
                        action = replace(action, rho=subject["rho"], axlambda=axl)
                        rec = {**rec, "rho": subject["rho"], "axlambda": subject["axlambda"]}
                    held = None if silent else (action, rec)
                    s_prev = state.image.detach()
                    t0 = time.perf_counter()
                    state = wm.step(state, action)
                    _sync_device(device)
                    t_step += time.perf_counter() - t0
                    frame = state.image.detach()
                    t1 = time.perf_counter()
                    a_map, q_map = _aux_maps(wm, state)
                    amp = np.zeros(max_elec, dtype=np.float32)
                    freq = np.zeros(max_elec, dtype=np.float32)
                    pdur = np.zeros(max_elec, dtype=np.float32)
                    amp[:n_e] = rec["amp"]
                    freq[:n_e] = rec["freq"]
                    pdur[:n_e] = rec["pdur"]
                    aux = np.stack([a_map, q_map], axis=0)
                    pending.append(
                        {
                            "s_t": s_prev.cpu().numpy(),
                            "s_tp1": frame.cpu().numpy(),
                            "aux": aux,
                            "amp": amp,
                            "freq": freq,
                            "pdur": pdur,
                            "rho": rec["rho"],
                            "axlambda": rec["axlambda"],
                            "cfg": cfg_idx,
                            "ep": ep_global,
                            "step": step,
                        }
                    )
                    n_steps += 1
                    if len(pending) >= slab:
                        flush()
                    t_io += time.perf_counter() - t1
                ep_global += 1
        t1 = time.perf_counter()
        flush()
        t_io += time.perf_counter() - t1

        meta.attrs["percept_scale"] = json.dumps(percept_scales)
    if timing_path is not None:
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        payload = {
            "topography_s": t_build,
            "gpu_step_s": t_step,
            "h5_write_s": t_io,
            "n_steps": n_steps,
            "samples_per_s": n_steps / max(t_step + t_io, 1e-9),
            "peak_gpu_mem_bytes": int(peak),
            "device": str(device),
            "write_slab": slab,
        }
        timing_path = Path(timing_path)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _pair(ctx, param, value):
    return (float(value[0]), float(value[1]))


@click.command(epilog="Docs: https://ascientist.github.io/sentionaut/user-guide/")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--model",
    "models",
    multiple=True,
    default=("axonmap", "scoreboard", "dynaphos"),
    show_default=True,
)
@click.option("--episodes", type=int, default=8, show_default=True)
@click.option("--sequence-length", type=int, default=4, show_default=True)
@click.option(
    "--silent-tail", type=int, default=2, show_default=True, help="Zero-drive fade steps."
)
@click.option("--dt-ms", type=float, default=20.0, show_default=True)
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
@click.option("--timing", "timing_path", type=click.Path(path_type=Path), default=None)
@click.option("--write-slab", type=int, default=32, show_default=True)
@click.option(
    "--stream", is_flag=True, help="Video episodes: held stimulation, gaps, one subject each."
)
@click.option("--hold-prob", type=float, default=0.5, show_default=True)
@click.option("--silent-prob", type=float, default=0.3, show_default=True)
def world_cli(
    output_path,
    models,
    episodes,
    sequence_length,
    silent_tail,
    dt_ms,
    xrange,
    yrange,
    xystep,
    device,
    seed,
    compression,
    timing_path,
    write_slab,
    stream,
    hold_prob,
    silent_prob,
):
    base = Config(
        model=list(models)[0],
        implant=DEFAULT_IMPLANT[list(models)[0]],
        xrange=xrange,
        yrange=yrange,
        xystep=xystep,
        device=device,
        dt_ms=dt_ms,
    )
    configs = build_configs(list(models), base)
    path = generate_world_dataset(
        output_path,
        configs,
        episodes,
        sequence_length,
        seed=seed,
        compression=compression,
        dt_ms=dt_ms,
        silent_tail=silent_tail,
        timing_path=timing_path,
        write_slab=write_slab,
        stream=StreamSchedule(hold_prob, silent_prob) if stream else None,
    )
    click.echo(str(path))


@click.command(epilog="Docs: https://ascientist.github.io/sentionaut/user-guide/")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--model", "model", type=str, default="axonmap", show_default=True)
@click.option("--samples", "samples", type=int, default=64, show_default=True)
@click.option("--dt-ms", type=float, default=20.0, show_default=True)
@click.option(
    "--xrange", nargs=2, type=float, callback=_pair, default=(-8.0, 8.0), show_default=True
)
@click.option(
    "--yrange", nargs=2, type=float, callback=_pair, default=(-8.0, 8.0), show_default=True
)
@click.option("--xystep", type=float, default=0.5, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
def cli(output_path, model, samples, dt_ms, xrange, yrange, xystep, device, seed):
    base = Config(
        model=model,
        implant=DEFAULT_IMPLANT[model],
        xrange=xrange,
        yrange=yrange,
        xystep=xystep,
        device=device,
        dt_ms=dt_ms,
    )
    path = generate_world_dataset(
        output_path,
        build_configs([model], base),
        episodes=samples,
        sequence_length=1,
        seed=seed,
        dt_ms=dt_ms,
    )
    click.echo(str(path))


if __name__ == "__main__":  # pragma: no cover
    world_cli()
