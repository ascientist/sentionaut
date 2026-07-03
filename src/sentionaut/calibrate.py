"""Subject-specific rho / axlambda calibration (Beyeler 2019 grid search)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .core.base import Action, Pose
from .core.config import Config
from .core.registry import build_components
from .learned.metrics import max_abs, mse, ssim


@dataclass
class SubjectCalibration:
    rho: float
    axlambda: float
    eye: str = "RE"
    fit_error: float = 0.0
    subject_id: str | None = None


def load_calibration(path: str | Path) -> SubjectCalibration:
    with open(path) as fh:
        d = json.load(fh)
    return SubjectCalibration(**{k: v for k, v in d.items() if k in SubjectCalibration.__dataclass_fields__})


def save_calibration(cal: SubjectCalibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(asdict(cal), fh, indent=2)
    return path


def _render_percept(cfg: Config, rho: float, axlambda: float, targets: list[dict]) -> list[np.ndarray]:
    device = torch.device("cpu")
    c = Config(**{**cfg.to_dict(), "rho": rho, "axlambda": axlambda, "model": "axonmap"})
    implant, _, model = build_components(c, device)
    percepts = []
    for t in targets:
        amp = torch.zeros(implant.n_electrodes)
        idx = implant.names.index(t["electrode"])
        amp[idx] = t.get("amp", 2.0)
        freq = torch.full((implant.n_electrodes,), t.get("freq", 30.0))
        pdur = torch.full((implant.n_electrodes,), t.get("phase_dur", 0.45))
        act = Action(
            amp=amp,
            freq=freq,
            phase_dur=pdur,
            rho=rho,
            axlambda=axlambda,
            pose=Pose(),
        )
        percepts.append(model.forward(act).detach().numpy())
    return percepts


def calibrate_subject(
    cfg: Config,
    targets: list[dict],
    *,
    rho_range: tuple[float, float] = (150.0, 350.0),
    axlambda_range: tuple[float, float] = (400.0, 700.0),
    n_grid: int = 11,
    holdout: list[int] | None = None,
) -> SubjectCalibration:
    """Grid search (rho, axlambda) minimizing MSE+SSIM error on held-out electrodes."""
    holdout = holdout or []
    train = [t for i, t in enumerate(targets) if i not in holdout]
    test = [t for i, t in enumerate(targets) if i in holdout] or train

    rhos = np.linspace(rho_range[0], rho_range[1], n_grid)
    axls = np.linspace(axlambda_range[0], axlambda_range[1], n_grid)
    best_err, best = float("inf"), (cfg.rho, cfg.axlambda)

    for rho in rhos:
        for axl in axls:
            preds = _render_percept(cfg, float(rho), float(axl), test)
            err = 0.0
            for pred, t in zip(preds, test):
                ref = np.asarray(t["target"], dtype=np.float32)
                err += mse(pred, ref) + (1.0 - ssim(pred, ref))
            if err < best_err:
                best_err, best = err, (float(rho), float(axl))

    return SubjectCalibration(
        rho=best[0],
        axlambda=best[1],
        eye=cfg.eye,
        fit_error=best_err / max(len(test), 1),
    )


def cli():  # pragma: no cover
    import click

    @click.command()
    @click.option("--implant", type=str, default="argusii")
    @click.option("--targets", type=click.Path(exists=True, path_type=Path), required=True)
    @click.option("--output", type=click.Path(path_type=Path), required=True)
    @click.option("--subject-id", type=str, default=None)
    def _cli(implant, targets, output, subject_id):
        with open(targets) as fh:
            data = json.load(fh)
        cfg = Config(model="axonmap", implant=implant)
        cal = calibrate_subject(cfg, data["targets"])
        cal.subject_id = subject_id
        save_calibration(cal, output)
        click.echo(str(output))

    _cli()
