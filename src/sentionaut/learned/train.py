"""Distillation training loop + specialist-vs-shared ablation harness."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.config import ScaleConfig
from ..core.device import get_device
from .dataset import WorldTransitionDataset, train_val_split
from .metrics import batch_metrics
from .model import UnifiedWorldModel


def build_model(dataset: WorldTransitionDataset, mode: str = "shared") -> UnifiedWorldModel:
    return UnifiedWorldModel(
        action_dim=dataset.action_dim,
        n_models=dataset.n_models,
        n_implants=dataset.n_implants,
        mode=mode,
        in_channels=3,
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def step_loss(model: UnifiedWorldModel, batch: dict) -> torch.Tensor:
    pred = model(
        batch["s_t"], batch["action"], batch["model_id"], batch["implant_id"], batch["topo_params"]
    )
    return F.mse_loss(pred, batch["s_tp1"])


def single_batch_step(model: UnifiedWorldModel, batch: dict, device: torch.device) -> float:
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _move(batch, device)
    opt.zero_grad()
    loss = step_loss(model, batch)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


def evaluate(
    model: UnifiedWorldModel, dataset: WorldTransitionDataset, device: torch.device
) -> dict:
    loader = DataLoader(dataset, batch_size=16)
    model.eval()
    totals = {"mse": 0.0, "ssim": 0.0, "max_abs": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            pred = model(
                batch["s_t"],
                batch["action"],
                batch["model_id"],
                batch["implant_id"],
                batch["topo_params"],
            )
            m = batch_metrics(pred, batch["s_tp1"])
            for k in totals:
                totals[k] += m[k]
            count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


def train(
    dataset_path: str,
    scale: ScaleConfig,
    mode: str = "shared",
    device: torch.device | None = None,
    *,
    log_path: Path | None = None,
) -> dict:
    device = device or get_device()
    full = WorldTransitionDataset(dataset_path)
    train_ds, val_ds = train_val_split(full)
    loader = DataLoader(train_ds, batch_size=scale.batch_size, shuffle=True)
    model = build_model(full, mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=scale.lr)
    history = []
    for _ in range(scale.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = _move(batch, device)
            opt.zero_grad()
            loss = step_loss(model, batch)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            count += 1
        history.append(total / max(count, 1))
    eval_metrics = evaluate(model, val_ds, device)
    result = {"mode": mode, "loss_history": history, "eval": eval_metrics, "model": model}
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as fh:
            json.dump({"mode": mode, "loss_history": history, "eval": eval_metrics}, fh, indent=2)
    return result


def ablate(
    dataset_path: str,
    scale: ScaleConfig,
    device: torch.device | None = None,
    *,
    log_dir: Path | None = None,
) -> dict:
    device = device or get_device()
    results = {}
    for mode in ("shared", "specialist", "shared_trunk"):
        log_path = Path(log_dir) / f"eval_{mode}.json" if log_dir else None
        out = train(dataset_path, scale, mode=mode, device=device, log_path=log_path)
        results[mode] = {"final_train_loss": out["loss_history"][-1], **out["eval"]}
    if log_dir is not None:
        with open(Path(log_dir) / "eval.json", "w") as fh:
            json.dump(results, fh, indent=2)
    return results


@click.group()
def cli():
    """Train / ablate the unified learned world model."""


@cli.command(name="train")
@click.option("--dataset", "dataset_path", type=click.Path(), required=True)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option(
    "--mode", type=click.Choice(["shared", "specialist", "shared_trunk"]), default="shared"
)
@click.option("--device", type=str, default=None)
@click.option("--log", "log_path", type=click.Path(path_type=Path), default=None)
def train_cmd(dataset_path, config_path, mode, device, log_path):
    scale = ScaleConfig.from_yaml(config_path) if config_path else ScaleConfig()
    dev = get_device(device)
    out = train(dataset_path, scale, mode=mode, device=dev, log_path=log_path)
    click.echo(f"loss_history={out['loss_history']} eval={out['eval']}")


@cli.command(name="ablate")
@click.option("--dataset", "dataset_path", type=click.Path(), required=True)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--device", type=str, default=None)
@click.option("--log-dir", type=click.Path(path_type=Path), default="logs")
def ablate_cmd(dataset_path, config_path, device, log_dir):
    scale = ScaleConfig.from_yaml(config_path) if config_path else ScaleConfig()
    dev = get_device(device)
    results = ablate(dataset_path, scale, device=dev, log_dir=log_dir)
    for mode, r in results.items():
        click.echo(f"{mode}: {r}")


if __name__ == "__main__":  # pragma: no cover
    cli()
