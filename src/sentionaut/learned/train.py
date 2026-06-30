"""Distillation training loop + specialist-vs-shared ablation harness.

Heavy/full-scale training is launched on the cluster; locally only the
single-batch smoke path is exercised (see tests).
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.config import ScaleConfig
from ..core.device import get_device
from .dataset import WorldTransitionDataset
from .model import UnifiedWorldModel


def build_model(dataset: WorldTransitionDataset, mode: str = "shared") -> UnifiedWorldModel:
    return UnifiedWorldModel(
        action_dim=dataset.action_dim,
        n_models=dataset.n_models,
        n_implants=dataset.n_implants,
        mode=mode,
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def step_loss(model: UnifiedWorldModel, batch: dict) -> torch.Tensor:
    pred = model(
        batch["s_t"], batch["action"], batch["model_id"], batch["implant_id"], batch["topo_params"]
    )
    return F.mse_loss(pred, batch["s_tp1"])


def single_batch_step(model: UnifiedWorldModel, batch: dict, device: torch.device) -> float:
    """One optimizer step on a single batch (used by the smoke test)."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _move(batch, device)
    opt.zero_grad()
    loss = step_loss(model, batch)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


def train(
    dataset_path: str,
    scale: ScaleConfig,
    mode: str = "shared",
    device: torch.device | None = None,
) -> dict:
    device = device or get_device()
    ds = WorldTransitionDataset(dataset_path)
    loader = DataLoader(ds, batch_size=scale.batch_size, shuffle=True)
    model = build_model(ds, mode=mode).to(device)
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
    return {"mode": mode, "loss_history": history, "model": model}


def evaluate(model: UnifiedWorldModel, dataset_path: str, device: torch.device) -> float:
    ds = WorldTransitionDataset(dataset_path)
    loader = DataLoader(ds, batch_size=16)
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            total += float(step_loss(model, batch).cpu())
            count += 1
    return total / max(count, 1)


def ablate(dataset_path: str, scale: ScaleConfig, device: torch.device | None = None) -> dict:
    """Run the shared-vs-specialist ablation, reporting parity-to-analytical MSE."""
    device = device or get_device()
    results = {}
    for mode in ("shared", "specialist"):
        out = train(dataset_path, scale, mode=mode, device=device)
        mse = evaluate(out["model"], dataset_path, device)
        results[mode] = {"final_train_loss": out["loss_history"][-1], "eval_mse": mse}
    return results


@click.group()
def cli():
    """Train / ablate the unified learned world model."""


@cli.command(name="train")
@click.option("--dataset", "dataset_path", type=click.Path(), required=True)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--mode", type=click.Choice(["shared", "specialist"]), default="shared")
@click.option("--device", type=str, default=None)
def train_cmd(dataset_path, config_path, mode, device):
    scale = ScaleConfig.from_yaml(config_path) if config_path else ScaleConfig()
    dev = get_device(device)
    out = train(dataset_path, scale, mode=mode, device=dev)
    click.echo(f"loss_history={out['loss_history']}")


@cli.command(name="ablate")
@click.option("--dataset", "dataset_path", type=click.Path(), required=True)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--device", type=str, default=None)
def ablate_cmd(dataset_path, config_path, device):
    scale = ScaleConfig.from_yaml(config_path) if config_path else ScaleConfig()
    dev = get_device(device)
    results = ablate(dataset_path, scale, device=dev)
    for mode, r in results.items():
        click.echo(f"{mode}: {r}")


if __name__ == "__main__":  # pragma: no cover
    cli()


def _smoke_dataset(tmp_path: Path) -> str:  # pragma: no cover - helper for tests
    from ..core.config import Config
    from ..generate import build_configs, generate_world_dataset

    base = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    out = tmp_path / "smoke.h5"
    generate_world_dataset(out, build_configs(["axonmap"], base), episodes=2, sequence_length=1)
    return str(out)
