"""Single-batch training step + single-batch inference for UnifiedWorldModel."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from sentionaut.core.config import Config
from sentionaut.generate import build_configs, generate_world_dataset
from sentionaut.learned.dataset import WorldTransitionDataset
from sentionaut.learned.model import UnifiedWorldModel
from sentionaut.learned.train import build_model, single_batch_step


def _tiny_dataset(tmp_path):
    base = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    out = tmp_path / "smoke.h5"
    generate_world_dataset(
        out,
        build_configs(["axonmap"], base),
        episodes=4,
        sequence_length=1,
        device=torch.device("cpu"),
        seed=0,
    )
    return WorldTransitionDataset(str(out))


def test_single_batch_train_step(tmp_path):
    ds = _tiny_dataset(tmp_path)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    model = build_model(ds, mode="shared")
    loss = single_batch_step(model, batch, torch.device("cpu"))
    assert math.isfinite(loss)


def test_single_batch_inference(tmp_path):
    ds = _tiny_dataset(tmp_path)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    model = build_model(ds, mode="shared").eval()
    with torch.no_grad():
        out = model(
            batch["s_t"],
            batch["action"],
            batch["model_id"],
            batch["implant_id"],
            batch["topo_params"],
        )
    assert out.shape == batch["s_tp1"].shape
    assert torch.isfinite(out).all()


def test_specialist_mode_builds(tmp_path):
    ds = _tiny_dataset(tmp_path)
    model = UnifiedWorldModel(ds.action_dim, ds.n_models, ds.n_implants, mode="specialist")
    batch = next(iter(DataLoader(ds, batch_size=2)))
    out = model(
        batch["s_t"], batch["action"], batch["model_id"], batch["implant_id"], batch["topo_params"]
    )
    assert out.shape == batch["s_tp1"].shape
