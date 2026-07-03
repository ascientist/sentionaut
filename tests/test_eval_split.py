"""Train/val split must not leak config ids."""

from __future__ import annotations

import torch

from sentionaut.core.config import Config
from sentionaut.generate import build_configs, generate_world_dataset
from sentionaut.learned.dataset import WorldTransitionDataset, train_val_split


def test_no_config_leakage(tmp_path):
    base = Config(model="axonmap", implant="argusii", xrange=(-4, 4), yrange=(-4, 4), xystep=2.0)
    out = tmp_path / "split.h5"
    generate_world_dataset(
        out,
        build_configs(["axonmap", "scoreboard"], base),
        episodes=3,
        sequence_length=2,
        device=torch.device("cpu"),
        seed=0,
    )
    full = WorldTransitionDataset(str(out))
    train_ds, val_ds = train_val_split(full, val_config_ids=[1])
    train_cfgs = set()
    val_cfgs = set()
    for ds, acc in ((train_ds, train_cfgs), (val_ds, val_cfgs)):
        for j in range(len(ds)):
            acc.add(int(ds[j]["config_id"]))
    assert train_cfgs.isdisjoint(val_cfgs)
    assert 1 in val_cfgs
