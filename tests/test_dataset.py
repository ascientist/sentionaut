"""Multi-config HDF5 round-trip on a tiny 2-episode dataset."""

from __future__ import annotations

import h5py
import torch

from sentionaut.core.config import Config
from sentionaut.generate import build_configs, generate_world_dataset
from sentionaut.learned.dataset import WorldTransitionDataset


def test_multi_config_roundtrip(tmp_path):
    base = Config(
        model="scoreboard",
        implant="orion",
        xrange=(-4, 4),
        yrange=(-4, 4),
        xystep=2.0,
        regions=("v1",),
    )
    configs = build_configs(["scoreboard", "dynaphos"], base)
    out = tmp_path / "world.h5"
    generate_world_dataset(
        out, configs, episodes=2, sequence_length=2, device=torch.device("cpu"), seed=0
    )

    with h5py.File(out, "r") as h5:
        n = h5["world"]["s_t"].shape[0]
        assert n == 2 * 2 * 2  # 2 configs * 2 episodes * 2 steps
        assert set(h5["world"].keys()) >= {"s_t", "s_tp1", "amp", "config_id"}

    ds = WorldTransitionDataset(str(out))
    assert len(ds) == 8
    sample = ds[0]
    H, W = ds.grid_shape
    assert sample["s_t"].shape == (1, H, W)
    assert sample["s_tp1"].shape == (1, H, W)
    assert sample["action"].shape[0] == ds.action_dim
    assert sample["model_id"].dtype == torch.long


def test_config_ids_cover_both(tmp_path):
    base = Config(
        model="scoreboard",
        implant="orion",
        xrange=(-4, 4),
        yrange=(-4, 4),
        xystep=2.0,
        regions=("v1",),
    )
    configs = build_configs(["scoreboard", "dynaphos"], base)
    out = tmp_path / "w.h5"
    generate_world_dataset(
        out, configs, episodes=1, sequence_length=2, device=torch.device("cpu"), seed=1
    )
    with h5py.File(out, "r") as h5:
        ids = set(int(x) for x in h5["world"]["config_id"][:])
    assert ids == {0, 1}
