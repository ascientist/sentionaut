"""Torch ``Dataset`` over the multi-config world HDF5."""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ..core.config import Config


class WorldTransitionDataset(Dataset):
    """Yields ``(s_t, s_tp1, action, model_id, implant_id, topo_params)`` transitions."""

    MODEL_IDS = {"axonmap": 0, "scoreboard": 1, "dynaphos": 2}
    IMPLANT_IDS = {
        "argusii": 0,
        "alphaims": 1,
        "alphaams": 2,
        "prima": 3,
        "grid": 4,
        "orion": 5,
        "cortivis": 6,
        "icvp": 7,
        "neuralink": 8,
    }

    def __init__(self, path: str, indices: list[int] | None = None):
        self.path = path
        with h5py.File(path, "r") as h5:
            table = json.loads(h5["metadata"].attrs["config_table"])
            self.configs = [Config.from_dict(d) for d in table]
            self.grid_shape = tuple(h5["metadata"].attrs["grid_shape"])
            self.max_elec = int(h5["metadata"].attrs["max_electrodes"])
            self.n = h5["world"]["s_t"].shape[0]
            scale_raw = h5["metadata"].attrs.get("percept_scale")
            self.percept_scale = (
                {int(k): float(v) for k, v in json.loads(scale_raw).items()}
                if scale_raw
                else {i: 1.0 for i in range(len(self.configs))}
            )
            self._has_aux = "aux_t" in h5["world"]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self.n

    @property
    def action_dim(self) -> int:
        return self.max_elec * 3 + 2

    @property
    def n_models(self) -> int:
        return len(self.MODEL_IDS)

    @property
    def n_implants(self) -> int:
        return len(self.IMPLANT_IDS)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i] if self.indices is not None else i
        with h5py.File(self.path, "r") as h5:
            g = h5["world"]
            cfg_idx = int(g["config_id"][idx])
            cfg = self.configs[cfg_idx]
            scale = self.percept_scale.get(cfg_idx, 1.0)
            s_t = g["s_t"][idx].astype(np.float32) / scale
            s_tp1 = g["s_tp1"][idx].astype(np.float32) / scale
            if self._has_aux:
                a_map = g["aux_t"][idx, 0].astype(np.float32) / scale
                q_map = g["aux_t"][idx, 1].astype(np.float32) / scale
            else:
                a_map = q_map = np.zeros_like(s_t)
            amp = g["amp"][idx]
            freq = g["freq"][idx]
            pdur = g["phase_dur"][idx]
            rho = float(g["rho"][idx])
            axl = float(g["axlambda"][idx])
        stacked = np.stack([s_t, a_map, q_map], axis=0)
        action = np.concatenate([amp, freq, pdur, [rho, axl]]).astype(np.float32)
        return {
            "s_t": torch.from_numpy(stacked),
            "s_tp1": torch.from_numpy(s_tp1[None].astype(np.float32)),
            "action": torch.from_numpy(action),
            "model_id": torch.tensor(self.MODEL_IDS[cfg.model], dtype=torch.long),
            "implant_id": torch.tensor(self.IMPLANT_IDS[cfg.implant], dtype=torch.long),
            "topo_params": torch.tensor([rho / 1000.0, axl / 1000.0], dtype=torch.float32),
            "config_id": torch.tensor(cfg_idx, dtype=torch.long),
        }


def train_val_split(
    dataset: WorldTransitionDataset,
    *,
    val_config_ids: list[int] | None = None,
    holdout_implant: str | None = "neuralink",
) -> tuple[WorldTransitionDataset, WorldTransitionDataset]:
    """Hold out rows by ``config_id`` (and optionally one implant)."""
    with h5py.File(dataset.path, "r") as h5:
        cfg_ids = h5["world"]["config_id"][:]
    n_cfgs = len(dataset.configs)
    if val_config_ids is None:
        val_config_ids = [n_cfgs - 1] if n_cfgs > 1 else []
    val_set = set(val_config_ids)
    train_idx, val_idx = [], []
    for i, cid in enumerate(cfg_ids):
        cfg = dataset.configs[int(cid)]
        if int(cid) in val_set or (holdout_implant and cfg.implant == holdout_implant):
            val_idx.append(i)
        else:
            train_idx.append(i)
    if not val_idx:
        val_idx = train_idx[-max(1, len(train_idx) // 5) :]
        train_idx = train_idx[: -len(val_idx)]
    return (
        WorldTransitionDataset(dataset.path, indices=train_idx),
        WorldTransitionDataset(dataset.path, indices=val_idx),
    )
