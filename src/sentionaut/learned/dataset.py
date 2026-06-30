"""Torch ``Dataset`` over the multi-config world HDF5."""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.config import Config


class WorldTransitionDataset(Dataset):
    """Yields ``(s_t, s_tp1, action, model_id, implant_id, topo_params)`` transitions."""

    MODEL_IDS = {"axonmap": 0, "scoreboard": 1, "dynaphos": 2}
    IMPLANT_IDS = {"argusii": 0, "alphaims": 1, "orion": 2, "cortivis": 3, "icvp": 4}

    def __init__(self, path: str):
        self.path = path
        with h5py.File(path, "r") as h5:
            table = json.loads(h5["metadata"].attrs["config_table"])
            self.configs = [Config.from_dict(d) for d in table]
            self.grid_shape = tuple(h5["metadata"].attrs["grid_shape"])
            self.max_elec = int(h5["metadata"].attrs["max_electrodes"])
            self.n = h5["world"]["s_t"].shape[0]
        self._h5: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    @property
    def action_dim(self) -> int:
        return self.max_elec * 3 + 2

    @property
    def n_models(self) -> int:
        return len(self.MODEL_IDS)

    @property
    def n_implants(self) -> int:
        return len(self.IMPLANT_IDS)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = self._file()["world"]
        cfg = self.configs[int(g["config_id"][i])]
        amp = g["amp"][i]
        freq = g["freq"][i]
        pdur = g["phase_dur"][i]
        rho = float(g["rho"][i])
        axl = float(g["axlambda"][i])
        action = np.concatenate([amp, freq, pdur, [rho, axl]]).astype(np.float32)
        return {
            "s_t": torch.from_numpy(g["s_t"][i][None].astype(np.float32)),
            "s_tp1": torch.from_numpy(g["s_tp1"][i][None].astype(np.float32)),
            "action": torch.from_numpy(action),
            "model_id": torch.tensor(self.MODEL_IDS[cfg.model], dtype=torch.long),
            "implant_id": torch.tensor(self.IMPLANT_IDS[cfg.implant], dtype=torch.long),
            "topo_params": torch.tensor([rho / 1000.0, axl / 1000.0], dtype=torch.float32),
        }
