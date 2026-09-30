"""Axon-map rollouts whose stimulation is read off a MNIST digit.

The published student was trained on random electrode draws. This generator
keeps the same teacher and the same HDF5 schema, and replaces that draw: each
episode starts from a digit painted on the percept grid, and each electrode's
amplitude is the brightness of the digit in that electrode's cell of the Argus
II layout. Pulse width, rate, rho, and lambda stay fixed.
"""

from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ..core.base import Action, State
from ..core.config import Config
from ..core.registry import build_components
from ..world import WorldModel

_MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"


def download_mnist(dest: Path) -> np.ndarray:
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz = dest.with_suffix(".gz")
    if not gz.exists():
        urllib.request.urlretrieve(_MNIST_URL, gz)
    with gzip.open(gz, "rb") as fh:
        _, n, rows, cols = struct.unpack(">IIII", fh.read(16))
        images = np.frombuffer(fh.read(), dtype=np.uint8).reshape(n, rows, cols)
    return images


def argus_cell(name: str) -> tuple[int, int] | None:
    if len(name) < 2 or not name[0].isalpha():
        return None
    row = ord(name[0].upper()) - ord("A")
    try:
        col = int(name[1:]) - 1
    except ValueError:
        return None
    if not (0 <= row < 6 and 0 <= col < 10):
        return None
    return row, col


def amps_from_digit(image: np.ndarray, names: list[str]) -> np.ndarray:
    """Map a ``(H, W)`` digit in ``[0, 1]`` onto Argus II amplitudes (× threshold)."""
    h, w = image.shape
    amp = np.zeros(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        cell = argus_cell(name)
        if cell is None:
            continue
        r, c = cell
        patch = image[r * h // 6 : (r + 1) * h // 6, c * w // 10 : (c + 1) * w // 10]
        if patch.size == 0 or float(patch.mean()) < 0.15:
            continue
        amp[i] = 0.5 + 2.5 * float(patch.mean())
    if amp.max() == 0:
        amp[int(np.argmax(image)) % len(names)] = 1.0
    return amp


def _resize(image: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
    out = F.interpolate(t, size=grid, mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def generate_mnist_dataset(
    output_path: Path,
    *,
    episodes: int = 512,
    stimulated: int = 4,
    silent_tail: int = 2,
    device: torch.device | None = None,
    cache_dir: Path | None = None,
    seed: int = 0,
) -> Path:
    device = device or torch.device("cpu")
    cfg = Config(
        model="axonmap",
        implant="argusii",
        xrange=(-12, 12),
        yrange=(-12, 12),
        xystep=0.25,
        device=str(device),
    )
    implant, _, teacher = build_components(cfg, device)
    wm = WorldModel(teacher, cfg)
    h, w = wm.grid_shape
    images = download_mnist((cache_dir or output_path.parent) / "mnist-images")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(images), size=episodes, replace=False)
    n_e = implant.n_electrodes
    steps = stimulated + silent_tail
    n_total = episodes * steps

    frames = []
    for ep, idx in enumerate(pick):
        digit = _resize(images[int(idx)], (h, w))
        amp = amps_from_digit(digit, implant.names)
        state = State(image=torch.from_numpy(digit).to(device))
        for step in range(steps):
            use = amp if step < stimulated else np.zeros(n_e, dtype=np.float32)
            action = Action(
                amp=torch.from_numpy(use).to(device),
                freq=torch.full((n_e,), 40.0, device=device),
                phase_dur=torch.full((n_e,), 0.2, device=device),
                rho=200.0,
                axlambda=500.0,
            )
            prev = state.image.detach()
            state = wm.step(state, action)
            frames.append((prev.cpu().numpy(), state.image.detach().cpu().numpy(), use, ep, step))
        if device.type == "cuda":
            torch.cuda.synchronize()

    scale = float(np.percentile([f[1] for f in frames[: min(32, len(frames))]], 99)) or 1.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["config_table"] = json.dumps([cfg.to_dict()])
        meta.attrs["grid_shape"] = (h, w)
        meta.attrs["max_electrodes"] = n_e
        meta.attrs["dt_ms"] = float(cfg.dt_ms)
        meta.attrs["percept_scale"] = json.dumps({"0": scale})
        meta.attrs["source"] = "mnist"
        g = h5.create_group("world")
        g.create_dataset("s_t", data=np.stack([f[0] for f in frames]).astype(np.float32))
        g.create_dataset("s_tp1", data=np.stack([f[1] for f in frames]).astype(np.float32))
        g.create_dataset("aux_t", data=np.zeros((n_total, 2, h, w), dtype=np.float32))
        amp_d = np.zeros((n_total, n_e), dtype=np.float32)
        for i, f in enumerate(frames):
            amp_d[i] = f[2]
        g.create_dataset("amp", data=amp_d)
        g.create_dataset("freq", data=np.full((n_total, n_e), 40.0, dtype=np.float32))
        g.create_dataset("phase_dur", data=np.full((n_total, n_e), 0.2, dtype=np.float32))
        g.create_dataset("rho", data=np.full((n_total,), 200.0, dtype=np.float32))
        g.create_dataset("axlambda", data=np.full((n_total,), 500.0, dtype=np.float32))
        g.create_dataset("config_id", data=np.zeros((n_total,), dtype=np.int32))
        g.create_dataset("episode_id", data=np.array([f[3] for f in frames], dtype=np.int32))
        g.create_dataset("step_in_episode", data=np.array([f[4] for f in frames], dtype=np.int32))
    return output_path
