"""Device selection: prefer Apple MPS, fall back to CPU."""

from __future__ import annotations

import os

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Order of preference: explicit ``prefer`` (or ``SENTIONAUT_DEVICE`` env var),
    then CUDA, then Apple MPS, then CPU.
    """
    requested = prefer or os.environ.get("SENTIONAUT_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype() -> torch.dtype:
    return torch.float32
