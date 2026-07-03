"""Registry coverage: every implant builds and yields electrode coords."""

from __future__ import annotations

import pytest
import torch

from sentionaut.core.config import CORTICAL_IMPLANTS, RETINAL_IMPLANTS, Config
from sentionaut.implants.registry import build_implant


@pytest.mark.parametrize("name", [n for n in sorted(RETINAL_IMPLANTS) if n != "prima"])
def test_retinal_implants_build(name):
    cfg = Config(model="axonmap", implant=name)
    imp = build_implant(cfg, torch.device("cpu"))
    coords = imp.electrode_coords()
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert imp.n_electrodes == coords.shape[0] > 0


def test_prima_axonmap_rejected():
    with pytest.raises(ValueError, match="PRIMA"):
        Config(model="axonmap", implant="prima")


@pytest.mark.parametrize("name", sorted(CORTICAL_IMPLANTS))
def test_cortical_implants_build(name):
    cfg = Config(model="scoreboard", implant=name)
    imp = build_implant(cfg, torch.device("cpu"))
    coords = imp.electrode_coords()
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert imp.n_electrodes == coords.shape[0] > 0
