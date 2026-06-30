"""Factory that builds a bound ``(implant, topography, model)`` from a ``Config``."""

from __future__ import annotations

import torch

from .base import Implant, PerceptModel, Topography
from .config import Config
from .device import get_device


def build_implant(config: Config, device: torch.device) -> Implant:
    from ..implants.registry import build_implant as _build

    return _build(config, device)


def build_topography(config: Config, device: torch.device) -> Topography:
    if config.is_cortical:
        if config.use_neuropythy:
            from ..topography.neuropythy import NeuropythyTopography

            return NeuropythyTopography.build(config, device)
        from ..topography.cortical import CorticalTopography

        return CorticalTopography.build(config, device)
    from ..topography.axon_map import AxonMapTopography

    return AxonMapTopography.build(config, device)


def build_model(config: Config) -> PerceptModel:
    if config.model == "axonmap":
        from ..models.axonmap import BiphasicAxonMapTorch

        return BiphasicAxonMapTorch(rho=config.rho, axlambda=config.axlambda)
    if config.model == "scoreboard":
        from ..models.scoreboard import ScoreboardTorch

        return ScoreboardTorch(rho=config.rho)
    if config.model == "dynaphos":
        from ..models.dynaphos import DynaphosTorch

        return DynaphosTorch()
    raise ValueError(f"Unknown model '{config.model}'.")


def build_components(
    config: Config, device: torch.device | None = None
) -> tuple[Implant, Topography, PerceptModel]:
    """Build and bind the three swappable axes for a config."""
    device = device or get_device(config.device)
    implant = build_implant(config, device)
    topography = build_topography(config, device)
    model = build_model(config).to(device)
    model.build(implant, topography)
    return implant, topography, model
