"""Shared interfaces for the swappable (Implant x Topography x PerceptModel) axes.

Every ``PerceptModel`` exposes the same world-model interface
``f(s_t, a_t) -> s_{t+1}`` so analytical physics models and the learned model are
interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class Pose:
    """Rigid pose applied to an implant: translation (microns) + rotation (rad)."""

    dx: float = 0.0
    dy: float = 0.0
    rot: float = 0.0


@dataclass
class Action:
    """Unified per-electrode action container.

    Fields not used by a given model are simply ignored by that model (e.g. a
    stateless spatial model has no use for ``delay``). All per-electrode tensors
    are shape ``(n_electrodes,)`` unless batched, in which case they are
    ``(batch, n_electrodes)``.
    """

    amp: torch.Tensor
    freq: torch.Tensor | None = None
    phase_dur: torch.Tensor | None = None
    delay: torch.Tensor | None = None
    # Model-level spatial params (scalars or per-batch); masked per model.
    rho: torch.Tensor | float | None = None
    axlambda: torch.Tensor | float | None = None
    pose: Pose = field(default_factory=Pose)

    def to(self, device: torch.device) -> "Action":
        def mv(t):
            return t.to(device) if isinstance(t, torch.Tensor) else t

        return Action(
            amp=mv(self.amp),
            freq=mv(self.freq),
            phase_dur=mv(self.phase_dur),
            delay=mv(self.delay),
            rho=mv(self.rho),
            axlambda=mv(self.axlambda),
            pose=self.pose,
        )


@dataclass
class State:
    """Percept state. ``image`` is ``(H, W)`` or ``(batch, H, W)``.

    ``aux`` carries optional temporal channels (e.g. Dynaphos charge/activation
    accumulators) threaded across ``step`` calls.
    """

    image: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class Implant(ABC):
    """Electrode geometry (microns, tissue/cortex space) + applied pose."""

    names: list[str]

    @abstractmethod
    def electrode_coords(self, pose: Pose | None = None) -> torch.Tensor:
        """Return electrode coordinates as ``(n_electrodes, 2)`` (x, y) in microns."""

    @property
    @abstractmethod
    def n_electrodes(self) -> int: ...


class Topography(ABC):
    """Maps percept grid points (dva) <-> tissue coordinates as GPU tensors."""

    grid_shape: tuple[int, int]

    @abstractmethod
    def to(self, device: torch.device) -> "Topography": ...


class PerceptModel(nn.Module, ABC):
    """Differentiable percept model with a world-model interface."""

    def __init__(self) -> None:
        super().__init__()
        self.implant: Implant | None = None
        self.topography: Topography | None = None
        self._built = False

    @abstractmethod
    def build(self, implant: Implant, topography: Topography) -> "PerceptModel":
        """Bind implant + topography and precompute cached tensors."""

    @abstractmethod
    def forward(self, action: Action) -> torch.Tensor:
        """Render a percept image ``(H, W)`` (or batched) from an action."""

    def step(self, state: State | None, action: Action) -> State:
        """Advance the world model one step. Stateless models ignore ``state``."""
        image = self.forward(action)
        return State(image=image)

    @property
    def is_built(self) -> bool:
        return self._built
