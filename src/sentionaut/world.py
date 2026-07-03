"""``WorldModel``: standardizes any ``PerceptModel`` to ``f(s_t, a_t) -> s_{t+1}``.

All three analytical models are stateful: Axon Map and Scoreboard carry a fading
brightness field via ``FadingTemporal``; Dynaphos additionally threads per-
electrode activation/charge in ``State.aux`` and exposes rasterized A/Q maps.
"""

from __future__ import annotations

import torch

from .core.base import Action, PerceptModel, State
from .core.config import Config
from .core.registry import build_components


class WorldModel:
    def __init__(self, model: PerceptModel, config: Config):
        self.model = model
        self.config = config

    @classmethod
    def from_config(cls, config: Config, device: torch.device | None = None) -> "WorldModel":
        _, _, model = build_components(config, device)
        return cls(model, config)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.model.topography.grid_shape

    def initial_state(self, device: torch.device | None = None) -> State:
        if hasattr(self.model, "initial_state"):
            return self.model.initial_state(device)
        H, W = self.grid_shape
        dev = device or self._device()
        return State(image=torch.zeros(H, W, device=dev))

    def _device(self) -> torch.device:
        topo = self.model.topography
        ref = getattr(topo, "coords", None)
        ref = ref if ref is not None else topo.grid_x
        return ref.device

    def step(self, state: State | None, action: Action) -> State:
        return self.model.step(state, action)

    def rollout(self, actions: list[Action], state: State | None = None) -> list[State]:
        states = []
        for action in actions:
            state = self.step(state, action)
            states.append(state)
        return states
