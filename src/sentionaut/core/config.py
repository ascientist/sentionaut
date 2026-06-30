"""Config dataclass + YAML loader so swapping components is a one-line change."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Canonical component ids. Implant family is implied by the model family:
# retinal models pair with retinal implants, cortical with cortical.
RETINAL_MODELS = {"axonmap"}
CORTICAL_MODELS = {"scoreboard", "dynaphos"}
RETINAL_IMPLANTS = {"argusii", "alphaims"}
CORTICAL_IMPLANTS = {"orion", "cortivis", "icvp"}


@dataclass
class Config:
    model: str = "axonmap"
    implant: str = "argusii"
    # Percept grid (degrees of visual angle). Small defaults keep M1 memory low.
    xrange: tuple[float, float] = (-12.0, 12.0)
    yrange: tuple[float, float] = (-12.0, 12.0)
    xystep: float = 0.5
    # Spatial params (microns). rho is shared; axlambda is retinal-only.
    rho: float = 200.0
    axlambda: float = 500.0
    # Cortical regions to simulate.
    regions: tuple[str, ...] = ("v1",)
    # Retinal eye.
    eye: str = "RE"
    device: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.model = self.model.lower()
        self.implant = self.implant.lower()
        if self.model in RETINAL_MODELS and self.implant not in RETINAL_IMPLANTS:
            raise ValueError(
                f"Retinal model '{self.model}' needs a retinal implant "
                f"({sorted(RETINAL_IMPLANTS)}), got '{self.implant}'."
            )
        if self.model in CORTICAL_MODELS and self.implant not in CORTICAL_IMPLANTS:
            raise ValueError(
                f"Cortical model '{self.model}' needs a cortical implant "
                f"({sorted(CORTICAL_IMPLANTS)}), got '{self.implant}'."
            )

    @property
    def is_cortical(self) -> bool:
        return self.model in CORTICAL_MODELS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["xrange"] = list(self.xrange)
        d["yrange"] = list(self.yrange)
        d["regions"] = list(self.regions)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        for key in ("xrange", "yrange"):
            if key in d and d[key] is not None:
                d[key] = tuple(d[key])
        if "regions" in d and d["regions"] is not None:
            d["regions"] = tuple(d["regions"])
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))


@dataclass
class ScaleConfig:
    """Config-driven scale knobs (small defaults for tests, large for cluster)."""

    episodes: int = 4
    sequence_length: int = 4
    batch_size: int = 8
    epochs: int = 1
    lr: float = 1e-3
    samples_per_config: int = 16
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScaleConfig":
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        fields = {f for f in cls.__dataclass_fields__}
        known = {k: v for k, v in d.items() if k in fields}
        known["extra"] = {k: v for k, v in d.items() if k not in fields}
        return cls(**known)
