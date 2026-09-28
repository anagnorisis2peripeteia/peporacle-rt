"""Model, training and dataset configuration, loaded from the TOML files in ``configs/``."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    n_blocks: int
    n_heads: int
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    weight_decay: float
    batch_size: int
    grad_accum: int
    grad_clip: float
    warmup_epochs: int
    epochs: int


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv: str
    digit_map: bool


def load_frozen(path: Path) -> tuple[ModelConfig, TrainConfig]:
    raw = tomllib.loads(Path(path).read_text())
    return ModelConfig(**raw["model"]), TrainConfig(**raw["train"])


def load_datasets(path: Path) -> dict[str, DatasetSpec]:
    raw = tomllib.loads(Path(path).read_text())
    return {name: DatasetSpec(name=name, **spec) for name, spec in raw.items()}


def as_dict(*configs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in configs:
        out.update(asdict(c))
    return out
