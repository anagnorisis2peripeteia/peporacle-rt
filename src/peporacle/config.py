"""Model, training and dataset configuration, loaded from the TOML files in ``configs/``."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
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


@dataclass(frozen=True)
class OperatorSettings:
    operators: tuple[str, ...]
    n_layers: int
    attention_blocks: dict[str, int]
    micro_batch: dict[str, dict[str, int]]

    def train_config(self, base: TrainConfig, operator: str, dataset: str) -> TrainConfig:
        """``base`` with the micro-batch this operator used on this dataset.

        Gradient accumulation is raised to match, so every update still sees
        ``base.batch_size * base.grad_accum`` peptides.
        """
        sizes = self.micro_batch.get(operator, {})
        micro = sizes.get(dataset, sizes.get("default", base.batch_size))
        per_update = base.batch_size * base.grad_accum
        if per_update % micro:
            raise ValueError(f"micro-batch {micro} does not divide {per_update}")
        return replace(base, batch_size=micro, grad_accum=per_update // micro)


def load_operator_settings(path: Path) -> OperatorSettings:
    raw = tomllib.loads(Path(path).read_text())
    return OperatorSettings(
        operators=tuple(raw["operators"]),
        n_layers=raw["n_layers"],
        attention_blocks=dict(raw["attention_blocks"]),
        micro_batch={op: dict(sizes) for op, sizes in raw.get("micro_batch", {}).items()},
    )


def as_dict(*configs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in configs:
        out.update(asdict(c))
    return out
