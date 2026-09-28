"""Train/validation/test splits and data loaders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from peporacle.cache import GraphCache


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_indices(
    sequences: Sequence[str], seed: int, test_frac: float = 0.1, val_frac: float = 0.1
) -> Split:
    """Hold out a test set, then carve a validation set from what remains.

    Splitting is grouped by sequence, so a peptide measured more than once never
    appears on both sides of a split. ``seed`` fixes the test set; the validation
    split uses ``seed + 1``. Model selection only ever sees the validation set.
    """
    group_of: dict[str, int] = {}
    groups = np.array([group_of.setdefault(s, len(group_of)) for s in sequences])
    outer = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train, test = next(outer.split(groups, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
    sub_train, sub_val = next(inner.split(train, groups=groups[train]))
    return Split(train=train[sub_train], val=train[sub_val], test=test)


@dataclass(frozen=True)
class LabelScaler:
    """Standardise retention times with training-set statistics."""

    mean: float
    std: float

    @classmethod
    def fit(cls, labels: np.ndarray) -> LabelScaler:
        return cls(mean=float(labels.mean()), std=float(labels.std()))

    def transform(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.float32)
        return (labels - np.float32(self.mean)) / (np.float32(self.std) + 1e-8)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * (self.std + 1e-8) + self.mean


class LabelledGraphs(Dataset):
    """Cache graphs at ``indices`` with standardised labels attached as ``y``."""

    def __init__(self, cache: GraphCache, indices: np.ndarray, scaler: LabelScaler) -> None:
        self.cache = cache
        self.indices = np.asarray(indices)
        self.y = torch.from_numpy(scaler.transform(cache.labels[self.indices]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Data:
        g = self.cache[int(self.indices[i])]
        g.y = self.y[i]
        return g


def make_loaders(
    cache: GraphCache, split: Split, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader, LabelScaler]:
    scaler = LabelScaler.fit(cache.labels[split.train])

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            LabelledGraphs(cache, idx, scaler), batch_size=batch_size, shuffle=shuffle
        )

    return loader(split.train, True), loader(split.val, False), loader(split.test, False), scaler
