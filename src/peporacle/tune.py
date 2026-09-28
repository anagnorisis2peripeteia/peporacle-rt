"""Hyperparameter search for atomic attention on a fixed 10,000-peptide subset of misc_dia.

Each trial trains for the full epoch budget and is scored by the validation R² of
its final-epoch weights. The validation set is carved from the subset's training
portion; the subset's test portion is never used.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import optuna
import torch

from peporacle.cache import GraphCache
from peporacle.config import ModelConfig, TrainConfig
from peporacle.data import Split, make_loaders, split_indices
from peporacle.model import AtomicAttention
from peporacle.train import fit

log = logging.getLogger(__name__)

SUBSET_SIZE = 10_000
SUBSET_SEED = 0xC0FFEE
SPLIT_SEED = 0


def subset_split(cache: GraphCache) -> Split:
    """Split of the tuning subset, as row indices into the full cache."""
    rng = np.random.default_rng(SUBSET_SEED)
    rows = np.sort(rng.choice(len(cache), size=SUBSET_SIZE, replace=False))
    local = split_indices([cache.sequences[i] for i in rows], SPLIT_SEED)
    return Split(train=rows[local.train], val=rows[local.val], test=rows[local.test])


def suggest(trial: optuna.Trial, base: TrainConfig) -> tuple[ModelConfig, TrainConfig]:
    model = ModelConfig(
        hidden_dim=trial.suggest_int("hidden_dim", 128, 192, step=64),
        n_blocks=trial.suggest_int("n_blocks", 1, 16),
        n_heads=trial.suggest_categorical("n_heads", [1, 2, 4]),
        dropout=trial.suggest_float("dropout", 0.015, 0.10, log=True),
    )
    train = replace(
        base,
        lr=trial.suggest_float("lr", 3e-5, 5e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 1e-3),
        grad_clip=trial.suggest_float("grad_clip", 3.0, 10.0),
        warmup_epochs=trial.suggest_int("warmup_epochs", 5, 30),
        grad_accum=trial.suggest_categorical("grad_accum", [1, 2, 4]),
    )
    return model, train


def tune(
    cache: GraphCache,
    *,
    base: TrainConfig,
    n_trials: int,
    storage: str | None,
    study_name: str,
    device: torch.device,
) -> optuna.Study:
    """Run trials until ``n_trials`` have completed; an existing study is resumed."""
    split = subset_split(cache)

    def objective(trial: optuna.Trial) -> float:
        model_cfg, train_cfg = suggest(trial, base)
        train_loader, val_loader, _, scaler = make_loaders(cache, split, train_cfg.batch_size)
        result = fit(
            AtomicAttention(cache.node_dim, model_cfg),
            train_loader=train_loader,
            val_loader=val_loader,
            val_labels=cache.labels[split.val],
            scaler=scaler,
            cfg=train_cfg,
            device=device,
        )
        return result.final_val_r2

    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage, load_if_exists=True
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(n_trials - done, 0))
    log.info(
        "best trial %d: val r2 %.5f %s",
        study.best_trial.number,
        study.best_value,
        study.best_params,
    )
    return study
