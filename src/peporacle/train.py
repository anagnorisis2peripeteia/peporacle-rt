"""Training loop with per-epoch checkpoints, so an interrupted run resumes where it stopped."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from peporacle.config import TrainConfig
from peporacle.data import LabelScaler

log = logging.getLogger(__name__)


def select_device(gpu_memory_fraction: float | None = None) -> torch.device:
    """CUDA if available, optionally capping the share of GPU memory this process may use."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_memory_fraction is not None:
        # The published runs shared a 12 GB card with the display. Without a cap the
        # caching allocator held memory up to the card's limit.
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
    return torch.device("cuda")


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_epochs: int, epochs: int) -> LambdaLR:
    """Linear warm-up from zero over ``warmup_epochs``, then cosine decay to zero."""

    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> float:
    """One pass over ``loader`` with gradient accumulation; returns the mean batch loss."""
    model.train()
    loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    total = 0.0
    for step, cpu_batch in enumerate(loader, start=1):
        batch = cpu_batch.to(device)
        loss = loss_fn(model(batch), batch.y) / cfg.grad_accum
        loss.backward()
        if step % cfg.grad_accum == 0 or step == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        total += loss.item() * cfg.grad_accum
    return total / len(loader)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, scaler: LabelScaler, device: torch.device
) -> np.ndarray:
    """Predictions in the original label units, in loader order."""
    model.eval()
    out = [model(batch.to(device)).cpu().numpy() for batch in loader]
    return scaler.inverse(np.concatenate(out))


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_val_r2: float
    final_val_r2: float
    resumed_from_epoch: int
    train_seconds: float


def fit(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_labels: np.ndarray,
    scaler: LabelScaler,
    cfg: TrainConfig,
    device: torch.device,
    checkpoint: Path | None = None,
) -> FitResult:
    """Train for ``cfg.epochs`` epochs and leave ``model`` holding its best-validation weights.

    If ``checkpoint`` is given, the full training state is written there after every
    epoch and training resumes from it when it exists.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs, cfg.epochs)
    state = {
        "epoch": 0,
        "best_epoch": -1,
        "best_val_r2": -math.inf,
        "best_weights": None,
        "val_r2": math.nan,
        "train_seconds": 0.0,
    }
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        scheduler.load_state_dict(saved.pop("scheduler"))
        torch.set_rng_state(saved.pop("rng_cpu"))
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved.pop("rng_cuda"))
        state.update(saved)
        log.info("resumed at epoch %d", state["epoch"])
    resumed_from = state["epoch"]

    for epoch in range(state["epoch"], cfg.epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, cfg, device)
        scheduler.step()
        val_r2 = regression_metrics(val_labels, predict(model, val_loader, scaler, device))["r2"]
        if val_r2 > state["best_val_r2"]:
            state["best_val_r2"], state["best_epoch"] = val_r2, epoch
            state["best_weights"] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        state.update(
            epoch=epoch + 1, val_r2=val_r2, train_seconds=state["train_seconds"] + time.time() - t0
        )
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            log.info("epoch %3d  loss %.4f  val r2 %.4f", epoch, loss, val_r2)
        if checkpoint is not None:
            _save(checkpoint, model, optimizer, scheduler, state)

    if state["best_weights"] is not None:
        model.load_state_dict(state["best_weights"])
    return FitResult(
        best_epoch=state["best_epoch"],
        best_val_r2=state["best_val_r2"],
        final_val_r2=state["val_r2"],
        resumed_from_epoch=resumed_from,
        train_seconds=round(state["train_seconds"], 1),
    )


def _save(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: LambdaLR, state: dict
) -> None:
    payload = dict(
        state,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        rng_cpu=torch.get_rng_state(),
    )
    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
