"""Train the frozen configuration on every dataset and seed; one JSON result per run."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from peporacle.cache import GraphCache
from peporacle.config import ModelConfig, TrainConfig, as_dict
from peporacle.data import make_loaders, split_indices
from peporacle.model import AtomicAttention
from peporacle.train import fit, predict, regression_metrics

log = logging.getLogger(__name__)


def environment() -> dict[str, str]:
    env = {pkg: version(pkg) for pkg in ("peporacle", "torch", "torch_geometric", "numpy")}
    env["python"] = platform.python_version()
    env["device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return env


def run_one(
    cache: GraphCache,
    *,
    seed: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    checkpoint: Path,
) -> dict:
    """Train one seed and return its test metrics, training record and provenance."""
    split = split_indices(cache.sequences, seed)
    train_loader, val_loader, test_loader, scaler = make_loaders(cache, split, train_cfg.batch_size)
    torch.manual_seed(seed)
    model = AtomicAttention(cache.node_dim, model_cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = fit(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_labels=cache.labels[split.val],
        scaler=scaler,
        cfg=train_cfg,
        device=device,
        checkpoint=checkpoint,
    )
    test = regression_metrics(cache.labels[split.test], predict(model, test_loader, scaler, device))
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    return {
        "seed": seed,
        **test,
        **asdict(result),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "wall_seconds": round(time.time() - t0, 1),
        "peak_gpu_gb": None if peak_gb is None else round(peak_gb, 2),
        "n_params": sum(p.numel() for p in model.parameters()),
        "config": as_dict(model_cfg, train_cfg),
        "cache_manifest_sha256": hashlib.sha256(
            (cache.dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "environment": environment(),
    }


def sweep(
    cache_root: Path,
    *,
    datasets: list[str],
    seeds: list[int],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Run every (dataset, seed) that has no result file in ``out_dir`` yet."""
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        todo = [s for s in seeds if not (out_dir / f"{dataset}_seed{s}.json").exists()]
        if not todo:
            continue
        cache = GraphCache(cache_root / dataset)
        for seed in todo:
            log.info("%s seed %d", dataset, seed)
            ckpt = ckpt_dir / f"{dataset}_seed{seed}.pt"
            record = {"dataset": dataset} | run_one(
                cache,
                seed=seed,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
                checkpoint=ckpt,
            )
            path = out_dir / f"{dataset}_seed{seed}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=1) + "\n")
            os.replace(tmp, path)
            ckpt.unlink()
            log.info("%s seed %d: test r2 %.4f", dataset, seed, record["r2"])


def summarise(results_dir: Path) -> list[dict]:
    """Mean and sample SD of the test metrics per dataset, over the seeds present."""
    by_dataset: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*_seed*.json")):
        rec = json.loads(path.read_text())
        by_dataset.setdefault(rec["dataset"], []).append(rec)
    rows = []
    for dataset, recs in by_dataset.items():
        row: dict = {"dataset": dataset, "n_seeds": len(recs)}
        for metric in ("r2", "mae", "rmse"):
            values = np.array([r[metric] for r in recs])
            row[metric] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return rows
