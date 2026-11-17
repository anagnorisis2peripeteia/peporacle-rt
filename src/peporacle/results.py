"""Run tables, per-dataset summaries and seed-paired comparisons between models.

Models are compared seed by seed: seed *k* has the same test peptides for every model, so
each comparison is the mean of three paired differences in test R², with a 90% t interval.
The verdict compares that interval with a margin of ±0.01 R², fixed before any model was
compared.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

MARGIN = 0.01
RUN_COLUMNS = (
    "model",
    "dataset",
    "seed",
    "test_r2",
    "test_mae",
    "test_rmse",
    "n_train",
    "n_val",
    "n_test",
    "n_params",
    "batch_size",
    "grad_accum",
    "epochs",
    "run_seconds",
)


def collect(results_dir: Path) -> list[dict]:
    """One row per result file written by :func:`peporacle.sweep.sweep`."""
    rows = []
    for path in sorted(Path(results_dir).glob("*_seed*.json")):
        r = json.loads(path.read_text())
        rows.append(
            {
                "model": r.get("model", "atomic attention"),
                "dataset": r["dataset"],
                "seed": r["seed"],
                "test_r2": r["r2"],
                "test_mae": r["mae"],
                "test_rmse": r["rmse"],
                "n_train": r["n_train"],
                "n_val": r["n_val"],
                "n_test": r["n_test"],
                "n_params": r["n_params"],
                "batch_size": r["config"]["batch_size"],
                "grad_accum": r["config"]["grad_accum"],
                "epochs": r["config"]["epochs"],
                "run_seconds": r["train_seconds"],
            }
        )
    return rows


def read_runs(path: Path) -> list[dict]:
    """Rows of a ``runs.csv``, either written by :func:`write_csv` or published in results/."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["model"] = r.get("model") or r.get("operator") or "atomic attention"
        r["seed"] = int(r["seed"])
        r["test_r2"] = float(r["test_r2"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict]) -> list[dict]:
    """Mean and sample SD of test R², mean MAE and mean run time per model and dataset."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["dataset"]), []).append(r)
    out = []
    for (model, dataset), runs in groups.items():
        r2 = np.array([r["test_r2"] for r in runs])
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "n_seeds": len(runs),
                "test_r2_mean": round(float(r2.mean()), 6),
                "test_r2_sd": round(float(r2.std(ddof=1)), 6) if len(runs) > 1 else None,
                "test_mae_mean": round(float(np.mean([r["test_mae"] for r in runs])), 4),
                "run_hours_mean": round(float(np.mean([r["run_seconds"] for r in runs])) / 3600, 2),
            }
        )
    return out


def verdict(diffs: list[float], a_name: str, b_name: str, margin: float = MARGIN) -> dict:
    """Mean, SD and 90% t interval of paired differences (a minus b), and the verdict."""
    d = np.asarray(diffs, dtype=float)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.95, len(d) - 1)) * sd / math.sqrt(len(d))
    lo, hi = mean - half, mean + half
    if -margin <= lo and hi <= margin:
        word = "equivalent"
    elif lo > margin:
        word = f"{a_name} higher"
    elif hi < -margin:
        word = f"{b_name} higher"
    else:
        word = "inconclusive"
    return {"diff_mean": mean, "diff_sd": sd, "ci90_low": lo, "ci90_high": hi, "verdict": word}


def paired(
    a: list[dict],
    b: list[dict],
    *,
    a_name: str,
    b_name: str,
    same_model: bool = False,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict]:
    """Compare every model and dataset in ``a`` with ``b``, seed by seed.

    With ``same_model`` each model in ``a`` is paired with the same model in ``b`` (an
    operator with and without attention); otherwise ``b`` must hold a single model.
    """
    if not same_model and len({r["model"] for r in b}) != 1:
        raise ValueError("b holds several models; pass same_model=True to pair them by name")
    r2_a = {(r["model"], r["dataset"], r["seed"]): r["test_r2"] for r in a}
    r2_b = {
        ((r["model"] if same_model else None), r["dataset"], r["seed"]): r["test_r2"] for r in b
    }
    out = []
    for model, dataset in dict.fromkeys((r["model"], r["dataset"]) for r in a):
        ref = model if same_model else None
        both = [s for s in seeds if (model, dataset, s) in r2_a and (ref, dataset, s) in r2_b]
        diffs = {s: r2_a[(model, dataset, s)] - r2_b[(ref, dataset, s)] for s in both}
        row = {"operator": model, "dataset": dataset}
        row |= {f"diff_seed{s}": round(diffs[s], 6) if s in diffs else None for s in seeds}
        if len(both) < len(seeds):
            row |= dict.fromkeys(("diff_mean", "diff_sd", "ci90_low", "ci90_high", "p_paired_t"))
            row["verdict"] = f"not assessed ({len(both)} of {len(seeds)} seeds)"
        else:
            v = verdict(list(diffs.values()), a_name, b_name)
            row |= {k: round(v[k], 6) for k in ("diff_mean", "diff_sd", "ci90_low", "ci90_high")}
            vals_a = [r2_a[(model, dataset, s)] for s in seeds]
            vals_b = [r2_b[(ref, dataset, s)] for s in seeds]
            row["p_paired_t"] = round(float(stats.ttest_rel(vals_a, vals_b).pvalue), 4)
            row["verdict"] = v["verdict"]
        out.append(row)
    return out
