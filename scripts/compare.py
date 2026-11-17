"""Step 6: compare the models seed by seed against the ±0.01 R² margin.

    python scripts/compare.py [--published]

Reads runs.csv from your sweeps (after summarise.py) and writes the paired comparisons
next to them. With --published, rebuilds the paired_*.csv files in results/ from the
published runs.csv files instead.
"""

import argparse
from pathlib import Path

from peporacle.results import paired, read_runs, write_csv

RESULTS = Path(__file__).resolve().parents[1] / "results"
# (compared, reference, names for the verdicts, pair operators by name, output file)
COMPARISONS = (
    (
        "operator_ablation",
        "atomic_attention",
        "operator",
        "atomic attention",
        False,
        "paired_vs_atomic_attention.csv",
    ),
    (
        "operator_attention",
        "atomic_attention",
        "operator + attention",
        "atomic attention",
        False,
        "paired_vs_atomic_attention.csv",
    ),
    (
        "operator_attention",
        "operator_ablation",
        "operator + attention",
        "operator only",
        True,
        "paired_vs_operator_only.csv",
    ),
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--published", action="store_true")
args = parser.parse_args()


def run_dir(arm: str) -> Path:
    if args.published:
        return RESULTS / arm
    return RESULTS / ("sweep" if arm == "atomic_attention" else f"sweep_{arm}")


for a, b, a_name, b_name, same_model, out in COMPARISONS:
    a_runs, b_runs = run_dir(a) / "runs.csv", run_dir(b) / "runs.csv"
    if not (a_runs.exists() and b_runs.exists()):
        print(f"skipped {a} vs {b}: no runs.csv yet")
        continue
    rows = paired(
        read_runs(a_runs), read_runs(b_runs), a_name=a_name, b_name=b_name, same_model=same_model
    )
    write_csv(run_dir(a) / out, rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"{a} vs {b}: " + ", ".join(f"{n} {v}" for v, n in counts.items()))
