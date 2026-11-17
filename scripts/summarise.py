"""Step 5: collect each sweep's result files into runs.csv and summary.csv.

    python scripts/summarise.py [SWEEP_DIR ...]

Default: whichever of results/sweep, results/sweep_operator_ablation and
results/sweep_operator_attention exist. Both tables are written into each directory.
"""

import argparse
from pathlib import Path

from peporacle.results import collect, summarise, write_csv

RESULTS = Path(__file__).resolve().parents[1] / "results"
SWEEPS = ("sweep", "sweep_operator_ablation", "sweep_operator_attention")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("dirs", nargs="*", type=Path)
args = parser.parse_args()

dirs = args.dirs or [RESULTS / s for s in SWEEPS if (RESULTS / s).is_dir()]
if not dirs:
    print("no sweep results in results/ yet")
for sweep_dir in dirs:
    runs = collect(sweep_dir)
    if not runs:
        print(f"{sweep_dir}: no results yet")
        continue
    summary = summarise(runs)
    write_csv(sweep_dir / "runs.csv", runs)
    write_csv(sweep_dir / "summary.csv", summary)
    for r in summary:
        sd = r["test_r2_sd"] or 0.0
        name = f"{r['model']} {r['dataset']}"
        print(f"{name:34s} R² {r['test_r2_mean']:.4f} ± {sd:.4f}  (n={r['n_seeds']})")
