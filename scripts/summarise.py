"""Step 4: mean and SD of test metrics per dataset -> results/summary.csv.

python scripts/summarise.py
"""

import csv
from pathlib import Path

from peporacle.sweep import summarise

RESULTS = Path(__file__).resolve().parents[1] / "results"

rows = summarise(RESULTS / "sweep")
with open(RESULTS / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for r in rows:
    print(f"{r['dataset']:16s} R² {r['r2']:.4f} ± {r['r2_sd'] or 0:.4f}  (n={r['n_seeds']})")
