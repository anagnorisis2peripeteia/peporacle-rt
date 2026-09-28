"""Step 2: search atomic-attention hyperparameters on the 10,000-peptide misc_dia subset.

    python scripts/tune.py [--trials 20]

Resumes an interrupted study from results/tuning/study.db.
"""

import argparse
import logging
from pathlib import Path

from peporacle.cache import GraphCache
from peporacle.config import load_frozen
from peporacle.train import select_device
from peporacle.tune import tune

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--gpu-memory-fraction", type=float)
args = parser.parse_args()

_, base = load_frozen(ROOT / "configs" / "frozen.toml")
tune(
    GraphCache(ROOT / "caches" / "misc_dia"),
    base=base,
    n_trials=args.trials,
    storage=f"sqlite:///{ROOT / 'results' / 'tuning' / 'study.db'}",
    study_name="atomic_attention",
    device=select_device(args.gpu_memory_fraction),
)
