"""Step 3: train the frozen configuration on each dataset and seed.

    python scripts/sweep.py [DATASET ...] [--seeds 0 1 2]

Writes results/sweep/<dataset>_seed<k>.json. Finished runs are skipped and an
interrupted run resumes from its last epoch.
"""

import argparse
import logging
from pathlib import Path

from peporacle.config import load_datasets, load_frozen
from peporacle.sweep import sweep
from peporacle.train import select_device

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("datasets", nargs="*", help="default: all nine")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--gpu-memory-fraction", type=float)
args = parser.parse_args()

model_cfg, train_cfg = load_frozen(ROOT / "configs" / "frozen.toml")
sweep(
    ROOT / "caches",
    datasets=args.datasets or list(load_datasets(ROOT / "configs" / "datasets.toml")),
    seeds=args.seeds,
    model_cfg=model_cfg,
    train_cfg=train_cfg,
    out_dir=ROOT / "results" / "sweep",
    device=select_device(args.gpu_memory_fraction),
)
