"""Step 1: featurise the dataset CSVs in data/ into graph caches in caches/.

    python scripts/build_caches.py [DATASET ...] [--workers 8]

Interrupted builds resume from their completed shards.
"""

import argparse
import logging
from pathlib import Path

from peporacle.build import build_cache
from peporacle.config import load_datasets
from peporacle.sequences import HELA_DIGIT_MAP

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="default: all nine")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "caches")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    specs = load_datasets(ROOT / "configs" / "datasets.toml")
    for name in args.datasets or list(specs):
        build_cache(
            csv_path=args.data_dir / specs[name].csv,
            out_dir=args.cache_dir / name,
            digit_map=HELA_DIGIT_MAP if specs[name].digit_map else None,
            workers=args.workers,
        )


if __name__ == "__main__":  # workers are spawned processes
    main()
