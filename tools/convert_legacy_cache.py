"""Convert a published pickle cache (graphs.pkl, labels.pkl, sequences.pkl) to the cache format.

    python tools/convert_legacy_cache.py LEGACY_DIR OUT_DIR

The published results were trained on these pickles. Converting them, rather than
rebuilding from the CSVs, keeps the training inputs bit-identical on any machine. The
pickles are trusted files from the original runs: unpickling can execute code, so do
not run this on files from elsewhere.
"""

import pickle
import sys
from pathlib import Path

from peporacle.cache import GraphCache, write_graphs

KEYS = ("x", "edge_index", "edge_attr", "residue_index")


def main() -> None:
    legacy, out = map(Path, sys.argv[1:3])
    parts = {}
    for name in ("graphs", "labels", "sequences"):
        with open(legacy / f"{name}.pkl", "rb") as f:
            parts[name] = pickle.load(f)  # noqa: S301 - trusted, see module docstring
    sequences = [str(s) for s in parts["sequences"]]
    write_graphs(out, parts["graphs"], [float(y) for y in parts["labels"]], sequences)

    cache = GraphCache(out)
    for i, old in enumerate(parts["graphs"]):
        new = cache[i]
        if not all(getattr(old, k).equal(getattr(new, k)) for k in KEYS):
            raise SystemExit(f"graph {i} differs after conversion")
    print(f"{out}: {len(cache)} graphs, identical to {legacy}")


if __name__ == "__main__":
    main()
