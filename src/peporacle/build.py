"""Build a graph cache from a CSV of sequences and labels.

Graphs are featurised in parallel into fingerprinted shards next to the output directory, so
an interrupted build resumes where it stopped; once every shard exists they are merged with
:func:`peporacle.cache.write_cache`.
"""

import hashlib
import json
import logging
import multiprocessing
import platform
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyPept.molecule
import pyPept.sequence
from rdkit import RDLogger

import peporacle.sequences
from peporacle import featurise
from peporacle.cache import sha256_file, write_cache, write_shard
from peporacle.featurise import DEFAULT_MONOMER_SDF, build_graph

PACKAGES = ("pyPept", "rdkit", "torch", "torch_geometric", "numpy", "pandas")

log = logging.getLogger(__name__)


def build_cache(
    *,
    csv_path: Path,
    out_dir: Path,
    monomer_sdf: Path | None = None,
    digit_map: Mapping[str, str] | None = None,
    sequence_column: str = "sequence",
    label_column: str = "RT",
    workers: int = 8,
    shard_size: int = 1000,
) -> dict:
    """Featurise every row of ``csv_path`` into a cache at ``out_dir``; return its manifest.

    ``monomer_sdf`` defaults to pyPept's bundled dictionary. A row that fails to featurise
    aborts the build; shards completed so far are kept and reused on the next call. Workers
    are spawned, so call this under ``if __name__ == "__main__":`` in scripts.
    """
    csv_path, out_dir = Path(csv_path).resolve(), Path(out_dir).resolve()
    monomer_sdf = Path(DEFAULT_MONOMER_SDF if monomer_sdf is None else monomer_sdf).resolve()
    frame = pd.read_csv(csv_path)
    sequences = frame[sequence_column].tolist()
    labels = frame[label_column].tolist()
    provenance = {
        "csv": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "monomer_sdf": {"path": str(monomer_sdf), "sha256": sha256_file(monomer_sdf)},
        "columns": {"sequence": sequence_column, "label": label_column},
        "digit_map": dict(digit_map or {}),
        "featuriser_sha256": _featuriser_sha256(),
        "versions": {"python": platform.python_version()}
        | {name: version(name) for name in PACKAGES},
        "platform": platform.platform(),
    }
    fingerprint = hashlib.sha256(
        json.dumps([provenance, shard_size], sort_keys=True).encode()
    ).hexdigest()
    shard_dir = out_dir.with_name(f"{out_dir.name}.shards") / fingerprint[:16]
    shard_dir.mkdir(parents=True, exist_ok=True)
    starts = range(0, len(sequences), shard_size)
    bounds = [(lo, min(lo + shard_size, len(sequences))) for lo in starts]
    paths = [shard_dir / f"{lo:07d}-{hi:07d}.npz" for lo, hi in bounds]

    todo = [(b, p) for b, p in zip(bounds, paths, strict=True) if not p.exists()]
    log.info(
        "%d rows, %d of %d shards to build in %s", len(sequences), len(todo), len(paths), shard_dir
    )
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_quiet_worker) as pool:
        for (lo, hi), path in todo:
            tasks = [(sequence, monomer_sdf, digit_map) for sequence in sequences[lo:hi]]
            write_shard(path, pool.map(_featurise, tasks, chunksize=4))
            log.info("wrote %s", path.name)

    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"refusing to merge: {len(missing)} shards missing, e.g. {missing[0]}")
    manifest = write_cache(out_dir, paths, labels, sequences, provenance)
    log.info("cache %s: %d graphs", out_dir, manifest["n_graphs"])
    return manifest


def _featuriser_sha256() -> str:
    """Digest of the code that determines the graphs, so shards of edited code are never reused."""
    sources = [
        Path(peporacle.sequences.__file__),
        Path(pyPept.molecule.__file__),
        Path(pyPept.sequence.__file__),
    ]
    sources += sorted(Path(featurise.__file__).parent.glob("*.py"))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def _quiet_worker() -> None:
    RDLogger.DisableLog("rdApp.*")


def _featurise(task: tuple[str, Path, Mapping[str, str] | None]) -> dict[str, np.ndarray | int]:
    """Graph of one sequence as NumPy arrays (cheap to send between processes)."""
    graph = build_graph(*task)
    arrays = {
        name: graph[name].numpy() for name in ("x", "edge_attr", "residue_index", "edge_index")
    }
    return arrays | {"num_residues": graph.num_residues}
