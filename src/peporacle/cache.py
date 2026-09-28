"""On-disk graph caches and the shards they are built from.

A cache is a directory of flat NumPy arrays plus a JSON manifest. Node and edge rows of
all graphs are concatenated; ``node_offsets`` and ``edge_offsets`` mark where each graph
starts. The arrays are memory-mapped when read, so a cache does not have to fit in RAM,
and nothing is unpickled.

Layout::

    x.npy              float32 (total_nodes, node_dim)
    edge_attr.npy      float32 (total_edges, edge_dim)
    edge_index.npy     int64   (2, total_edges)   node indices local to each graph
    residue_index.npy  int64   (total_nodes,)
    node_offsets.npy   int64   (n_graphs + 1,)
    edge_offsets.npy   int64   (n_graphs + 1,)
    num_residues.npy   int64   (n_graphs,)
    labels.npy         float64 (n_graphs,)
    sequences.txt      one sequence per line, in array order
    manifest.json      format, counts, dimensions, provenance and the SHA-256 of every file

A shard is one ``.npz`` holding the same node and edge arrays for a run of consecutive
graphs, plus their ``num_nodes``, ``num_edges`` and ``num_residues``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

FORMAT = "peporacle-cache-v1"
ROWS = {"x": np.float32, "edge_attr": np.float32, "residue_index": np.int64}
ARRAYS = (
    "x",
    "edge_attr",
    "edge_index",
    "residue_index",
    "node_offsets",
    "edge_offsets",
    "num_residues",
    "labels",
)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_shard(path: Path, graphs: Sequence[Any]) -> None:
    """Write ``graphs`` (``Data`` or dicts with the same keys) to one shard, atomically."""
    arrays = {name: np.concatenate([np.asarray(g[name]) for g in graphs]) for name in ROWS}
    arrays["edge_index"] = np.concatenate([np.asarray(g["edge_index"]) for g in graphs], axis=1)
    arrays["num_nodes"] = np.array([len(g["x"]) for g in graphs])
    arrays["num_edges"] = np.array([np.asarray(g["edge_index"]).shape[1] for g in graphs])
    arrays["num_residues"] = np.array([int(g["num_residues"]) for g in graphs])
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def write_cache(
    out_dir: Path,
    shards: Sequence[Path],
    labels: Sequence[float],
    sequences: Sequence[str],
    provenance: dict[str, Any] | None = None,
) -> dict:
    """Merge ``shards`` into a cache at ``out_dir`` and return its manifest.

    Two passes over the shards: the first reads only the per-graph counts, the second copies
    the arrays into pre-sized memory-mapped files, so peak memory is one shard.
    """
    out_dir = Path(out_dir)
    counts = {"num_nodes": [], "num_edges": [], "num_residues": []}
    dims = {}
    for path in shards:
        with np.load(path) as shard:
            for name, values in counts.items():
                values.append(shard[name])
            dims = dims or {k: tuple(map(int, shard[k].shape[1:])) for k in ("x", "edge_attr")}
    n_nodes, n_edges, n_res = (np.concatenate(v) for v in counts.values())
    total_nodes, total_edges = int(n_nodes.sum()), int(n_edges.sum())
    if not len(n_res) == len(labels) == len(sequences):
        raise ValueError(f"{len(n_res)} graphs, {len(labels)} labels, {len(sequences)} sequences")

    tmp = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shapes = {
        "x": (total_nodes, *dims["x"]),
        "edge_attr": (total_edges, *dims["edge_attr"]),
        "residue_index": (total_nodes,),
        "edge_index": (2, total_edges),
    }
    dtypes = ROWS | {"edge_index": np.int64}
    out = {
        name: np.lib.format.open_memmap(
            tmp / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape
        )
        for name, shape in shapes.items()
    }
    node, edge = 0, 0
    for path in shards:
        with np.load(path) as shard:
            nn, ne = len(shard["residue_index"]), shard["edge_index"].shape[1]
            out["x"][node : node + nn] = shard["x"]
            out["residue_index"][node : node + nn] = shard["residue_index"]
            out["edge_attr"][edge : edge + ne] = shard["edge_attr"]
            out["edge_index"][:, edge : edge + ne] = shard["edge_index"]
        node, edge = node + nn, edge + ne
    for arr in out.values():
        arr.flush()
    del out

    small = {
        "node_offsets": np.concatenate([[0], np.cumsum(n_nodes)]).astype(np.int64),
        "edge_offsets": np.concatenate([[0], np.cumsum(n_edges)]).astype(np.int64),
        "num_residues": n_res.astype(np.int64),
        "labels": np.asarray(labels, dtype=np.float64),
    }
    for name, arr in small.items():
        np.save(tmp / f"{name}.npy", arr)
    (tmp / "sequences.txt").write_text("".join(f"{s}\n" for s in sequences))

    manifest = {
        "format": FORMAT,
        "n_graphs": len(n_res),
        "node_dim": int(dims["x"][0]),
        "edge_dim": int(dims["edge_attr"][0]),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "files": {p.name: sha256_file(p) for p in sorted(tmp.iterdir())},
        "provenance": provenance or {},
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    return manifest


def write_graphs(
    out_dir: Path, graphs: Iterable[Data], labels: Sequence[float], sequences: Sequence[str]
) -> dict:
    """Write an in-memory list of graphs as a cache (one shard; for tests and small sets)."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shard = out_dir.with_name(out_dir.name + ".shard.npz")
    write_shard(shard, list(graphs))
    try:
        return write_cache(out_dir, [shard], labels, sequences)
    finally:
        shard.unlink()


def verify_cache(cache_dir: Path) -> None:
    """Raise ``ValueError`` if any file differs from the hash recorded in the manifest."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{cache_dir}: unsupported format {manifest.get('format')!r}")
    for name, expected in manifest["files"].items():
        actual = sha256_file(cache_dir / name)
        if actual != expected:
            raise ValueError(f"{cache_dir / name}: sha256 {actual} != manifest {expected}")


class GraphCache:
    """Read-only, memory-mapped view of a cache directory.

    ``cache[i]`` returns a ``Data`` whose tensors are views into the mapped arrays; pages
    are read from disk only when a batch is collated.
    """

    def __init__(self, cache_dir: Path, verify: bool = True) -> None:
        self.dir = Path(cache_dir)
        if verify:
            verify_cache(self.dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        # Copy-on-write mapping: torch requires writable arrays; the files are never changed.
        a = {name: np.load(self.dir / f"{name}.npy", mmap_mode="c") for name in ARRAYS}
        self._x = torch.from_numpy(a["x"])
        self._edge_attr = torch.from_numpy(a["edge_attr"])
        self._edge_index = torch.from_numpy(a["edge_index"])
        self._residue_index = torch.from_numpy(a["residue_index"])
        self._node_off = np.asarray(a["node_offsets"])
        self._edge_off = np.asarray(a["edge_offsets"])
        self._num_residues = np.asarray(a["num_residues"])
        self.labels = np.asarray(a["labels"])
        self.sequences = (self.dir / "sequences.txt").read_text().splitlines()
        if not len(self.sequences) == len(self.labels) == len(self):
            raise ValueError(f"{self.dir}: inconsistent graph, label and sequence counts")

    def __len__(self) -> int:
        return len(self._num_residues)

    @property
    def node_dim(self) -> int:
        return int(self.manifest["node_dim"])

    def __getitem__(self, i: int) -> Data:
        n0, n1 = int(self._node_off[i]), int(self._node_off[i + 1])
        e0, e1 = int(self._edge_off[i]), int(self._edge_off[i + 1])
        return Data(
            x=self._x[n0:n1],
            edge_attr=self._edge_attr[e0:e1],
            edge_index=self._edge_index[:, e0:e1],
            residue_index=self._residue_index[n0:n1],
            num_residues=int(self._num_residues[i]),
        )
