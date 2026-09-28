import numpy as np
import pytest
import torch

from peporacle.cache import GraphCache, verify_cache, write_cache, write_graphs, write_shard


def test_round_trip(tmp_path, graphs):
    labels = np.arange(len(graphs), dtype=float) * 1.5
    seqs = [f"SEQ{i}" for i in range(len(graphs))]
    write_graphs(tmp_path / "c", graphs, labels, seqs)
    cache = GraphCache(tmp_path / "c")
    assert len(cache) == len(graphs) and cache.sequences == seqs
    assert np.array_equal(cache.labels, labels)
    for g, h in zip(graphs, (cache[i] for i in range(len(cache))), strict=True):
        for key in ("x", "edge_index", "edge_attr", "residue_index"):
            assert torch.equal(getattr(g, key), getattr(h, key))
        assert g.num_residues == h.num_residues


def test_corruption_is_detected(tmp_path, graphs):
    write_graphs(tmp_path / "c", graphs, [0.0] * len(graphs), ["S"] * len(graphs))
    x = np.load(tmp_path / "c" / "x.npy")
    x[0, 0] += 1
    np.save(tmp_path / "c" / "x.npy", x)
    with pytest.raises(ValueError, match="sha256"):
        verify_cache(tmp_path / "c")


def test_merging_several_shards_equals_one(tmp_path, graphs):
    labels, seqs = [float(i) for i in range(len(graphs))], [f"S{i}" for i in range(len(graphs))]
    write_graphs(tmp_path / "one", graphs, labels, seqs)
    shards = []
    for k, part in enumerate((graphs[:3], graphs[3:4], graphs[4:])):
        shards.append(tmp_path / f"part{k}.npz")
        write_shard(shards[-1], part)
    write_cache(tmp_path / "three", shards, labels, seqs)
    for name in (
        "x.npy",
        "edge_attr.npy",
        "edge_index.npy",
        "residue_index.npy",
        "node_offsets.npy",
        "edge_offsets.npy",
        "num_residues.npy",
    ):
        assert np.array_equal(np.load(tmp_path / "one" / name), np.load(tmp_path / "three" / name))
