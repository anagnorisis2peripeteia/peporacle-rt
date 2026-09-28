"""Golden-graph tests: the featuriser must reproduce the published cache graphs exactly."""

import hashlib
import json
import platform
from pathlib import Path

import pytest
import torch

from peporacle.cache import sha256_file
from peporacle.featurise import DEFAULT_MONOMER_SDF, EDGE_FEATURES, NODE_FEATURES, build_graph
from peporacle.sequences import HELA_DIGIT_MAP, decode_sequence

GOLDEN = json.loads((Path(__file__).parent / "data" / "featurise_golden.json").read_text())
DIGIT_MAPS = {"hela": HELA_DIGIT_MAP, None: None}
# Bond lengths come from 2D depiction coordinates, whose last bits differ between CPU
# architectures. The published caches were built on Apple silicon, so the full check is
# exact only there; everywhere else every other feature must still match exactly.
GEOMETRY = [EDGE_FEATURES.index(name) for name in GOLDEN["geometry_edge_features"]]
BUILT_HERE = f"{platform.system()} {platform.machine()}" == GOLDEN["built_on"]


def graph_hash(graph, portable: bool = False) -> str:
    """SHA-256 of the graph tensors; ``portable`` leaves out the geometry edge columns."""
    keep = [i for i in range(len(EDGE_FEATURES)) if i not in GEOMETRY]
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "edge_attr", "residue_index"):
        tensor = getattr(graph, name)
        if portable and name == "edge_attr":
            tensor = tensor[:, keep]
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_monomer_dictionary_is_the_published_one():
    assert sha256_file(DEFAULT_MONOMER_SDF) == GOLDEN["monomer_sdf_sha256"]


@pytest.mark.parametrize("case", GOLDEN["graphs"], ids=lambda case: case["sequence"])
def test_golden_graph(case):
    graph = build_graph(case["sequence"], digit_map=DIGIT_MAPS[case["digit_map"]])
    assert graph.num_residues == case["num_residues"]
    assert graph.x.shape == (case["num_atoms"], len(NODE_FEATURES))
    assert graph.edge_attr.shape == (case["num_edges"], len(EDGE_FEATURES))
    assert graph_hash(graph, portable=True) == case["sha256_portable"]
    if BUILT_HERE:
        assert graph_hash(graph) == case["sha256"]


def test_feature_names_match_tensors():
    graph = build_graph("ACDEFGHIKLMNPQRSTVWY")
    assert len(set(NODE_FEATURES)) == len(NODE_FEATURES) == graph.x.shape[1] == GOLDEN["node_dim"]
    assert len(set(EDGE_FEATURES)) == len(EDGE_FEATURES) == graph.edge_attr.shape[1]
    assert graph.edge_attr.shape[1] == GOLDEN["edge_dim"]
    assert graph.x.dtype == graph.edge_attr.dtype == torch.float32
    assert graph.edge_index.dtype == graph.residue_index.dtype == torch.int64
    assert sorted(graph.keys()) == ["edge_attr", "edge_index", "num_residues", "residue_index", "x"]


def test_decode_sequence():
    assert decode_sequence("la1k", HELA_DIGIT_MAP) == "L-A-Met_O-K"
    with pytest.raises(ValueError, match="unknown residue code"):
        decode_sequence("LA1K")


def test_unknown_monomer_raises():
    with pytest.raises(ValueError, match="cannot parse"):
        build_graph("LAXK", digit_map={"X": "not_a_monomer"})


def test_monomer_override(tmp_path):
    sdf = tmp_path / "monomers.sdf"
    sdf.write_bytes(DEFAULT_MONOMER_SDF.read_bytes())
    case = GOLDEN["graphs"][0]
    graph = build_graph(case["sequence"], sdf, HELA_DIGIT_MAP)
    assert graph_hash(graph, portable=True) == case["sha256_portable"]
    with pytest.raises(FileNotFoundError):
        build_graph(case["sequence"], tmp_path / "missing.sdf", HELA_DIGIT_MAP)
