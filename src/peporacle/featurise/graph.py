"""Peptide sequence to PyTorch Geometric graph."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch_geometric.data import Data

from peporacle.featurise.atoms import NODE_FEATURES, node_features
from peporacle.featurise.bonds import EDGE_FEATURES, edge_features
from peporacle.featurise.molecule import DEFAULT_MONOMER_SDF, build_peptide
from peporacle.sequences import decode_sequence


def build_graph(
    sequence: str,
    monomer_sdf: Path | None = None,
    digit_map: Mapping[str, str] | None = None,
) -> Data:
    """Atom-level graph of a peptide given as one-letter codes plus optional digit codes.

    ``monomer_sdf`` defaults to pyPept's bundled monomer dictionary. Returns ``Data`` with
    ``x`` (atoms x ``NODE_FEATURES``), ``edge_index``, ``edge_attr`` (directed edges x
    ``EDGE_FEATURES``), the per-atom ``residue_index`` and ``num_residues``.
    """
    sdf = DEFAULT_MONOMER_SDF if monomer_sdf is None else Path(monomer_sdf)
    peptide = build_peptide(decode_sequence(sequence, digit_map), sdf)
    edges, edge_rows = edge_features(peptide.mol)
    x = torch.tensor(node_features(peptide), dtype=torch.float32)
    edge_attr = torch.tensor(edge_rows, dtype=torch.float32)
    assert x.shape[1] == len(NODE_FEATURES) and edge_attr.shape[1] == len(EDGE_FEATURES)
    return Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.int64).t().contiguous(),
        edge_attr=edge_attr,
        residue_index=torch.tensor(peptide.residue_index, dtype=torch.int64),
        num_residues=peptide.num_residues,
    )
