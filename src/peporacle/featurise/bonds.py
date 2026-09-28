"""Edge (bond) features; every bond gives two directed edges with identical features."""

import numpy as np
from rdkit import Chem

BT, BD = Chem.BondType, Chem.BondDir
BOND_TYPES = (BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC)
BOND_ORDER = {BT.SINGLE: 1.0, BT.DOUBLE: 2.0, BT.TRIPLE: 3.0, BT.AROMATIC: 1.5}
BOND_DIRECTIONS = (BD.NONE, BD.BEGINWEDGE, BD.BEGINDASH, BD.ENDDOWNRIGHT)
# Bond lengths are scaled by a nominal maximum of 5 angstrom.
MAX_LENGTH = 5.0
# The begin atom's z coordinate decides whether the conformer counts as 3D.
_PLANAR_TOLERANCE = 0.001

EDGE_FEATURES = (
    "single",
    "double",
    "triple",
    "aromatic",
    "bond_order",
    *(f"direction_{direction.name.lower()}" for direction in BOND_DIRECTIONS),
    "length",
    "graph_distance",
    "length_3d",
)


def edge_features(mol: Chem.Mol) -> tuple[list[list[int]], list[list[float]]]:
    """Directed edge list (begin, end), (end, begin) per bond, and one feature row per edge."""
    positions = mol.GetConformer().GetPositions()
    edges, rows = [], []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type, direction = bond.GetBondType(), bond.GetBondDir()
        length = min(np.linalg.norm(positions[begin] - positions[end]) / MAX_LENGTH, 1.0)
        row = [
            *(bond_type == value for value in BOND_TYPES),
            BOND_ORDER.get(bond_type, 1.0) / 3.0,
            *(direction == value for value in BOND_DIRECTIONS),
            length,
            1.0,
            length if abs(positions[begin][2]) > _PLANAR_TOLERANCE else 0.0,
        ]
        edges += [[begin, end], [end, begin]]
        rows += [row, row]
    return edges, rows
