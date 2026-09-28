"""Per-residue structural properties."""

import numpy as np
from rdkit import Chem

from peporacle.featurise.hydrophobicity import residue_atom_hydrophobicity


def residue_properties(
    mol: Chem.Mol, residue_index: list[int], charges: list[float]
) -> dict[int, tuple[float, float, float, float, float]]:
    """Scaled (hydrophobicity, net charge, size, branching, aromaticity) of every residue.

    Hydrophobicity is the mean per-atom estimate; net charge the formal plus half the
    Gasteiger charge, over 3 and clipped to [-1, 1]; size the atom count over 30; branching
    the mean degree over 4; aromaticity the aromatic fraction of atoms.
    """
    members: dict[int, list[Chem.Atom]] = {}
    for atom in mol.GetAtoms():
        members.setdefault(residue_index[atom.GetIdx()], []).append(atom)
    properties = {}
    for residue, atoms in members.items():
        hydrophobicity = float(
            np.mean([residue_atom_hydrophobicity(a, charges[a.GetIdx()]) for a in atoms])
        )
        charge = 0.0
        for atom in atoms:
            charge += atom.GetFormalCharge()
            charge += charges[atom.GetIdx()] * 0.5
        branching = float(np.mean([atom.GetDegree() for atom in atoms]))
        aromaticity = sum(atom.GetIsAromatic() for atom in atoms) / len(atoms)
        properties[residue] = (
            hydrophobicity,
            max(-1.0, min(1.0, charge / 3.0)),
            min(1.0, len(atoms) / 30.0),
            min(1.0, branching / 4.0),
            aromaticity,
        )
    return properties
