"""Atomic hydrophobicity estimates from polarisability, electronegativity and size."""

from rdkit import Chem

from peporacle.featurise.elements import ELECTRONEGATIVITY, POLARIZABILITY, VDW_RADIUS

# N or O with fewer heavy-atom neighbours than this counts as hydrogen-bonding.
_HBOND_DEGREE_LIMIT = 3
# (|Gasteiger charge| threshold, contribution), largest threshold first
_ATOM_CHARGE_STEPS = ((0.3, -0.5), (0.2, -0.3), (0.1, -0.1))
_RESIDUE_CHARGE_STEPS = ((0.3, -0.5), (0.2, -0.3))


def _element_terms(z: int) -> tuple[float, float, float]:
    """logP-like dispersion, electronegativity and cavity contributions of an element."""
    dispersion = (POLARIZABILITY[z] - 0.5) / 4.5 * 0.3
    electronegativity = -(ELECTRONEGATIVITY[z] - 2.0) / 1.5 * 0.2
    cavity = (VDW_RADIUS[z] - 1.2) / 0.8 * 0.15
    return dispersion, electronegativity, cavity


def _charge_term(
    atom: Chem.Atom, charge: float, formal_term: float, steps: tuple[tuple[float, float], ...]
) -> float:
    """``formal_term`` if the atom is formally charged, else the first step |charge| exceeds."""
    if atom.GetFormalCharge() != 0:
        return formal_term
    return next((term for threshold, term in steps if abs(charge) > threshold), 0.0)


def atom_hydrophobicity(atom: Chem.Atom, charge: float) -> float:
    """Node-level hydrophobicity on [-1, 1]."""
    dispersion, electronegativity, cavity = _element_terms(atom.GetAtomicNum())
    polarity = electronegativity + _charge_term(atom, charge, -1.0, _ATOM_CHARGE_STEPS)
    aromatic = 0.13 if atom.GetIsAromatic() else 0.0
    polar = atom.GetSymbol() in ("N", "O") and atom.GetDegree() < _HBOND_DEGREE_LIMIT
    hbond = -0.3 if polar else 0.0
    return max(-1.0, min(1.0, (dispersion + polarity + cavity + aromatic + hbond) / 2.0))


def residue_atom_hydrophobicity(atom: Chem.Atom, charge: float) -> float:
    """Per-atom term of the residue hydrophobicity, on [-1, 1]."""
    dispersion, electronegativity, cavity = _element_terms(atom.GetAtomicNum())
    aromatic = 0.2 if atom.GetIsAromatic() else 0.0
    charged = _charge_term(atom, charge, -0.8, _RESIDUE_CHARGE_STEPS)
    return max(-1.0, min(1.0, dispersion + electronegativity + cavity + aromatic + charged))
