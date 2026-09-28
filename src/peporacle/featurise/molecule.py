"""Peptide assembly with pyPept, atom-to-residue assignment and partial charges."""

import math
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from pyPept.molecule import Molecule
from pyPept.sequence import Sequence
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from peporacle.featurise.elements import SUPPORTED_ELEMENTS

# pyPept's bundled dictionary, byte-identical to the rt_monomers.sdf of the published caches.
DEFAULT_MONOMER_SDF = Path(str(files("pyPept.data") / "monomers.sdf"))

# Monomer match scoring: matched atoms count 100 each, every atom already claimed by an
# earlier residue costs 10000, and distance from the monomer's expected offset breaks ties.
_MATCH_ATOM_SCORE = 100
_CLAIMED_ATOM_PENALTY = 10_000
_OFFSET_WEIGHT = 0.1
_MIN_MATCH_SCORE = -5_000


@dataclass(frozen=True)
class Peptide:
    """Sanitised peptide with 2D coordinates, per-atom residue index and Gasteiger charges."""

    mol: Chem.Mol
    residue_index: list[int]
    num_residues: int
    charges: list[float]


def build_peptide(biln: str, monomer_sdf: Path = DEFAULT_MONOMER_SDF) -> Peptide:
    """Assemble a BILN peptide from the monomer dictionary ``monomer_sdf``."""
    if not monomer_sdf.is_file():
        raise FileNotFoundError(monomer_sdf)
    try:
        sequence = Sequence(biln, path=str(monomer_sdf.parent), monomer_lib=monomer_sdf.name)
    except SystemExit as error:  # pyPept exits when a monomer is missing from the dictionary
        raise ValueError(f"pyPept cannot parse {biln!r} (exit status {error.code})") from error
    molecule = Molecule(sequence)
    mol = molecule.mol
    elements = {atom.GetAtomicNum(): atom.GetSymbol() for atom in mol.GetAtoms()}
    if unsupported := sorted(elements[z] for z in elements.keys() - SUPPORTED_ELEMENTS):
        raise ValueError(f"{biln!r} contains unsupported elements {unsupported}")
    residue_index = assign_residues(molecule.unsanitized_mol, molecule.monomers, molecule.offset)
    return Peptide(mol, residue_index, len(sequence.s_monomers), gasteiger_charges(mol))


def assign_residues(mol: Chem.Mol, monomers: list[dict], offsets: list[int]) -> list[int]:
    """Residue index of every atom, found by matching each monomer as a substructure.

    ``mol`` is the assembled peptide before sanitisation and ``offsets[i]`` the index of the
    first atom of monomer ``i`` before its R-group placeholders were removed. Monomers are
    matched in sequence order and an atom keeps the first residue that claims it. A monomer
    whose best match overlaps claimed atoms claims nothing; its atoms are then assigned from
    their neighbours.
    """
    owner: dict[int, int] = {}
    for residue, monomer in enumerate(monomers):
        query = Chem.RWMol(monomer["m_romol"])
        placeholders = [a.GetIdx() for a in query.GetAtoms() if a.GetAtomicNum() == 0]
        for idx in sorted(placeholders, reverse=True):
            query.RemoveAtom(idx)
        best, best_score = None, -math.inf
        for match in mol.GetSubstructMatches(query, uniquify=False):
            claimed = sum(idx in owner for idx in match)
            score = len(match) * _MATCH_ATOM_SCORE - claimed * _CLAIMED_ATOM_PENALTY
            score -= abs(sum(match) / len(match) - offsets[residue]) * _OFFSET_WEIGHT
            if score > best_score:
                best, best_score = match, score
        if best is not None and best_score > _MIN_MATCH_SCORE:
            for idx in best:
                owner.setdefault(idx, residue)

    num_atoms, num_residues = mol.GetNumAtoms(), len(monomers)
    # Unclaimed atoms take the most common residue of their assigned neighbours, else a residue
    # proportional to their index. They are visited in this set's iteration order, which
    # decides ties between unclaimed neighbours.
    for idx in set(range(num_atoms)) - set(owner):
        neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
        residues = [owner[n] for n in neighbours if n in owner]
        if residues:
            owner[idx] = Counter(residues).most_common(1)[0][0]
        else:
            owner[idx] = min(int(idx / (num_atoms / num_residues)), num_residues - 1)
    return [owner[idx] for idx in range(num_atoms)]


def gasteiger_charges(mol: Chem.Mol) -> list[float]:
    """Gasteiger partial charge of every atom; raises if any is undefined."""
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
    if any(math.isnan(q) for q in charges):
        raise ValueError(f"undefined Gasteiger charge in {Chem.MolToSmiles(mol)}")
    return charges
