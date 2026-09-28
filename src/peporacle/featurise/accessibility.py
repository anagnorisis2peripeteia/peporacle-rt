"""Graph-only proxies for side-chain solvent accessibility, after the Janin scale."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from peporacle.featurise.elements import ATOM_VOLUME, ELECTRONEGATIVITY, VDW_RADIUS

SINGLE, DOUBLE, TRIPLE = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE

# A hydrogen-bond acceptor has fewer heavy-atom neighbours than this (it keeps a lone pair).
ACCEPTOR_DEGREE_LIMIT = {"N": 3, "O": 2}
_BRANCH_POINT_DEGREE = 3
_PACKING_RADIUS = 3  # bonds
_RIGID_BOND_WEIGHT = {DOUBLE: 0.2, TRIPLE: 0.3}


@dataclass(frozen=True)
class Accessibility:
    """Accessibility descriptors of one atom; all but ``side_chain_volume`` lie in [0, 1]."""

    side_chain_volume: float
    branching: float
    backbone_distance: float
    surface_to_volume: float
    packing_density: float
    conformational_entropy: float
    hbond_network: float
    charge_density: float
    geometric_constraint: float
    accessibility: float


def is_acceptor(atom: Chem.Atom) -> bool:
    """N or O that still has a lone pair available."""
    return atom.GetDegree() < ACCEPTOR_DEGREE_LIMIT.get(atom.GetSymbol(), 0)


def accessibility_descriptors(mol: Chem.Mol, charges: list[float]) -> list[Accessibility]:
    """Accessibility descriptors for every atom of ``mol``."""
    backbone = [_is_backbone(atom) for atom in mol.GetAtoms()]
    distances = Chem.GetDistanceMatrix(mol)
    to_backbone = distances[:, np.array(backbone)].min(axis=1)
    packed = ((distances > 0) & (distances <= _PACKING_RADIUS)).sum(axis=1)
    return [
        _describe(atom, backbone, to_backbone[atom.GetIdx()], packed[atom.GetIdx()], charge)
        for atom, charge in zip(mol.GetAtoms(), charges, strict=True)
    ]


def _is_backbone(atom: Chem.Atom) -> bool:
    """Carbonyl O, C bonded to both N and O, or N bonded to two or more C."""
    symbol = atom.GetSymbol()
    neighbours = [n.GetSymbol() for n in atom.GetNeighbors()]
    if symbol == "O":
        bond = atom.GetBonds()[0]
        return bond.GetBondType() == DOUBLE and bond.GetOtherAtom(atom).GetSymbol() == "C"
    if symbol == "C":
        return "N" in neighbours and "O" in neighbours
    if symbol == "N":
        return neighbours.count("C") >= 2  # noqa: PLR2004 - amide N: bonded to C-alpha and C'
    return False


def _side_chain(atom: Chem.Atom, backbone: list[bool]) -> list[Chem.Atom]:
    """Atoms reachable from ``atom`` without entering the backbone, in breadth-first order."""
    if backbone[atom.GetIdx()]:
        return []
    chain, seen, queue = [atom], {atom.GetIdx()}, deque([atom])
    while queue:
        for neighbour in queue.popleft().GetNeighbors():
            idx = neighbour.GetIdx()
            if idx not in seen and not backbone[idx]:
                seen.add(idx)
                chain.append(neighbour)
                queue.append(neighbour)
    return chain


def _rotatable_bonds(chain: list[Chem.Atom]) -> int:
    """Acyclic single bonds inside the side chain between two non-terminal atoms."""
    members = {atom.GetIdx() for atom in chain}
    bonds = {
        bond.GetIdx(): bond
        for atom in chain
        for bond in atom.GetBonds()
        if bond.GetOtherAtomIdx(atom.GetIdx()) in members
    }
    return sum(
        bond.GetBondType() == SINGLE
        and not bond.IsInRing()
        and bond.GetBeginAtom().GetDegree() > 1
        and bond.GetEndAtom().GetDegree() > 1
        for bond in bonds.values()
    )


def _describe(
    atom: Chem.Atom, backbone: list[bool], to_backbone: float, packed: int, charge: float
) -> Accessibility:
    """Descriptors of one atom, given its graph distance to the backbone and its neighbour
    count within ``_PACKING_RADIUS``."""
    z = atom.GetAtomicNum()
    chain = _side_chain(atom, backbone)
    volume = sum(ATOM_VOLUME[a.GetAtomicNum()] for a in chain)
    rotatable = _rotatable_bonds(chain)
    ring_atoms = sum(a.IsInRing() for a in chain)

    branching = min(sum(a.GetDegree() >= _BRANCH_POINT_DEGREE for a in chain) / 3.0, 1.0)
    backbone_distance = min(to_backbone / 7.0, 1.0)
    # SA/V of a sphere is 3/r; the typical range [1.5, 2.5] per angstrom maps onto [0, 1].
    surface_to_volume = max(0.0, min(1.0, (3.0 / VDW_RADIUS[z] - 1.5) / (2.5 - 1.5)))
    packing_density = min(packed / 30.0, 1.0)
    entropy = min(max((rotatable - ring_atoms * 0.5) / 6.0, 0.0), 1.0) if chain else 1.0

    hbond = 0.0
    if is_acceptor(atom):
        hbond += 0.5
    for neighbour in atom.GetNeighbors():
        if neighbour.GetSymbol() in ("N", "O"):
            hbond += 0.1
    hbond_network = min(hbond, 1.0)

    charge_density = min((abs(atom.GetFormalCharge()) + abs(charge)) / ATOM_VOLUME[z] / 0.5, 1.0)

    constraint = 0.0
    if atom.IsInRing():
        constraint += 0.4
    for bond in atom.GetBonds():
        constraint += _RIGID_BOND_WEIGHT.get(bond.GetBondType(), 0.0)
    if atom.GetIsAromatic():
        constraint += 0.3
    geometric_constraint = min(constraint, 1.0)

    # Accessibility: side-chain size, flexibility, polarity, backbone and aromaticity factors,
    # shifted by the descriptors that indicate exposure or burial.
    polarity = min((ELECTRONEGATIVITY[z] - 2.0) / 1.5 + min(abs(charge) * 2, 1.0), 1.0)
    base = (
        (1.0 - min(volume / 200.0, 1.0))
        * (0.5 + 0.5 * min(rotatable / 5.0, 1.0))
        * (0.5 + 0.5 * polarity)
        * (0.3 if backbone[atom.GetIdx()] else 1.0)
        * (1.0 - (0.3 if atom.GetIsAromatic() else 0.0))
    )
    exposed = (
        0.15 * surface_to_volume
        + 0.10 * backbone_distance
        + 0.15 * entropy
        + 0.10 * hbond_network
        + 0.10 * charge_density
    )
    buried = 0.15 * branching + 0.15 * packing_density + 0.10 * geometric_constraint
    return Accessibility(
        side_chain_volume=volume,
        branching=branching,
        backbone_distance=backbone_distance,
        surface_to_volume=surface_to_volume,
        packing_density=packing_density,
        conformational_entropy=entropy,
        hbond_network=hbond_network,
        charge_density=charge_density,
        geometric_constraint=geometric_constraint,
        accessibility=max(0.0, min(1.0, base + (exposed - buried))),
    )
