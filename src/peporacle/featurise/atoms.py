"""Node (atom) features."""

from rdkit import Chem

from peporacle.featurise.accessibility import Accessibility, accessibility_descriptors, is_acceptor
from peporacle.featurise.elements import ELECTRONEGATIVITY, VDW_RADIUS
from peporacle.featurise.hydrophobicity import atom_hydrophobicity
from peporacle.featurise.molecule import Peptide
from peporacle.featurise.residues import residue_properties

HT = Chem.HybridizationType
CT = Chem.ChiralType

# One-hot element columns; any other element sets element_other.
ELEMENTS = (
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "H",
    "B",
    "Si",
    "Se",
    "Te",
    "As",
    "Na",
    "K",
    "Ca",
    "Mg",
)
FORMAL_CHARGES = (-2, -1, 0, 1, 2)
HYBRIDISATIONS = (HT.SP, HT.SP2, HT.SP3)
RING_SIZES = (3, 4, 5, 6, 7, 8)  # smallest ring containing the atom; 8 stands for 8 or more
POLAR_ELEMENTS = ("N", "O", "P", "S")
MULTIPLE_BONDS = (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
# Positional features are scaled by a nominal maximum peptide length.
MAX_RESIDUES = 100.0

NODE_FEATURES = (
    *(f"element_{symbol}" for symbol in ELEMENTS),
    "element_other",
    "atomic_number",
    "formal_charge",
    *(f"formal_charge_{charge:+d}" for charge in FORMAL_CHARGES),
    "partial_charge",
    "electronegativity",
    "vdw_radius",
    "sp",
    "sp2",
    "sp3",
    "hybridisation_other",
    "aromatic",
    "in_ring",
    *(f"smallest_ring_{size}" for size in RING_SIZES),
    "ring_count",
    "valence",
    "degree",
    "hydrogens",
    "chiral",
    "chiral_cw",
    "chiral_ccw",
    "chiral_unspecified",
    "hydrophobicity",
    "hbond_donor",
    "hbond_acceptor",
    "aromatic_neighbours",
    "accessibility",
    "degree_exposure",
    "degree_6",
    "polar",
    "polar_neighbour_fraction",
    "ring_rigid",
    "multiple_bonds",
    "electrons",
    "formal_charge_per_neighbour",
    "abs_formal_charge",
    "branching",
    "backbone_distance",
    "surface_to_volume",
    "packing_density",
    "conformational_entropy",
    "hbond_network",
    "charge_density",
    "geometric_constraint",
    "charge_x_volume",
    "electronegativity_x_charge",
    "branching_x_packing",
    "surface_to_volume_x_entropy",
    "charge_x_surface_to_volume",
    "packing_x_entropy",
    "relative_position",
    "position_per_1000",
    "position_per_100",
    "length_per_100",
    "residues_to_c_terminus_per_100",
    "residue_hydrophobicity",
    "residue_charge",
    "residue_size",
    "residue_branching",
    "residue_aromaticity",
)


def node_features(peptide: Peptide) -> list[list[float]]:
    """One row of ``NODE_FEATURES`` per atom."""
    mol, charges, num_residues = peptide.mol, peptide.charges, peptide.num_residues
    rings = mol.GetRingInfo().AtomRings()
    residues = residue_properties(mol, peptide.residue_index, charges)
    descriptors = accessibility_descriptors(mol, charges)
    rows = []
    for atom, charge, access, residue in zip(
        mol.GetAtoms(), charges, descriptors, peptide.residue_index, strict=True
    ):
        position = [
            residue / max(num_residues - 1, 1),
            residue / 1000.0,
            residue / MAX_RESIDUES,
            num_residues / MAX_RESIDUES,
            (num_residues - 1 - residue) / MAX_RESIDUES,
        ]
        rows.append([*_atom_features(atom, rings, charge, access), *position, *residues[residue]])
    return rows


def _atom_features(atom: Chem.Atom, rings: tuple, charge: float, access: Accessibility) -> list:
    z, symbol = atom.GetAtomicNum(), atom.GetSymbol()
    formal, degree = atom.GetFormalCharge(), atom.GetDegree()
    hybridisation, chirality = atom.GetHybridization(), atom.GetChiralTag()
    ring_sizes = [len(ring) for ring in rings if atom.GetIdx() in ring]
    smallest_ring = min(*ring_sizes, RING_SIZES[-1]) if ring_sizes else 0
    neighbours = atom.GetNeighbors()
    multiple_bonds = sum(b.GetBondType() in MULTIPLE_BONDS for b in atom.GetBonds())
    volume = min(access.side_chain_volume / 200.0, 1.0)
    partial = abs(charge)
    return [
        *(symbol == element for element in ELEMENTS),
        symbol not in ELEMENTS,
        z / 100.0,
        formal / 2.0,
        *(formal == value for value in FORMAL_CHARGES),
        max(-1.0, min(1.0, charge)),
        ELECTRONEGATIVITY[z] / 4.0,
        VDW_RADIUS[z] / 3.0,
        *(hybridisation == value for value in HYBRIDISATIONS),
        hybridisation not in HYBRIDISATIONS,
        atom.GetIsAromatic(),
        atom.IsInRing(),
        *(smallest_ring == size for size in RING_SIZES),
        min(len(ring_sizes) / 5.0, 1.0),
        min(atom.GetTotalValence() / 6.0, 1.0),
        min(degree / 4.0, 1.0),
        min(atom.GetTotalNumHs() / 4.0, 1.0),
        chirality != CT.CHI_UNSPECIFIED,
        chirality == CT.CHI_TETRAHEDRAL_CW,
        chirality == CT.CHI_TETRAHEDRAL_CCW,
        chirality == CT.CHI_UNSPECIFIED,
        atom_hydrophobicity(atom, charge),
        symbol in ("N", "O") and atom.GetTotalNumHs() > 0,
        is_acceptor(atom),
        min(sum(n.GetIsAromatic() for n in neighbours) / 4.0, 1.0),
        access.accessibility,
        1.0 - min(degree / 6.0, 1.0),
        min(degree / 6.0, 1.0),
        symbol in POLAR_ELEMENTS,
        sum(n.GetSymbol() in POLAR_ELEMENTS for n in neighbours) / len(neighbours),
        atom.IsInRing(),
        min(multiple_bonds / 3.0, 1.0),
        min(z / 50.0, 1.0),
        min(abs(formal) / len(neighbours), 1.0),
        abs(formal) / 2.0,
        access.branching,
        access.backbone_distance,
        access.surface_to_volume,
        access.packing_density,
        access.conformational_entropy,
        access.hbond_network,
        access.charge_density,
        access.geometric_constraint,
        partial * volume,
        ELECTRONEGATIVITY[z] / 4.0 * partial,
        access.branching * access.packing_density,
        access.surface_to_volume * access.conformational_entropy,
        partial * access.surface_to_volume,
        access.packing_density * access.conformational_entropy,
    ]
