"""Per-element constants, keyed by atomic number."""

import math

# The descriptor tables below are defined for the elements of the RT datasets only.
SUPPORTED_ELEMENTS = frozenset({6, 7, 8, 15, 16})

# Pauling electronegativity (Allred, J. Inorg. Nucl. Chem. 17, 215, 1961)
ELECTRONEGATIVITY = {6: 2.55, 7: 3.04, 8: 3.44, 15: 2.19, 16: 2.58}

# van der Waals radius in angstrom (Bondi, J. Phys. Chem. 68, 441, 1964)
VDW_RADIUS = {6: 1.70, 7: 1.55, 8: 1.52, 15: 1.80, 16: 1.80}

# Static dipole polarisability in cubic angstrom; C, N, O and P as in the CRC Handbook,
# S is 3.30 (CRC: 2.90) in the published feature set.
POLARIZABILITY = {6: 1.76, 7: 1.10, 8: 0.80, 15: 3.63, 16: 3.30}

# Volume of the van der Waals sphere
ATOM_VOLUME = {z: (4 / 3) * math.pi * r**3 for z, r in VDW_RADIUS.items()}
