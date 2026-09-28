"""Raw dataset sequences to pyPept BILN strings."""

from collections.abc import Mapping

STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Modification codes of the DeepRT+ HeLa set (hela_mod); values are monomer symbols in the SDF.
HELA_DIGIT_MAP: dict[str, str] = {
    "1": "Met_O",  # methionine sulfoxide
    "2": "Ser_PO3H2",  # phosphoserine
    "3": "Thr_PO3H2",  # phosphothreonine
    "4": "Tyr_PO3H2",  # phosphotyrosine
}


def decode_sequence(sequence: str, digit_map: Mapping[str, str] | None = None) -> str:
    """Return the BILN form of a one-letter sequence, e.g. ``"LA1K"`` -> ``"L-A-Met_O-K"``."""
    digit_map = digit_map or {}
    monomers = []
    for code in sequence.strip().upper():
        if code in STANDARD_AMINO_ACIDS:
            monomers.append(code)
        elif code in digit_map:
            monomers.append(digit_map[code])
        else:
            raise ValueError(f"unknown residue code {code!r} in {sequence!r}")
    return "-".join(monomers)
