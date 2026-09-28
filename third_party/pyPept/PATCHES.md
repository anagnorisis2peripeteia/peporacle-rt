# pyPept: local changes

This is [pyPept](https://github.com/Boehringer-Ingelheim/pyPept) at commit `4214ca54c0d0`
(MIT licence, see `LICENSE`): the package source and packaging files, without upstream's
`docs/`, `examples/`, `tests/` and linter configuration. Install with
`pip install ./third_party/pyPept`.

The changes below are also in `peporacle.diff` (`patch -p1` in an upstream checkout
reproduces this tree). They are the smallest set that lets peporacle rebuild the published
graph caches bit for bit; the evidence is in `featurise_verification.json`.

## Changes

1. **`sequence.py`: monomer dictionary from a directory.** Upstream resolves `path` with
   `importlib.resources.files`, which accepts only importable packages, so a dictionary
   outside the package cannot be loaded. A `path` that is a directory is now joined with
   `monomer_lib`. peporacle passes the directory and file name of the dictionary it uses
   (by default pyPept's own `data/monomers.sdf`).

2. **`sequence.py`: pandas 3.** `get_monomer_info` stores Python lists in the three R-group
   columns. pandas 3 gives those columns a string dtype, and `df.loc[idx, col] = list`
   raises `ValueError: Must have equal len keys and value when setting with an iterable`.
   The columns are cast to `object` and cells are set with `.at`; under pandas 3.0.6
   neither change works alone.

3. **`molecule.py`: `Molecule.unsanitized_mol`.** A copy of the assembled molecule after the
   R-group placeholders are deleted and before `SanitizeMol`. peporacle assigns atoms to
   residues on this molecule (`peporacle.featurise.molecule.assign_residues`), the point at
   which the code that built the published caches did it. That code called the featuriser
   from inside `molecule.py`, a circular import that this attribute removes.

## Changes in the as-run copy that were not kept

The copy that built the published caches (`pep_core/pypept`) carried further edits. For all
217,298 distinct peptides of the nine RT datasets, the molecules and residue maps it produces
(atoms, bonds, stereo and ring flags, 2D coordinates, atom-to-residue map) are identical to
those of this tree with peporacle's `assign_residues`, and this tree raises for none of them
(`tools/check_featuriser.py --level molecule` reruns the comparison). The edits were dropped:

- a manual atom-and-bond copy when `Chem.CombineMols` raises, and a
  `SetNoImplicit`-and-retry when `RWMol.AddBond` raises, both meant for protected monomers,
  plus `try`/`except` blocks with warnings around these two steps: neither call ever raises;
- removing the `[#0]` placeholders with an `RWMol.RemoveAtom` loop instead of
  `Chem.DeleteSubstructs`: the resulting molecules are identical;
- `UpdatePropertyCache`, a forced `RingInfo` initialisation, and ignoring `SanitizeMol`
  failures: sanitisation never fails;
- the `PYPEPT_MONOMER_SDF` environment variable and a file path accepted as `path`: replaced
  by change 1;
- the in-`molecule.py` atom-to-residue mapping and its offset-ratio fallback (used only if
  the featuriser could not be imported): moved to peporacle, see change 3.
