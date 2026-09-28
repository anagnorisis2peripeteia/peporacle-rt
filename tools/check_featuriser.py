"""Check that peporacle.featurise reproduces reference graphs bit for bit.

Two references are supported:

* ``--hashes FILE``: graph hashes of a published cache, ``{dataset: [[sequence, sha256, ...]]}``.
  Every listed sequence is featurised and compared.
* ``--asrun DIR``: the as-run code tree (holding ``pep_core``, ``RT`` and ``data``). Its
  featuriser is run on ``--sample`` random rows (``--seed``) of
  ``DIR/RT/<dataset>/data/<dataset>.csv``, both sides reading ``DIR``'s monomer dictionary.

The graph hash is SHA-256 over, for x, edge_index, edge_attr and residue_index in turn, the
name, the shape tuple and the raw bytes. With ``--level molecule`` the as-run comparison stops
after pyPept: it hashes the assembled molecule (atoms, bonds, rings, 2D coordinates) and the
atom-to-residue map, which checks the pyPept patch much faster. ``--sample 0`` takes every row.
Writes a JSON report; exits non-zero on a mismatch.

    python tools/check_featuriser.py --hashes published_cache_graph_hashes.json
    python tools/check_featuriser.py --asrun ~/Documents/pep-oracle --sample 150 --seed 0 \\
        --datasets luna_hilic luna_silica atlantis_silica xbridge_amide misc_dia
"""

import argparse
import csv
import hashlib
import importlib
import json
import multiprocessing
import random
import sys
import warnings
from pathlib import Path

from rdkit import RDLogger

from peporacle.sequences import HELA_DIGIT_MAP, decode_sequence

DIGIT_MAPS = {"hela_mod": HELA_DIGIT_MAP}
HASHED = ("x", "edge_index", "edge_attr", "residue_index")


def graph_hash(graph: object) -> str:
    digest = hashlib.sha256()
    for name in HASHED:
        tensor = getattr(graph, name)
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def molecule_hash(mol: object, residue_index: list[int]) -> str:
    """SHA-256 over atom and bond attributes, rings, 2D coordinates and the residue map."""
    atoms = [
        (
            a.GetAtomicNum(),
            a.GetFormalCharge(),
            a.GetIsAromatic(),
            a.GetHybridization(),
            a.GetChiralTag(),
            a.GetTotalNumHs(),
            a.GetNumExplicitHs(),
            a.GetNoImplicit(),
        )
        for a in mol.GetAtoms()
    ]
    bonds = [
        (
            b.GetBeginAtomIdx(),
            b.GetEndAtomIdx(),
            b.GetBondType(),
            b.GetBondDir(),
            b.GetStereo(),
            b.GetIsAromatic(),
            b.GetIsConjugated(),
        )
        for b in mol.GetBonds()
    ]
    digest = hashlib.sha256(repr((atoms, bonds, mol.GetRingInfo().AtomRings())).encode())
    digest.update(mol.GetConformer().GetPositions().tobytes())
    digest.update(repr(list(residue_index)).encode())
    return digest.hexdigest()


# The featurisers are imported inside the worker processes: the as-run and the release tree
# each provide a top-level ``pyPept``, so every process must import exactly one of them.


def _quiet() -> None:
    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore", message="fmt must be one of")


def _use_asrun(root: str) -> None:
    """Worker initialiser: import the as-run featuriser, a plain directory tree."""
    _quiet()
    sys.path.insert(0, root)
    importlib.import_module("pep_core.build_graph")  # also imports its vendored pyPept
    if not sys.modules["pyPept"].__file__.startswith(root):
        raise RuntimeError(f"pyPept was imported from {sys.modules['pyPept'].__file__}")


def _new_hash(task: tuple[str, str | None, dict | None]) -> str:
    featurise = importlib.import_module("peporacle.featurise")
    return graph_hash(featurise.build_graph(*task))


def _asrun_hash(task: tuple[str, str, dict | None]) -> str:
    return graph_hash(importlib.import_module("pep_core.build_graph").build_graph(*task))


def _new_molecule_hash(task: tuple[str, str | None, dict | None]) -> str:
    molecule = importlib.import_module("peporacle.featurise.molecule")
    sequence, sdf, digit_map = task
    sdf = Path(sdf or molecule.DEFAULT_MONOMER_SDF)
    peptide = molecule.build_peptide(decode_sequence(sequence, digit_map), sdf)
    return molecule_hash(peptide.mol, peptide.residue_index)


def _asrun_molecule_hash(task: tuple[str, str, dict | None]) -> str:
    sequence, sdf, digit_map = task
    biln = importlib.import_module("pep_core.sequence_parser").decode_sequence(sequence, digit_map)
    sequence_type = importlib.import_module("pyPept.sequence").Sequence
    molecule = importlib.import_module("pyPept.molecule").Molecule(sequence_type(biln, sdf))
    residue_map = molecule.atom_to_residue_map
    return molecule_hash(molecule.mol, [residue_map[i] for i in range(molecule.mol.GetNumAtoms())])


WORKERS = {  # (new code, as-run code) per comparison level
    "graph": (_new_hash, _asrun_hash),
    "molecule": (_new_molecule_hash, _asrun_molecule_hash),
}


def _hashes(worker, tasks: list, workers: int, initializer, *initargs: str) -> list[str]:
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=initializer, initargs=initargs) as pool:
        return pool.map(worker, tasks, chunksize=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--hashes", type=Path, help="published cache hashes (JSON)")
    reference.add_argument("--asrun", type=Path, help="root of the as-run code tree")
    parser.add_argument("--sdf", type=Path, help="monomer dictionary for the new code")
    parser.add_argument("--datasets", nargs="+", help="default: all datasets in --hashes")
    parser.add_argument("--level", choices=sorted(WORKERS), default="graph")
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("featurise_check.json"))
    args = parser.parse_args()
    sdf = args.sdf and str(args.sdf)
    new_worker, asrun_worker = WORKERS[args.level]
    if args.hashes and args.level != "graph":
        parser.error("published hashes are graph hashes; use --level graph")
    if args.asrun and not args.datasets:
        parser.error("--asrun needs --datasets")

    if args.hashes:
        published = json.loads(args.hashes.read_text())
        datasets = args.datasets or list(published)
        cases = {ds: [(row[0], row[1]) for row in published[ds]] for ds in datasets}
    else:
        root = args.asrun.expanduser().resolve()
        sdf = sdf or str(root / "data" / "dictionaries" / "rt_monomers.sdf")
        cases = {}
        for ds in args.datasets:
            with (root / "RT" / ds / "data" / f"{ds}.csv").open() as handle:
                rows = [row["sequence"] for row in csv.DictReader(handle)]
            size = args.sample or len(rows)
            picked = sorted(random.Random(args.seed).sample(range(len(rows)), size))
            tasks = [(rows[i], sdf, DIGIT_MAPS.get(ds)) for i in picked]
            expected = _hashes(asrun_worker, tasks, args.workers, _use_asrun, str(root))
            cases[ds] = [(rows[i], h) for i, h in zip(picked, expected, strict=True)]

    report, failed = {}, False
    for ds, pairs in cases.items():
        tasks = [(sequence, sdf, DIGIT_MAPS.get(ds)) for sequence, _ in pairs]
        actual = _hashes(new_worker, tasks, args.workers, _quiet)
        mismatches = [s for (s, h), a in zip(pairs, actual, strict=True) if a != h]
        report[ds] = {
            "reference": "published cache" if args.hashes else f"as-run code ({args.level})",
            "checked": len(pairs),
            "identical": len(pairs) - len(mismatches),
            "mismatches": mismatches,
        } | ({} if args.hashes else {"sample": args.sample, "seed": args.seed})
        failed |= bool(mismatches)
        print(f"{ds}: {report[ds]['identical']}/{len(pairs)} identical")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
