# PepOracle: atomic attention for peptide retention time

PepOracle predicts the chromatographic retention time of a peptide from its atoms. Each
peptide is turned into a graph of heavy atoms with 88 features per atom, and a transformer
lets every atom attend to every other atom in the peptide ("atomic attention"). There is
no message passing along bonds.

This repository contains everything behind the results below: the graph builder, the
hyperparameter search, and the sweep that trains one frozen configuration on nine datasets
with three seeds each.

## Results

Test R² of the frozen configuration. Each seed holds out a different random 10% of the
peptides as its test set, so the spread includes which peptides land in the test set.

| dataset | peptides | test R² (mean ± SD, 3 seeds) | training time per run |
|---|---:|---|---:|
| HeLa (`hela_mod3`) | 1,170 | 0.9618 ± 0.0087 | 6 min |
| HeLa, modified residues kept (`hela_mod`) | 3,413 | 0.9585 ± 0.0069 | 16 min |
| Yeast (`yeast_unmod`) | 14,361 | 0.9644 ± 0.0020 | 62 min |
| SCX (`scx`) | 30,482 | 0.9947 ± 0.0008 | 2.2 h |
| Luna HILIC (`luna_hilic`) | 36,271 | 0.9826 ± <0.0001 | 2.6 h |
| Luna Silica (`luna_silica`) | 37,110 | 0.9824 ± 0.0005 | 2.6 h |
| Atlantis Silica (`atlantis_silica`) | 39,091 | 0.9803 ± 0.0004 | 2.8 h |
| XBridge Amide (`xbridge_amide`) | 40,290 | 0.9862 ± 0.0003 | 2.9 h |
| Misc (`misc_dia`) | 146,587 | 0.9780 ± 0.0006 | 10.4 h |

Training times are on one NVIDIA RTX 5070 (12 GB). The per-run records are in
[`results/atomic_attention/`](results/atomic_attention/).
They were produced by the original scripts; [Verification](#verification-against-the-original-runs)
explains how this code was checked against them.

## Method

**Graphs.** Sequences are assembled atom by atom with [pyPept](https://github.com/Boehringer-Ingelheim/pyPept)
and RDKit (`peporacle.featurise`). Each heavy atom gets 88 features: element, charge
(formal and Gasteiger), hybridisation, ring membership, hydrogen-bonding and hydrophobicity
descriptors, and the position and properties of the residue it belongs to. The full list is
`peporacle.featurise.NODE_FEATURES`. Bonds are stored with 12 features for use by other
models, but atomic attention does not read them, so bond structure reaches the model only
through the atom features.

**Model** (`peporacle.model`). A linear atom embedding, 15 post-norm transformer blocks
(4 heads, width 128) over all atoms of each peptide, mean and max pooling, and a small MLP
head: 3.0 M parameters.

**Splits** (`peporacle.data`). For seed *k*, 10% of peptides are held out as the test set
(grouped by sequence, `GroupShuffleSplit`, `random_state=k`), and 10% of the rest become the
validation set (`random_state=k+1`). Retention times are standardised with training-set
statistics.

**Training** (`peporacle.train`). MSE loss, AdamW, 100 epochs, batch 32 with 4-step gradient
accumulation, gradient clipping, and a learning rate that warms up linearly from zero over
12 epochs and then follows a cosine decay. The test score uses the weights from the epoch
with the best validation R².

**Tuning** (`peporacle.tune`). One configuration is used for every dataset. It was chosen
by a 20-trial Optuna search on a fixed random subset of 10,000 peptides from Misc. Each trial
trained 100 epochs and was scored on the subset's validation split with its final-epoch
weights; the subset's test split was never used. The winning trial is
[`configs/frozen.toml`](configs/frozen.toml), and the search record is in
[`results/tuning/`](results/tuning/).

## Reproducing

Python 3.12. Install the bundled pyPept and this package (a CUDA build of PyTorch 2.11 is
needed for GPU training; see [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install ./third_party/pyPept -e ".[dev]"
pytest
```

Then run the steps in order:

```bash
python scripts/download_data.py     # the nine dataset CSVs -> data/ (checksum-verified)
python scripts/build_caches.py      # CSVs -> graph caches in caches/ (resumable)
python scripts/tune.py              # optional: repeat the 20-trial search
python scripts/sweep.py             # frozen config x 9 datasets x seeds 0-2 -> results/sweep/
python scripts/summarise.py         # mean and SD per dataset -> results/summary.csv
```

`sweep.py` and `tune.py` take dataset names, `--seeds` and `--trials` to run a subset; every
run checkpoints each epoch and resumes after an interruption. Building all nine caches takes
a few hours on 8 CPU cores; the whole sweep took 75 hours (about 3 days) on one RTX 5070. On a GPU
that also drives a display, `--gpu-memory-fraction 0.65` keeps PyTorch from holding memory
up to the card's limit.

## Verification against the original runs

The results above were produced by earlier research scripts. This code is a rewrite, checked
against them component by component:

| component | check | result |
|---|---|---|
| Graphs | new featuriser vs the published caches (HeLa and HeLa-modified in full; the first 400 of Yeast and SCX) | 5,383 / 5,383 bit-identical |
| Graphs | new vs original featuriser, 150 random peptides from each of the other five datasets | 750 / 750 bit-identical |
| Molecules | pyPept molecules and atom-to-residue assignments with the trimmed pyPept patch vs the original, all 217,298 distinct peptides (500 per dataset with `tools/check_featuriser.py`) | identical |
| Caches | `scripts/build_caches.py` from the CSVs vs the published HeLa caches | 4,583 / 4,583 graphs identical |
| Splits | all 27 dataset × seed splits | identical |
| Model | outputs from identical weights | bit-identical |
| Training | original vs new training loop, same starting weights and batches, 3 epochs on HeLa (CPU) | losses, validation R² and all weights bit-identical |
| Tuning | search space, subset (10,000 rows, seed `0xC0FFEE`), split and scoring rule vs the launch record | identical settings |

The tools are in [`tools/`](tools/); they need the original scripts, which are not
included in this repository. Two differences remain:

- The original runs set no random seed, and GPU training is not deterministic, so their
  exact values cannot be regenerated by any code. This code seeds each run from its seed
  number, so its own runs repeat on the same hardware; they should agree with the table
  within the seed spread. For the same reason `tune.py` repeats the search, not its exact trials.
- The original sweep used early stopping with a patience of 50 epochs. It never triggered in
  the 16 runs whose logs survive; for the other 11 (HeLa, HeLa-modified and Yeast seeds 0-2,
  SCX seeds 0-1) the epoch count was not recorded. Since the test score always came from the
  best-validation epoch, early stopping could only have changed a result if it stopped a run
  that would have improved later. This code trains the full 100 epochs.

## Limitations

- Three seeds per dataset. On the two HeLa sets (117 and 342 test peptides) the spread is
  dominated by which peptides are in the test set.
- One configuration, tuned on Misc only, is applied to every dataset.
- Atoms are assigned to residues by substructure matching. For 72% of peptides some atoms are
  instead assigned from their neighbours or by position, so the residue-level atom features
  are approximate.
- The published caches were built on Apple silicon. The two bond-length columns (`length`,
  `length_3d`) come from 2D depiction coordinates, whose last bits can depend on the CPU
  architecture: an earlier check on x86 found differences there in some graphs, although the
  10 golden test graphs match bit for bit on x86. The tests therefore check those two columns
  exactly only on Apple silicon and every other feature exactly everywhere. Atomic attention
  does not read bond features, so its results are unaffected.

## Layout

```
configs/        frozen configuration and dataset list
data/           dataset sources, citations and checksums (CSVs are downloaded)
scripts/        the steps above
src/peporacle/  featurise/ (sequence -> graph), cache, data, model, train, tune, sweep
results/        published runs (atomic_attention/), tuning record (tuning/)
tests/          unit tests, including golden graphs from the published caches
third_party/    pyPept (MIT, Boehringer Ingelheim) with the patches in PATCHES.md
tools/          checks against the original scripts
```

## Citation and licence

See [`CITATION.cff`](CITATION.cff). The code is MIT-licensed; pyPept is MIT-licensed by
Boehringer Ingelheim. The datasets are not included; their sources, which should be cited
when the data are used, are listed in [`data/README.md`](data/README.md).
