# PepOracle: atomic attention for peptide retention time

PepOracle predicts the chromatographic retention time of a peptide from its atoms. Each
peptide is turned into a graph of heavy atoms with 88 features per atom, and a transformer
lets every atom attend to every other atom in the peptide ("atomic attention"). There is
no message passing along bonds.

This repository contains the graph builder, hyperparameter search, a sweep that trains
one frozen configuration on nine datasets with three seeds each, and the message-passing
models it is compared with.

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

## Operator comparison

The same nine datasets and seeds with five message-passing operators in place of atomic
attention (15 layers, same frozen settings, no attention). Test R², mean ± SD over 3 seeds:

| dataset | atomic attention | TransformerConv | GCNConv | SAGEConv | ARMAConv | NNConv |
|---|---:|---:|---:|---:|---:|---:|
| HeLa | 0.9618 ± 0.0087 | 0.8864 ± 0.0267 | 0.9114 ± 0.0072 | 0.9277 ± 0.0153 | 0.9173 ± 0.0230 | 0.9176 ± 0.0080 |
| HeLa, modified | 0.9585 ± 0.0069 | 0.8654 ± 0.0363 | 0.8909 ± 0.0243 | 0.9359 ± 0.0058 | 0.9233 ± 0.0131 | 0.8994 ± 0.0139 |
| Yeast | 0.9644 ± 0.0020 | 0.9501 ± 0.0037 | 0.9495 ± 0.0064 | 0.9536 ± 0.0048 | 0.9660 ± 0.0026 | 0.9537 ± 0.0076 |
| SCX | 0.9947 ± 0.0008 | 0.9553 ± 0.0045 | 0.9170 ± 0.0101 | 0.9921 ± 0.0007 | 0.9751 ± 0.0019 | 0.9090 ± 0.0102 |
| Luna HILIC | 0.9826 ± <0.0001 | 0.9850 ± 0.0007 | 0.9837 ± 0.0008 | 0.9801 ± 0.0007 | 0.9871 ± 0.0006 | 0.9835 ± 0.0013 |
| Luna Silica | 0.9824 ± 0.0005 | 0.9815 ± 0.0009 | 0.9806 ± 0.0013 | 0.9780 ± 0.0003 | 0.9832 ± 0.0014 | 0.9762 ± 0.0019 |
| Atlantis Silica | 0.9803 ± 0.0004 | 0.9769 ± 0.0011 | 0.9779 ± 0.0028 | 0.9762 ± 0.0008 | 0.9828 ± 0.0011 | 0.9732 ± 0.0025 |
| XBridge Amide | 0.9862 ± 0.0003 | 0.9885 ± 0.0015 | 0.9878 ± 0.0004 | 0.9836 ± 0.0010 | 0.9912 ± 0.0005 | 0.9894 ± 0.0002 |
| Misc | 0.9780 ± 0.0006 | 0.9869 ± 0.0009 | 0.9742 ± 0.0006 | 0.9637 ± 0.0005 | 0.9905 ± 0.0005 | 0.9841 ± 0.0009 |

Models, protocol and the paired comparisons with atomic attention are in
[`results/operator_ablation/`](results/operator_ablation/).

## Operator + attention

The same five operators, each followed by 5 atomic-attention blocks (15 operator layers, then
5 attention blocks, same frozen settings). Test R², mean ± SD over 3 seeds:

| dataset | atomic attention | TransformerConv | GCNConv | SAGEConv | ARMAConv | NNConv |
|---|---:|---:|---:|---:|---:|---:|
| HeLa | 0.9618 ± 0.0087 | 0.9542 ± 0.0134 | 0.9576 ± 0.0119 | 0.9633 ± 0.0055 | 0.9460 ± 0.0261 | 0.9515 ± 0.0067 |
| HeLa, modified | 0.9585 ± 0.0069 | 0.9527 ± 0.0056 | 0.9583 ± 0.0062 | 0.9581 ± 0.0057 | 0.9407 ± 0.0131 | 0.9515 ± 0.0038 |
| Yeast | 0.9644 ± 0.0020 | 0.9679 ± 0.0032 | 0.9662 ± 0.0023 | 0.9640 ± 0.0037 | 0.9716 ± 0.0011 | 0.9691 ± 0.0032 |
| SCX | 0.9947 ± 0.0008 | 0.9947 ± 0.0004 | 0.9945 ± 0.0006 | 0.9953 ± 0.0003 | 0.9950 ± 0.0003 | 0.9937 ± 0.0006 |
| Luna HILIC | 0.9826 ± <0.0001 | 0.9873 ± 0.0012 | 0.9854 ± 0.0006 | 0.9835 ± 0.0005 | 0.9885 ± 0.0002 | 0.9859 ± 0.0006 |
| Luna Silica | 0.9824 ± 0.0005 | 0.9855 ± 0.0006 | 0.9849 ± 0.0007 | 0.9829 ± 0.0004 | 0.9867 ± 0.0005 | 0.9854 ± 0.0006 |
| Atlantis Silica | 0.9803 ± 0.0004 | 0.9850 ± 0.0009 | 0.9829 ± 0.0023 | 0.9809 ± 0.0008 | 0.9861 ± 0.0006 | 0.9836 ± 0.0009 |
| XBridge Amide | 0.9862 ± 0.0003 | 0.9914 ± 0.0004 | 0.9898 ± 0.0008 | 0.9868 ± 0.0005 | 0.9927 ± 0.0003 | 0.9905 ± 0.0001 |
| Misc | 0.9780 ± 0.0006 | 0.9894 ± 0.0007 | 0.9826 ± 0.0010 | 0.9777 ± 0.0006 | 0.9920 ± 0.0008 | 0.9842 ± 0.0015 |

Models, protocol and the paired comparisons with atomic attention and with each operator alone
are in [`results/operator_attention/`](results/operator_attention/).

## Method

**Graphs.** A [patched copy](third_party/pyPept/PATCHES.md) of
[pyPept](https://github.com/Boehringer-Ingelheim/pyPept) assembles peptide molecules from
monomer structures. PepOracle computes graph features using RDKit (`peporacle.featurise`).
Each heavy atom gets 88 features: element, charge
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

**Operator models** (`peporacle.operators`). Five PyTorch Geometric operators
(TransformerConv, GCNConv, SAGEConv, ARMAConv, NNConv), each as 15 layers on the same atom
embedding, pooling and head, with no attention blocks (operator ablation) or 5 attention
blocks after them (operator + attention). TransformerConv and NNConv read the 12 bond
features; the others use only which atoms are bonded. They use the frozen settings
unchanged; [`configs/operators.toml`](configs/operators.toml) lists the few settings of
their own.

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
python scripts/sweep.py             # atomic attention x 9 datasets x seeds 0-2 -> results/sweep/
python scripts/sweep_operators.py operator_ablation    # -> results/sweep_operator_ablation/
python scripts/sweep_operators.py operator_attention   # -> results/sweep_operator_attention/
python scripts/summarise.py         # runs.csv and summary.csv in each results/sweep*/
python scripts/compare.py           # paired comparisons and verdicts, as in results/
```

The sweeps take dataset names and `--seeds` (and `--operators` for the operator sweeps) to
run a subset; every run checkpoints each epoch and resumes after an interruption. Building
all nine caches takes a few hours on 8 CPU cores. On one RTX 5070 the atomic-attention sweep
took 75 GPU hours, the operator ablation 425 and operator + attention 657; NNConv and
ARMAConv account for most of the operator time. On a GPU that also drives a display,
`--gpu-memory-fraction 0.65` keeps PyTorch from holding memory up to the card's limit; the
published runs used it throughout.

Seed *k* fixes the split, so a rerun has the same test peptides as the published runs. The
scripts also seed the initial weights and batch order with *k*; the published runs did not,
so rerun scores will differ from the published ones by run-to-run training variation.
`python scripts/compare.py --published` rebuilds the paired comparisons in `results/` from
the published `runs.csv` files.

## Layout

```
configs/        frozen configuration, operator settings and dataset list
data/           dataset sources, citations and checksums (CSVs are downloaded)
scripts/        the steps above
src/peporacle/  featurise/ (sequence -> graph), cache, data, model, operators, train,
                tune, sweep, results
results/        published runs (atomic_attention/, operator_ablation/, operator_attention/),
                tuning record (tuning/)
tests/          unit tests, including golden graphs from the published caches
third_party/    pyPept (MIT, Boehringer Ingelheim) with the patches in PATCHES.md
tools/          checks against the original scripts
```

## Citation and licence

See [`CITATION.cff`](CITATION.cff). The code is MIT-licensed; pyPept is MIT-licensed by
Boehringer Ingelheim. The datasets are not included; their sources, which should be cited
when the data are used, are listed in [`data/README.md`](data/README.md).

pyPept: Ochoa R, Brown JB, Fox T (2023).
[pyPept: a python library to generate atomistic 2D and 3D representations of peptides](https://doi.org/10.1186/s13321-023-00748-2).
*Journal of Cheminformatics* 15, 79.
