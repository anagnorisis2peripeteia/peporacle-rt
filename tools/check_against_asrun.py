"""Check the split and the model against the code that produced the published results.

Usage: python tools/check_against_asrun.py AS_RUN_DIR DATA_DIR
AS_RUN_DIR holds the original ``pep_core`` package; DATA_DIR holds the dataset CSVs.
Split sizes are compared with results/atomic_attention/runs.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Batch, Data

from peporacle.config import ModelConfig
from peporacle.data import split_indices
from peporacle.model import AtomicAttention

asrun_dir, data_dir = map(Path, sys.argv[1:3])
RUNS = Path(__file__).resolve().parents[1] / "results" / "atomic_attention" / "runs.csv"
sys.path.insert(0, str(asrun_dir))
from pep_core.models import GATWithGTModel  # noqa: E402


def asrun_split(sequences, seed, test_frac=0.1, val_frac=0.1):
    """Verbatim logic of pep_core.dataset.random_split (group_by_seq, early_stop)."""
    uniq = list(dict.fromkeys(sequences))
    s2g = {s: i for i, s in enumerate(uniq)}
    groups = np.array([s2g[s] for s in sequences])
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train, test = next(gss.split(np.arange(len(sequences)), groups=groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
    sub_tr, sub_val = next(gss.split(np.arange(len(train)), groups=groups[train]))
    return train[sub_tr], train[sub_val], test


with open(RUNS) as f:
    published = {(r["dataset"], int(r["seed"])): r for r in csv.DictReader(f)}
for (dataset, seed), rec in sorted(published.items()):
    with open(data_dir / f"{dataset}.csv") as f:
        seqs = [row["sequence"] for row in csv.DictReader(f)]
    new = split_indices(seqs, seed)
    old = asrun_split(seqs, seed)
    same = all(
        np.array_equal(a, b) for a, b in zip((new.train, new.val, new.test), old, strict=True)
    )
    counts = (len(new.train), len(new.val), len(new.test))
    recorded = tuple(int(rec[k]) if rec[k] else None for k in ("n_train", "n_val", "n_test"))
    ok = "not recorded" if None in recorded else counts == recorded
    print(f"split {dataset:16s} seed {seed}: identical={same} counts {counts} match_record={ok}")

cfg = ModelConfig(hidden_dim=128, n_blocks=15, n_heads=4, dropout=0.0594)
torch.manual_seed(0)
old = GATWithGTModel(88, 12, hidden_dim=128, n_local=0, n_glob=15, n_heads=4, dropout=0.0594).eval()
new = AtomicAttention(88, cfg).eval()
renamed = {
    k.replace("global_layers.", "blocks.").replace("head.mlp.", "head."): v
    for k, v in old.state_dict().items()
}
new.load_state_dict(renamed, strict=True)
graphs = [
    Data(
        x=torch.randn(n, 88),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 12),
    )
    for n in (31, 157, 88, 212, 64)
]
batch = Batch.from_data_list(graphs)
with torch.no_grad():
    a, b = old(batch), new(batch)
print(
    f"model: params old {sum(p.numel() for p in old.parameters())} "
    f"new {sum(p.numel() for p in new.parameters())} | outputs bit-identical: {torch.equal(a, b)}"
    f" | max abs diff {float((a - b).abs().max()):.3g}"
)
