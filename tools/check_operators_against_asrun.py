"""Check the operator models against the code that produced the published operator runs.

    python tools/check_operators_against_asrun.py AS_RUN_DIR

AS_RUN_DIR holds the original ``pep_core`` package. For each operator, with and without
attention blocks, the as-run model's weights are loaded into ``OperatorModel``; the two must
give bit-identical predictions, and bit-identical weights after one training epoch from the
same random state on the CPU. Parameter counts are compared with the published runs.csv files.
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from peporacle.config import load_frozen, load_operator_settings
from peporacle.operators import OPERATORS, OperatorModel
from peporacle.train import train_epoch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(sys.argv[1])))
from pep_core.models import MODELS  # noqa: E402
from pep_core.train import _train_epoch  # noqa: E402

ASRUN_NAME = {"TransformerConv": "gatgt", "GCNConv": "gcnconv", "SAGEConv": "sageconv"}
ASRUN_NAME |= {"ARMAConv": "arma", "NNConv": "nnconv"}
RENAME = {"global_layers.": "blocks.", "head.mlp.": "head.", "local_layers.": "convs."}
RENAME |= {"local_norms.": "norms."}
NODE_DIM, EDGE_DIM = 88, 12

torch.set_num_threads(1)  # single-threaded kernels, so both runs reduce in the same order
model_cfg, base = load_frozen(ROOT / "configs" / "frozen.toml")
settings = load_operator_settings(ROOT / "configs" / "operators.toml")


def rename(key: str) -> str:
    for old, new in RENAME.items():
        key = key.replace(old, new)
    return key


def peptide_like_graph(n_atoms: int) -> Data:
    """A backbone chain with a side branch every third atom, bonds stored in both directions."""
    pairs = [(i, i + 1) for i in range(n_atoms // 2 - 1)]
    pairs += [
        (3 * j, n_atoms // 2 + j) for j in range(n_atoms - n_atoms // 2) if 3 * j < n_atoms // 2
    ]
    src, dst = map(torch.tensor, zip(*pairs, strict=True))
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1)
    return Data(
        x=torch.randn(n_atoms, NODE_DIM),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.shape[1], EDGE_DIM),
        y=torch.randn(()),
    )


def published_params(arm: str) -> dict[str, int]:
    with open(ROOT / "results" / arm / "runs.csv") as f:
        return {r["operator"]: int(r["n_params"]) for r in csv.DictReader(f)}


torch.manual_seed(0)
graphs = [peptide_like_graph(n) for n in torch.randint(20, 90, (96,)).tolist()]
probe = Batch.from_data_list(graphs[:8])
cpu = torch.device("cpu")
all_ok = True
for arm, n_attention in settings.attention_blocks.items():
    params = published_params(arm)
    for op in OPERATORS:
        torch.manual_seed(1)
        old = MODELS[ASRUN_NAME[op]](
            NODE_DIM,
            EDGE_DIM,
            hidden_dim=model_cfg.hidden_dim,
            n_local=settings.n_layers,
            n_glob=n_attention,
            n_heads=model_cfg.n_heads,
            dropout=model_cfg.dropout,
        )
        new = OperatorModel(
            op,
            NODE_DIM,
            EDGE_DIM,
            model_cfg,
            n_layers=settings.n_layers,
            n_attention_blocks=n_attention,
        )
        new.load_state_dict({rename(k): v for k, v in old.state_dict().items()}, strict=True)
        with torch.no_grad():
            same_out = torch.equal(old.eval()(probe), new.eval()(probe))

        cfg = settings.train_config(base, op, "any")
        start = {k: v.clone() for k, v in new.state_dict().items()}
        rng = torch.get_rng_state()
        trained = []
        for model in (old, new):
            torch.set_rng_state(rng)
            opt = AdamW(model.parameters(), lr=base.lr, weight_decay=base.weight_decay)
            loader = DataLoader(graphs, batch_size=cfg.batch_size, shuffle=True)
            if model is old:
                old_cfg = SimpleNamespace(
                    task="regression", grad_accum=cfg.grad_accum, grad_clip=cfg.grad_clip
                )
                _train_epoch(model, loader, opt, torch.nn.MSELoss(), old_cfg, cpu)
            else:
                train_epoch(model, loader, opt, cfg, cpu)
            trained.append(model.state_dict())
        same_train = all(torch.equal(v, trained[1][rename(k)]) for k, v in trained[0].items())
        moved = any(not torch.equal(v, start[k]) for k, v in trained[1].items())
        n = sum(p.numel() for p in new.parameters())
        ok = same_out and same_train and moved and n == params[op]
        all_ok &= ok
        print(
            f"{arm:18s} {op:15s} params {n:>10,} (published {params[op]:>10,}) | "
            f"predictions identical {same_out} | weights after 1 epoch identical {same_train}"
        )
print("all checks passed" if all_ok else "CHECK FAILED")
sys.exit(0 if all_ok else 1)
