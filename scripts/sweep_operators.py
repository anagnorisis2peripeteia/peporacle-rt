"""Step 4: train the operator models on each dataset and seed with the frozen settings.

    python scripts/sweep_operators.py {operator_ablation,operator_attention} [DATASET ...]
        [--operators NNConv ...] [--seeds 0 1 2]

operator_ablation: 15 operator layers, no attention. operator_attention: the same layers
followed by 5 atomic-attention blocks (configs/operators.toml). Writes
results/sweep_<arm>/<operator>_<dataset>_seed<k>.json; finished runs are skipped and an
interrupted run resumes from its last epoch.
"""

import argparse
import logging
from pathlib import Path

from peporacle.config import load_datasets, load_frozen, load_operator_settings
from peporacle.operators import OperatorModel
from peporacle.sweep import sweep
from peporacle.train import select_device

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

settings = load_operator_settings(ROOT / "configs" / "operators.toml")
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("arm", choices=sorted(settings.attention_blocks))
parser.add_argument("datasets", nargs="*", help="default: all nine")
parser.add_argument(
    "--operators", nargs="+", choices=settings.operators, default=settings.operators
)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--gpu-memory-fraction", type=float)
args = parser.parse_args()

model_cfg, base = load_frozen(ROOT / "configs" / "frozen.toml")
n_attention = settings.attention_blocks[args.arm]
device = select_device(args.gpu_memory_fraction)
for operator in args.operators:
    for dataset in args.datasets or list(load_datasets(ROOT / "configs" / "datasets.toml")):
        sweep(
            ROOT / "caches",
            datasets=[dataset],
            seeds=args.seeds,
            build_model=lambda cache, op=operator: OperatorModel(
                op,
                cache.node_dim,
                cache.edge_dim,
                model_cfg,
                n_layers=settings.n_layers,
                n_attention_blocks=n_attention,
            ),
            model_config={
                "operator": operator,
                "hidden_dim": model_cfg.hidden_dim,
                "n_heads": model_cfg.n_heads,
                "dropout": model_cfg.dropout,
                "n_layers": settings.n_layers,
                "n_attention_blocks": n_attention,
            },
            train_cfg=settings.train_config(base, operator, dataset),
            out_dir=ROOT / "results" / f"sweep_{args.arm}",
            device=device,
            model=operator,
        )
