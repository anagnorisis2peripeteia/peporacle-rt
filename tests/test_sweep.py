import torch

from peporacle.cache import write_graphs
from peporacle.config import ModelConfig, TrainConfig
from peporacle.operators import OperatorModel
from peporacle.results import collect, summarise
from peporacle.sweep import sweep

CFG = ModelConfig(hidden_dim=16, n_blocks=0, n_heads=2, dropout=0.0)
TRAIN = TrainConfig(
    lr=1e-3, weight_decay=0.0, batch_size=8, grad_accum=2, grad_clip=5.0, warmup_epochs=0, epochs=1
)


def test_operator_sweep_writes_one_record_per_seed(tmp_path, graphs):
    many = graphs * 4
    labels = [float(g.x[:, 0].mean()) for g in many]
    write_graphs(tmp_path / "caches" / "tiny", many, labels, [f"S{i}" for i in range(len(many))])
    out = tmp_path / "results"
    sweep(
        tmp_path / "caches",
        datasets=["tiny"],
        seeds=[0, 1],
        build_model=lambda cache: OperatorModel(
            "GCNConv", cache.node_dim, cache.edge_dim, CFG, n_layers=1, n_attention_blocks=1
        ),
        model_config={"operator": "GCNConv"},
        train_cfg=TRAIN,
        out_dir=out,
        device=torch.device("cpu"),
        model="GCNConv",
    )
    assert sorted(p.name for p in out.glob("*.json")) == [
        "GCNConv_tiny_seed0.json",
        "GCNConv_tiny_seed1.json",
    ]
    rows = collect(out)
    assert {(r["model"], r["seed"], r["batch_size"], r["epochs"]) for r in rows} == {
        ("GCNConv", 0, 8, 1),
        ("GCNConv", 1, 8, 1),
    }
    (summary,) = summarise(rows)
    assert (summary["model"], summary["n_seeds"]) == ("GCNConv", 2)
