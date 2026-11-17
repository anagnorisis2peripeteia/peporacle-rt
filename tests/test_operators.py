from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from peporacle.config import ModelConfig, load_frozen, load_operator_settings
from peporacle.operators import OPERATORS, READS_BONDS, OperatorModel

ROOT = Path(__file__).resolve().parents[1]
CFG = ModelConfig(hidden_dim=32, n_blocks=0, n_heads=4, dropout=0.1)


def model(operator: str, n_attention: int = 1) -> OperatorModel:
    torch.manual_seed(0)
    return OperatorModel(operator, 88, 12, CFG, n_layers=2, n_attention_blocks=n_attention).eval()


@pytest.mark.parametrize("operator", OPERATORS)
def test_one_prediction_per_graph(operator, graphs):
    assert model(operator)(Batch.from_data_list(graphs)).shape == (len(graphs),)


@pytest.mark.parametrize("operator", OPERATORS)
@torch.no_grad()
def test_bonds_matter_and_bond_features_only_where_read(operator, graphs):
    m, g = model(operator), graphs[3]
    unbonded, recoloured = g.clone(), g.clone()
    unbonded.edge_index, unbonded.edge_attr = g.edge_index[:, :0], g.edge_attr[:0]
    recoloured.edge_attr = torch.randn_like(g.edge_attr)
    out = m(Batch.from_data_list([g]))
    assert not torch.allclose(out, m(Batch.from_data_list([unbonded])))
    same_bonds = torch.allclose(out, m(Batch.from_data_list([recoloured])))
    assert same_bonds == (operator not in READS_BONDS)


def test_published_sizes():
    """Parameter counts at the published settings (88 atom and 12 bond features)."""
    cfg, _ = load_frozen(ROOT / "configs" / "frozen.toml")
    settings = load_operator_settings(ROOT / "configs" / "operators.toml")
    published = {
        "operator_ablation": (1_047_553, 281_473, 527_233, 10_396_033, 3_476_353),
        "operator_attention": (2_038_913, 1_272_833, 1_518_593, 11_387_393, 4_467_713),
    }
    for arm, sizes in published.items():
        for op, size in zip(OPERATORS, sizes, strict=True):
            m = OperatorModel(
                op,
                88,
                12,
                cfg,
                n_layers=settings.n_layers,
                n_attention_blocks=settings.attention_blocks[arm],
            )
            assert sum(p.numel() for p in m.parameters()) == size, (arm, op)


def test_micro_batch_keeps_128_per_update():
    _, base = load_frozen(ROOT / "configs" / "frozen.toml")
    settings = load_operator_settings(ROOT / "configs" / "operators.toml")
    for op, dataset, micro in (
        ("NNConv", "scx", 8),
        ("NNConv", "misc_dia", 4),
        ("GCNConv", "scx", 32),
    ):
        cfg = settings.train_config(base, op, dataset)
        assert (cfg.batch_size, cfg.batch_size * cfg.grad_accum) == (micro, 128)
