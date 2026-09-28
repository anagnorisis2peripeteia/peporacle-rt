import torch
from torch_geometric.data import Batch

from peporacle.config import ModelConfig
from peporacle.model import AtomicAttention

CFG = ModelConfig(hidden_dim=32, n_blocks=2, n_heads=4, dropout=0.1)


def model() -> AtomicAttention:
    torch.manual_seed(0)
    return AtomicAttention(88, CFG).eval()


def test_one_prediction_per_graph(graphs):
    assert model()(Batch.from_data_list(graphs)).shape == (len(graphs),)


@torch.no_grad()
def test_prediction_does_not_depend_on_the_rest_of_the_batch(graphs):
    m = model()
    together = m(Batch.from_data_list(graphs))
    alone = torch.cat([m(Batch.from_data_list([g])) for g in graphs])
    assert torch.allclose(together, alone, atol=1e-5)


@torch.no_grad()
def test_atom_order_and_bonds_are_ignored(graphs):
    """Atomic attention sees a peptide as a set of atoms: no bonds, no atom order."""
    m, g = model(), graphs[3]
    shuffled = g.clone()
    shuffled.x = g.x[torch.randperm(g.num_nodes)]
    shuffled.edge_index = shuffled.edge_index[:, :0]
    assert torch.allclose(
        m(Batch.from_data_list([g])), m(Batch.from_data_list([shuffled])), atol=1e-5
    )
