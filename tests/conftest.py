import pytest
import torch
from torch_geometric.data import Data


def random_graph(n_atoms: int, node_dim: int = 88, edge_dim: int = 12, n_residues: int = 5) -> Data:
    src = torch.arange(n_atoms - 1)
    edge_index = torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], 1)
    return Data(
        x=torch.randn(n_atoms, node_dim),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.shape[1], edge_dim),
        residue_index=torch.arange(n_atoms) % n_residues,
        num_residues=n_residues,
    )


@pytest.fixture
def graphs() -> list[Data]:
    torch.manual_seed(0)
    return [random_graph(n) for n in (12, 40, 25, 60, 33, 18, 51, 29)]
