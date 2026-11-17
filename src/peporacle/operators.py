"""Message-passing operators from PyTorch Geometric, alone or followed by atomic attention.

These are the comparison models in ``results/operator_ablation/`` (no attention blocks) and
``results/operator_attention/`` (attention blocks after the operator layers). Each operator
layer adds its output, after an activation and dropout, to its input and then applies layer
normalisation. The atom embedding, attention blocks, pooling and head are those of
:class:`peporacle.model.AtomicAttention`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import ARMAConv, GCNConv, NNConv, SAGEConv, TransformerConv

from peporacle.config import ModelConfig
from peporacle.model import AttentionBlock, predict_from_atoms, prediction_head

OPERATORS = ("TransformerConv", "GCNConv", "SAGEConv", "ARMAConv", "NNConv")
READS_BONDS = frozenset({"TransformerConv", "NNConv"})


def make_layer(operator: str, dim: int, edge_dim: int, cfg: ModelConfig) -> nn.Module:
    """One operator layer of width ``dim`` with the settings the published runs used."""
    if operator == "TransformerConv":
        return TransformerConv(
            dim,
            dim // cfg.n_heads,
            heads=cfg.n_heads,
            edge_dim=edge_dim,
            dropout=cfg.dropout,
            concat=True,
        )
    if operator == "GCNConv":
        return GCNConv(dim, dim, normalize=True)
    if operator == "SAGEConv":
        return SAGEConv(dim, dim, normalize=True)
    if operator == "ARMAConv":
        return ARMAConv(dim, dim, num_stacks=3, num_layers=7, dropout=cfg.dropout)
    if operator == "NNConv":
        return NNConv(dim, dim, nn.Linear(edge_dim, dim * dim), aggr="mean")
    raise ValueError(f"unknown operator {operator!r}; choose from {', '.join(OPERATORS)}")


class OperatorModel(nn.Module):
    """Atom embedding, ``n_layers`` operator layers, ``n_attention_blocks`` attention blocks,
    mean+max pooling, MLP head.

    Width, heads and dropout come from ``cfg``; ``cfg.n_blocks`` is not used.
    """

    def __init__(
        self,
        operator: str,
        node_dim: int,
        edge_dim: int,
        cfg: ModelConfig,
        *,
        n_layers: int,
        n_attention_blocks: int,
    ) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.reads_bonds = operator in READS_BONDS
        self.input_proj = nn.Linear(node_dim, d)
        self.convs = nn.ModuleList(make_layer(operator, d, edge_dim, cfg) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            AttentionBlock(d, cfg.n_heads, cfg.dropout) for _ in range(n_attention_blocks)
        )
        self.head = prediction_head(d, cfg.dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        bonds = (data.edge_index, data.edge_attr) if self.reads_bonds else (data.edge_index,)
        h = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = norm(h + self.drop(self.act(conv(h, *bonds))))
        for block in self.blocks:
            h = block(h, data.batch)
        return predict_from_atoms(self.head, h, data.batch)
