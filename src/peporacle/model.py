"""Atomic attention: a transformer over all atoms of a peptide.

Every atom attends to every other atom in the same peptide. The model reads only
the node features; bonds and residue order reach it through what those features
encode, not through the graph structure.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from peporacle.config import ModelConfig


class AttentionBlock(nn.Module):
    """Post-norm transformer block on padded atoms."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        dense, mask = to_dense_batch(x, batch)
        out, _ = self.attn(dense, dense, dense, key_padding_mask=~mask)
        dense = self.norm1(dense + out)
        dense = self.norm2(dense + self.ffn(dense))
        return dense[mask]


class AtomicAttention(nn.Module):
    """Atom embedding, ``n_blocks`` attention blocks, mean+max pooling, MLP head."""

    def __init__(self, node_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.input_proj = nn.Linear(node_dim, d)
        self.blocks = nn.ModuleList(
            AttentionBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_blocks)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        h = self.input_proj(data.x)
        for block in self.blocks:
            h = block(h, data.batch)
        pooled = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch)], -1)
        return self.head(pooled).squeeze(-1)
