"""SPC-HGT MatchNet model skeleton."""
from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


class SPCHGTMatchNet(nn.Module):  # type: ignore[misc]
    """Skill-Path Contrastive Heterogeneous Graph Transformer.

    TODO: Implement HGT/HeteroConv encoder and matching head.
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self, batch: object) -> object:
        """Forward pass for candidate-job matching."""
        raise NotImplementedError
