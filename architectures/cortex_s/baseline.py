from __future__ import annotations

import torch
from torch import nn


class TransformerStateBaseline(nn.Module):
    """Causal Transformer baseline for the CPU state-machine primitive.

    It receives exactly the same operation/value sequence and per-step targets as
    CortexSCore. The default size is within ~0.2% of the CORTEX-S tiny config.
    """

    def __init__(
        self,
        *,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 136,
        num_ops: int = 3,
        num_values: int = 16,
        max_length: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.op_embedding = nn.Embedding(num_ops, d_model)
        self.value_embedding = nn.Embedding(num_values, d_model)
        self.input_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.state_head = nn.Linear(d_model, num_values)

    def forward(self, ops: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if ops.shape != values.shape or ops.ndim != 2:
            raise ValueError("ops and values must both be [batch, sequence]")
        _, length = ops.shape
        positions = torch.arange(length, device=ops.device)
        x = self.input_projection(
            torch.cat([self.op_embedding(ops), self.value_embedding(values)], dim=-1)
        )
        x = x + self.position_embedding(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.ones(length, length, device=ops.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.encoder(x, mask=causal_mask)
        return self.state_head(self.output_norm(hidden))
