from __future__ import annotations

import torch
import torch.nn as nn


class WorldTransitionHead(nn.Module):
    """Small action-conditioned latent dynamics model for controlled world-model tests."""

    def __init__(self, d_model: int, action_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError("state/action leading dimensions must match")
        return self.net(torch.cat((state, action), dim=-1))
