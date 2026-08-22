from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import FastMemoryState


class DeltaFastMemory(nn.Module):
    """Session-local delta-rule associative memory.

    The update is local to the memory state and does not mutate base parameters:

        prediction = k @ M
        error      = target - prediction
        M          = decay*M + eta*strength*outer(k, error)

    Sequential writes let repeated keys replace stale values while approximately
    preserving mappings for orthogonal/independent keys. This is still a bounded
    memory, so interference must be measured rather than hand-waved away.
    """

    def __init__(
        self,
        d_model: int,
        memory_dim: int,
        *,
        lr: float = 0.2,
        decay: float = 0.999,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.lr = lr
        self.decay = decay
        self.q = nn.Linear(d_model, memory_dim, bias=False)
        self.k = nn.Linear(d_model, memory_dim, bias=False)
        self.v = nn.Linear(d_model, memory_dim, bias=False)
        self.out = nn.Linear(memory_dim, d_model, bias=False)

    def empty_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> FastMemoryState:
        return FastMemoryState(
            torch.zeros(batch_size, self.memory_dim, self.memory_dim, device=device, dtype=dtype)
        )

    def read(self, x: torch.Tensor, state: FastMemoryState) -> torch.Tensor:
        query = F.normalize(self.q(x), dim=-1)
        recalled = torch.einsum("btd,bdm->btm", query, state.matrix)
        return self.out(recalled)

    @torch.no_grad()
    def local_update(
        self,
        x: torch.Tensor,
        write_strength: torch.Tensor,
        state: FastMemoryState,
    ) -> FastMemoryState:
        if write_strength.shape != (*x.shape[:-1], 1):
            raise ValueError("write_strength must be [batch,time,1]")
        keys = F.normalize(self.k(x.detach()), dim=-1)
        targets = torch.tanh(self.v(x.detach()))
        strength = write_strength.detach().clamp(0.0, 1.0)
        matrix = state.matrix.detach().clone()

        for t in range(x.size(1)):
            matrix = self.decay * matrix
            key = keys[:, t]
            target = targets[:, t]
            pred = torch.einsum("bi,bij->bj", key, matrix)
            error = target - pred
            eta = self.lr * strength[:, t]
            matrix = matrix + torch.einsum("bi,bj->bij", key * eta, error)
        return FastMemoryState(matrix.detach())
