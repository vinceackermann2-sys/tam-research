from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import ChunkLatentReasoner, HardwareAERAConfig
from .aera_hardware_core_v4 import (
    HardSparsePredictiveStreamStage,
    HardwareAwareAERATextLMV4,
)


class DtypeSafeChunkLatentReasoner(ChunkLatentReasoner):
    """Adaptive latent reasoner with explicit dtype boundaries for autocast.

    GRUCell kernels may emit BF16 under CUDA autocast even when the running latent
    state is FP32 (for example after LayerNorm).  index_copy requires exact dtype
    equality, so the recurrent update is restored to the running state's dtype
    before sparse hard-depth accumulation.  The cast is differentiable and does
    not change the mathematical routing decision.
    """

    def forward(self, summary: torch.Tensor, depth_logits: torch.Tensor, *, hard: bool) -> torch.Tensor:
        if depth_logits.shape != (summary.size(0), self.max_steps):
            raise ValueError("depth_logits shape mismatch")
        probs = F.softmax(depth_logits.float(), dim=-1).to(summary.dtype)
        values = torch.arange(
            1,
            self.max_steps + 1,
            device=summary.device,
            dtype=summary.dtype,
        )
        self.last_expected = (probs * values[None]).sum(dim=-1).detach().cpu()

        if not hard:
            current = summary
            states = []
            for _ in range(self.max_steps):
                updated = self.cell(current, current)
                current = updated.to(dtype=current.dtype)
                states.append(current)
            stacked = torch.stack(states, dim=1)
            self.last_steps = None
            return (stacked * probs[:, :, None]).sum(dim=1)

        chosen = depth_logits.argmax(dim=-1) + 1
        current = summary
        for step in range(1, self.max_steps + 1):
            idx = (chosen >= step).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break
            selected = current.index_select(0, idx)
            updated = self.cell(selected, selected).to(dtype=current.dtype)
            current = current.index_copy(0, idx, updated)
        self.last_steps = chosen.detach().cpu()
        return current


class MixedPrecisionSafeAERAStage(HardSparsePredictiveStreamStage):
    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__(cfg)
        self.reasoner = DtypeSafeChunkLatentReasoner(cfg.d_model, cfg.max_reason_steps)


class HardwareAwareAERATextLMV5(HardwareAwareAERATextLMV4):
    """Frozen pre-scale AERA candidate with BF16-safe adaptive latent depth."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            MixedPrecisionSafeAERAStage(cfg) for _ in range(cfg.n_stages)
        )
