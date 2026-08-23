from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig, StackedChunkExpertBank
from .aera_hardware_core_v3 import (
    HardwareAwareAERATextLMV3,
    PredictiveStreamAERAStage,
)


class TrulySparseStackedChunkExpertBank(StackedChunkExpertBank):
    """Stacked expert bank whose hard top-1 decision actually skips expert #2.

    Soft training intentionally reuses the differentiable top-2 reference path.
    Hard inference executes the first selected expert for every chunk, then groups
    only chunks that requested two experts for the second batched GEMM.  This makes
    expert-count routing a real compute decision rather than a zero-valued mask over
    already-executed work.
    """

    def forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        count_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if not hard:
            return super().forward(x, expert_logits, count_logits, hard=False)

        b, t, d = x.shape
        if expert_logits.shape != (b, self.n_experts):
            raise ValueError("expert_logits must be [batch,n_experts]")
        if count_logits.shape != (b, 2):
            raise ValueError("count_logits must be [batch,2]")

        route_probs = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
        selected_probs, idx = torch.topk(route_probs, self.max_active, dim=-1)
        chosen_count = count_logits.argmax(dim=-1) + 1

        # First selected expert runs for every chunk.
        first_idx = idx[:, 0]
        first_w1 = self.w1[first_idx]
        first_w2 = self.w2[first_idx]
        first_hidden = torch.einsum("btd,bhd->bth", x, first_w1)
        first_hidden = F.gelu(first_hidden)
        first_out = torch.einsum("bth,bdh->btd", first_hidden, first_w2)

        if self.max_active == 1:
            out = first_out
        else:
            p1 = selected_probs[:, 0]
            p2 = selected_probs[:, 1]
            denom = (p1 + p2).clamp_min(1e-6)
            use_second = chosen_count >= 2
            first_weight = torch.where(use_second, p1 / denom, torch.ones_like(p1))
            out = first_out * first_weight[:, None, None]

            # Only chunks choosing top-2 are gathered for the second expert GEMMs.
            second_batch = use_second.nonzero(as_tuple=False).squeeze(-1)
            if second_batch.numel() > 0:
                x2 = x.index_select(0, second_batch)
                second_idx = idx[second_batch, 1]
                second_w1 = self.w1[second_idx]
                second_w2 = self.w2[second_idx]
                second_hidden = torch.einsum("btd,bhd->bth", x2, second_w1)
                second_hidden = F.gelu(second_hidden)
                second_out = torch.einsum("bth,bdh->btd", second_hidden, second_w2)
                second_weight = (p2[second_batch] / denom[second_batch])[:, None, None]
                out = out.index_add(
                    0,
                    second_batch,
                    (second_out * second_weight).to(out.dtype),
                )

        self.last_counts = chosen_count.detach().cpu()
        self.last_route_probs = route_probs.detach().float().mean(dim=0).cpu()
        return out


class HardSparsePredictiveStreamStage(PredictiveStreamAERAStage):
    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__(cfg)
        self.experts = TrulySparseStackedChunkExpertBank(cfg)


class HardwareAwareAERATextLMV4(HardwareAwareAERATextLMV3):
    """Pre-100M AERA core with predictive state and truly sparse hard experts."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            HardSparsePredictiveStreamStage(cfg) for _ in range(cfg.n_stages)
        )
