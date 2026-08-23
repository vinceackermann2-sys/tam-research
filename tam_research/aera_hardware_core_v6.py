from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v4 import TrulySparseStackedChunkExpertBank
from .aera_hardware_core_v5 import (
    HardwareAwareAERATextLMV5,
    MixedPrecisionSafeAERAStage,
)


class BMMHardSparseExpertBank(TrulySparseStackedChunkExpertBank):
    """Hard top-1/top-2 expert execution using contiguous batched GEMMs.

    The prior hard path used routed einsums after advanced indexing. On GPU that
    produced poor kernels despite doing fewer FLOPs than the dense all-expert
    control. This version gathers selected weights into contiguous [B,H,D]/[B,D,H]
    tensors and uses torch.bmm, which maps the selected-expert operation directly
    to strided batched GEMMs. The second expert is still executed only for chunks
    whose controller chose top-2.
    """

    def _run_selected(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        w1 = self.w1.index_select(0, expert_ids).contiguous()
        w2 = self.w2.index_select(0, expert_ids).contiguous()
        hidden = torch.bmm(x, w1.transpose(1, 2))
        hidden = F.gelu(hidden)
        return torch.bmm(hidden, w2.transpose(1, 2))

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

        b, _, _ = x.shape
        if expert_logits.shape != (b, self.n_experts):
            raise ValueError("expert_logits must be [batch,n_experts]")
        if count_logits.shape != (b, 2):
            raise ValueError("count_logits must be [batch,2]")

        route_probs = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
        selected_probs, idx = torch.topk(route_probs, self.max_active, dim=-1)
        chosen_count = count_logits.argmax(dim=-1) + 1

        first_out = self._run_selected(x, idx[:, 0])
        if self.max_active == 1:
            out = first_out
        else:
            p1, p2 = selected_probs[:, 0], selected_probs[:, 1]
            denom = (p1 + p2).clamp_min(1e-6)
            use_second = chosen_count >= 2
            first_weight = torch.where(use_second, p1 / denom, torch.ones_like(p1))
            out = first_out * first_weight[:, None, None]

            second_batch = use_second.nonzero(as_tuple=False).squeeze(-1)
            if second_batch.numel() > 0:
                x2 = x.index_select(0, second_batch).contiguous()
                second_ids = idx.index_select(0, second_batch)[:, 1]
                second_out = self._run_selected(x2, second_ids)
                second_weight = (p2.index_select(0, second_batch) / denom.index_select(0, second_batch))[:, None, None]
                out = out.index_add(0, second_batch, (second_out * second_weight).to(out.dtype))

        self.last_counts = chosen_count.detach().cpu()
        self.last_route_probs = route_probs.detach().float().mean(dim=0).cpu()
        return out


class BMMHardSparseAERAStage(MixedPrecisionSafeAERAStage):
    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__(cfg)
        self.experts = BMMHardSparseExpertBank(cfg)


class HardwareAwareAERATextLMV6(HardwareAwareAERATextLMV5):
    """Pre-scale AERA candidate with BF16-safe depth and BMM sparse experts."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(BMMHardSparseAERAStage(cfg) for _ in range(cfg.n_stages))
