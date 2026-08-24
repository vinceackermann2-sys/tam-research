from __future__ import annotations

import torch
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v16 import HardwareAwareAERATextLMV16


class HardwareAwareAERATextLMV17(HardwareAwareAERATextLMV16):
    """AERA-v17: separate difficulty ordering from hard compute prevalence.

    V16 aligned the budget term with deployment hard execution and revived the
    middle optional stage, but seed8311 still left the deepest stage at 0% hard
    execution. The remaining teacher is batch-relative and low-resolution:
    `MICRO_BATCH=8` turns the nominal 50% / 1/3 / 1/6 targets into 4/8, 3/8,
    and 1/8 binary labels per chunk. The deepest router therefore learns from a
    noisy "hardest one in this batch" classification target.

    V17 keeps architecture, inference, target rates, hard threshold, task-gradient
    isolation, data, experts, memory, and parameter count unchanged. It separates
    two jobs that were previously entangled:

    * difficulty: every optional-stage router learns to rank harder chunks above
      easier chunks using all pairwise comparisons inside each chunk-position batch;
    * prevalence: V16's straight-through hard-execution budget is retained, but is
      matched directly to the exact nominal 0.50 / 1/3 / 1/6 rates rather than to
      quantized binary-label means.

    The pairwise teacher is training-only. Inference remains the same causal router
    using first-event + carried-stream state and the unchanged p>=0.5 decision.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)

    @staticmethod
    def pairwise_difficulty_ranking_loss(
        routing_logits: torch.Tensor,
        chunk_losses: torch.Tensor,
    ) -> torch.Tensor:
        """Logistic pairwise ranking loss, independently within each chunk position.

        Args:
            routing_logits: [batch, chunks, optional_stages]
            chunk_losses: [batch, chunks], detached or detach-able language losses
        """
        if routing_logits.ndim != 3:
            raise ValueError("routing_logits must be [batch,chunks,optional_stages]")
        if chunk_losses.ndim != 2 or routing_logits.shape[:2] != chunk_losses.shape:
            raise ValueError("chunk loss shape must match routing batch/chunk axes")
        if routing_logits.size(0) < 2:
            raise ValueError("pairwise ranking requires at least two examples")

        difficulty = chunk_losses.detach().float()
        pieces: list[torch.Tensor] = []
        for chunk_index in range(routing_logits.size(1)):
            d = difficulty[:, chunk_index]
            harder = d[:, None] > d[None, :]
            if not bool(harder.any()):
                continue
            for stage_index in range(routing_logits.size(2)):
                scores = routing_logits[:, chunk_index, stage_index].float()
                score_delta = scores[:, None] - scores[None, :]
                # If i is harder than j, score_i should exceed score_j.
                pieces.append(F.softplus(-score_delta[harder]).mean())
        if not pieces:
            return routing_logits.float().sum() * 0.0
        return torch.stack(pieces).mean()

    def routing_supervision(
        self,
        output: dict[str, object],
        chunk_losses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        probabilities = self.optional_stage_probabilities(output).float()
        optional_stages = self.cfg.n_stages - 1
        if probabilities.size(1) % optional_stages:
            raise ValueError("optional stage probability history does not divide by stage count")
        chunks = probabilities.size(1) // optional_stages
        if chunk_losses.shape != (probabilities.size(0), chunks):
            raise ValueError("chunk_losses shape does not match routing history")

        p = probabilities.reshape(probabilities.size(0), chunks, optional_stages)
        eps = 1e-6
        routing_logits = torch.logit(p.clamp(eps, 1.0 - eps))
        rank_loss = self.pairwise_difficulty_ranking_loss(routing_logits, chunk_losses)

        hard = (p >= 0.5).to(p.dtype)
        straight_through_hard = hard.detach() - p.detach() + p
        per_stage_hard_fraction_st = straight_through_hard.mean(dim=(0, 1))
        per_stage_hard_fraction = hard.mean(dim=(0, 1)).detach()
        exact_target_rates = torch.tensor(
            self.OPTIONAL_STAGE_RUN_RATES,
            device=p.device,
            dtype=torch.float32,
        )
        per_stage_hard_budget_error = (
            per_stage_hard_fraction_st - exact_target_rates
        ).square()
        polarization = (p * (1.0 - p)).mean()

        # `soft_objective` in the inherited v12 path consumes the historical
        # `stage_difficulty_bce` scalar. Keep that compatibility key while making
        # its semantics explicit through `stage_difficulty_rank`.
        return {
            "stage_difficulty_bce": rank_loss,
            "stage_difficulty_rank": rank_loss,
            "stage_budget": per_stage_hard_budget_error.mean(),
            "stage_polarization": polarization,
            "optional_stage_hard_run_fractions": per_stage_hard_fraction,
            "optional_stage_hard_run_fractions_st": per_stage_hard_fraction_st,
            "optional_stage_target_fractions": exact_target_rates,
            "optional_stage_budget_errors": per_stage_hard_budget_error,
        }
