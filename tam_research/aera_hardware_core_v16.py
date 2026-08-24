from __future__ import annotations

import torch

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v15 import HardwareAwareAERATextLMV15


class HardwareAwareAERATextLMV16(HardwareAwareAERATextLMV15):
    """AERA-v16: align router budget training with hard inference execution.

    V15 constrains each optional stage's mean sigmoid probability to the existing
    target prevalence. Hard inference, however, executes a stage only when p>=0.5.
    For rare target rates (1/3 and 1/6), a calibrated router can satisfy the soft
    probability budget while every score remains below 0.5, producing zero hard
    executions. V15 seed8291 exhibited exactly this failure.

    V16 preserves architecture, target rates, inference threshold, data, parameter
    count, and v14 task-gradient isolation. Only the budget surrogate changes: its
    forward value is the actual hard run/skip gate, while gradients flow through
    the sigmoid probability with a straight-through estimator.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)

    def routing_supervision(
        self,
        output: dict[str, object],
        chunk_losses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Reuse V15 for difficulty BCE, polarization and diagnostics, then replace
        # only the stage-budget surrogate with deployment-aligned hard execution.
        terms = super().routing_supervision(output, chunk_losses)
        probabilities = self.optional_stage_probabilities(output).float()
        targets = self.chunk_difficulty_stage_targets(chunk_losses).to(
            device=probabilities.device,
            dtype=torch.float32,
        )
        optional_stages = self.cfg.n_stages - 1
        if probabilities.size(1) % optional_stages:
            raise ValueError("optional stage probability history does not divide by stage count")
        chunks = probabilities.size(1) // optional_stages
        p = probabilities.reshape(probabilities.size(0), chunks, optional_stages)
        t = targets.reshape(targets.size(0), chunks, optional_stages)

        hard = (p >= 0.5).to(p.dtype)
        straight_through_hard = hard.detach() - p.detach() + p
        per_stage_hard_fraction_st = straight_through_hard.mean(dim=(0, 1))
        per_stage_hard_fraction = hard.mean(dim=(0, 1)).detach()
        per_stage_target_fraction = t.mean(dim=(0, 1))
        per_stage_hard_budget_error = (
            per_stage_hard_fraction_st - per_stage_target_fraction
        ).square()

        return {
            **terms,
            "stage_budget": per_stage_hard_budget_error.mean(),
            "optional_stage_hard_run_fractions": per_stage_hard_fraction,
            "optional_stage_hard_run_fractions_st": per_stage_hard_fraction_st,
            "optional_stage_target_fractions": per_stage_target_fraction,
            "optional_stage_budget_errors": per_stage_hard_budget_error,
        }
