from __future__ import annotations

import torch

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v14 import HardwareAwareAERATextLMV14


class HardwareAwareAERATextLMV15(HardwareAwareAERATextLMV14):
    """AERA-v15: close the pooled-budget redistribution loophole.

    V14 proved held-out difficulty-dependent whole-stage routing, but the optional
    hierarchy collapsed to roughly 55.6% / 2.7% / 0.0% stage use. The v12-v14
    budget term constrained only the pooled mean over all optional stages. Since
    the intended optional run rates sum to one stage/chunk, a stage1-heavy policy
    can satisfy that pooled scalar while later stages die.

    V15 preserves the architecture, targets, task-gradient isolation, inference,
    data contract, and parameter count. It replaces only the pooled budget scalar
    with a per-stage budget match against the existing 0.50 / 1/3 / 1/6 targets.
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
        per_stage_mean_probability = p.mean(dim=(0, 1))
        per_stage_target_fraction = t.mean(dim=(0, 1))
        per_stage_budget_error = (per_stage_mean_probability - per_stage_target_fraction).square()

        # Preserve the scalar key consumed by the inherited soft objective, but make
        # it the mean of stage-specific errors rather than an error on the pooled mean.
        return {
            **terms,
            "stage_budget": per_stage_budget_error.mean(),
            "optional_stage_mean_probabilities": per_stage_mean_probability,
            "optional_stage_target_fractions": per_stage_target_fraction,
            "optional_stage_budget_errors": per_stage_budget_error,
        }
