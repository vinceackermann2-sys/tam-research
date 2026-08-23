from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v11 import HardwareAwareAERATextLMV11


class HardwareAwareAERATextLMV12(HardwareAwareAERATextLMV11):
    """AERA-v12: explicit difficulty-supervised whole-stage budgeting.

    v11 fixed the initialization/chunk-geometry confounds, but its learned stage
    probabilities moved *up* during calibration and hard inference still executed
    every stage.  v12 preserves the v11 runtime core and changes only how the stage
    routers are constrained/trained:

    * stage 0 is a permanently available foundation stage;
    * stages 1..N-1 remain optional and are the only learned whole-stage gates;
    * dense/calibration passes may supervise those optional gates from a detached
      per-chunk language-loss difficulty target;
    * the target allocates progressively fewer optional stages to easier chunks;
    * an explicit budget term keeps mean optional run probability aligned with the
      target instead of relying on a tiny uncalibrated linear compute coefficient;
    * a small polarization term discourages all probabilities from hovering around
      the 0.5 hard-routing threshold.

    The difficulty target is training supervision only.  At inference the router
    still receives only its causal first-event + carried-stream inputs, so future
    chunk tokens are never an inference-time feature.
    """

    # With four stages this gives expected optional use 1.0 stage per chunk:
    # 0.50 + 1/3 + 1/6 = 1.0.  Together with the mandatory foundation stage,
    # the calibration target is therefore 2/4 = 50% mean stage execution, matching
    # the previously validated forced half-stage systems geometry.
    OPTIONAL_STAGE_RUN_RATES: tuple[float, ...] = (0.50, 1.0 / 3.0, 1.0 / 6.0)
    FOUNDATION_STAGE = 0

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        if cfg.n_stages < 2:
            raise ValueError("AERA-v12 requires at least two stages")
        if cfg.n_stages - 1 != len(self.OPTIONAL_STAGE_RUN_RATES):
            raise ValueError(
                "AERA-v12 reference schedule currently requires four total stages"
            )

        # Stage 0 is deliberately not a learned skip decision.  A large positive
        # frozen bias makes both calibration and hard inference execute it, avoiding
        # the degenerate all-stages-skipped path while leaving parameter count intact.
        foundation = self.stage_routers[self.FOUNDATION_STAGE]
        with torch.no_grad():
            foundation.proj.weight.zero_()
            foundation.proj.bias.fill_(12.0)
        for parameter in foundation.parameters():
            parameter.requires_grad_(False)

    def set_optional_stage_routers_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze only optional stage routers; stage 0 stays fixed."""
        for stage_index, router in enumerate(self.stage_routers):
            enabled = bool(trainable and stage_index != self.FOUNDATION_STAGE)
            for parameter in router.parameters():
                parameter.requires_grad_(enabled)

    @staticmethod
    def _rank_hardness(losses: torch.Tensor) -> torch.Tensor:
        """Convert detached per-example losses to deterministic [0,1] hardness ranks."""
        if losses.ndim != 1 or losses.numel() < 1:
            raise ValueError("losses must be a non-empty [batch] tensor")
        values = losses.detach().float()
        order = torch.argsort(values, stable=True)
        ranks = torch.empty_like(values)
        ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
        return (ranks + 0.5) / float(values.numel())

    @classmethod
    def chunk_difficulty_stage_targets(cls, chunk_losses: torch.Tensor) -> torch.Tensor:
        """Return binary optional-stage targets matching route-history order.

        Args:
            chunk_losses: [batch, chunks] detached or detach-able language losses.

        Returns:
            [batch, chunks * optional_stages] binary targets.  For each chunk, the
            hardest ~50%, ~33%, and ~17% of examples request optional stages 1, 2,
            and 3 respectively.  Small batches use deterministic rank quantization.
        """
        if chunk_losses.ndim != 2 or chunk_losses.size(0) < 1 or chunk_losses.size(1) < 1:
            raise ValueError("chunk_losses must be non-empty [batch,chunks]")
        per_chunk: list[torch.Tensor] = []
        for chunk_index in range(chunk_losses.size(1)):
            hardness = cls._rank_hardness(chunk_losses[:, chunk_index])
            stage_targets = []
            for run_rate in cls.OPTIONAL_STAGE_RUN_RATES:
                threshold = 1.0 - float(run_rate)
                stage_targets.append((hardness >= threshold).to(chunk_losses.dtype))
            per_chunk.append(torch.stack(stage_targets, dim=1))
        return torch.cat(per_chunk, dim=1).detach()

    def optional_stage_probabilities(self, output: dict[str, object]) -> torch.Tensor:
        """Collect optional stage run probabilities as [batch, chunks*(stages-1)]."""
        routes = output.get("stage_routes")
        if not isinstance(routes, list) or not routes:
            raise ValueError("output missing stage_routes")
        collected: list[torch.Tensor] = []
        batch_size: int | None = None
        for chunk in routes:
            if not isinstance(chunk, list) or len(chunk) != self.cfg.n_stages:
                raise ValueError("invalid stage route history")
            for stage_index, item in enumerate(chunk):
                if stage_index == self.FOUNDATION_STAGE:
                    continue
                probability = item.get("stage_route_probability") if isinstance(item, dict) else None
                if not isinstance(probability, torch.Tensor) or probability.ndim != 2 or probability.size(1) != 1:
                    raise ValueError("invalid stage_route_probability")
                if batch_size is None:
                    batch_size = probability.size(0)
                elif probability.size(0) != batch_size:
                    raise ValueError("stage route batch mismatch")
                collected.append(probability.float())
        if not collected:
            raise ValueError("no optional stage probabilities found")
        return torch.cat(collected, dim=1)

    def routing_supervision(
        self,
        output: dict[str, object],
        chunk_losses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute explicit adaptive-stage supervision without touching base task loss."""
        probabilities = self.optional_stage_probabilities(output)
        targets = self.chunk_difficulty_stage_targets(chunk_losses).to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        if targets.shape != probabilities.shape:
            raise ValueError(
                f"target/probability shape mismatch: {targets.shape} vs {probabilities.shape}"
            )
        eps = torch.finfo(probabilities.dtype).eps
        p = probabilities.clamp(eps, 1.0 - eps)
        difficulty_bce = F.binary_cross_entropy(p, targets)
        mean_probability = probabilities.mean()
        mean_target = targets.mean()
        budget = (mean_probability - mean_target).square()
        polarization = (probabilities * (1.0 - probabilities)).mean()
        return {
            "stage_difficulty_bce": difficulty_bce,
            "stage_budget": budget,
            "stage_polarization": polarization,
            "optional_stage_mean_probability": mean_probability,
            "optional_stage_target_fraction": mean_target,
        }

    def soft_objective(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
        *,
        chunk_losses: torch.Tensor | None = None,
        stage_difficulty_weight: float = 0.10,
        stage_budget_weight: float = 0.05,
        stage_polarization_weight: float = 0.01,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Disable v8/v11's weak linear mean-probability penalty.  v12 replaces it
        # with an explicit target-budget constraint and difficulty supervision.
        kwargs.pop("stage_compute_weight", None)
        terms = super().soft_objective(
            tokens,
            output,
            stage_compute_weight=0.0,
            **kwargs,
        )
        if chunk_losses is None:
            return terms
        routing = self.routing_supervision(output, chunk_losses)
        total = (
            terms["total"]
            + stage_difficulty_weight * routing["stage_difficulty_bce"]
            + stage_budget_weight * routing["stage_budget"]
            + stage_polarization_weight * routing["stage_polarization"]
        )
        return {**terms, **routing, "total": total}
