from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig, HardwareAwareAERATextLM
from .aera_hardware_core_v9 import HardwareAwareAERATextLMV9


class TiedStreamForecastProjector(nn.Module):
    """Predict several future latent token states, then decode with tied embeddings.

    Earlier AERA revisions used an independent d_model -> K*vocab projection for
    every stage. That made the auxiliary predictive-state objective absurdly
    parameter-heavy at realistic vocabularies. This head instead learns only
    d_model -> K*d_model and reuses the model's tied token embedding as the decoder.
    It preserves the predictive compression objective while making its train-only
    parameter cost independent of vocabulary size.
    """

    def __init__(self, d_model: int, forecast_tokens: int):
        super().__init__()
        self.d_model = d_model
        self.forecast_tokens = forecast_tokens
        self.proj = nn.Linear(d_model, forecast_tokens * d_model, bias=False)

    def latent(self, stream: torch.Tensor) -> torch.Tensor:
        return self.proj(stream).view(
            stream.size(0), self.forecast_tokens, self.d_model
        )


class HardwareAwareAERATextLMV10(HardwareAwareAERATextLMV9):
    """AERA pre-scale candidate with parameter-efficient predictive state.

    Runtime architecture is unchanged from v9. The only change is the train-time
    auxiliary decoder used to teach recurrent stream compression: per-stage giant
    vocabulary heads are replaced by small latent projectors decoded through the
    shared tied token embedding. This substantially reduces stored/training params
    without removing the state-forecast learning signal.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        # v3-v9 constructed large per-stage vocabulary forecast heads. Remove them
        # from the final parameter graph and replace with vocabulary-independent
        # latent projectors. (A later frozen implementation can avoid their transient
        # construction entirely; they are not retained or optimized in v10.)
        del self.stream_forecast_heads
        self.stream_forecast_projectors = nn.ModuleList(
            TiedStreamForecastProjector(cfg.d_model, stream_forecast_tokens)
            for _ in range(cfg.n_stages)
        )

    def _stream_forecast_loss(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
    ) -> torch.Tensor:
        history = output.get("stream_history")
        if not isinstance(history, list):
            raise ValueError("v10 output missing stream_history")
        terms: list[torch.Tensor] = []
        for chunk_index, stage_streams in enumerate(history[:-1]):
            next_start = (chunk_index + 1) * self.cfg.chunk_size
            remaining = tokens.size(1) - next_start
            k = min(self.stream_forecast_tokens, remaining)
            if k <= 0:
                continue
            target = tokens[:, next_start : next_start + k]
            for stage_index, stream in enumerate(stage_streams):
                latent = self.stream_forecast_projectors[stage_index].latent(stream)[:, :k]
                # Reuse the same token geometry as the LM head; no independent
                # K*vocab matrix is stored.
                pred = F.linear(latent, self.token_emb.weight)
                terms.append(
                    F.cross_entropy(
                        pred.float().reshape(-1, self.cfg.vocab_size),
                        target.reshape(-1),
                    )
                )
        return (
            torch.stack(terms).mean()
            if terms
            else torch.zeros((), device=tokens.device, dtype=torch.float32)
        )

    def soft_objective(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
        *,
        stage_compute_weight: float = 0.002,
        event_weight: float = 0.05,
        compute_weight: float = 0.002,
        balance_weight: float = 0.02,
        block_weight: float = 0.25,
        stream_forecast_weight: float = 0.20,
    ) -> dict[str, torch.Tensor]:
        if output.get("routing_mode") == "hard_sparse":
            raise ValueError("use hard_sparse_task_loss for hard_sparse training steps")

        routes = output["stage_routes"]
        assert isinstance(routes, list)
        base_controls: list[list[dict[str, dict[str, torch.Tensor]]]] = []
        stage_probs: list[torch.Tensor] = []
        for chunk in routes:
            base_chunk = []
            for item in chunk:
                start = item["start"]
                end = item["end"]
                assert isinstance(start, dict) and isinstance(end, dict)
                base_chunk.append({"start": start, "end": end})
                stage_probs.append(item["stage_route_probability"].float().mean())
            base_controls.append(base_chunk)

        base_output = dict(output)
        base_output["controls"] = base_controls
        # Call the pre-predictive base objective explicitly so the removed legacy
        # vocabulary forecast heads are never referenced.
        terms = HardwareAwareAERATextLM.objective(
            self,
            tokens,
            base_output,
            event_weight=event_weight,
            compute_weight=compute_weight,
            balance_weight=balance_weight,
            block_weight=block_weight,
        )
        forecast = self._stream_forecast_loss(tokens, output)
        stage_compute = torch.stack(stage_probs).mean()
        total = (
            terms["total"]
            + stream_forecast_weight * forecast
            + stage_compute_weight * stage_compute
        )
        return {
            **terms,
            "stream_forecast": forecast,
            "stage_compute": stage_compute,
            "total": total,
        }

    def predictive_head_accounting(self) -> dict[str, int | float]:
        new_params = sum(p.numel() for p in self.stream_forecast_projectors.parameters())
        legacy_equivalent = (
            self.cfg.n_stages
            * self.cfg.d_model
            * self.stream_forecast_tokens
            * self.cfg.vocab_size
        )
        return {
            "tied_forecast_parameters": new_params,
            "legacy_equivalent_parameters": legacy_equivalent,
            "parameter_reduction": legacy_equivalent - new_params,
            "fraction_of_legacy": new_params / legacy_equivalent,
        }
