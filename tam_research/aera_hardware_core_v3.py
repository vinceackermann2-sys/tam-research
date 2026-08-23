from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig, HardwareAERAState
from .aera_hardware_core_v2 import (
    HardwareAwareAERATextLMV2,
    UnconditionalStreamAERAStage,
)


class PredictiveStreamAERAStage(UnconditionalStreamAERAStage):
    """Hardware-aware stage with a direct observation path into recurrent state.

    The compressed stream should preserve what happened in the chunk, not only the
    result of latent reasoning over the chunk summary.  The state update therefore
    receives both the final causal chunk representation and the latent-reasoned
    summary.  The model-level v3 objective additionally trains this state to forecast
    the prefix of the next chunk.
    """

    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__(cfg)
        self.stream_input_norm = nn.LayerNorm(cfg.d_model)

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ):
        if events.ndim != 3 or events.size(1) > self.cfg.chunk_size:
            raise ValueError("events must be [batch,time,d_model] within chunk_size")
        if state is None:
            state = self.empty_state(events)

        h = self.norm(events)
        start_control = self.controller(h[:, 0], state.stream)
        self.last_start_controls = {
            k: v.detach() for k, v in start_control.items() if "logits" not in k
        }

        memory_read = self.memory.read(h[:, :1], state.memory).squeeze(1)
        carried = self.state_to_chunk(state.stream)
        # Stream state is the mandatory working-memory path. Fast neural memory is
        # optional and remains controller-gated.
        h = h + (carried + start_control["memory_read"] * memory_read)[:, None, :]

        h = h + self.attn(h)
        h = h + self.experts(
            h,
            start_control["expert_logits"],
            start_control["expert_count_logits"],
            hard=hard,
        )

        # Last event is a causal summary of the entire current chunk.
        end_summary = h[:, -1]
        end_control = self.controller(end_summary, state.stream)
        self.last_end_controls = {
            k: v.detach() for k, v in end_control.items() if "logits" not in k
        }
        reasoned = self.reasoner(end_summary, end_control["depth_logits"], hard=hard)

        # Latent reasoning may alter the boundary prediction and future state, but
        # never earlier logits in the already-causal chunk.
        last_mask = torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
        last_mask[-1] = 1
        h = h + self.reason_to_chunk(reasoned)[:, None, :] * last_mask[None, :, None]
        h = self.out_norm(h)

        # Preserve both observed chunk content and the derived latent lesson.
        # This is still one fused recurrent transition per chunk.
        stream_input = self.stream_input_norm(end_summary + reasoned)
        final_stream = self.stream_cell(stream_input, state.stream)

        memory_state = state.memory
        if update_memory:
            write = (end_control["novelty"] * end_control["memory_write"]).clamp(0.0, 1.0)
            memory_state = self.memory.local_update(
                reasoned[:, None, :], write[:, None, :], state.memory
            )

        return h, AERAState(final_stream, memory_state), {
            "start": start_control,
            "end": end_control,
        }


class HardwareAwareAERATextLMV3(HardwareAwareAERATextLMV2):
    """Canonical pre-scale AERA core with predictive compressed stream state.

    Besides ordinary next-token / block / compute objectives, every completed
    recurrent stream is trained to forecast a short prefix of the *next* chunk.
    The future prefix is a training target only: inference receives no future data,
    so this objective cannot introduce causal leakage.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg)
        if stream_forecast_tokens < 1:
            raise ValueError("stream_forecast_tokens must be >=1")
        self.stream_forecast_tokens = stream_forecast_tokens
        self.stages = nn.ModuleList(PredictiveStreamAERAStage(cfg) for _ in range(cfg.n_stages))
        self.stream_forecast_heads = nn.ModuleList(
            nn.Linear(
                cfg.d_model,
                stream_forecast_tokens * cfg.vocab_size,
                bias=False,
            )
            for _ in range(cfg.n_stages)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        state: HardwareAERAState | None = None,
        *,
        hard: bool = False,
        update_memory: bool = False,
        return_block_logits: bool = False,
    ) -> dict[str, object]:
        if tokens.ndim != 2 or tokens.size(1) < 1:
            raise ValueError("tokens must be nonempty [batch,time]")
        if state is None:
            state = self.empty_state(tokens)
        if len(state.stages) != len(self.stages):
            raise ValueError("state stage count mismatch")

        outputs: list[torch.Tensor] = []
        control_history: list[list[dict[str, dict[str, torch.Tensor]]]] = []
        stream_history: list[list[torch.Tensor]] = []
        current_state = state

        for start in range(0, tokens.size(1), self.cfg.chunk_size):
            chunk = tokens[:, start : start + self.cfg.chunk_size]
            pos = torch.arange(chunk.size(1), device=tokens.device)
            x = self.token_emb(chunk) + self.local_pos(pos)[None]
            new_states: list[AERAState] = []
            stage_controls: list[dict[str, dict[str, torch.Tensor]]] = []
            stage_streams: list[torch.Tensor] = []
            for stage, stage_state in zip(self.stages, current_state.stages):
                x, new_state, controls = stage.forward_chunk(
                    x,
                    stage_state,
                    hard=hard,
                    update_memory=update_memory,
                )
                new_states.append(new_state)
                stage_controls.append(controls)
                stage_streams.append(new_state.stream)
            outputs.append(x)
            control_history.append(stage_controls)
            stream_history.append(stage_streams)
            current_state = HardwareAERAState(new_states)

        hidden = self.norm(torch.cat(outputs, dim=1))
        result: dict[str, object] = {
            "logits": self.lm_head(hidden),
            "hidden": hidden,
            "state": current_state,
            "next_event_prediction": self.next_event(hidden),
            "controls": control_history,
            "stream_history": stream_history,
        }
        if return_block_logits:
            result["block_logits"] = self.block_draft(hidden, self.lm_head)
        return result

    def objective(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
        *,
        event_weight: float = 0.05,
        compute_weight: float = 0.002,
        balance_weight: float = 0.02,
        block_weight: float = 0.25,
        stream_forecast_weight: float = 0.20,
    ) -> dict[str, torch.Tensor]:
        base = super().objective(
            tokens,
            output,
            event_weight=event_weight,
            compute_weight=compute_weight,
            balance_weight=balance_weight,
            block_weight=block_weight,
        )
        history = output.get("stream_history")
        if not isinstance(history, list):
            raise ValueError("v3 output missing stream_history")

        terms: list[torch.Tensor] = []
        # A stream produced by chunk c forecasts a short prefix of chunk c+1.
        for chunk_index, stage_streams in enumerate(history[:-1]):
            next_start = (chunk_index + 1) * self.cfg.chunk_size
            remaining = tokens.size(1) - next_start
            k = min(self.stream_forecast_tokens, remaining)
            if k <= 0:
                continue
            target = tokens[:, next_start : next_start + k]
            for stage_index, stream in enumerate(stage_streams):
                pred = self.stream_forecast_heads[stage_index](stream)
                pred = pred.view(
                    stream.size(0),
                    self.stream_forecast_tokens,
                    self.cfg.vocab_size,
                )[:, :k]
                terms.append(
                    F.cross_entropy(
                        pred.float().reshape(-1, self.cfg.vocab_size),
                        target.reshape(-1),
                    )
                )

        forecast = (
            torch.stack(terms).mean()
            if terms
            else torch.zeros((), device=tokens.device, dtype=torch.float32)
        )
        total = base["total"] + stream_forecast_weight * forecast
        result = dict(base)
        result["stream_forecast"] = forecast
        result["total"] = total
        return result
