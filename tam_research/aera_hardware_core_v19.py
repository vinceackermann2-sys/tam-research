from __future__ import annotations

import torch
import torch.nn as nn

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v18 import (
    HardwareAwareAERATextLMV18,
    PretrainableDeltaFastMemory,
)


class TokenwiseFastMemoryStage(nn.Module):
    """Semantics-preserving stage wrapper with token-wise prior-memory reads.

    V18 performs one associative-memory query from the first representation of a
    chunk and broadcasts that recalled vector across every token. V19 keeps the
    memory state, write equation, write timing, controller gates, recurrent stream,
    experts, attention and latent reasoning unchanged, but queries the *fixed
    prior-chunk memory* independently from every current causal token
    representation.

    Because the memory matrix is not updated until the chunk has finished, all
    token-wise reads are parallel and cannot contain information from future tokens
    in the current chunk.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        required = (
            "cfg",
            "norm",
            "controller",
            "state_to_chunk",
            "attn",
            "experts",
            "reasoner",
            "stream_cell",
            "memory",
            "reason_to_chunk",
            "out_norm",
        )
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(f"source stage missing required modules: {missing}")

        self.cfg = source.cfg
        self.norm = source.norm
        self.controller = source.controller
        self.state_to_chunk = source.state_to_chunk
        self.attn = source.attn
        self.experts = source.experts
        self.reasoner = source.reasoner
        self.stream_cell = source.stream_cell
        self.memory = source.memory
        self.reason_to_chunk = source.reason_to_chunk
        self.out_norm = source.out_norm
        if not isinstance(self.memory, PretrainableDeltaFastMemory):
            raise TypeError("v19 requires v18 PretrainableDeltaFastMemory")
        self.last_start_controls = getattr(source, "last_start_controls", None)
        self.last_end_controls = getattr(source, "last_end_controls", None)

    def empty_state(self, x: torch.Tensor) -> AERAState:
        b = x.size(0)
        return AERAState(
            stream=torch.zeros(b, self.cfg.d_model, device=x.device, dtype=x.dtype),
            memory=self.memory.empty_state(b, x.device, x.dtype),
        )

    def _tokenwise_context(
        self,
        h: torch.Tensor,
        state: AERAState,
        start_control: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [batch,time,d_model] context and raw token-wise memory recall."""
        memory_read = self.memory.read(h, state.memory)
        carried = self.state_to_chunk(state.stream)
        context = (
            start_control["state_read"][:, None, :] * carried[:, None, :]
            + start_control["memory_read"][:, None, :] * memory_read
        )
        return context, memory_read

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, dict[str, torch.Tensor]]]:
        if events.ndim != 3 or events.size(1) > self.cfg.chunk_size:
            raise ValueError("events must be [batch,time,d_model] within chunk_size")
        if state is None:
            state = self.empty_state(events)

        h = self.norm(events)
        # The scalar read/write policy remains a chunk-start causal decision.
        start_control = self.controller(h[:, 0], state.stream)
        self.last_start_controls = {
            k: v.detach() for k, v in start_control.items() if "logits" not in k
        }

        context, _ = self._tokenwise_context(h, state, start_control)
        h = h + context

        h = h + self.attn(h)
        h = h + self.experts(
            h,
            start_control["expert_logits"],
            start_control["expert_count_logits"],
            hard=hard,
        )

        end_summary = h[:, -1]
        end_control = self.controller(end_summary, state.stream)
        self.last_end_controls = {
            k: v.detach() for k, v in end_control.items() if "logits" not in k
        }
        reasoned = self.reasoner(end_summary, end_control["depth_logits"], hard=hard)

        last_mask = torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
        last_mask[-1] = 1
        h = h + self.reason_to_chunk(reasoned)[:, None, :] * last_mask[None, :, None]
        h = self.out_norm(h)

        final_stream = self.stream_cell(reasoned, state.stream)
        memory_state = state.memory
        if update_memory:
            write = (end_control["novelty"] * end_control["memory_write"]).clamp(0.0, 1.0)
            memory_state = self.memory.local_update(
                reasoned[:, None, :], write[:, None, :], state.memory
            )

        controls = {"start": start_control, "end": end_control}
        return h, AERAState(final_stream, memory_state), controls

    def stats(self) -> dict[str, object]:
        def means(x: dict[str, torch.Tensor] | None) -> dict[str, float]:
            if not x:
                return {}
            return {k: float(v.float().mean()) for k, v in x.items()}

        return {
            "experts": self.experts.stats(),
            "reasoning": self.reasoner.stats(),
            "start_controls": means(self.last_start_controls),
            "end_controls": means(self.last_end_controls),
        }


class HardwareAwareAERATextLMV19(HardwareAwareAERATextLMV18):
    """V18 with token-wise content-addressed reads from prior-chunk fast memory."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        # Rewrap the exact initialized modules instead of reconstructing them. This
        # preserves checkpoint keys, parameter values, sparse expert backend and all
        # routing/runtime machinery; only stage.forward_chunk read addressing changes.
        self.stages = nn.ModuleList(TokenwiseFastMemoryStage(stage) for stage in self.stages)


def memory_addressing_protocol() -> dict[str, object]:
    return {
        "version": "aera-v19-tokenwise-prior-memory-read",
        "memory_equation_changed": False,
        "memory_write_rule_changed": False,
        "memory_write_timing_changed": False,
        "memory_dimension_changed": False,
        "stored_parameter_count_changed": False,
        "routing_changed_from_v17": False,
        "read_gate_changed": False,
        "read_granularity": "token-wise",
        "read_query": "current causal token representation",
        "memory_state_during_current_chunk": "fixed prior-chunk state",
        "current_chunk_future_information_in_memory": False,
        "deployment_local_update_detached": True,
        "base_pretraining_update_differentiable": True,
        "gpu_authorized": False,
    }
