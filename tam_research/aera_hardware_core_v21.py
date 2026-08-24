from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState, FastMemoryState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v18 import PretrainableDeltaFastMemory
from .aera_hardware_core_v19 import HardwareAwareAERATextLMV19, TokenwiseFastMemoryStage


class EventPairFastMemoryStage(TokenwiseFastMemoryStage):
    """V19 stage with bounded adjacent-event memory writes.

    A completed chunk exposes at most ``T-1`` causal transition candidates.  The
    candidate at t stores address ``base_h[t]`` -> payload ``base_h[t+1]`` using
    the existing delta-memory k/v projections and correction equation.  A tiny
    learned pair gate modulates each transition; the existing chunk-level
    novelty*memory_write gate remains the outer write budget.

    Writes are applied only after the chunk output has been produced, so they can
    affect future chunks only.  The v19 token-wise read path is unchanged.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__(source)
        self.pair_write_gate = nn.Linear(2 * self.cfg.d_model, 1, bias=True)
        # Neutral 0.5 start: no arbitrary preference among token transitions.
        nn.init.zeros_(self.pair_write_gate.weight)
        nn.init.zeros_(self.pair_write_gate.bias)
        self.last_pair_gate: torch.Tensor | None = None
        self.last_pair_strength: torch.Tensor | None = None
        self.last_candidate_count: int = 0

    def _event_pair_update(
        self,
        address_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: FastMemoryState,
    ) -> FastMemoryState:
        memory = self.memory
        if not isinstance(memory, PretrainableDeltaFastMemory):
            raise TypeError("v21 requires PretrainableDeltaFastMemory")
        if address_source.shape != payload_source.shape or address_source.ndim != 3:
            raise ValueError("address/payload sources must match [batch,candidates,d_model]")
        if write_strength.shape != (*address_source.shape[:-1], 1):
            raise ValueError("write_strength must be [batch,candidates,1]")
        if state.matrix.ndim != 3 or state.matrix.size(0) != address_source.size(0):
            raise ValueError("memory state batch mismatch")

        def update(matrix: torch.Tensor, *, detach_inputs: bool) -> torch.Tensor:
            a = address_source.detach() if detach_inputs else address_source
            p = payload_source.detach() if detach_inputs else payload_source
            s = write_strength.detach() if detach_inputs else write_strength
            keys = F.normalize(memory.k(a), dim=-1)
            targets = torch.tanh(memory.v(p))
            strengths = s.clamp(0.0, 1.0)
            for t in range(a.size(1)):
                matrix = memory.decay * matrix
                key = keys[:, t]
                target = targets[:, t]
                pred = torch.einsum("bi,bij->bj", key, matrix)
                error = target - pred
                eta = memory.lr * strengths[:, t]
                matrix = matrix + torch.einsum("bi,bj->bij", key * eta, error)
            return matrix

        if memory.differentiable_pretraining:
            return FastMemoryState(update(state.matrix, detach_inputs=False))
        with torch.no_grad():
            matrix = update(state.matrix.detach().clone(), detach_inputs=True)
        return FastMemoryState(matrix.detach())

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, dict[str, torch.Tensor]]]:
        prior_state = state if state is not None else self.empty_state(events)
        # Preserve v19's exact compute/read/stream path and replace only the
        # completed-chunk future-memory write.
        h_out, next_state, controls = super().forward_chunk(
            events,
            prior_state,
            hard=hard,
            update_memory=False,
        )
        if not update_memory or events.size(1) < 2:
            self.last_candidate_count = max(int(events.size(1)) - 1, 0)
            return h_out, next_state, controls

        base_h = self.norm(events)
        address_source = base_h[:, :-1]
        payload_source = base_h[:, 1:]
        pair_features = torch.cat((address_source, payload_source), dim=-1)
        pair_gate = torch.sigmoid(self.pair_write_gate(pair_features))
        end_control = controls["end"]
        chunk_strength = (
            end_control["novelty"] * end_control["memory_write"]
        ).clamp(0.0, 1.0)
        write_strength = pair_gate * chunk_strength[:, None, :]

        self.last_pair_gate = pair_gate.detach()
        self.last_pair_strength = write_strength.detach()
        self.last_candidate_count = int(address_source.size(1))
        memory_state = self._event_pair_update(
            address_source,
            payload_source,
            write_strength,
            prior_state.memory,
        )
        return h_out, AERAState(next_state.stream, memory_state), controls

    def stats(self) -> dict[str, object]:
        result = super().stats()
        result["event_pair_write_candidates"] = self.last_candidate_count
        if self.last_pair_gate is not None:
            result["event_pair_gate_mean"] = float(self.last_pair_gate.float().mean())
        if self.last_pair_strength is not None:
            result["event_pair_write_strength_mean"] = float(
                self.last_pair_strength.float().mean()
            )
        return result


class HardwareAwareAERATextLMV21(HardwareAwareAERATextLMV19):
    """V19 routing/read core with bounded adjacent-event fast-memory writes."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(EventPairFastMemoryStage(stage) for stage in self.stages)


def event_pair_memory_protocol() -> dict[str, object]:
    return {
        "version": "aera-v21-adjacent-event-pair-fast-memory-write",
        "routing_changed_from_v19": False,
        "tokenwise_read_changed_from_v19": False,
        "predictive_stream_changed_from_v19": False,
        "memory_matrix_shape_changed": False,
        "memory_dimension_changed": False,
        "delta_correction_equation_changed": False,
        "write_timing": "future-only after completed chunk",
        "write_representation": "adjacent causal event transitions h_t -> h_t+1",
        "max_write_candidates_per_stage_chunk": "chunk_size-1",
        "pair_gate": "sigmoid learned per adjacent event pair",
        "chunk_gate": "existing novelty * memory_write",
        "deployment_local_update_detached": True,
        "base_pretraining_update_differentiable": True,
        "gpu_authorized": False,
    }
