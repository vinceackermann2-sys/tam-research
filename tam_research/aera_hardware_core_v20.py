from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState, FastMemoryState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v18 import PretrainableDeltaFastMemory
from .aera_hardware_core_v19 import HardwareAwareAERATextLMV19, TokenwiseFastMemoryStage


class FactorizedWriteExtractor(nn.Module):
    """One bounded address/payload write candidate from a completed chunk.

    V19 asks one latent summary to be both the associative address and payload.
    V20 keeps one write per stage/chunk but factorizes those two jobs.  Both
    poolers see only the already-completed causal chunk representation and can
    therefore affect future chunks only.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.address_score = nn.Linear(d_model, 1, bias=False)
        self.payload_score = nn.Linear(d_model, 1, bias=False)
        # Neutral/uniform start avoids injecting an arbitrary positional bias.
        nn.init.zeros_(self.address_score.weight)
        nn.init.zeros_(self.payload_score.weight)
        self.last_address_weights: torch.Tensor | None = None
        self.last_payload_weights: torch.Tensor | None = None

    def forward(self, base_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if base_h.ndim != 3:
            raise ValueError("factorized write extractor expects [batch,time,d_model]")
        address_weights = F.softmax(self.address_score(base_h).squeeze(-1).float(), dim=-1).to(base_h.dtype)
        payload_weights = F.softmax(self.payload_score(base_h).squeeze(-1).float(), dim=-1).to(base_h.dtype)
        address_source = torch.einsum("bt,btd->bd", address_weights, base_h)
        payload_source = torch.einsum("bt,btd->bd", payload_weights, base_h)
        self.last_address_weights = address_weights.detach()
        self.last_payload_weights = payload_weights.detach()
        return address_source, payload_source

    def stats(self) -> dict[str, object] | None:
        if self.last_address_weights is None or self.last_payload_weights is None:
            return None
        return {
            "address_weight_sum_mean": float(self.last_address_weights.float().sum(dim=-1).mean()),
            "payload_weight_sum_mean": float(self.last_payload_weights.float().sum(dim=-1).mean()),
            "address_weight_max_mean": float(self.last_address_weights.float().max(dim=-1).values.mean()),
            "payload_weight_max_mean": float(self.last_payload_weights.float().max(dim=-1).values.mean()),
        }


class FactorizedFastMemoryStage(TokenwiseFastMemoryStage):
    """V19 stage with separate address and payload sources for one memory write."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__(source)
        self.write_extractor = FactorizedWriteExtractor(self.cfg.d_model)
        self.last_factorized_write_strength: torch.Tensor | None = None

    def _factorized_update(
        self,
        address_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: FastMemoryState,
    ) -> FastMemoryState:
        memory = self.memory
        if not isinstance(memory, PretrainableDeltaFastMemory):
            raise TypeError("v20 requires PretrainableDeltaFastMemory")
        if address_source.shape != payload_source.shape:
            raise ValueError("address/payload sources must have identical [batch,d_model] shape")
        if write_strength.shape != (address_source.size(0), 1):
            raise ValueError("write_strength must be [batch,1]")
        if state.matrix.ndim != 3 or state.matrix.size(0) != address_source.size(0):
            raise ValueError("memory state batch mismatch")

        def update(matrix: torch.Tensor, *, detach_inputs: bool) -> torch.Tensor:
            a = address_source.detach() if detach_inputs else address_source
            p = payload_source.detach() if detach_inputs else payload_source
            s = write_strength.detach() if detach_inputs else write_strength
            key = F.normalize(memory.k(a), dim=-1)
            target = torch.tanh(memory.v(p))
            strength = s.clamp(0.0, 1.0)
            matrix = memory.decay * matrix
            pred = torch.einsum("bi,bij->bj", key, matrix)
            error = target - pred
            eta = memory.lr * strength
            return matrix + torch.einsum("bi,bj->bij", key * eta, error)

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
        # Preserve the exact v19 compute/read/stream path.  We intentionally call
        # it with update_memory=False and replace only the future-memory write.
        prior_state = state if state is not None else self.empty_state(events)
        h_out, next_state, controls = super().forward_chunk(
            events,
            prior_state,
            hard=hard,
            update_memory=False,
        )
        if not update_memory:
            return h_out, next_state, controls

        base_h = self.norm(events)
        address_source, payload_source = self.write_extractor(base_h)
        end_control = controls["end"]
        write_strength = (end_control["novelty"] * end_control["memory_write"]).clamp(0.0, 1.0)
        self.last_factorized_write_strength = write_strength.detach()
        memory_state = self._factorized_update(
            address_source,
            payload_source,
            write_strength,
            prior_state.memory,
        )
        return h_out, AERAState(next_state.stream, memory_state), controls

    def stats(self) -> dict[str, object]:
        result = super().stats()
        result["factorized_write"] = self.write_extractor.stats()
        if self.last_factorized_write_strength is not None:
            result["factorized_write_strength_mean"] = float(
                self.last_factorized_write_strength.float().mean()
            )
        return result


class HardwareAwareAERATextLMV20(HardwareAwareAERATextLMV19):
    """V19 routing/read core with factorized bounded fast-memory writes."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(FactorizedFastMemoryStage(stage) for stage in self.stages)


def factorized_memory_protocol() -> dict[str, object]:
    return {
        "version": "aera-v20-factorized-fast-memory-write",
        "routing_changed_from_v19": False,
        "tokenwise_read_changed_from_v19": False,
        "predictive_stream_changed_from_v19": False,
        "memory_matrix_shape_changed": False,
        "memory_dimension_changed": False,
        "delta_correction_equation_changed": False,
        "write_timing": "one future-only write per executed stage/chunk",
        "write_representation": "separate learned address and payload pools over completed causal base_h",
        "write_slots_per_stage_chunk": 1,
        "deployment_local_update_detached": True,
        "base_pretraining_update_differentiable": True,
        "gpu_authorized": False,
    }
