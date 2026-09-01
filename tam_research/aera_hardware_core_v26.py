from __future__ import annotations

"""AERA-v26 CPU-first coalesced sparse runtime.

Preregistered by issue #398 after the #381/v25.1 systems gate was exhausted without
an authoritative timing result. V26 is an execution architecture boundary, not a
scientific-architecture change: learned parameters, routing decisions, FICEM
equations, recurrent state semantics, write budget, and durable session state are
preserved.

This CPU reference introduces two execution interfaces:
1. coalesced optional-stage state movement: floating stage state is packed into one
   ephemeral tensor for selected-population gather/merge, with validity kept as the
   sole boolean tensor;
2. a FICEM backend boundary whose reference implementation delegates to the exact
   final v25.1 stable-compaction equations. A future separately-preregistered CUDA
   backend may fuse these primitives without changing their semantics.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v23 import sparse_write_budget
from .aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    _as_epi,
    _restore_epi_dtype,
    episodic_state_bytes_per_session,
)
from .aera_hardware_core_v25_1 import _known_empty_hint, _set_known_empty_hint
from .aera_hardware_core_v25_1_compact import (
    StableCompactExecutionEquivalentFactorizedIdentityContextMemory,
)
from .aera_hardware_core_v25_1_nohost import (
    HardwareAwareAERATextLMV251NoHostTelemetry,
    NoHostAdaptiveTelemetryStableCompactFICEMStage,
    no_host_adaptive_telemetry_v25_1_protocol,
)
from .aera_hardware_core_v8 import StageRouteGate


@dataclass(frozen=True)
class PackedEpisodicRuntimeState:
    """Ephemeral coalesced floating state plus exact boolean validity.

    This is never persisted in session state. ``floating`` concatenates, in order:
    stream, keys, values, strengths. Unpacking returns views into the packed tensor.
    """

    floating: torch.Tensor
    valid: torch.Tensor
    d_model: int
    capacity: int
    memory_dim: int


def pack_ephemeral_epi_state(state: AERAState) -> PackedEpisodicRuntimeState:
    memory = _as_epi(state.memory)
    batch = state.stream.size(0)
    if memory.keys.shape != memory.values.shape:
        raise ValueError("episodic key/value state mismatch")
    if memory.keys.size(0) != batch:
        raise ValueError("stream/memory batch mismatch")
    if memory.strengths.shape != memory.valid.shape:
        raise ValueError("episodic strength/valid shape mismatch")
    if memory.keys.shape[:2] != memory.valid.shape:
        raise ValueError("episodic slot/valid shape mismatch")
    if not (
        state.stream.dtype == memory.keys.dtype
        == memory.values.dtype
        == memory.strengths.dtype
    ):
        raise ValueError("coalesced floating state requires one floating dtype")

    capacity = int(memory.keys.size(1))
    memory_dim = int(memory.keys.size(2))
    d_model = int(state.stream.size(1))
    floating = torch.cat(
        (
            state.stream,
            memory.keys.reshape(batch, capacity * memory_dim),
            memory.values.reshape(batch, capacity * memory_dim),
            memory.strengths,
        ),
        dim=1,
    )
    return PackedEpisodicRuntimeState(
        floating=floating,
        valid=memory.valid,
        d_model=d_model,
        capacity=capacity,
        memory_dim=memory_dim,
    )


def unpack_ephemeral_epi_state(packed: PackedEpisodicRuntimeState) -> AERAState:
    floating = packed.floating
    if floating.ndim != 2 or packed.valid.ndim != 2:
        raise ValueError("packed episodic state must be rank-2")
    if floating.size(0) != packed.valid.size(0):
        raise ValueError("packed float/valid batch mismatch")
    if packed.valid.size(1) != packed.capacity:
        raise ValueError("packed validity capacity mismatch")

    batch = floating.size(0)
    stream_end = packed.d_model
    keys_end = stream_end + packed.capacity * packed.memory_dim
    values_end = keys_end + packed.capacity * packed.memory_dim
    strengths_end = values_end + packed.capacity
    if floating.size(1) != strengths_end:
        raise ValueError("packed floating width mismatch")

    stream = floating[:, :stream_end]
    keys = floating[:, stream_end:keys_end].view(
        batch, packed.capacity, packed.memory_dim
    )
    values = floating[:, keys_end:values_end].view(
        batch, packed.capacity, packed.memory_dim
    )
    strengths = floating[:, values_end:strengths_end]
    return AERAState(
        stream=stream,
        memory=ContextualEpisodicMemoryState(
            keys=keys,
            values=values,
            strengths=strengths,
            valid=packed.valid,
        ),
    )


def select_packed_epi_state(
    packed: PackedEpisodicRuntimeState,
    idx: torch.Tensor,
) -> PackedEpisodicRuntimeState:
    if idx.ndim != 1:
        raise ValueError("selected population index must be rank-1")
    return PackedEpisodicRuntimeState(
        floating=packed.floating.index_select(0, idx),
        valid=packed.valid.index_select(0, idx),
        d_model=packed.d_model,
        capacity=packed.capacity,
        memory_dim=packed.memory_dim,
    )


def merge_packed_epi_state(
    base: PackedEpisodicRuntimeState,
    update: PackedEpisodicRuntimeState,
    idx: torch.Tensor,
) -> PackedEpisodicRuntimeState:
    if (
        base.d_model != update.d_model
        or base.capacity != update.capacity
        or base.memory_dim != update.memory_dim
    ):
        raise ValueError("packed episodic geometries differ")
    if update.floating.size(0) != idx.numel() or update.valid.size(0) != idx.numel():
        raise ValueError("packed update batch must match selected population")
    return PackedEpisodicRuntimeState(
        floating=base.floating.index_copy(0, idx, update.floating),
        valid=base.valid.index_copy(0, idx, update.valid),
        d_model=base.d_model,
        capacity=base.capacity,
        memory_dim=base.memory_dim,
    )


@dataclass(frozen=True)
class FICEMReadPrimitive:
    recalled: torch.Tensor
    projected_query: torch.Tensor | None
    normalized_old_keys: torch.Tensor | None


class FICEMExecutionBackend(Protocol):
    """Fixed-geometry execution boundary for exact v25.1 FICEM semantics."""

    name: str

    def read(
        self,
        memory: "CoalescedFICEMMemory",
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> FICEMReadPrimitive: ...

    def update(
        self,
        memory: "CoalescedFICEMMemory",
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState: ...

    def update_from_projected(
        self,
        memory: "CoalescedFICEMMemory",
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState: ...


class TorchFICEMReferenceBackend:
    """Pure-PyTorch reference backend; delegates to exact final-v25.1 operations."""

    name = "torch-reference-v25.1-exact"

    def read(
        self,
        memory: "CoalescedFICEMMemory",
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> FICEMReadPrimitive:
        recalled, projected_query, normalized_old_keys = (
            memory._reference_read_with_reuse(identity_source, context_source, state)
        )
        return FICEMReadPrimitive(
            recalled=recalled,
            projected_query=projected_query,
            normalized_old_keys=normalized_old_keys,
        )

    def update(
        self,
        memory: "CoalescedFICEMMemory",
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        return memory._reference_update_block(
            identity_source,
            context_source,
            payload_source,
            write_strength,
            state,
        )

    def update_from_projected(
        self,
        memory: "CoalescedFICEMMemory",
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        return memory._reference_update_block_from_projected(
            projected_new_keys,
            normalized_old_keys,
            payload_source,
            write_strength,
            state,
        )


class CoalescedFICEMMemory(
    StableCompactExecutionEquivalentFactorizedIdentityContextMemory
):
    """Final v25.1 FICEM math behind the explicit v26 execution-backend boundary."""

    def __init__(
        self,
        source: StableCompactExecutionEquivalentFactorizedIdentityContextMemory,
    ) -> None:
        super().__init__(source)
        self._execution_backend: FICEMExecutionBackend = TorchFICEMReferenceBackend()
        self.backend_read_calls = 0
        self.backend_update_calls = 0
        self.backend_projected_update_calls = 0

    @property
    def execution_backend_name(self) -> str:
        return self._execution_backend.name

    def _reference_read_with_reuse(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return super().read_with_reuse(identity_source, context_source, state)

    def _reference_update_block(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        return super().update_block(
            identity_source,
            context_source,
            payload_source,
            write_strength,
            state,
        )

    def _reference_update_block_from_projected(
        self,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        return super().update_block_from_projected(
            projected_new_keys,
            normalized_old_keys,
            payload_source,
            write_strength,
            state,
        )

    def read_with_reuse(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        result = self._execution_backend.read(
            self, identity_source, context_source, state
        )
        self.backend_read_calls += 1
        return result.recalled, result.projected_query, result.normalized_old_keys

    def update_block(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        result = self._execution_backend.update(
            self,
            identity_source,
            context_source,
            payload_source,
            write_strength,
            state,
        )
        self.backend_update_calls += 1
        return result

    def update_block_from_projected(
        self,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        result = self._execution_backend.update_from_projected(
            self,
            projected_new_keys,
            normalized_old_keys,
            payload_source,
            write_strength,
            state,
        )
        self.backend_projected_update_calls += 1
        return result


class CoalescedRuntimeFICEMStage(NoHostAdaptiveTelemetryStableCompactFICEMStage):
    """Final v25.1 stage with only the FICEM execution interface replaced."""

    def __init__(self, source: NoHostAdaptiveTelemetryStableCompactFICEMStage) -> None:
        super().__init__(source)
        self.memory = CoalescedFICEMMemory(self.memory)


class HardwareAwareAERATextLMV26(HardwareAwareAERATextLMV251NoHostTelemetry):
    """CPU-first v26 runtime with coalesced optional-stage state movement."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            CoalescedRuntimeFICEMStage(stage) for stage in self.stages
        )
        self.set_memory_pretraining_mode(False)
        self.coalesced_float_state_select_calls = 0
        self.coalesced_valid_select_calls = 0
        self.coalesced_float_state_merge_calls = 0
        self.coalesced_valid_merge_calls = 0
        self.coalesced_pack_calls = 0
        self.legacy_float_component_selects_avoided = 0
        self.legacy_float_component_merges_avoided = 0

    def _route_one_stage(
        self,
        x: torch.Tensor,
        stage: nn.Module,
        stage_state: AERAState,
        router: StageRouteGate,
        *,
        route_mode: str,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, object]]:
        is_foundation = router is self.stage_routers[self.FOUNDATION_STAGE]
        if route_mode != "hard_sparse" or is_foundation:
            return super()._route_one_stage(
                x,
                stage,
                stage_state,
                router,
                route_mode=route_mode,
                update_memory=update_memory,
            )

        gate, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        prob = torch.sigmoid(logits)
        run_idx = (gate[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
        if run_idx.numel() == 0:
            return x, stage_state, {
                "stage_route_probability": prob,
                "stage_route_gate": gate,
                "executed_fraction": 0.0,
                "start": None,
                "end": None,
            }

        base_packed = pack_ephemeral_epi_state(stage_state)
        self.coalesced_pack_calls += 1
        selected_packed = select_packed_epi_state(base_packed, run_idx)
        self.coalesced_float_state_select_calls += 1
        self.coalesced_valid_select_calls += 1
        # Legacy v25.1 selected stream, keys, values and strengths independently.
        self.legacy_float_component_selects_avoided += 3

        selected_state = unpack_ephemeral_epi_state(selected_packed)
        prior_known_empty = _known_empty_hint(stage_state.memory)
        if prior_known_empty:
            _set_known_empty_hint(selected_state.memory, True)

        selected_y, selected_new_state, selected_controls = stage.forward_chunk(
            x.index_select(0, run_idx),
            selected_state,
            hard=True,
            update_memory=update_memory,
        )
        selected_y = selected_y.to(dtype=x.dtype)
        selected_new_state = _restore_epi_dtype(selected_state, selected_new_state)

        update_packed = pack_ephemeral_epi_state(selected_new_state)
        self.coalesced_pack_calls += 1
        merged_packed = merge_packed_epi_state(base_packed, update_packed, run_idx)
        self.coalesced_float_state_merge_calls += 1
        self.coalesced_valid_merge_calls += 1
        self.legacy_float_component_merges_avoided += 3
        merged_state = unpack_ephemeral_epi_state(merged_packed)
        if prior_known_empty and not update_memory:
            _set_known_empty_hint(merged_state.memory, True)

        return (
            x.index_copy(0, run_idx, selected_y),
            merged_state,
            {
                "stage_route_probability": prob,
                "stage_route_gate": gate,
                "executed_fraction": float(run_idx.numel() / x.size(0)),
                "start": selected_controls["start"],
                "end": selected_controls["end"],
            },
        )


def coalesced_runtime_v26_protocol() -> dict[str, Any]:
    protocol = dict(no_host_adaptive_telemetry_v25_1_protocol())
    protocol.update(
        {
            "version": "aera-v26-coalesced-sparse-runtime-cpu-reference",
            "research_issue": 398,
            "source_version": "aera-v25.1-execution-equivalent-no-host-adaptive-telemetry",
            "learned_equations_changed": False,
            "learned_parameter_count_changed": False,
            "state_dict_schema_changed": False,
            "routing_policy_changed": False,
            "optional_stage_skipping_changed": False,
            "expert_sparsity_changed": False,
            "reasoning_sparsity_changed": False,
            "coalesced_optional_state": True,
            "coalesced_floating_components": [
                "stream",
                "episodic_keys",
                "episodic_values",
                "episodic_strengths",
            ],
            "validity_remains_separate_bool_tensor": True,
            "selected_population_float_state_index_selects_target": 1,
            "selected_population_validity_index_selects_target": 1,
            "selected_population_float_state_index_copies_target": 1,
            "selected_population_validity_index_copies_target": 1,
            "ficem_backend_interface": True,
            "ficem_reference_backend": TorchFICEMReferenceBackend.name,
            "ficem_equations_changed": False,
            "read_top_k_changed": False,
            "read_temperature_changed": False,
            "write_budget_changed": False,
            "real_language_selected_writes": sparse_write_budget(255),
            "vectorized_update_calls_per_completed_stage_chunk": 1,
            "persistent_state_bytes_real_language_four_stage_memory_dim50": episodic_state_bytes_per_session(
                n_stages=4, memory_dim=50
            ),
            "persistent_runtime_pack_state": False,
            "gpu_authorized": False,
            "scientific_training_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol
