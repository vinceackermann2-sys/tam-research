from __future__ import annotations

"""AERA-v26.1 CPU-first zero-pack sparse-state transport.

Preregistered by issue #406 after the authoritative #400 synthetic L4 result
showed that v26's ephemeral packing was exact and reduced CUDA device events, but
was slower in every frozen latency row.  V26.1 changes no scientific equation or
persistent session state.  It replaces only optional-stage hard-sparse state
movement with an explicit transport backend whose semantic inputs/outputs remain
stream + FICEM keys/values/strengths/validity as separate tensors.

The CPU backend intentionally uses the exact existing componentwise PyTorch
select/merge implementation.  It makes no performance claim; it freezes the
interface required by a future, separately-preregistered zero-pack fused CUDA
backend that can consume the semantic tensors as separate pointers without first
materializing a packed tensor.
"""

from typing import Any, Protocol

import torch
import torch.nn as nn

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v23 import sparse_write_budget
from .aera_hardware_core_v24 import (
    _merge_epi_state,
    _restore_epi_dtype,
    _select_epi_state,
    episodic_state_bytes_per_session,
)
from .aera_hardware_core_v25_1 import _known_empty_hint, _set_known_empty_hint
from .aera_hardware_core_v26 import (
    HardwareAwareAERATextLMV26,
    coalesced_runtime_v26_protocol,
)
from .aera_hardware_core_v8 import StageRouteGate


class SparseStageStateTransportBackend(Protocol):
    """Semantic state movement boundary for optional hard-sparse stages.

    Implementations consume and return the existing AERAState structure.  They may
    use transient output/scratch storage for the current call, but may not add a
    durable packed state, cache, shadow session state, or execute skipped examples.
    """

    name: str

    def select(self, state: AERAState, run_idx: torch.Tensor) -> AERAState: ...

    def merge(
        self,
        base_state: AERAState,
        selected_update: AERAState,
        run_idx: torch.Tensor,
    ) -> AERAState: ...


class TorchComponentwiseStateTransport:
    """Exact CPU reference semantics with no ephemeral packing.

    This deliberately delegates to the pre-v26 componentwise helpers.  The later
    CUDA backend is expected to fuse the same semantic copies into one gather and
    one merge launch while writing the five semantic output tensors directly.
    """

    name = "torch-componentwise-exact-zero-pack"

    def select(self, state: AERAState, run_idx: torch.Tensor) -> AERAState:
        return _select_epi_state(state, run_idx)

    def merge(
        self,
        base_state: AERAState,
        selected_update: AERAState,
        run_idx: torch.Tensor,
    ) -> AERAState:
        return _merge_epi_state(base_state, selected_update, run_idx)


class HardwareAwareAERATextLMV261ZeroPackTransport(HardwareAwareAERATextLMV26):
    """Final v26 semantics with optional hard-sparse state transport abstracted.

    Foundation-stage dispatch and all non-hard calibration paths remain inherited.
    Only optional hard-sparse selected-population state movement is redirected to
    ``SparseStageStateTransportBackend``.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self._state_transport_backend: SparseStageStateTransportBackend = (
            TorchComponentwiseStateTransport()
        )
        self.zero_pack_transport_select_calls = 0
        self.zero_pack_transport_merge_calls = 0
        self.zero_pack_optional_stage_calls = 0

    @property
    def state_transport_backend_name(self) -> str:
        return self._state_transport_backend.name

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

        selected_state = self._state_transport_backend.select(stage_state, run_idx)
        self.zero_pack_transport_select_calls += 1
        self.zero_pack_optional_stage_calls += 1

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

        merged_state = self._state_transport_backend.merge(
            stage_state, selected_new_state, run_idx
        )
        self.zero_pack_transport_merge_calls += 1
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


def zero_pack_transport_v26_1_protocol() -> dict[str, Any]:
    protocol = dict(coalesced_runtime_v26_protocol())
    protocol.update(
        {
            "version": "aera-v26.1-zero-pack-sparse-state-transport-cpu-reference",
            "research_issue": 406,
            "source_version": "aera-v26-coalesced-sparse-runtime-cpu-reference",
            "source_issue400_authoritative_decision": "FAIL",
            "source_issue400_batch64_geomean_latency_ratio": 1.1147760789777879,
            "source_issue400_kernel_event_ratio_each_row": 8.0 / 15.0,
            "learned_equations_changed": False,
            "learned_parameter_count_changed": False,
            "state_dict_schema_changed": False,
            "routing_policy_changed": False,
            "optional_stage_skipping_changed": False,
            "expert_sparsity_changed": False,
            "reasoning_sparsity_changed": False,
            "ficem_equations_changed": False,
            "state_transport_backend_interface": True,
            "state_transport_cpu_reference_backend": TorchComponentwiseStateTransport.name,
            "hard_sparse_ephemeral_pack_state": False,
            "hard_sparse_pack_helper_calls_target": 0,
            "hard_sparse_state_cat_calls_target": 0,
            "hard_sparse_state_stack_calls_target": 0,
            "persistent_state_format_changed": False,
            "persistent_state_extra_tensors": 0,
            "persistent_state_bytes_real_language_four_stage_memory_dim50": episodic_state_bytes_per_session(
                n_stages=4, memory_dim=50
            ),
            "real_language_selected_writes": sparse_write_budget(255),
            "future_cuda_transport_target": {
                "selected_population_gather_launches": 1,
                "selected_population_merge_launches": 1,
                "semantic_output_tensors": [
                    "stream",
                    "episodic_keys",
                    "episodic_values",
                    "episodic_strengths",
                    "episodic_valid",
                ],
                "input_pack_required": False,
                "dense_masked_stage_execution": False,
            },
            "cuda_backend_implemented": False,
            "gpu_authorized": False,
            "scientific_training_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol
