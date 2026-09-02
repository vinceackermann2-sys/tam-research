from __future__ import annotations

"""AERA-v26.5 mixed-field-dtype successor for the fused FICEM WRITE tail.

Issue #511 is a CPU-first integration repair after the exhausted #508 systems
attempt reached the L4 and exposed a candidate-only WRITE validator overconstraint.
The historical v26.4 backend and its two Triton kernels remain byte-for-byte
unchanged.  This version reuses those exact kernels and changes only the wrapper's
floating-dtype contract: each WRITE field may be FP32 or BF16 independently, while
all projected/value/strength/duplicate math is computed before materialization.
Durable outputs keep the corresponding prior-state field dtype via empty_like.
"""

from typing import Any, Iterable

import torch

from .aera_hardware_core_v25_1 import _set_known_empty_hint
from .aera_hardware_core_v26 import CoalescedFICEMMemory
from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v26_4_ficem_write_triton import (
    TritonFICEMReadWriteBackend,
    WRITE_CAPACITY,
    WRITE_COUNT,
    WRITE_DUPLICATE_SIMILARITY,
    WRITE_MEMORY_DIM,
    _write_adjudicate_map_kernel,
    _write_materialize_kernel,
    fused_ficem_read_write_v26_4_protocol,
    triton,
)

RESEARCH_ISSUE = 511
SOURCE_MAIN = "1d475a199cfd2b14d5e94e5cffa29e05ac868ab1"
PREDECESSOR_WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
SUPPORTED_WRITE_FLOAT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
)


def supported_write_field_dtypes(dtypes: Iterable[torch.dtype]) -> bool:
    """Pure CPU-checkable dtype contract for the inference WRITE wrapper."""

    values = tuple(dtypes)
    return bool(values) and all(dtype in SUPPORTED_WRITE_FLOAT_DTYPES for dtype in values)


def _validate_mixed_write_tail_inputs(
    incoming_similarity: torch.Tensor,
    old_similarity: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    new_strengths: torch.Tensor,
    new_valid: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> int:
    if new_keys.ndim != 3 or new_keys.shape != new_values.shape:
        raise ValueError("FICEM new key/value tensors must match [batch,16,50]")
    batch, k_count, memory_dim = map(int, new_keys.shape)
    if (k_count, memory_dim) != (WRITE_COUNT, WRITE_MEMORY_DIM):
        raise ValueError("v26.5 mixed-dtype fused write is frozen to K16/memory_dim50")
    if state.keys.shape != state.values.shape:
        raise ValueError("FICEM old key/value state mismatch")
    if state.keys.shape != (batch, WRITE_CAPACITY, WRITE_MEMORY_DIM):
        raise ValueError("v26.5 mixed-dtype fused write is frozen to capacity48/memory_dim50")
    if state.strengths.shape != (batch, WRITE_CAPACITY):
        raise ValueError("FICEM old strength shape mismatch")
    if state.valid.shape != (batch, WRITE_CAPACITY):
        raise ValueError("FICEM old validity shape mismatch")
    if new_strengths.shape != (batch, WRITE_COUNT):
        raise ValueError("FICEM new strength shape mismatch")
    if new_valid.shape != (batch, WRITE_COUNT) or new_valid.dtype is not torch.bool:
        raise ValueError("FICEM new validity must be boolean [batch,16]")
    if state.valid.dtype is not torch.bool:
        raise TypeError("FICEM old validity must be boolean")
    if incoming_similarity.shape != (batch, WRITE_COUNT, WRITE_COUNT):
        raise ValueError("incoming similarity must be [batch,16,16]")
    if old_similarity.shape != (batch, WRITE_COUNT, WRITE_CAPACITY):
        raise ValueError("new/old similarity must be [batch,16,48]")

    floating = (
        incoming_similarity,
        old_similarity,
        new_keys,
        new_values,
        new_strengths,
        state.keys,
        state.values,
        state.strengths,
    )
    if not supported_write_field_dtypes(t.dtype for t in floating):
        raise TypeError("v26.5 fused FICEM write supports only float32/bfloat16 fields")
    if any(t.device.type != "cuda" for t in (*floating, new_valid, state.valid)):
        raise RuntimeError("v26.5 mixed-dtype fused FICEM write requires CUDA tensors")
    devices = {t.device for t in (*floating, new_valid, state.valid)}
    if len(devices) != 1:
        raise ValueError("FICEM write-tail tensors must share one CUDA device")
    if not all(t.is_contiguous() for t in (*floating, new_valid, state.valid)):
        raise ValueError("v26.5 mixed-dtype fused FICEM write requires contiguous tensors")
    return batch


def fused_ficem_write_tail_mixed_dtype(
    incoming_similarity: torch.Tensor,
    old_similarity: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    new_strengths: torch.Tensor,
    new_valid: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> ContextualEpisodicMemoryState:
    """Run the exact historical two-kernel tail with field-wise FP32/BF16 dtypes."""

    if triton is None or _write_adjudicate_map_kernel is None or _write_materialize_kernel is None:
        raise RuntimeError("Triton FICEM write kernels are unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fused FICEM write is unavailable")
    batch = _validate_mixed_write_tail_inputs(
        incoming_similarity,
        old_similarity,
        new_keys,
        new_values,
        new_strengths,
        new_valid,
        state,
    )

    source_map = torch.empty(
        (batch, WRITE_CAPACITY),
        device=new_keys.device,
        dtype=torch.int32,
    )
    durable_valid = torch.empty(
        (batch, WRITE_CAPACITY),
        device=new_keys.device,
        dtype=torch.bool,
    )
    out_keys = torch.empty_like(state.keys)
    out_values = torch.empty_like(state.values)
    out_strengths = torch.empty_like(state.strengths)

    _write_adjudicate_map_kernel[(batch,)](
        incoming_similarity,
        old_similarity,
        new_valid,
        state.valid,
        source_map,
        durable_valid,
        K=WRITE_COUNT,
        CAPACITY=WRITE_CAPACITY,
        K_BLOCK=16,
        CAPACITY_BLOCK=64,
        DUPLICATE_THRESHOLD=WRITE_DUPLICATE_SIMILARITY,
        num_warps=4,
    )
    _write_materialize_kernel[(batch * WRITE_CAPACITY,)](
        source_map,
        new_keys,
        new_values,
        new_strengths,
        state.keys,
        state.values,
        state.strengths,
        out_keys,
        out_values,
        out_strengths,
        K=WRITE_COUNT,
        CAPACITY=WRITE_CAPACITY,
        MEMORY_DIM=WRITE_MEMORY_DIM,
        DIM_BLOCK=64,
        num_warps=1,
    )
    return ContextualEpisodicMemoryState(
        keys=out_keys,
        values=out_values,
        strengths=out_strengths,
        valid=durable_valid,
    )


class MixedDtypeTritonFICEMReadWriteBackend(TritonFICEMReadWriteBackend):
    """Historical repair5 READ + v26.4 kernels with mixed-field WRITE support."""

    name = "triton-fused-ficem-read-repair5-write-tail-v26.5-mixed-dtype"

    def _inference_update_from_projected(
        self,
        memory: CoalescedFICEMMemory,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        new_keys = projected_new_keys.detach()
        normalized_old = normalized_old_keys.detach()
        payload = payload_source.detach()
        strength = write_strength.detach()
        old_keys = state.keys.detach()
        old_values = state.values.detach()
        old_strengths = state.strengths.detach()
        old_valid = state.valid.detach()

        if new_keys.shape != (new_keys.size(0), WRITE_COUNT, WRITE_MEMORY_DIM):
            raise ValueError("v26.5 projected write keys must be [batch,16,50]")
        if payload.shape[:2] != new_keys.shape[:2] or payload.ndim != 3:
            raise ValueError("v26.5 payload must match projected write batch/K axes")
        if strength.shape != (*new_keys.shape[:-1], 1):
            raise ValueError("v26.5 write strength must be [batch,16,1]")
        if normalized_old.shape != state.keys.shape:
            raise ValueError("v26.5 normalized old keys must match old state keys")

        # Preserve exact v26.4/reference learned math and duplicate decisions.
        # No dtype conversion occurs before either similarity decision or the tail.
        new_values = torch.tanh(memory.v(payload))
        new_strengths = strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0
        incoming_similarity = torch.einsum("bkd,bjd->bkj", new_keys, new_keys)
        old_similarity = torch.einsum("bkd,bsd->bks", new_keys, normalized_old)

        next_state = fused_ficem_write_tail_mixed_dtype(
            incoming_similarity.contiguous(),
            old_similarity.contiguous(),
            new_keys.contiguous(),
            new_values.contiguous(),
            new_strengths.contiguous(),
            new_valid.contiguous(),
            ContextualEpisodicMemoryState(
                keys=old_keys.contiguous(),
                values=old_values.contiguous(),
                strengths=old_strengths.contiguous(),
                valid=old_valid.contiguous(),
            ),
        )
        _set_known_empty_hint(next_state, False)
        return next_state


def mixed_dtype_ficem_read_write_v26_5_protocol() -> dict[str, Any]:
    protocol = dict(fused_ficem_read_write_v26_4_protocol())
    protocol.update(
        {
            "version": "aera-v26.5-triton-ficem-write-mixed-field-dtype",
            "research_issue": RESEARCH_ISSUE,
            "source_main": SOURCE_MAIN,
            "backend_name": MixedDtypeTritonFICEMReadWriteBackend.name,
            "predecessor_write_backend_blob": PREDECESSOR_WRITE_BACKEND_BLOB,
            "historical_v26_4_backend_mutated": False,
            "write_global_cross_field_dtype_equality_required": False,
            "write_supported_float_dtypes": ["float32", "bfloat16"],
            "write_fieldwise_mixed_dtype_supported": True,
            "write_materialization_output_follows_durable_state_field_dtype": True,
            "write_duplicate_decisions_before_materialization": True,
            "write_explicit_pre_tail_cast_kernels": 0,
            "write_triton_kernel_bodies_changed": False,
            "write_tail_triton_launches_target": 2,
            "read_backend_changed_by_v26_5": False,
            "write_similarity_einsums_changed_by_v26_5": False,
            "write_value_projection_changed_by_v26_5": False,
            "write_strength_semantics_changed_by_v26_5": False,
            "write_duplicate_semantics_changed_by_v26_5": False,
            "write_state_schema_changed_by_v26_5": False,
            "write_persistent_state_changed_by_v26_5": False,
            "write_training_backend_changed_by_v26_5": False,
            "mixed_dtype_gpu_gate_authorized": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol
