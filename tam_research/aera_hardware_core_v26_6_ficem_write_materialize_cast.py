from __future__ import annotations

"""AERA-v26.6 mixed-source materialization repair for the fused FICEM WRITE tail.

Issue #517 follows the authoritative #514 mixed-field-dtype primitive FAIL.  The
failure partition was exact: corresponding new/durable key, value, and strength
dtypes had to match, while similarity dtypes were irrelevant.  This successor
keeps adjudication, learned math, routing, state schema, and the two-launch WRITE
topology unchanged.  The only Triton change is a versioned materialization kernel
that converts both source branches to each durable output pointer element type
before branch selection.
"""

from typing import Any, Iterable

import torch

from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v25_1 import _set_known_empty_hint
from .aera_hardware_core_v26 import CoalescedFICEMMemory
from .aera_hardware_core_v26_4_ficem_write_triton import (
    TritonFICEMReadWriteBackend,
    WRITE_CAPACITY,
    WRITE_COUNT,
    WRITE_DUPLICATE_SIMILARITY,
    WRITE_MEMORY_DIM,
    _write_adjudicate_map_kernel,
    fused_ficem_read_write_v26_4_protocol,
    triton,
    tl,
)

RESEARCH_ISSUE = 517
SOURCE_MAIN = "a67ba825cd71ed78cc7294c7c9fed7532a5100ca"
PREDECESSOR_WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
FAILED_V26_5_BACKEND_BLOB = "dab24c733eff7aa08e5f818614f7504eaac48dc3"
ISSUE514_PROBE_BLOB = "e48dc308bef3b4ef9b6590ab0324db4a50c0f7af"
ISSUE514_RESULT_SHA256 = "c1a8936458c57e975787a27288d3caf494e360ec8ae8acb8d0f5742aef6bf505"
ISSUE514_RUN = 33664645415
ISSUE514_JOB = 100363263710
SUPPORTED_WRITE_FLOAT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
)


def supported_write_field_dtypes(dtypes: Iterable[torch.dtype]) -> bool:
    """CPU-checkable field-wise contract retained exactly from v26.5."""

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
        raise ValueError("v26.6 mixed-dtype fused write is frozen to K16/memory_dim50")
    if state.keys.shape != state.values.shape:
        raise ValueError("FICEM old key/value state mismatch")
    if state.keys.shape != (batch, WRITE_CAPACITY, WRITE_MEMORY_DIM):
        raise ValueError("v26.6 mixed-dtype fused write is frozen to capacity48/memory_dim50")
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
        raise TypeError("v26.6 fused FICEM write supports only float32/bfloat16 fields")
    if any(t.device.type != "cuda" for t in (*floating, new_valid, state.valid)):
        raise RuntimeError("v26.6 mixed-dtype fused FICEM write requires CUDA tensors")
    devices = {t.device for t in (*floating, new_valid, state.valid)}
    if len(devices) != 1:
        raise ValueError("FICEM write-tail tensors must share one CUDA device")
    if not all(t.is_contiguous() for t in (*floating, new_valid, state.valid)):
        raise ValueError("v26.6 mixed-dtype fused FICEM write requires contiguous tensors")
    return batch


_write_materialize_cast_kernel = None

if triton is not None:

    @triton.jit
    def _write_materialize_cast_kernel(
        source_map_ptr,
        new_keys_ptr,
        new_values_ptr,
        new_strengths_ptr,
        old_keys_ptr,
        old_values_ptr,
        old_strengths_ptr,
        out_keys_ptr,
        out_values_ptr,
        out_strengths_ptr,
        K: tl.constexpr,
        CAPACITY: tl.constexpr,
        MEMORY_DIM: tl.constexpr,
        DIM_BLOCK: tl.constexpr,
    ):
        """Materialize retained sources after durable-type branch unification."""

        output_row = tl.program_id(0)
        batch_row = output_row // CAPACITY
        output_slot = output_row - batch_row * CAPACITY

        encoded_source = tl.load(
            source_map_ptr + batch_row * CAPACITY + output_slot
        ).to(tl.int32)
        from_new = encoded_source < K
        safe_new = tl.where(from_new, encoded_source, 0)
        safe_old = tl.where(from_new, 0, encoded_source - K)

        dim = tl.arange(0, DIM_BLOCK)
        dim_mask = dim < MEMORY_DIM

        new_base = (batch_row * K + safe_new) * MEMORY_DIM
        old_base = (batch_row * CAPACITY + safe_old) * MEMORY_DIM
        new_key = tl.load(
            new_keys_ptr + new_base + dim,
            mask=dim_mask & from_new,
            other=0.0,
        ).to(out_keys_ptr.dtype.element_ty)
        old_key = tl.load(
            old_keys_ptr + old_base + dim,
            mask=dim_mask & ~from_new,
            other=0.0,
        ).to(out_keys_ptr.dtype.element_ty)
        new_value = tl.load(
            new_values_ptr + new_base + dim,
            mask=dim_mask & from_new,
            other=0.0,
        ).to(out_values_ptr.dtype.element_ty)
        old_value = tl.load(
            old_values_ptr + old_base + dim,
            mask=dim_mask & ~from_new,
            other=0.0,
        ).to(out_values_ptr.dtype.element_ty)

        out_base = output_row * MEMORY_DIM
        tl.store(
            out_keys_ptr + out_base + dim,
            tl.where(from_new, new_key, old_key),
            mask=dim_mask,
        )
        tl.store(
            out_values_ptr + out_base + dim,
            tl.where(from_new, new_value, old_value),
            mask=dim_mask,
        )

        new_strength = tl.load(
            new_strengths_ptr + batch_row * K + safe_new,
            mask=from_new,
            other=0.0,
        ).to(out_strengths_ptr.dtype.element_ty)
        old_strength = tl.load(
            old_strengths_ptr + batch_row * CAPACITY + safe_old,
            mask=~from_new,
            other=0.0,
        ).to(out_strengths_ptr.dtype.element_ty)
        tl.store(
            out_strengths_ptr + output_row,
            tl.where(from_new, new_strength, old_strength),
        )


def fused_ficem_write_tail_materialize_cast(
    incoming_similarity: torch.Tensor,
    old_similarity: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    new_strengths: torch.Tensor,
    new_valid: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> ContextualEpisodicMemoryState:
    """Run unchanged adjudication plus the v26.6 durable-type materializer."""

    if triton is None or _write_adjudicate_map_kernel is None or _write_materialize_cast_kernel is None:
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
    _write_materialize_cast_kernel[(batch * WRITE_CAPACITY,)](
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


class MaterializeCastTritonFICEMReadWriteBackend(TritonFICEMReadWriteBackend):
    """Repair5 READ + unchanged adjudication + v26.6 WRITE materialization."""

    name = "triton-fused-ficem-read-repair5-write-tail-v26.6-materialize-cast"

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
            raise ValueError("v26.6 projected write keys must be [batch,16,50]")
        if payload.shape[:2] != new_keys.shape[:2] or payload.ndim != 3:
            raise ValueError("v26.6 payload must match projected write batch/K axes")
        if strength.shape != (*new_keys.shape[:-1], 1):
            raise ValueError("v26.6 write strength must be [batch,16,1]")
        if normalized_old.shape != state.keys.shape:
            raise ValueError("v26.6 normalized old keys must match old state keys")

        # Exact v26.4/v26.5 learned math and duplicate decisions.  No materialized
        # field conversion is allowed before the two-kernel tail.
        new_values = torch.tanh(memory.v(payload))
        new_strengths = strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0
        incoming_similarity = torch.einsum("bkd,bjd->bkj", new_keys, new_keys)
        old_similarity = torch.einsum("bkd,bsd->bks", new_keys, normalized_old)

        next_state = fused_ficem_write_tail_materialize_cast(
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


def materialize_cast_ficem_read_write_v26_6_protocol() -> dict[str, Any]:
    protocol = dict(fused_ficem_read_write_v26_4_protocol())
    protocol.update(
        {
            "version": "aera-v26.6-triton-ficem-write-materialize-cast",
            "research_issue": RESEARCH_ISSUE,
            "source_main": SOURCE_MAIN,
            "backend_name": MaterializeCastTritonFICEMReadWriteBackend.name,
            "predecessor_write_backend_blob": PREDECESSOR_WRITE_BACKEND_BLOB,
            "failed_v26_5_backend_blob": FAILED_V26_5_BACKEND_BLOB,
            "issue514_probe_blob": ISSUE514_PROBE_BLOB,
            "issue514_result_sha256": ISSUE514_RESULT_SHA256,
            "issue514_run": ISSUE514_RUN,
            "issue514_job": ISSUE514_JOB,
            "historical_v26_4_backend_mutated": False,
            "failed_v26_5_backend_mutated": False,
            "write_global_cross_field_dtype_equality_required": False,
            "write_supported_float_dtypes": ["float32", "bfloat16"],
            "write_fieldwise_mixed_dtype_supported": True,
            "write_materialization_output_follows_durable_state_field_dtype": True,
            "write_duplicate_decisions_before_materialization": True,
            "write_materialize_both_branches_cast_to_output_element_type": True,
            "write_materialize_cast_numeric_not_bitcast": True,
            "write_explicit_pre_tail_cast_kernels": 0,
            "write_new_triton_kernels": 1,
            "write_adjudicate_kernel_changed_by_v26_6": False,
            "write_materialize_kernel_versioned_by_v26_6": True,
            "write_tail_triton_launches_target": 2,
            "read_backend_changed_by_v26_6": False,
            "write_similarity_einsums_changed_by_v26_6": False,
            "write_value_projection_changed_by_v26_6": False,
            "write_strength_semantics_changed_by_v26_6": False,
            "write_duplicate_semantics_changed_by_v26_6": False,
            "write_state_schema_changed_by_v26_6": False,
            "write_persistent_state_changed_by_v26_6": False,
            "write_training_backend_changed_by_v26_6": False,
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
