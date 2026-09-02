from __future__ import annotations

"""AERA-v26.4 inference-only fused FICEM write tail.

The repair5 READ backend is inherited unchanged. For writes, address projection,
old-key normalization, learned value projection/tanh, strength clamp, and both
similarity einsums remain the exact PyTorch operations from the stable-compaction
reference. Only the post-similarity duplicate/newest-wins adjudication and bounded
state rebuild are replaced by two fixed-geometry Triton kernels.

Preregistered by issue #485 after the authoritative repair5 READ primitive PASS.
This module authorizes no GPU experiment by itself; GPU validation requires a
separate post-merge preregistration.
"""

from typing import Any

import torch
import torch.nn.functional as F

from .aera_hardware_core_v24 import (
    DUPLICATE_SIMILARITY,
    ContextualEpisodicMemoryState,
)
from .aera_hardware_core_v25_1 import _set_known_empty_hint
from .aera_hardware_core_v26 import CoalescedFICEMMemory
from .aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
    fused_ficem_read_v26_3_protocol,
    triton,
    tl,
)

WRITE_COUNT = 16
WRITE_CAPACITY = 48
WRITE_MEMORY_DIM = 50
WRITE_DUPLICATE_SIMILARITY = 0.95
WRITE_SOURCE_COUNT = WRITE_COUNT + WRITE_CAPACITY

_write_adjudicate_map_kernel = None
_write_materialize_kernel = None

if triton is not None:

    @triton.jit
    def _write_adjudicate_map_kernel(
        incoming_similarity_ptr,
        old_similarity_ptr,
        new_valid_ptr,
        old_valid_ptr,
        source_map_ptr,
        durable_valid_ptr,
        K: tl.constexpr,
        CAPACITY: tl.constexpr,
        K_BLOCK: tl.constexpr,
        CAPACITY_BLOCK: tl.constexpr,
        DUPLICATE_THRESHOLD: tl.constexpr,
    ):
        """Compute exact newest-wins survivors and stable-compaction map."""
        batch_row = tl.program_id(0)

        incoming = tl.arange(0, K_BLOCK)
        other_incoming = tl.arange(0, K_BLOCK)
        incoming_valid = tl.load(new_valid_ptr + batch_row * K + incoming) != 0
        other_valid = tl.load(new_valid_ptr + batch_row * K + other_incoming) != 0

        incoming_similarity = tl.load(
            incoming_similarity_ptr
            + batch_row * K * K
            + incoming[:, None] * K
            + other_incoming[None, :]
        )
        # Reference direction: position i is shadowed by a valid later position j.
        later = other_incoming[None, :] > incoming[:, None]
        shadowed = (
            tl.sum(
                (
                    (incoming_similarity >= DUPLICATE_THRESHOLD)
                    & incoming_valid[:, None]
                    & other_valid[None, :]
                    & later
                ).to(tl.int32),
                axis=1,
            )
            > 0
        )
        surviving_new = incoming_valid & ~shadowed

        old_slot = tl.arange(0, CAPACITY_BLOCK)
        old_slot_mask = old_slot < CAPACITY
        old_valid = (
            tl.load(
                old_valid_ptr + batch_row * CAPACITY + old_slot,
                mask=old_slot_mask,
                other=0,
            )
            != 0
        ) & old_slot_mask

        new_old_similarity = tl.load(
            old_similarity_ptr
            + batch_row * K * CAPACITY
            + incoming[:, None] * CAPACITY
            + old_slot[None, :],
            mask=old_slot_mask[None, :],
            other=-float("inf"),
        )
        duplicate_old = (
            tl.sum(
                (
                    (new_old_similarity >= DUPLICATE_THRESHOLD)
                    & surviving_new[:, None]
                    & old_valid[None, :]
                ).to(tl.int32),
                axis=0,
            )
            > 0
        )
        keep_old = old_valid & ~duplicate_old

        # New writes are conceptually flipped before stable compaction. Instead of
        # physically flipping them, compute each original incoming write's rank in
        # that reversed source order. Higher original indices are newer/earlier.
        source_new_order = other_incoming[None, :] >= incoming[:, None]
        new_invalid = ~surviving_new
        new_valid_rank = (
            tl.sum(
                (surviving_new[None, :] & source_new_order).to(tl.int32), axis=1
            )
            - 1
        )
        new_invalid_rank = (
            tl.sum((new_invalid[None, :] & source_new_order).to(tl.int32), axis=1)
            - 1
        )

        new_valid_count = tl.sum(surviving_new.to(tl.int32), axis=0)
        old_valid_count = tl.sum(keep_old.to(tl.int32), axis=0)
        total_valid = new_valid_count + old_valid_count
        new_invalid_count = K - new_valid_count

        new_destination = tl.where(
            surviving_new,
            new_valid_rank,
            total_valid + new_invalid_rank,
        )
        # Encoded 0..K-1 denotes the original incoming index. Materialization uses
        # this exact original index; reversed ordering is represented by destination.
        tl.store(
            source_map_ptr + batch_row * CAPACITY + new_destination,
            incoming.to(tl.int32),
            mask=new_destination < CAPACITY,
        )

        # Stable old-source prefix ranks. This is the exact valid-first/invalid-next
        # stable partition after all reversed incoming writes.
        old_other = tl.arange(0, CAPACITY_BLOCK)
        prefix = old_other[None, :] <= old_slot[:, None]
        old_valid_prefix = (
            tl.sum((keep_old[None, :] & prefix).to(tl.int32), axis=1) - 1
        )
        old_invalid = old_slot_mask & ~keep_old
        old_invalid_prefix = (
            tl.sum((old_invalid[None, :] & prefix).to(tl.int32), axis=1) - 1
        )
        old_destination = tl.where(
            keep_old,
            new_valid_count + old_valid_prefix,
            total_valid + new_invalid_count + old_invalid_prefix,
        )
        tl.store(
            source_map_ptr + batch_row * CAPACITY + old_destination,
            (K + old_slot).to(tl.int32),
            mask=old_slot_mask & (old_destination < CAPACITY),
        )

        output_slot = tl.arange(0, CAPACITY_BLOCK)
        output_mask = output_slot < CAPACITY
        durable_valid = output_slot < tl.minimum(total_valid, CAPACITY)
        tl.store(
            durable_valid_ptr + batch_row * CAPACITY + output_slot,
            durable_valid,
            mask=output_mask,
        )

    @triton.jit
    def _write_materialize_kernel(
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
        """Materialize exact retained source storage into the durable 48 slots."""
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
        )
        old_key = tl.load(
            old_keys_ptr + old_base + dim,
            mask=dim_mask & ~from_new,
            other=0.0,
        )
        new_value = tl.load(
            new_values_ptr + new_base + dim,
            mask=dim_mask & from_new,
            other=0.0,
        )
        old_value = tl.load(
            old_values_ptr + old_base + dim,
            mask=dim_mask & ~from_new,
            other=0.0,
        )

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
        )
        old_strength = tl.load(
            old_strengths_ptr + batch_row * CAPACITY + safe_old,
            mask=~from_new,
            other=0.0,
        )
        tl.store(
            out_strengths_ptr + output_row,
            tl.where(from_new, new_strength, old_strength),
        )


def triton_ficem_write_available() -> bool:
    return (
        triton is not None
        and _write_adjudicate_map_kernel is not None
        and _write_materialize_kernel is not None
        and torch.cuda.is_available()
    )


def _validate_write_tail_inputs(
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
        raise ValueError("v26.4 fused write is frozen to K16/memory_dim50")
    if state.keys.shape != state.values.shape:
        raise ValueError("FICEM old key/value state mismatch")
    if state.keys.shape != (batch, WRITE_CAPACITY, WRITE_MEMORY_DIM):
        raise ValueError("v26.4 fused write is frozen to capacity48/memory_dim50")
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
    if any(t.device.type != "cuda" for t in (*floating, new_valid, state.valid)):
        raise RuntimeError("v26.4 fused FICEM write requires CUDA tensors")
    devices = {t.device for t in (*floating, new_valid, state.valid)}
    if len(devices) != 1:
        raise ValueError("FICEM write-tail tensors must share one CUDA device")
    if not all(t.is_contiguous() for t in (*floating, new_valid, state.valid)):
        raise ValueError("v26.4 fused FICEM write requires contiguous tensors")
    if not (
        new_keys.dtype
        == new_values.dtype
        == new_strengths.dtype
        == state.keys.dtype
        == state.values.dtype
        == state.strengths.dtype
    ):
        raise TypeError("FICEM write state/value floating dtypes must match")
    return batch


def fused_ficem_write_tail(
    incoming_similarity: torch.Tensor,
    old_similarity: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    new_strengths: torch.Tensor,
    new_valid: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> ContextualEpisodicMemoryState:
    """Run the two-kernel post-similarity newest-wins stable write tail."""
    if triton is None or _write_adjudicate_map_kernel is None or _write_materialize_kernel is None:
        raise RuntimeError("Triton FICEM write kernels are unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fused FICEM write is unavailable")
    batch = _validate_write_tail_inputs(
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


class TritonFICEMReadWriteBackend(TritonFICEMReadBackend):
    """Repair5 read backend plus v26.4 inference-only stable write fusion."""

    name = "triton-fused-ficem-read-repair5-write-tail-v26.4"

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
            raise ValueError("v26.4 projected write keys must be [batch,16,50]")
        if payload.shape[:2] != new_keys.shape[:2] or payload.ndim != 3:
            raise ValueError("v26.4 payload must match projected write batch/K axes")
        if strength.shape != (*new_keys.shape[:-1], 1):
            raise ValueError("v26.4 write strength must be [batch,16,1]")
        if normalized_old.shape != state.keys.shape:
            raise ValueError("v26.4 normalized old keys must match old state keys")

        # Intentionally unchanged PyTorch value/strength and similarity semantics.
        new_values = torch.tanh(memory.v(payload))
        new_strengths = strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0
        incoming_similarity = torch.einsum("bkd,bjd->bkj", new_keys, new_keys)
        old_similarity = torch.einsum("bkd,bsd->bks", new_keys, normalized_old)

        next_state = fused_ficem_write_tail(
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

    def update_from_projected(
        self,
        memory: CoalescedFICEMMemory,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        if torch.is_grad_enabled() or memory.differentiable_pretraining:
            return self._reference.update_from_projected(
                memory,
                projected_new_keys,
                normalized_old_keys,
                payload_source,
                write_strength,
                state,
            )
        if projected_new_keys.device.type != "cuda":
            raise RuntimeError("v26.4 fused FICEM write requires CUDA inference tensors")
        return self._inference_update_from_projected(
            memory,
            projected_new_keys,
            normalized_old_keys,
            payload_source,
            write_strength,
            state,
        )

    def update(
        self,
        memory: CoalescedFICEMMemory,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        if torch.is_grad_enabled() or memory.differentiable_pretraining:
            return self._reference.update(
                memory,
                identity_source,
                context_source,
                payload_source,
                write_strength,
                state,
            )
        if identity_source.device.type != "cuda":
            raise RuntimeError("v26.4 fused FICEM write requires CUDA inference tensors")
        identity = identity_source.detach()
        context = context_source.detach()
        _, _, new_keys = memory.address_factors(identity, context)
        normalized_old = F.normalize(state.keys.detach(), dim=-1)
        return self._inference_update_from_projected(
            memory,
            new_keys,
            normalized_old,
            payload_source,
            write_strength,
            state,
        )


def fused_ficem_read_write_v26_4_protocol() -> dict[str, Any]:
    read_protocol = fused_ficem_read_v26_3_protocol()
    return {
        **read_protocol,
        "version": "aera-v26.4-triton-fused-ficem-write-tail",
        "research_issue": 485,
        "backend_name": TritonFICEMReadWriteBackend.name,
        "source_main": "8ab27e55270a4d0ff26e9d21674c58ac3a4ce612",
        "repair5_read_backend_blob": "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
        "v26_interface_blob": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
        "stable_compaction_reference_blob": "4e336b6e1a6238dac782fa320751d68281493ee1",
        "write_count": WRITE_COUNT,
        "capacity": WRITE_CAPACITY,
        "memory_dim": WRITE_MEMORY_DIM,
        "duplicate_similarity": WRITE_DUPLICATE_SIMILARITY,
        "write_tail_triton_launches_target": 2,
        "read_backend_changed_by_v26_4": False,
        "write_similarity_einsums_changed": False,
        "write_value_projection_changed": False,
        "write_strength_semantics_changed": False,
        "write_duplicate_semantics_changed": False,
        "write_incoming_order_changed": False,
        "write_stable_compaction_semantics_changed": False,
        "write_invalid_storage_semantics_changed": False,
        "write_training_backend_changed": False,
        "write_persistent_state_changed": False,
        "write_persistent_cache": False,
        "write_gpu_gate_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
