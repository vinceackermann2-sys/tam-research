from __future__ import annotations

"""AERA-v26.2 fused zero-pack state transport for the issue #408 primitive gate.

This module implements the v26.1 semantic ``select``/``merge`` transport boundary
with two direct Triton kernels.  It is inference/systems infrastructure only: it
adds no learned parameters, persistent session tensors, caches, model equations or
training path.  The issue #408 benchmark is the sole GPU-authorized caller until a
later boundary explicitly integrates the backend into end-to-end AERA execution.
"""

from typing import Any

import torch

from .aera import AERAState
from .aera_hardware_core_v24 import ContextualEpisodicMemoryState, _as_epi
from .aera_hardware_core_v26_1 import SparseStageStateTransportBackend

try:  # CPU CI intentionally does not require Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU environments without Triton.
    triton = None
    tl = None


_fused_select_kernel = None
_fused_merge_kernel = None

if triton is not None:

    @triton.jit
    def _fused_select_kernel(
        stream_ptr,
        keys_ptr,
        values_ptr,
        strengths_ptr,
        valid_ptr,
        run_idx_ptr,
        out_stream_ptr,
        out_keys_ptr,
        out_values_ptr,
        out_strengths_ptr,
        out_valid_ptr,
        STREAM_WIDTH: tl.constexpr,
        KV_WIDTH: tl.constexpr,
        CAPACITY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        selected_row = tl.program_id(0)
        source_row = tl.load(run_idx_ptr + selected_row)
        offsets = tl.arange(0, BLOCK_SIZE)

        stream_mask = offsets < STREAM_WIDTH
        stream_value = tl.load(
            stream_ptr + source_row * STREAM_WIDTH + offsets,
            mask=stream_mask,
            other=0.0,
        )
        tl.store(
            out_stream_ptr + selected_row * STREAM_WIDTH + offsets,
            stream_value,
            mask=stream_mask,
        )

        kv_mask = offsets < KV_WIDTH
        key_value = tl.load(
            keys_ptr + source_row * KV_WIDTH + offsets,
            mask=kv_mask,
            other=0.0,
        )
        value_value = tl.load(
            values_ptr + source_row * KV_WIDTH + offsets,
            mask=kv_mask,
            other=0.0,
        )
        tl.store(
            out_keys_ptr + selected_row * KV_WIDTH + offsets,
            key_value,
            mask=kv_mask,
        )
        tl.store(
            out_values_ptr + selected_row * KV_WIDTH + offsets,
            value_value,
            mask=kv_mask,
        )

        capacity_mask = offsets < CAPACITY
        strength_value = tl.load(
            strengths_ptr + source_row * CAPACITY + offsets,
            mask=capacity_mask,
            other=0.0,
        )
        valid_value = tl.load(
            valid_ptr + source_row * CAPACITY + offsets,
            mask=capacity_mask,
            other=0,
        )
        tl.store(
            out_strengths_ptr + selected_row * CAPACITY + offsets,
            strength_value,
            mask=capacity_mask,
        )
        tl.store(
            out_valid_ptr + selected_row * CAPACITY + offsets,
            valid_value,
            mask=capacity_mask,
        )


    @triton.jit
    def _fused_merge_kernel(
        base_stream_ptr,
        base_keys_ptr,
        base_values_ptr,
        base_strengths_ptr,
        base_valid_ptr,
        update_stream_ptr,
        update_keys_ptr,
        update_values_ptr,
        update_strengths_ptr,
        update_valid_ptr,
        run_idx_ptr,
        out_stream_ptr,
        out_keys_ptr,
        out_values_ptr,
        out_strengths_ptr,
        out_valid_ptr,
        n_selected,
        STREAM_WIDTH: tl.constexpr,
        KV_WIDTH: tl.constexpr,
        CAPACITY: tl.constexpr,
        INDEX_BLOCK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        base_row = tl.program_id(0)

        index_offsets = tl.arange(0, INDEX_BLOCK)
        selected_ids = tl.load(
            run_idx_ptr + index_offsets,
            mask=index_offsets < n_selected,
            other=-1,
        )
        encoded_position = tl.where(
            selected_ids == base_row,
            index_offsets + 1,
            0,
        )
        selected_position = tl.max(encoded_position, axis=0) - 1
        use_update = selected_position >= 0

        offsets = tl.arange(0, BLOCK_SIZE)

        stream_mask = offsets < STREAM_WIDTH
        base_stream = tl.load(
            base_stream_ptr + base_row * STREAM_WIDTH + offsets,
            mask=stream_mask,
            other=0.0,
        )
        update_stream = tl.load(
            update_stream_ptr + selected_position * STREAM_WIDTH + offsets,
            mask=stream_mask & use_update,
            other=0.0,
        )
        tl.store(
            out_stream_ptr + base_row * STREAM_WIDTH + offsets,
            tl.where(use_update, update_stream, base_stream),
            mask=stream_mask,
        )

        kv_mask = offsets < KV_WIDTH
        base_keys = tl.load(
            base_keys_ptr + base_row * KV_WIDTH + offsets,
            mask=kv_mask,
            other=0.0,
        )
        base_values = tl.load(
            base_values_ptr + base_row * KV_WIDTH + offsets,
            mask=kv_mask,
            other=0.0,
        )
        update_keys = tl.load(
            update_keys_ptr + selected_position * KV_WIDTH + offsets,
            mask=kv_mask & use_update,
            other=0.0,
        )
        update_values = tl.load(
            update_values_ptr + selected_position * KV_WIDTH + offsets,
            mask=kv_mask & use_update,
            other=0.0,
        )
        tl.store(
            out_keys_ptr + base_row * KV_WIDTH + offsets,
            tl.where(use_update, update_keys, base_keys),
            mask=kv_mask,
        )
        tl.store(
            out_values_ptr + base_row * KV_WIDTH + offsets,
            tl.where(use_update, update_values, base_values),
            mask=kv_mask,
        )

        capacity_mask = offsets < CAPACITY
        base_strengths = tl.load(
            base_strengths_ptr + base_row * CAPACITY + offsets,
            mask=capacity_mask,
            other=0.0,
        )
        base_valid = tl.load(
            base_valid_ptr + base_row * CAPACITY + offsets,
            mask=capacity_mask,
            other=0,
        )
        update_strengths = tl.load(
            update_strengths_ptr + selected_position * CAPACITY + offsets,
            mask=capacity_mask & use_update,
            other=0.0,
        )
        update_valid = tl.load(
            update_valid_ptr + selected_position * CAPACITY + offsets,
            mask=capacity_mask & use_update,
            other=0,
        )
        tl.store(
            out_strengths_ptr + base_row * CAPACITY + offsets,
            tl.where(use_update, update_strengths, base_strengths),
            mask=capacity_mask,
        )
        tl.store(
            out_valid_ptr + base_row * CAPACITY + offsets,
            tl.where(use_update, update_valid, base_valid),
            mask=capacity_mask,
        )


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def triton_transport_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _validate_semantic_state(state: AERAState, *, require_cuda: bool) -> tuple[ContextualEpisodicMemoryState, int, int, int]:
    memory = _as_epi(state.memory)
    if state.stream.ndim != 2 or memory.keys.ndim != 3:
        raise ValueError("state transport expects stream [B,D] and episodic keys [B,C,M]")
    if memory.values.shape != memory.keys.shape:
        raise ValueError("episodic key/value geometry mismatch")
    if memory.strengths.shape != memory.valid.shape:
        raise ValueError("episodic strength/valid geometry mismatch")
    if memory.keys.shape[:2] != memory.valid.shape:
        raise ValueError("episodic capacity geometry mismatch")
    batch = int(state.stream.size(0))
    if memory.keys.size(0) != batch:
        raise ValueError("stream/memory batch mismatch")
    if not (
        state.stream.dtype == memory.keys.dtype
        == memory.values.dtype
        == memory.strengths.dtype
    ):
        raise ValueError("floating state dtypes must match")
    if state.stream.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("fused transport supports fp32/bf16/fp16 floating state")
    if memory.valid.dtype is not torch.bool:
        raise TypeError("episodic validity must be boolean")
    tensors = (state.stream, memory.keys, memory.values, memory.strengths, memory.valid)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all state tensors must share a device")
    if require_cuda and state.stream.device.type != "cuda":
        raise RuntimeError("Triton fused transport requires CUDA state tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Triton fused transport requires contiguous semantic state tensors")
    return memory, batch, int(state.stream.size(1)), int(memory.keys.size(2))


def _allocate_state_like(state: AERAState, batch_size: int) -> AERAState:
    memory, _, d_model, memory_dim = _validate_semantic_state(state, require_cuda=False)
    capacity = int(memory.keys.size(1))
    device = state.stream.device
    dtype = state.stream.dtype
    return AERAState(
        stream=torch.empty((batch_size, d_model), device=device, dtype=dtype),
        memory=ContextualEpisodicMemoryState(
            keys=torch.empty((batch_size, capacity, memory_dim), device=device, dtype=dtype),
            values=torch.empty((batch_size, capacity, memory_dim), device=device, dtype=dtype),
            strengths=torch.empty((batch_size, capacity), device=device, dtype=dtype),
            valid=torch.empty((batch_size, capacity), device=device, dtype=torch.bool),
        ),
    )


class TritonFusedStateTransport(SparseStageStateTransportBackend):
    """Two-kernel functional transport over the five separate semantic tensors."""

    name = "triton-fused-zero-pack-v26.2"

    def __init__(self, *, max_batch: int = 64) -> None:
        if triton is None:
            raise RuntimeError("Triton is not installed; fused transport is unavailable")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; fused transport is unavailable")
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.max_batch = int(max_batch)
        self.index_block = _next_power_of_two(self.max_batch)

    def _validate_index(self, run_idx: torch.Tensor, *, batch_size: int) -> int:
        if run_idx.ndim != 1 or run_idx.dtype != torch.long:
            raise TypeError("run_idx must be rank-1 torch.long")
        if run_idx.device.type != "cuda" or not run_idx.is_contiguous():
            raise ValueError("run_idx must be a contiguous CUDA tensor")
        selected = int(run_idx.numel())
        if batch_size > self.max_batch:
            raise ValueError("batch exceeds fused transport max_batch")
        if selected < 1 or selected > batch_size:
            raise ValueError("fused transport requires 1 <= selected <= batch")
        return selected

    def select(self, state: AERAState, run_idx: torch.Tensor) -> AERAState:
        memory, batch, d_model, memory_dim = _validate_semantic_state(
            state, require_cuda=True
        )
        selected = self._validate_index(run_idx, batch_size=batch)
        capacity = int(memory.keys.size(1))
        kv_width = capacity * memory_dim
        block_size = _next_power_of_two(max(d_model, kv_width, capacity))
        num_warps = 8 if block_size >= 2048 else 4
        output = _allocate_state_like(state, selected)
        out_memory = _as_epi(output.memory)
        if _fused_select_kernel is None:
            raise RuntimeError("Triton select kernel is unavailable")
        _fused_select_kernel[(selected,)](
            state.stream,
            memory.keys,
            memory.values,
            memory.strengths,
            memory.valid,
            run_idx,
            output.stream,
            out_memory.keys,
            out_memory.values,
            out_memory.strengths,
            out_memory.valid,
            STREAM_WIDTH=d_model,
            KV_WIDTH=kv_width,
            CAPACITY=capacity,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output

    def merge(
        self,
        base_state: AERAState,
        selected_update: AERAState,
        run_idx: torch.Tensor,
    ) -> AERAState:
        base_memory, batch, d_model, memory_dim = _validate_semantic_state(
            base_state, require_cuda=True
        )
        update_memory, update_batch, update_d_model, update_memory_dim = (
            _validate_semantic_state(selected_update, require_cuda=True)
        )
        selected = self._validate_index(run_idx, batch_size=batch)
        if update_batch != selected:
            raise ValueError("selected update batch must match run_idx length")
        if update_d_model != d_model or update_memory_dim != memory_dim:
            raise ValueError("selected update geometry differs from base state")
        capacity = int(base_memory.keys.size(1))
        if int(update_memory.keys.size(1)) != capacity:
            raise ValueError("selected update capacity differs from base state")
        if selected_update.stream.dtype != base_state.stream.dtype:
            raise TypeError("selected update dtype differs from base state")

        kv_width = capacity * memory_dim
        block_size = _next_power_of_two(max(d_model, kv_width, capacity))
        num_warps = 8 if block_size >= 2048 else 4
        output = _allocate_state_like(base_state, batch)
        out_memory = _as_epi(output.memory)
        if _fused_merge_kernel is None:
            raise RuntimeError("Triton merge kernel is unavailable")
        _fused_merge_kernel[(batch,)](
            base_state.stream,
            base_memory.keys,
            base_memory.values,
            base_memory.strengths,
            base_memory.valid,
            selected_update.stream,
            update_memory.keys,
            update_memory.values,
            update_memory.strengths,
            update_memory.valid,
            run_idx,
            output.stream,
            out_memory.keys,
            out_memory.values,
            out_memory.strengths,
            out_memory.valid,
            selected,
            STREAM_WIDTH=d_model,
            KV_WIDTH=kv_width,
            CAPACITY=capacity,
            INDEX_BLOCK=self.index_block,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output


def fused_triton_transport_v26_2_protocol() -> dict[str, Any]:
    return {
        "version": "aera-v26.2-triton-fused-zero-pack-transport",
        "research_issue": 408,
        "source_main": "27fc272e495bc3c125f7a1786c09581557670b3d",
        "source_transport_interface": "aera-v26.1-zero-pack-sparse-state-transport-cpu-reference",
        "backend_name": TritonFusedStateTransport.name,
        "semantic_state_tensors": [
            "stream",
            "episodic_keys",
            "episodic_values",
            "episodic_strengths",
            "episodic_valid",
        ],
        "select_triton_launches_target": 1,
        "merge_triton_launches_target": 1,
        "auxiliary_row_map_kernel": False,
        "persistent_pack_state": False,
        "persistent_cache": False,
        "training_backend_authorized": False,
        "model_integration_authorized": False,
        "gpu_execution_authorized_only_by_issue408_gate": True,
        "scientific_training_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
