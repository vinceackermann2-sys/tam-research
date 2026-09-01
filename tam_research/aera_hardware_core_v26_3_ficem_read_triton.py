from __future__ import annotations

"""AERA-v26.3 inference-only fused FICEM read tail.

The learned identity/context projections, key normalization, similarity einsum and
learned output projection remain the exact merged PyTorch operations. Only the
nonempty post-similarity top-4 read tail is replaced by one direct Triton kernel.
Writes and every differentiable/training path delegate to the final v26 reference.

Issue #414 made the frozen read constants explicit Triton constexpr parameters.
Issue #426 keeps the same one-kernel architecture and adds a BF16-only execution
branch whose visible rounding checkpoints follow the frozen PyTorch diagnostic
reference localized by #423. The float32 execution path remains unchanged.
"""

from typing import Any

import torch
import torch.nn.functional as F

from .aera_hardware_core_v24 import (
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
    ContextualEpisodicMemoryState,
)
from .aera_hardware_core_v25_1 import _known_empty_hint
from .aera_hardware_core_v26 import (
    CoalescedFICEMMemory,
    FICEMReadPrimitive,
    TorchFICEMReferenceBackend,
)

try:  # CPU CI intentionally does not require Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - expected on CPU-only CI images.
    triton = None
    tl = None


_ficem_read_tail_kernel = None

if triton is not None:

    @triton.jit
    def _ficem_read_tail_kernel(
        similarity_ptr,
        strengths_ptr,
        valid_ptr,
        values_ptr,
        recalled_ptr,
        top_indices_ptr,
        TIME: tl.constexpr,
        CAPACITY: tl.constexpr,
        MEMORY_DIM: tl.constexpr,
        SLOT_BLOCK: tl.constexpr,
        DIM_BLOCK: tl.constexpr,
        WRITE_INDICES: tl.constexpr,
        IS_BF16: tl.constexpr,
        MIN_STRENGTH: tl.constexpr,
        READ_TEMPERATURE: tl.constexpr,
        READ_TOP_K: tl.constexpr,
    ):
        query_row = tl.program_id(0)
        batch_row = query_row // TIME

        slot_offsets = tl.arange(0, SLOT_BLOCK)
        slot_mask = slot_offsets < CAPACITY
        similarity = tl.load(
            similarity_ptr + query_row * CAPACITY + slot_offsets,
            mask=slot_mask,
            other=-float("inf"),
        )
        strengths = tl.load(
            strengths_ptr + batch_row * CAPACITY + slot_offsets,
            mask=slot_mask,
            other=MIN_STRENGTH,
        )
        valid = tl.load(
            valid_ptr + batch_row * CAPACITY + slot_offsets,
            mask=slot_mask,
            other=0,
        )

        # Historical #411/#414 source-contract anchor: exact two-sided clamp.
        clamped_strengths = tl.minimum(tl.maximum(strengths, MIN_STRENGTH), 1.0)

        if IS_BF16:
            # #423 localized two distinct-boundary mismatches to the region where
            # the old candidate promoted BF16 source values before logit creation.
            # Evaluate transcendental math with sufficient internal precision, then
            # expose the same BF16-visible checkpoints as the frozen reference.
            similarity_visible = similarity.to(tl.bfloat16)
            clamped_visible = clamped_strengths.to(tl.bfloat16)
            strength_bias = tl.log(clamped_visible.to(tl.float32)).to(tl.bfloat16)
            logits = (
                (similarity_visible + strength_bias).to(tl.bfloat16)
                / READ_TEMPERATURE
            ).to(tl.bfloat16)
        else:
            # Frozen pre-#426 float32/float16 execution behavior: promote the
            # candidate arithmetic to FP32 before logit construction.
            similarity_visible = similarity.to(tl.float32)
            clamped_visible = clamped_strengths.to(tl.float32)
            strength_bias = tl.log(clamped_visible)
            logits = (similarity_visible + strength_bias) / READ_TEMPERATURE

        logits = tl.where(slot_mask & valid, logits, -float("inf"))

        index0 = tl.argmax(logits, axis=0, tie_break_left=True)
        logit0 = tl.max(logits, axis=0)
        remaining1 = tl.where(slot_offsets == index0, -float("inf"), logits)
        index1 = tl.argmax(remaining1, axis=0, tie_break_left=True)
        logit1 = tl.max(remaining1, axis=0)
        remaining2 = tl.where(slot_offsets == index1, -float("inf"), remaining1)
        index2 = tl.argmax(remaining2, axis=0, tie_break_left=True)
        logit2 = tl.max(remaining2, axis=0)
        remaining3 = tl.where(slot_offsets == index2, -float("inf"), remaining2)
        index3 = tl.argmax(remaining3, axis=0, tie_break_left=True)
        logit3 = tl.max(remaining3, axis=0)

        valid0 = tl.load(valid_ptr + batch_row * CAPACITY + index0)
        valid1 = tl.load(valid_ptr + batch_row * CAPACITY + index1)
        valid2 = tl.load(valid_ptr + batch_row * CAPACITY + index2)
        valid3 = tl.load(valid_ptr + batch_row * CAPACITY + index3)

        if IS_BF16:
            # Reference boundary: selected BF16 logits become FP32 only for softmax.
            safe0 = tl.where(valid0, logit0, -1.0e9).to(tl.bfloat16).to(tl.float32)
            safe1 = tl.where(valid1, logit1, -1.0e9).to(tl.bfloat16).to(tl.float32)
            safe2 = tl.where(valid2, logit2, -1.0e9).to(tl.bfloat16).to(tl.float32)
            safe3 = tl.where(valid3, logit3, -1.0e9).to(tl.bfloat16).to(tl.float32)
        else:
            safe0 = tl.where(valid0, logit0, -1.0e9).to(tl.float32)
            safe1 = tl.where(valid1, logit1, -1.0e9).to(tl.float32)
            safe2 = tl.where(valid2, logit2, -1.0e9).to(tl.float32)
            safe3 = tl.where(valid3, logit3, -1.0e9).to(tl.float32)

        maximum = tl.maximum(tl.maximum(safe0, safe1), tl.maximum(safe2, safe3))
        exp0 = tl.exp(safe0 - maximum)
        exp1 = tl.exp(safe1 - maximum)
        exp2 = tl.exp(safe2 - maximum)
        exp3 = tl.exp(safe3 - maximum)
        softmax_sum = exp0 + exp1 + exp2 + exp3

        weight0 = tl.where(valid0, exp0 / softmax_sum, 0.0)
        weight1 = tl.where(valid1, exp1 / softmax_sum, 0.0)
        weight2 = tl.where(valid2, exp2 / softmax_sum, 0.0)
        weight3 = tl.where(valid3, exp3 / softmax_sum, 0.0)

        if IS_BF16:
            # Frozen reference performs softmax in FP32 then returns weights to the
            # source dtype before validity masking / renormalization. Keep those
            # BF16-visible values, accumulate their sum in FP32, round the visible
            # denominator back to BF16, and expose BF16 final weights.
            weight0_visible = weight0.to(tl.bfloat16)
            weight1_visible = weight1.to(tl.bfloat16)
            weight2_visible = weight2.to(tl.bfloat16)
            weight3_visible = weight3.to(tl.bfloat16)
            valid_weight_sum = (
                weight0_visible.to(tl.float32)
                + weight1_visible.to(tl.float32)
                + weight2_visible.to(tl.float32)
                + weight3_visible.to(tl.float32)
            ).to(tl.bfloat16)
            denominator = tl.maximum(
                valid_weight_sum.to(tl.float32), 1.0e-9
            ).to(tl.bfloat16)
            weight0 = (
                weight0_visible.to(tl.float32) / denominator.to(tl.float32)
            ).to(tl.bfloat16)
            weight1 = (
                weight1_visible.to(tl.float32) / denominator.to(tl.float32)
            ).to(tl.bfloat16)
            weight2 = (
                weight2_visible.to(tl.float32) / denominator.to(tl.float32)
            ).to(tl.bfloat16)
            weight3 = (
                weight3_visible.to(tl.float32) / denominator.to(tl.float32)
            ).to(tl.bfloat16)
        else:
            valid_weight_sum = weight0 + weight1 + weight2 + weight3
            denominator = tl.maximum(valid_weight_sum, 1.0e-9)
            weight0 = weight0 / denominator
            weight1 = weight1 / denominator
            weight2 = weight2 / denominator
            weight3 = weight3 / denominator

        dim_offsets = tl.arange(0, DIM_BLOCK)
        dim_mask = dim_offsets < MEMORY_DIM
        base = batch_row * CAPACITY * MEMORY_DIM
        value0 = tl.load(
            values_ptr + base + index0 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        value1 = tl.load(
            values_ptr + base + index1 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        value2 = tl.load(
            values_ptr + base + index2 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        value3 = tl.load(
            values_ptr + base + index3 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )

        # Both branches accumulate in FP32. In the BF16 branch both multiplicands
        # are first constrained to their reference-visible BF16 values.
        recalled = (
            weight0.to(tl.float32) * value0.to(tl.float32)
            + weight1.to(tl.float32) * value1.to(tl.float32)
            + weight2.to(tl.float32) * value2.to(tl.float32)
            + weight3.to(tl.float32) * value3.to(tl.float32)
        )
        tl.store(
            recalled_ptr + query_row * MEMORY_DIM + dim_offsets,
            recalled,
            mask=dim_mask,
        )

        if WRITE_INDICES:
            tl.store(top_indices_ptr + query_row * READ_TOP_K + 0, index0)
            tl.store(top_indices_ptr + query_row * READ_TOP_K + 1, index1)
            tl.store(top_indices_ptr + query_row * READ_TOP_K + 2, index2)
            tl.store(top_indices_ptr + query_row * READ_TOP_K + 3, index3)


def triton_ficem_read_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _validate_read_tail_inputs(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
) -> tuple[int, int, int, int]:
    if similarity.ndim != 3:
        raise ValueError("FICEM similarity must be [batch,time,capacity]")
    if values.ndim != 3:
        raise ValueError("FICEM values must be [batch,capacity,memory_dim]")
    if strengths.ndim != 2 or valid.ndim != 2:
        raise ValueError("FICEM strengths/valid must be [batch,capacity]")
    batch, time, capacity = map(int, similarity.shape)
    if values.shape[:2] != (batch, capacity):
        raise ValueError("FICEM similarity/value geometry mismatch")
    if strengths.shape != (batch, capacity) or valid.shape != (batch, capacity):
        raise ValueError("FICEM strength/valid geometry mismatch")
    memory_dim = int(values.size(2))
    if capacity != 48 or memory_dim != 50:
        raise ValueError("v26.3 fused read is frozen to capacity48/memory_dim50")
    if valid.dtype is not torch.bool:
        raise TypeError("FICEM validity must be boolean")
    if similarity.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("unsupported FICEM similarity dtype")
    if strengths.dtype != values.dtype or similarity.dtype != values.dtype:
        raise TypeError("FICEM read-tail floating dtypes must match")
    tensors = (similarity, strengths, valid, values)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("v26.3 fused FICEM read requires CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("FICEM read-tail tensors must share a device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("v26.3 fused FICEM read requires contiguous tensors")
    return batch, time, capacity, memory_dim


def fused_ficem_read_tail(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
    *,
    return_top_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the one-kernel post-similarity FICEM read tail."""
    if triton is None or _ficem_read_tail_kernel is None:
        raise RuntimeError("Triton FICEM read kernel is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fused FICEM read is unavailable")
    batch, time, capacity, memory_dim = _validate_read_tail_inputs(
        similarity, strengths, valid, values
    )
    recalled = torch.empty(
        (batch, time, memory_dim), device=values.device, dtype=values.dtype
    )
    if return_top_indices:
        top_indices = torch.empty(
            (batch, time, READ_TOP_K), device=values.device, dtype=torch.int32
        )
        index_pointer = top_indices
    else:
        top_indices = None
        # WRITE_INDICES is constexpr false, so the pointer is never dereferenced.
        index_pointer = recalled

    _ficem_read_tail_kernel[(batch * time,)](
        similarity,
        strengths,
        valid,
        values,
        recalled,
        index_pointer,
        TIME=time,
        CAPACITY=capacity,
        MEMORY_DIM=memory_dim,
        SLOT_BLOCK=64,
        DIM_BLOCK=64,
        WRITE_INDICES=return_top_indices,
        IS_BF16=similarity.dtype is torch.bfloat16,
        MIN_STRENGTH=MIN_STRENGTH,
        READ_TEMPERATURE=READ_TEMPERATURE,
        READ_TOP_K=READ_TOP_K,
        num_warps=4,
    )
    return recalled, top_indices


class TritonFICEMReadBackend:
    """Inference read fusion behind the existing v26 FICEM execution interface."""

    name = "triton-fused-ficem-read-tail-v26.3"

    def __init__(self) -> None:
        if triton is None:
            raise RuntimeError("Triton is not installed; fused FICEM read is unavailable")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; fused FICEM read is unavailable")
        self._reference = TorchFICEMReferenceBackend()

    def read(
        self,
        memory: CoalescedFICEMMemory,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> FICEMReadPrimitive:
        if torch.is_grad_enabled() or memory.differentiable_pretraining:
            return self._reference.read(memory, identity_source, context_source, state)
        if identity_source.ndim != 3 or context_source.shape != identity_source.shape:
            raise ValueError("FICEM read sources must match [batch,time,d_model]")
        if identity_source.device.type != "cuda":
            raise RuntimeError("v26.3 fused FICEM read requires CUDA inference tensors")

        if _known_empty_hint(state):
            return FICEMReadPrimitive(
                recalled=torch.zeros(
                    identity_source.size(0),
                    identity_source.size(1),
                    memory.out.out_features,
                    device=identity_source.device,
                    dtype=identity_source.dtype,
                ),
                projected_query=None,
                normalized_old_keys=None,
            )

        _, _, query = memory.address_factors(identity_source, context_source)
        keys = F.normalize(state.keys, dim=-1)
        similarity = torch.einsum("btd,bsd->bts", query, keys)
        recalled, _ = fused_ficem_read_tail(
            similarity.contiguous(),
            state.strengths.contiguous(),
            state.valid.contiguous(),
            state.values.contiguous(),
            return_top_indices=False,
        )
        return FICEMReadPrimitive(
            recalled=memory.out(recalled),
            projected_query=query,
            normalized_old_keys=keys,
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
        return self._reference.update(
            memory,
            identity_source,
            context_source,
            payload_source,
            write_strength,
            state,
        )

    def update_from_projected(
        self,
        memory: CoalescedFICEMMemory,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        return self._reference.update_from_projected(
            memory,
            projected_new_keys,
            normalized_old_keys,
            payload_source,
            write_strength,
            state,
        )


def fused_ficem_read_v26_3_protocol() -> dict[str, Any]:
    return {
        "version": "aera-v26.3-triton-fused-ficem-read-tail",
        "research_issue": 411,
        "source_main": "8227367a9c53cbb2c3ad14be426f4e9d95f46c89",
        "source_ficem_interface": "aera-v26-coalesced-sparse-runtime-cpu-reference",
        "backend_name": TritonFICEMReadBackend.name,
        "capacity": 48,
        "memory_dim": 50,
        "read_top_k": READ_TOP_K,
        "read_temperature": READ_TEMPERATURE,
        "min_strength": MIN_STRENGTH,
        "read_tail_triton_launches_target": 1,
        "bf16_reference_rounding_repair3": True,
        "float32_path_changed_by_repair3": False,
        "address_projection_changed": False,
        "key_normalization_changed": False,
        "similarity_einsum_changed": False,
        "learned_out_projection_changed": False,
        "write_backend_changed": False,
        "training_backend_changed": False,
        "known_empty_fastpath_preserved": True,
        "same_call_query_key_reuse_preserved": True,
        "persistent_state_changed": False,
        "persistent_cache": False,
        "persistent_packed_state": False,
        "gpu_authorized_by_module": False,
        "scientific_training_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
