from __future__ import annotations

"""AERA-v26.7 mixed-dtype integration successor for the fused FICEM READ tail.

Issue #551 follows the consumed #545/#550 end-to-end systems abort.  The exact
repair5 READ primitive remains historical PASS on its frozen same-tail-dtype
surface, but integrated BF16-autocast inference reached the repair5 validator
with compute/read similarity and durable episodic state at different floating
precisions.  The historical validator rejected that call before launching the
already-proven one-kernel READ tail.

This successor keeps the repair5 Triton kernel byte-for-byte and preserves the
v26.6 materialize-cast WRITE backend.  It versions only the READ validation and
dispatch wrapper needed to admit the preregistered FP32/BF16 compute-versus-
durable layouts.  Strengths and values must still share one durable dtype; this
module does not generalize to arbitrary independently mixed state fields.
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
from .aera_hardware_core_v26 import CoalescedFICEMMemory, FICEMReadPrimitive
from .aera_hardware_core_v26_3_ficem_read_triton import (
    _ficem_read_tail_kernel,
    fused_ficem_read_v26_3_protocol,
    triton,
)
from .aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
    materialize_cast_ficem_read_write_v26_6_protocol,
)

RESEARCH_ISSUE = 551
SOURCE_MAIN = "383444afa414fa955c46f98f11cf733ddcef656f"
HISTORICAL_REPAIR5_READ_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BACKEND_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
ISSUE530_SYSTEMS_EVALUATOR_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
ISSUE545_TRIGGER = 550
ISSUE545_RUN = 33686037672
ISSUE545_JOB = 100433658768
ISSUE545_FAILURE = "FICEM read-tail floating dtypes must match"
ISSUE545_AUTHORITATIVE_RESULT_EMITTED = False
ISSUE545_SCIENTIFIC_SEED_CONSUMED = False

# Preserve the historical all-FP16 same-dtype validator route, but the only new
# mixed layouts authorized by #551 are compute/durable FP32<->BF16.
HISTORICAL_READ_FLOAT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
)
MIXED_COMPUTE_DURABLE_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
)


def supported_read_dtype_layout(
    similarity_dtype: torch.dtype,
    strengths_dtype: torch.dtype,
    values_dtype: torch.dtype,
) -> bool:
    """CPU-checkable #551 READ dtype contract.

    State strengths and values remain one durable dtype.  Historical same-dtype
    FP32/BF16/FP16 calls stay accepted.  Cross compute/durable precision is new
    only for FP32/BF16 in either direction.
    """

    if (
        similarity_dtype not in HISTORICAL_READ_FLOAT_DTYPES
        or strengths_dtype not in HISTORICAL_READ_FLOAT_DTYPES
        or values_dtype not in HISTORICAL_READ_FLOAT_DTYPES
    ):
        return False
    if strengths_dtype != values_dtype:
        return False
    if similarity_dtype == values_dtype:
        return True
    return (
        similarity_dtype in MIXED_COMPUTE_DURABLE_DTYPES
        and values_dtype in MIXED_COMPUTE_DURABLE_DTYPES
    )


def _validate_mixed_read_tail_inputs(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Mirror repair5 geometry/device guards without its global dtype equality."""

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
        raise ValueError("v26.7 fused read is frozen to capacity48/memory_dim50")
    if valid.dtype is not torch.bool:
        raise TypeError("FICEM validity must be boolean")
    if not supported_read_dtype_layout(
        similarity.dtype,
        strengths.dtype,
        values.dtype,
    ):
        raise TypeError(
            "v26.7 fused FICEM read supports same-dtype FP32/BF16/FP16 and "
            "mixed FP32/BF16 compute-versus-durable state only"
        )
    tensors = (similarity, strengths, valid, values)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("v26.7 mixed-dtype fused FICEM read requires CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("FICEM read-tail tensors must share a device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("v26.7 mixed-dtype fused FICEM read requires contiguous tensors")
    return batch, time, capacity, memory_dim


def fused_ficem_read_tail_mixed_dtype(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
    *,
    return_top_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the exact repair5 kernel under the versioned #551 dtype contract."""

    if triton is None or _ficem_read_tail_kernel is None:
        raise RuntimeError("Triton FICEM read kernel is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fused FICEM read is unavailable")
    batch, time, capacity, memory_dim = _validate_mixed_read_tail_inputs(
        similarity,
        strengths,
        valid,
        values,
    )

    # Repair5 arithmetic mode is selected by similarity/compute precision.  BF16
    # durable values also require FP32 storage for the existing repair5 output
    # contract.  Historical all-FP16 remains FP16 exactly as before.
    recalled_dtype = (
        torch.float32
        if similarity.dtype is torch.bfloat16 or values.dtype is torch.bfloat16
        else values.dtype
    )
    recalled = torch.empty(
        (batch, time, memory_dim),
        device=values.device,
        dtype=recalled_dtype,
    )
    if return_top_indices:
        top_indices = torch.empty(
            (batch, time, READ_TOP_K),
            device=values.device,
            dtype=torch.int32,
        )
        index_pointer = top_indices
    else:
        top_indices = None
        # WRITE_INDICES is constexpr false, so this pointer is never dereferenced.
        index_pointer = recalled

    # Exact historical repair5 Triton kernel and launch geometry.  No host cast,
    # copy, materialization kernel, or second accelerated READ launch is added.
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


class MixedDtypeTritonFICEMReadWriteBackend(MaterializeCastTritonFICEMReadWriteBackend):
    """v26.7 mixed-dtype repair5 READ + exact v26.6 materialize-cast WRITE."""

    name = "triton-ficem-read-v26.7-mixed-dtype-write-v26.6-materialize-cast"

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
            raise RuntimeError("v26.7 fused FICEM read requires CUDA inference tensors")

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
        recalled, _ = fused_ficem_read_tail_mixed_dtype(
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


def mixed_dtype_ficem_read_write_v26_7_protocol() -> dict[str, Any]:
    """Versioned static protocol; this CPU repair authorizes no GPU execution."""

    protocol = dict(materialize_cast_ficem_read_write_v26_6_protocol())
    repair5 = fused_ficem_read_v26_3_protocol()
    protocol.update(
        {
            "version": "aera-v26.7-ficem-read-mixed-dtype",
            "research_issue": RESEARCH_ISSUE,
            "source_main": SOURCE_MAIN,
            "backend_name": MixedDtypeTritonFICEMReadWriteBackend.name,
            "historical_repair5_read_blob": HISTORICAL_REPAIR5_READ_BLOB,
            "v26_6_write_backend_blob": V26_6_WRITE_BACKEND_BLOB,
            "issue530_systems_evaluator_blob": ISSUE530_SYSTEMS_EVALUATOR_BLOB,
            "issue545_trigger": ISSUE545_TRIGGER,
            "issue545_run": ISSUE545_RUN,
            "issue545_job": ISSUE545_JOB,
            "issue545_failure": ISSUE545_FAILURE,
            "issue545_authoritative_result_emitted": ISSUE545_AUTHORITATIVE_RESULT_EMITTED,
            "issue545_scientific_seed_consumed": ISSUE545_SCIENTIFIC_SEED_CONSUMED,
            "historical_repair5_backend_mutated": False,
            "read_kernel_reused_from_repair5": True,
            "read_new_triton_kernels": 0,
            "read_tail_triton_launches_target": 1,
            "read_global_cross_field_dtype_equality_required": False,
            "read_strengths_values_dtype_equality_required": True,
            "read_supported_new_mixed_dtypes": ["float32", "bfloat16"],
            "read_historical_same_dtype_float16_preserved": True,
            "read_arbitrary_strengths_values_mixing_authorized": False,
            "read_bf16_mode_selected_by_similarity_dtype": True,
            "read_host_pre_tail_cast_kernels": 0,
            "read_arithmetic_changed_by_v26_7": False,
            "read_topology_changed_by_v26_7": False,
            "read_known_empty_changed_by_v26_7": False,
            "read_training_backend_changed_by_v26_7": False,
            "write_backend_changed_by_v26_7": False,
            "write_backend_name_preserved_from_v26_6": materialize_cast_ficem_read_write_v26_6_protocol()[
                "backend_name"
            ],
            "repair5_bf16_actual_autocast_tail_preserved": bool(
                repair5["bf16_actual_autocast_tail_repair5"]
            ),
            "mixed_dtype_read_gpu_gate_authorized": False,
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
