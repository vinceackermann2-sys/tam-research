from __future__ import annotations

"""AERA-v26.8 mixed-dtype FICEM READ strength-precision successor.

Issue #556 follows the authoritative #553 mixed-dtype READ FAIL.  The complete
historical repair5 same-dtype surface still passed through v26.7, while every
ordinary mixed compute/durable row failed with the one-kernel topology intact.

The v26.7 mixed wrapper reused the repair5 kernel with one ``IS_BF16`` flag
derived from similarity dtype.  That flag legitimately controls similarity and
softmax-weight visibility, but it also controlled strength-source visibility.
That coupling is valid only when similarity/strength/value dtypes are equal.

V26.8 leaves every same-dtype call on the exact historical repair5 helper and
adds exactly one mixed-only Triton kernel for the two preregistered FP32/BF16
compute-versus-durable layouts.  The mixed kernel separates compute precision
from durable-strength source precision without adding a host cast or a second
tail launch.
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
    fused_ficem_read_tail,
    tl,
    triton,
)
from .aera_hardware_core_v26_7_ficem_read_mixed_dtype import (
    MixedDtypeTritonFICEMReadWriteBackend,
    mixed_dtype_ficem_read_write_v26_7_protocol,
    supported_read_dtype_layout,
)

RESEARCH_ISSUE = 556
SOURCE_MAIN = "e4866dd6d4556fc090b556c09ae49fcf4c59105f"
SOURCE_TREE = "86af0b1dd4e6d3fd2cdb6412460b7e4f5cbbd6ff"

V26_7_BACKEND_BLOB = "d8133c6b204b1ee5f23955255fb2fb09d09bd723"
REPAIR5_READ_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
ISSUE553_PROBE_BLOB = "ff9a47f510be07e8adeff018f327338147163cdb"
ISSUE553_LAUNCHER_BLOB = "b3630e065c56c93a1b7f6f164416f068ccb2ecac"
ISSUE553_WORKFLOW_BLOB = "eef7826f1a76a853d9cf745243612dd457d79a10"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_PROBE_BLOB = "6fd6518e10ed1ef4115863f98ac591ffd77ce903"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"

ISSUE553_TRIGGER = 555
ISSUE553_RUN = 33727540468
ISSUE553_JOB = 100559866985
ISSUE553_RESULT_PATH = "/vol/aera-v26/issue553-ficem-read-mixed-dtype/result.json"
ISSUE553_RESULT_SHA256 = (
    "009af31baf70e46eb93b6e7489d62f356a02b727521d3fabe4a7dab2dcf5ab47"
)
ISSUE553_DECISION = "FAIL"

ISSUE545_TRIGGER = 550
ISSUE545_RUN = 33686037672
ISSUE545_JOB = 100433658768
ISSUE545_FAILURE = "FICEM read-tail floating dtypes must match"

ISSUE479_TRIGGER = 484
ISSUE479_RUN = 33618950619
ISSUE479_JOB = 100211244996

ISSUE529_TRIGGER = 529
ISSUE529_RUN = 33680028132
ISSUE529_JOB = 100414089065
ISSUE529_RESULT_SHA256 = (
    "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
)

MIXED_LAYOUTS: tuple[tuple[torch.dtype, torch.dtype], ...] = (
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
)


def is_v26_8_mixed_layout(
    similarity_dtype: torch.dtype,
    strengths_dtype: torch.dtype,
    values_dtype: torch.dtype,
) -> bool:
    """Return true only for the two #556 compute/durable mixed layouts."""

    if strengths_dtype != values_dtype:
        return False
    return (similarity_dtype, values_dtype) in MIXED_LAYOUTS


def read_dispatch_kind(
    similarity_dtype: torch.dtype,
    strengths_dtype: torch.dtype,
    values_dtype: torch.dtype,
) -> str:
    """CPU-testable dispatch contract.

    Every accepted same-dtype call stays on the exact repair5 helper.  Only the
    two FP32/BF16 mixed layouts enter the v26.8 mixed-only kernel.
    """

    if (
        similarity_dtype == strengths_dtype == values_dtype
        and supported_read_dtype_layout(
            similarity_dtype,
            strengths_dtype,
            values_dtype,
        )
    ):
        return "historical-repair5"
    if is_v26_8_mixed_layout(
        similarity_dtype,
        strengths_dtype,
        values_dtype,
    ):
        return "mixed-strength-precision-v26.8"
    return "unsupported"


_mixed_strength_precision_kernel = None

if triton is not None:

    @triton.jit
    def _mixed_strength_precision_kernel(
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
        IS_BF16_COMPUTE: tl.constexpr,
        DURABLE_IS_BF16: tl.constexpr,
        MIN_STRENGTH: tl.constexpr,
        READ_TEMPERATURE: tl.constexpr,
        READ_TOP_K: tl.constexpr,
    ):
        """One mixed-only tail with independent compute/strength visibility."""

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

        # Keep clamp semantics exact while preserving the durable source's visible
        # precision.  The result of native BF16 torch.log outside autocast is BF16;
        # the BF16-durable/FP32-compute branch explicitly recreates that boundary.
        if DURABLE_IS_BF16:
            strengths_source = strengths.to(tl.bfloat16)
            clamped_strengths = tl.minimum(
                tl.maximum(strengths_source, MIN_STRENGTH),
                1.0,
            ).to(tl.bfloat16)
        else:
            strengths_source = strengths.to(tl.float32)
            clamped_strengths = tl.minimum(
                tl.maximum(strengths_source, MIN_STRENGTH),
                1.0,
            ).to(tl.float32)

        if IS_BF16_COMPUTE:
            # The reference caller is CUDA BF16 autocast.  Similarity has the
            # repair5 BF16 source boundary.  FP32 durable strengths, however, must
            # stay FP32 through log; they must NOT be rounded to BF16 merely because
            # the compute source is BF16.
            similarity_visible = similarity.to(tl.bfloat16)
            if DURABLE_IS_BF16:
                strength_bias = tl.log(clamped_strengths.to(tl.float32))
            else:
                strength_bias = tl.log(clamped_strengths)
            logits = (
                similarity_visible.to(tl.float32)
                + strength_bias.to(tl.float32)
            ) / READ_TEMPERATURE
        else:
            # #556's only non-BF16-compute mixed layout is FP32 similarity with
            # BF16 durable state under *no autocast*.  Preserve the native BF16
            # log-result visibility before the normal FP32 promotion at addition.
            similarity_visible = similarity.to(tl.float32)
            if DURABLE_IS_BF16:
                strength_bias_visible = tl.log(
                    clamped_strengths.to(tl.float32)
                ).to(tl.bfloat16)
                strength_bias = strength_bias_visible.to(tl.float32)
            else:
                strength_bias = tl.log(clamped_strengths.to(tl.float32))
            logits = (
                similarity_visible + strength_bias.to(tl.float32)
            ) / READ_TEMPERATURE

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
        soft0 = exp0 / softmax_sum
        soft1 = exp1 / softmax_sum
        soft2 = exp2 / softmax_sum
        soft3 = exp3 / softmax_sum

        if IS_BF16_COMPUTE:
            # Exact repair5 compute-derived visibility: only the *initial* weights
            # are BF16-visible.  Denominator and final normalized weights are FP32.
            weight0_visible = soft0.to(tl.bfloat16)
            weight1_visible = soft1.to(tl.bfloat16)
            weight2_visible = soft2.to(tl.bfloat16)
            weight3_visible = soft3.to(tl.bfloat16)
            weight0_valid = tl.where(valid0, weight0_visible, 0.0).to(tl.bfloat16)
            weight1_valid = tl.where(valid1, weight1_visible, 0.0).to(tl.bfloat16)
            weight2_valid = tl.where(valid2, weight2_visible, 0.0).to(tl.bfloat16)
            weight3_valid = tl.where(valid3, weight3_visible, 0.0).to(tl.bfloat16)
            valid_weight_sum = (
                weight0_valid.to(tl.float32)
                + weight1_valid.to(tl.float32)
                + weight2_valid.to(tl.float32)
                + weight3_valid.to(tl.float32)
            )
            denominator = tl.maximum(valid_weight_sum, 1.0e-9)
            weight0 = weight0_valid.to(tl.float32) / denominator
            weight1 = weight1_valid.to(tl.float32) / denominator
            weight2 = weight2_valid.to(tl.float32) / denominator
            weight3 = weight3_valid.to(tl.float32) / denominator
        else:
            # `.to(similarity.dtype)` is an FP32 no-op for this mixed direction.
            weight0 = tl.where(valid0, soft0, 0.0).to(tl.float32)
            weight1 = tl.where(valid1, soft1, 0.0).to(tl.float32)
            weight2 = tl.where(valid2, soft2, 0.0).to(tl.float32)
            weight3 = tl.where(valid3, soft3, 0.0).to(tl.float32)
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

        # Preserve selected durable values in their source dtype until in-kernel
        # arithmetic.  Both authorized mixed reference paths reduce to FP32 recalled.
        if DURABLE_IS_BF16:
            value0_visible = value0.to(tl.bfloat16)
            value1_visible = value1.to(tl.bfloat16)
            value2_visible = value2.to(tl.bfloat16)
            value3_visible = value3.to(tl.bfloat16)
        else:
            value0_visible = value0.to(tl.float32)
            value1_visible = value1.to(tl.float32)
            value2_visible = value2.to(tl.float32)
            value3_visible = value3.to(tl.float32)

        product0 = weight0.to(tl.float32) * value0_visible.to(tl.float32)
        product1 = weight1.to(tl.float32) * value1_visible.to(tl.float32)
        product2 = weight2.to(tl.float32) * value2_visible.to(tl.float32)
        product3 = weight3.to(tl.float32) * value3_visible.to(tl.float32)
        recalled = product0 + product1 + product2 + product3

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


def _validate_v26_8_mixed_inputs(
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
        raise ValueError("v26.8 mixed read is frozen to capacity48/memory_dim50")
    if valid.dtype is not torch.bool:
        raise TypeError("FICEM validity must be boolean")
    if not is_v26_8_mixed_layout(
        similarity.dtype,
        strengths.dtype,
        values.dtype,
    ):
        raise TypeError(
            "v26.8 mixed kernel accepts only BF16-compute/FP32-durable or "
            "FP32-compute/BF16-durable layouts"
        )
    tensors = (similarity, strengths, valid, values)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("v26.8 mixed FICEM read requires CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("FICEM read-tail tensors must share a device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("v26.8 mixed FICEM read requires contiguous tensors")
    return batch, time, capacity, memory_dim


def fused_ficem_read_tail_v26_8(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
    *,
    return_top_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch same-dtype reads to repair5; execute only mixed rows in v26.8."""

    dispatch = read_dispatch_kind(
        similarity.dtype,
        strengths.dtype,
        values.dtype,
    )
    if dispatch == "historical-repair5":
        # Exact historical helper, validator, kernel and same-dtype output contract.
        return fused_ficem_read_tail(
            similarity,
            strengths,
            valid,
            values,
            return_top_indices=return_top_indices,
        )
    if dispatch != "mixed-strength-precision-v26.8":
        raise TypeError("unsupported v26.8 FICEM read dtype layout")
    if (
        triton is None
        or tl is None
        or _mixed_strength_precision_kernel is None
    ):
        raise RuntimeError("Triton v26.8 mixed FICEM read kernel is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; v26.8 mixed FICEM read is unavailable")

    batch, time, capacity, memory_dim = _validate_v26_8_mixed_inputs(
        similarity,
        strengths,
        valid,
        values,
    )
    recalled = torch.empty(
        (batch, time, memory_dim),
        device=values.device,
        dtype=torch.float32,
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
        # WRITE_INDICES is constexpr false, so the pointer is never dereferenced.
        index_pointer = recalled

    _mixed_strength_precision_kernel[(batch * time,)](
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
        IS_BF16_COMPUTE=similarity.dtype is torch.bfloat16,
        DURABLE_IS_BF16=strengths.dtype is torch.bfloat16,
        MIN_STRENGTH=MIN_STRENGTH,
        READ_TEMPERATURE=READ_TEMPERATURE,
        READ_TOP_K=READ_TOP_K,
        num_warps=4,
    )
    return recalled, top_indices


class StrengthPrecisionTritonFICEMReadWriteBackend(
    MixedDtypeTritonFICEMReadWriteBackend
):
    """v26.8 READ strength-precision repair + unchanged v26.6 WRITE."""

    name = (
        "triton-ficem-read-v26.8-mixed-strength-precision-"
        "write-v26.6-materialize-cast"
    )

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
            raise RuntimeError("v26.8 fused FICEM read requires CUDA inference tensors")

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
        recalled, _ = fused_ficem_read_tail_v26_8(
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


def mixed_strength_precision_v26_8_protocol() -> dict[str, Any]:
    protocol = dict(mixed_dtype_ficem_read_write_v26_7_protocol())
    protocol.update(
        {
            "version": "aera-v26.8-ficem-read-mixed-strength-precision",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue556": SOURCE_MAIN,
            "source_tree_issue556": SOURCE_TREE,
            "backend_name": StrengthPrecisionTritonFICEMReadWriteBackend.name,
            "v26_7_backend_blob": V26_7_BACKEND_BLOB,
            "repair5_read_blob": REPAIR5_READ_BLOB,
            "v26_6_write_blob": V26_6_WRITE_BLOB,
            "issue553_probe_blob": ISSUE553_PROBE_BLOB,
            "issue553_launcher_blob": ISSUE553_LAUNCHER_BLOB,
            "issue553_workflow_blob": ISSUE553_WORKFLOW_BLOB,
            "historical_probe_blob": HISTORICAL_PROBE_BLOB,
            "repair5_probe_blob": REPAIR5_PROBE_BLOB,
            "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
            "v26_interface_blob": V26_INTERFACE_BLOB,
            "stable_reference_blob": STABLE_REFERENCE_BLOB,
            "issue553_trigger": ISSUE553_TRIGGER,
            "issue553_run": ISSUE553_RUN,
            "issue553_job": ISSUE553_JOB,
            "issue553_result_path": ISSUE553_RESULT_PATH,
            "issue553_result_sha256": ISSUE553_RESULT_SHA256,
            "issue553_decision": ISSUE553_DECISION,
            "issue553_consumed": True,
            "issue553_historical_surface_pass": True,
            "issue553_mixed_regular_rows_pass": 0,
            "issue553_mixed_regular_rows_total": 8,
            "issue553_mixed_topology_pass": True,
            "issue553_mixed_near_tie_pass": True,
            "issue553_mixed_known_empty_pass": True,
            "issue545_trigger": ISSUE545_TRIGGER,
            "issue545_run": ISSUE545_RUN,
            "issue545_job": ISSUE545_JOB,
            "issue545_failure": ISSUE545_FAILURE,
            "issue479_trigger": ISSUE479_TRIGGER,
            "issue479_run": ISSUE479_RUN,
            "issue479_job": ISSUE479_JOB,
            "issue529_trigger": ISSUE529_TRIGGER,
            "issue529_run": ISSUE529_RUN,
            "issue529_job": ISSUE529_JOB,
            "issue529_result_sha256": ISSUE529_RESULT_SHA256,
            "same_dtype_dispatch": "historical-repair5",
            "same_dtype_arithmetic_changed_by_v26_8": False,
            "same_dtype_kernel_changed_by_v26_8": False,
            "mixed_new_triton_kernels": 1,
            "mixed_tail_triton_launches_target": 1,
            "mixed_compute_precision_control_separate": True,
            "mixed_durable_strength_precision_control_separate": True,
            "mixed_strengths_values_dtype_equality_required": True,
            "mixed_arbitrary_strengths_values_mixing_authorized": False,
            "mixed_fp16_authorized": False,
            "bf16_compute_fp32_strength_prelog_bf16_cast": False,
            "fp32_compute_bf16_strength_bias_bf16_visibility": True,
            "softmax_weight_visibility_controlled_by_compute_dtype": True,
            "mixed_selected_value_source_dtype_preserved": True,
            "mixed_recalled_pre_out_dtype": "float32",
            "mixed_host_pre_tail_cast_kernels": 0,
            "write_backend_changed_by_v26_8": False,
            "training_backend_changed_by_v26_8": False,
            "gpu_authorized_by_issue556": False,
            "mixed_dtype_read_gpu_gate_authorized": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
            "scientific_seed_consumed": False,
        }
    )
    return protocol


def cpu_contract_preflight_issue556() -> dict[str, Any]:
    expected_dispatch = {
        (torch.float32, torch.float32, torch.float32): "historical-repair5",
        (torch.bfloat16, torch.bfloat16, torch.bfloat16): "historical-repair5",
        (torch.float16, torch.float16, torch.float16): "historical-repair5",
        (torch.bfloat16, torch.float32, torch.float32):
            "mixed-strength-precision-v26.8",
        (torch.float32, torch.bfloat16, torch.bfloat16):
            "mixed-strength-precision-v26.8",
    }
    observed = {
        layout: read_dispatch_kind(*layout)
        for layout in expected_dispatch
    }
    if observed != expected_dispatch:
        raise RuntimeError("issue556 READ dispatch matrix drifted")

    rejected = (
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.float16, torch.float16),
        (torch.bfloat16, torch.float16, torch.float16),
        (torch.float32, torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.float64, torch.float64, torch.float64),
    )
    if any(read_dispatch_kind(*layout) != "unsupported" for layout in rejected):
        raise RuntimeError("issue556 unsupported dtype layout became admitted")

    protocol = mixed_strength_precision_v26_8_protocol()
    higher_false = (
        "gpu_authorized_by_issue556",
        "mixed_dtype_read_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    )
    if any(protocol[key] for key in higher_false):
        raise RuntimeError("issue556 higher authorization drifted")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "protocol": protocol,
        "dispatch": {
            "/".join(str(dtype) for dtype in layout): kind
            for layout, kind in observed.items()
        },
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }
