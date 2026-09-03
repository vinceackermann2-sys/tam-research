from __future__ import annotations

"""AERA-v26.9 integrated FICEM READ identity-weight-visibility successor.

Issue #600 follows the authoritative #594/#597 integrated localization.  V26.8
correctly separated similarity/compute precision from durable-strength source
precision, but still used similarity dtype to choose the initial softmax-weight
visibility boundary.  The exact Torch reference instead performs
``softmax(...).to(identity_source.dtype)``.

V26.9 preserves the exact historical repair5 same-dtype path and all v26.8 mixed
strength/logit/value arithmetic.  Its only mixed-tail semantic delta is a separate
constexpr controlling the initial softmax-weight visibility, derived from the
original identity-source dtype rather than the projected similarity dtype.
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
from .aera_hardware_core_v26_8_ficem_read_mixed_strength_precision import (
    StrengthPrecisionTritonFICEMReadWriteBackend,
    _validate_v26_8_mixed_inputs,
    mixed_strength_precision_v26_8_protocol,
    read_dispatch_kind as read_dispatch_kind_v26_8,
)

RESEARCH_ISSUE = 600
SOURCE_MAIN = "d4128a1b4e021ef998491e45ab2355586ea07b04"
SOURCE_TREE = "d00a7d3915d1e787758ed5df75a4214d8a848fed"

V26_8_BACKEND_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_7_BACKEND_BLOB = "d8133c6b204b1ee5f23955255fb2fb09d09bd723"
REPAIR5_READ_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
FACTORIZED_V25_BLOB = "f8cce87fa4dcae69fd171ba95fcbdab50e743a2f"
ISSUE558_PROBE_BLOB = "99ab8252f2b594404aae1ca86752eaa902eb80a5"
ISSUE558_TRIGGER = 561
ISSUE558_BOUND_MAIN = "75987bfb7976c6a970d63801c6e81b5b4993f544"
ISSUE558_DECISION = "PASS"
ISSUE594_TRIGGER = 596
ISSUE594_RUN = 33772104621
ISSUE594_JOB = 100704667286
ISSUE594_RESULT_PATH = "/vol/aera-v26/issue594-stage0-post-read-amplification-localizer/result.json"
ISSUE594_RESULT_SHA256 = "c950d8fa50e70a48ec64a87f860d70d854cf1a2b58e1acbdfbcb0052495e809e"
ISSUE597_TRIGGER = 599
ISSUE597_RUN = 33774062361
ISSUE597_JOB = 100711243436

SUPPORTED_IDENTITY_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
)


def read_dispatch_kind(
    identity_dtype: torch.dtype,
    similarity_dtype: torch.dtype,
    strengths_dtype: torch.dtype,
    values_dtype: torch.dtype,
) -> str:
    """CPU-testable v26.9 dispatch contract.

    Same-tail-dtype calls remain byte-for-byte on historical repair5.  Only the
    two v26.8 mixed compute/durable layouts enter the v26.9 mixed kernel, and for
    those rows the identity dtype is carried independently for weight visibility.
    """

    inherited = read_dispatch_kind_v26_8(
        similarity_dtype,
        strengths_dtype,
        values_dtype,
    )
    if inherited == "historical-repair5":
        return inherited
    if inherited != "mixed-strength-precision-v26.8":
        return "unsupported"
    if identity_dtype not in SUPPORTED_IDENTITY_DTYPES:
        return "unsupported"
    return "mixed-identity-weight-visibility-v26.9"


def initial_weight_visibility_kind(
    identity_dtype: torch.dtype,
    similarity_dtype: torch.dtype,
    strengths_dtype: torch.dtype,
    values_dtype: torch.dtype,
) -> str:
    """Expose the exact initial-weight visibility contract without CUDA."""

    dispatch = read_dispatch_kind(
        identity_dtype,
        similarity_dtype,
        strengths_dtype,
        values_dtype,
    )
    if dispatch == "historical-repair5":
        return "historical-repair5"
    if dispatch != "mixed-identity-weight-visibility-v26.9":
        return "unsupported"
    return "bfloat16" if identity_dtype is torch.bfloat16 else "float32"


_mixed_identity_weight_visibility_kernel = None

if triton is not None:

    @triton.jit
    def _mixed_identity_weight_visibility_kernel(
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
        WEIGHT_VISIBLE_BF16: tl.constexpr,
        MIN_STRENGTH: tl.constexpr,
        READ_TEMPERATURE: tl.constexpr,
        READ_TOP_K: tl.constexpr,
    ):
        """V26.8 mixed arithmetic with identity-derived initial weight visibility."""

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

        # Exact v26.8 strength-source visibility semantics.
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

        # Exact v26.8 similarity/logit arithmetic semantics.
        if IS_BF16_COMPUTE:
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

        # This is the only v26.9 arithmetic delta.  The exact Torch reference is
        # `.to(identity_source.dtype)`, not `.to(similarity.dtype)`.
        if WEIGHT_VISIBLE_BF16:
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

        # Exact v26.8 selected-value source visibility and FP32 mixed reduction.
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


def fused_ficem_read_tail_v26_9(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
    *,
    identity_dtype: torch.dtype,
    return_top_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use repair5 for same-dtype tails; use v26.9 only for v26.8 mixed rows."""

    dispatch = read_dispatch_kind(
        identity_dtype,
        similarity.dtype,
        strengths.dtype,
        values.dtype,
    )
    if dispatch == "historical-repair5":
        return fused_ficem_read_tail(
            similarity,
            strengths,
            valid,
            values,
            return_top_indices=return_top_indices,
        )
    if dispatch != "mixed-identity-weight-visibility-v26.9":
        raise TypeError("unsupported v26.9 FICEM read dtype layout")
    if triton is None or tl is None or _mixed_identity_weight_visibility_kernel is None:
        raise RuntimeError("Triton v26.9 mixed FICEM read kernel is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; v26.9 mixed FICEM read is unavailable")

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
        index_pointer = recalled

    _mixed_identity_weight_visibility_kernel[(batch * time,)](
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
        WEIGHT_VISIBLE_BF16=identity_dtype is torch.bfloat16,
        MIN_STRENGTH=MIN_STRENGTH,
        READ_TEMPERATURE=READ_TEMPERATURE,
        READ_TOP_K=READ_TOP_K,
        num_warps=4,
    )
    return recalled, top_indices


class IdentityWeightVisibilityTritonFICEMReadWriteBackend(
    StrengthPrecisionTritonFICEMReadWriteBackend
):
    """v26.9 READ identity-weight visibility + unchanged v26.8/v26.6 behavior."""

    name = (
        "triton-ficem-read-v26.9-identity-weight-visibility-"
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
            raise RuntimeError("v26.9 fused FICEM read requires CUDA inference tensors")

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
        recalled, _ = fused_ficem_read_tail_v26_9(
            similarity.contiguous(),
            state.strengths.contiguous(),
            state.valid.contiguous(),
            state.values.contiguous(),
            identity_dtype=identity_source.dtype,
            return_top_indices=False,
        )
        return FICEMReadPrimitive(
            recalled=memory.out(recalled),
            projected_query=query,
            normalized_old_keys=keys,
        )


def identity_weight_visibility_v26_9_protocol() -> dict[str, Any]:
    protocol = dict(mixed_strength_precision_v26_8_protocol())
    protocol.update(
        {
            "version": "aera-v26.9-ficem-read-identity-weight-visibility",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue600": SOURCE_MAIN,
            "source_tree_issue600": SOURCE_TREE,
            "backend_name": IdentityWeightVisibilityTritonFICEMReadWriteBackend.name,
            "v26_8_backend_blob": V26_8_BACKEND_BLOB,
            "v26_7_backend_blob": V26_7_BACKEND_BLOB,
            "repair5_read_blob": REPAIR5_READ_BLOB,
            "v26_6_write_blob": V26_6_WRITE_BLOB,
            "v26_interface_blob": V26_INTERFACE_BLOB,
            "stable_reference_blob": STABLE_REFERENCE_BLOB,
            "factorized_v25_blob": FACTORIZED_V25_BLOB,
            "issue558_probe_blob": ISSUE558_PROBE_BLOB,
            "issue558_trigger": ISSUE558_TRIGGER,
            "issue558_bound_main": ISSUE558_BOUND_MAIN,
            "issue558_decision": ISSUE558_DECISION,
            "issue558_preserved_as_authoritative_pass": True,
            "issue558_identity_context_followed_compute_dtype": True,
            "issue558_covered_fp32_identity_bf16_similarity_fp32_durable": False,
            "issue594_trigger": ISSUE594_TRIGGER,
            "issue594_run": ISSUE594_RUN,
            "issue594_job": ISSUE594_JOB,
            "issue594_result_path": ISSUE594_RESULT_PATH,
            "issue594_result_sha256": ISSUE594_RESULT_SHA256,
            "issue597_trigger": ISSUE597_TRIGGER,
            "issue597_run": ISSUE597_RUN,
            "issue597_job": ISSUE597_JOB,
            "same_dtype_dispatch": "historical-repair5",
            "mixed_new_triton_kernels": 1,
            "mixed_tail_triton_launches_target": 1,
            "mixed_strength_precision_semantics_changed_by_v26_9": False,
            "mixed_selected_value_semantics_changed_by_v26_9": False,
            "softmax_weight_visibility_controlled_by_compute_dtype": False,
            "softmax_weight_visibility_controlled_by_identity_dtype": True,
            "identity_weight_visibility_control_separate": True,
            "similarity_compute_precision_control_separate": True,
            "durable_strength_precision_control_separate": True,
            "integrated_fp32_identity_bf16_similarity_fp32_durable_weight_visibility": "float32",
            "historical_bf16_identity_bf16_similarity_fp32_durable_weight_visibility": "bfloat16",
            "fp32_identity_fp32_similarity_bf16_durable_weight_visibility": "float32",
            "mixed_recalled_pre_out_dtype": "float32",
            "write_backend_changed_by_v26_9": False,
            "training_backend_changed_by_v26_9": False,
            "gpu_authorized_by_issue600": False,
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


def cpu_contract_preflight_issue600() -> dict[str, Any]:
    expected = {
        (torch.float32, torch.bfloat16, torch.float32, torch.float32): (
            "mixed-identity-weight-visibility-v26.9",
            "float32",
        ),
        (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32): (
            "mixed-identity-weight-visibility-v26.9",
            "bfloat16",
        ),
        (torch.float32, torch.float32, torch.bfloat16, torch.bfloat16): (
            "mixed-identity-weight-visibility-v26.9",
            "float32",
        ),
        (torch.float32, torch.float32, torch.float32, torch.float32): (
            "historical-repair5",
            "historical-repair5",
        ),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16): (
            "historical-repair5",
            "historical-repair5",
        ),
        (torch.float16, torch.float16, torch.float16, torch.float16): (
            "historical-repair5",
            "historical-repair5",
        ),
    }
    observed = {
        layout: (
            read_dispatch_kind(*layout),
            initial_weight_visibility_kind(*layout),
        )
        for layout in expected
    }
    if observed != expected:
        raise RuntimeError("issue600 identity-weight visibility matrix drifted")

    rejected = (
        (torch.float16, torch.bfloat16, torch.float32, torch.float32),
        (torch.float32, torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.bfloat16, torch.bfloat16, torch.float32),
    )
    if any(read_dispatch_kind(*layout) != "unsupported" for layout in rejected):
        raise RuntimeError("issue600 unsupported dtype layout became admitted")

    protocol = identity_weight_visibility_v26_9_protocol()
    higher_false = (
        "gpu_authorized_by_issue600",
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
        raise RuntimeError("issue600 higher authorization drifted")

    return {
        "protocol": protocol,
        "dispatch_and_weight_visibility": observed,
        "gpu_authorized_by_cpu_preflight": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "scientific_seed_consumed": False,
        "repair_scope": "mixed-read initial softmax-weight visibility only",
    }
