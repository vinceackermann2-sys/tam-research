from __future__ import annotations

"""Issue #439 localization-only BF16 arithmetic diagnostic after repair3.

This module reconstructs only the original frozen `bfloat16_batch8_mixed` #411/#418
fixture.  The production repair3 backend and frozen probe remain untouched.  A
separate diagnostic-only Triton kernel mirrors the merged repair3 arithmetic while
writing intermediate checkpoints so the exact PyTorch reference can be compared
stage by stage on NVIDIA L4.

No timing, profiling, model/checkpoint/corpus/training, optimizer/backward, or
scientific-seed path is present here.
"""

from typing import Any

import torch

from . import aera_v26_3_ficem_read_probe as probe

try:  # CPU CI intentionally does not require Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - expected on CPU-only CI images.
    triton = None
    tl = None


RESEARCH_ISSUE = 439
SOURCE_MAIN = "1ec7229f976b360440171b979bde63dedd8e9697"
SOURCE_FAILED_ISSUE = 436
SOURCE_FAILED_TRIGGER = 438
SOURCE_FAILED_ACTIONS_RUN = 33510242472
SOURCE_FAILED_JOB = 99863885932

TARGET_ROW = "bfloat16_batch8_mixed"
TARGET_DTYPE = "bfloat16"
TARGET_BATCH = 8
TARGET_VALIDITY = "mixed"
TARGET_ORDINAL = 5
MAX_EXAMPLES = 16

FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR3_BACKEND_GIT_BLOB = "b6b37f0379b280eea4e5c2b16f349951dadc4df9"


def cpu_contract_preflight() -> dict[str, Any]:
    if probe.DESIGN_SEED != 408_411:
        raise RuntimeError("#439 requires the original #411 design seed")
    if probe.DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("#439 dtype order drifted")
    if probe.BATCH_SIZES != (8, 64):
        raise RuntimeError("#439 batch order drifted")
    if probe.VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("#439 validity order drifted")
    if (probe.BF16_ATOL, probe.BF16_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("#439 frozen BF16 diagnostic tolerance drifted")
    if (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) != (
        200,
        256,
        48,
        50,
    ):
        raise RuntimeError("#439 production geometry drifted")
    rows = [
        probe._row_key(dtype_name, batch_size, validity_kind)
        for dtype_name in probe.DTYPE_NAMES
        for batch_size in probe.BATCH_SIZES
        for validity_kind in probe.VALIDITY_KINDS
    ]
    if rows[TARGET_ORDINAL - 1] != TARGET_ROW:
        raise RuntimeError("#439 target ordinal no longer reproduces failed row")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_failed_issue": SOURCE_FAILED_ISSUE,
        "source_failed_trigger": SOURCE_FAILED_TRIGGER,
        "source_failed_actions_run": SOURCE_FAILED_ACTIONS_RUN,
        "source_failed_job": SOURCE_FAILED_JOB,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "design_seed": probe.DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "bf16_atol": probe.BF16_ATOL,
        "bf16_rtol": probe.BF16_RTOL,
        "frozen_probe_git_blob": FROZEN_PROBE_GIT_BLOB,
        "frozen_repair3_backend_git_blob": FROZEN_REPAIR3_BACKEND_GIT_BLOB,
        "original_global_case_order_preserved": True,
        "resampling": False,
        "rejection_sampling": False,
        "fixture_nudging": False,
        "alternate_seed": False,
        "localization_only": True,
        "timing_authorized": False,
        "profiling_authorized": False,
        "performance_decision_authorized": False,
        "production_backend_modified": False,
        "production_probe_modified": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def reconstruct_target_case(device: torch.device):
    """Replay the original global #411 ordinary-case stream through ordinal five."""
    generator = torch.Generator().manual_seed(probe.DESIGN_SEED)
    ordinal = 0
    for dtype_name in probe.DTYPE_NAMES:
        for batch_size in probe.BATCH_SIZES:
            for validity_kind in probe.VALIDITY_KINDS:
                ordinal += 1
                case = probe.make_case(
                    dtype_name=dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    generator=generator,
                    device=device,
                )
                key = probe._row_key(dtype_name, batch_size, validity_kind)
                if ordinal == TARGET_ORDINAL:
                    if key != TARGET_ROW:
                        raise RuntimeError("#439 replay reached the wrong target row")
                    return case
    raise RuntimeError("#439 could not reconstruct the target row")


_repair3_checkpoint_kernel = None

if triton is not None:

    @triton.jit
    def _repair3_checkpoint_kernel(
        similarity_ptr,
        strengths_ptr,
        valid_ptr,
        values_ptr,
        similarity_visible_ptr,
        clamped_visible_ptr,
        strength_bias_ptr,
        logits_ptr,
        masked_logits_ptr,
        top_indices_ptr,
        top_logits_ptr,
        safe_logits_fp32_ptr,
        softmax_fp32_ptr,
        weight_bf16_ptr,
        weight_valid_bf16_ptr,
        weight_sum_bf16_ptr,
        denominator_bf16_ptr,
        final_weight_bf16_ptr,
        selected_value_bf16_ptr,
        product_fp32_ptr,
        recalled_fp32_ptr,
        recalled_bf16_ptr,
        TIME: tl.constexpr,
        CAPACITY: tl.constexpr,
        MEMORY_DIM: tl.constexpr,
        SLOT_BLOCK: tl.constexpr,
        DIM_BLOCK: tl.constexpr,
        MIN_STRENGTH: tl.constexpr,
        READ_TEMPERATURE: tl.constexpr,
        READ_TOP_K: tl.constexpr,
    ):
        query_row = tl.program_id(0)
        batch_row = query_row // TIME

        slot_offsets = tl.arange(0, SLOT_BLOCK)
        slot_mask = slot_offsets < CAPACITY

        similarity_loaded = tl.load(
            similarity_ptr + query_row * CAPACITY + slot_offsets,
            mask=slot_mask,
            other=-float("inf"),
        )
        strengths_loaded = tl.load(
            strengths_ptr + batch_row * CAPACITY + slot_offsets,
            mask=slot_mask,
            other=MIN_STRENGTH,
        )
        valid = tl.load(
            valid_ptr + batch_row * CAPACITY + slot_offsets,
            mask=slot_mask,
            other=0,
        )

        # Mirror the merged repair3 BF16 branch exactly while making each visible
        # arithmetic boundary observable. This kernel is diagnostic-only.
        similarity_visible = similarity_loaded.to(tl.bfloat16)
        clamped = tl.minimum(tl.maximum(strengths_loaded, MIN_STRENGTH), 1.0)
        clamped_visible = clamped.to(tl.bfloat16)
        strength_bias = tl.log(clamped_visible.to(tl.float32)).to(tl.bfloat16)
        logits = (
            (similarity_visible + strength_bias).to(tl.bfloat16)
            / READ_TEMPERATURE
        ).to(tl.bfloat16)
        masked_logits = tl.where(slot_mask & valid, logits, -float("inf"))

        slot_base = query_row * CAPACITY + slot_offsets
        tl.store(similarity_visible_ptr + slot_base, similarity_visible, mask=slot_mask)
        tl.store(clamped_visible_ptr + slot_base, clamped_visible, mask=slot_mask)
        tl.store(strength_bias_ptr + slot_base, strength_bias, mask=slot_mask)
        tl.store(logits_ptr + slot_base, logits, mask=slot_mask)
        tl.store(masked_logits_ptr + slot_base, masked_logits, mask=slot_mask)

        index0 = tl.argmax(masked_logits, axis=0, tie_break_left=True)
        logit0 = tl.max(masked_logits, axis=0)
        remaining1 = tl.where(slot_offsets == index0, -float("inf"), masked_logits)
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

        safe0 = tl.where(valid0, logit0, -1.0e9).to(tl.bfloat16).to(tl.float32)
        safe1 = tl.where(valid1, logit1, -1.0e9).to(tl.bfloat16).to(tl.float32)
        safe2 = tl.where(valid2, logit2, -1.0e9).to(tl.bfloat16).to(tl.float32)
        safe3 = tl.where(valid3, logit3, -1.0e9).to(tl.bfloat16).to(tl.float32)

        maximum = tl.maximum(tl.maximum(safe0, safe1), tl.maximum(safe2, safe3))
        exp0 = tl.exp(safe0 - maximum)
        exp1 = tl.exp(safe1 - maximum)
        exp2 = tl.exp(safe2 - maximum)
        exp3 = tl.exp(safe3 - maximum)
        softmax_sum = exp0 + exp1 + exp2 + exp3
        softmax0 = exp0 / softmax_sum
        softmax1 = exp1 / softmax_sum
        softmax2 = exp2 / softmax_sum
        softmax3 = exp3 / softmax_sum

        weight0_cast = softmax0.to(tl.bfloat16)
        weight1_cast = softmax1.to(tl.bfloat16)
        weight2_cast = softmax2.to(tl.bfloat16)
        weight3_cast = softmax3.to(tl.bfloat16)
        weight0_valid = tl.where(valid0, weight0_cast, 0.0).to(tl.bfloat16)
        weight1_valid = tl.where(valid1, weight1_cast, 0.0).to(tl.bfloat16)
        weight2_valid = tl.where(valid2, weight2_cast, 0.0).to(tl.bfloat16)
        weight3_valid = tl.where(valid3, weight3_cast, 0.0).to(tl.bfloat16)

        valid_weight_sum = (
            weight0_valid.to(tl.float32)
            + weight1_valid.to(tl.float32)
            + weight2_valid.to(tl.float32)
            + weight3_valid.to(tl.float32)
        ).to(tl.bfloat16)
        denominator = tl.maximum(
            valid_weight_sum.to(tl.float32), 1.0e-9
        ).to(tl.bfloat16)
        final0 = (
            weight0_valid.to(tl.float32) / denominator.to(tl.float32)
        ).to(tl.bfloat16)
        final1 = (
            weight1_valid.to(tl.float32) / denominator.to(tl.float32)
        ).to(tl.bfloat16)
        final2 = (
            weight2_valid.to(tl.float32) / denominator.to(tl.float32)
        ).to(tl.bfloat16)
        final3 = (
            weight3_valid.to(tl.float32) / denominator.to(tl.float32)
        ).to(tl.bfloat16)

        top_base = query_row * READ_TOP_K
        tl.store(top_indices_ptr + top_base + 0, index0)
        tl.store(top_indices_ptr + top_base + 1, index1)
        tl.store(top_indices_ptr + top_base + 2, index2)
        tl.store(top_indices_ptr + top_base + 3, index3)
        tl.store(top_logits_ptr + top_base + 0, logit0)
        tl.store(top_logits_ptr + top_base + 1, logit1)
        tl.store(top_logits_ptr + top_base + 2, logit2)
        tl.store(top_logits_ptr + top_base + 3, logit3)
        tl.store(safe_logits_fp32_ptr + top_base + 0, safe0)
        tl.store(safe_logits_fp32_ptr + top_base + 1, safe1)
        tl.store(safe_logits_fp32_ptr + top_base + 2, safe2)
        tl.store(safe_logits_fp32_ptr + top_base + 3, safe3)
        tl.store(softmax_fp32_ptr + top_base + 0, softmax0)
        tl.store(softmax_fp32_ptr + top_base + 1, softmax1)
        tl.store(softmax_fp32_ptr + top_base + 2, softmax2)
        tl.store(softmax_fp32_ptr + top_base + 3, softmax3)
        tl.store(weight_bf16_ptr + top_base + 0, weight0_cast)
        tl.store(weight_bf16_ptr + top_base + 1, weight1_cast)
        tl.store(weight_bf16_ptr + top_base + 2, weight2_cast)
        tl.store(weight_bf16_ptr + top_base + 3, weight3_cast)
        tl.store(weight_valid_bf16_ptr + top_base + 0, weight0_valid)
        tl.store(weight_valid_bf16_ptr + top_base + 1, weight1_valid)
        tl.store(weight_valid_bf16_ptr + top_base + 2, weight2_valid)
        tl.store(weight_valid_bf16_ptr + top_base + 3, weight3_valid)
        tl.store(weight_sum_bf16_ptr + query_row, valid_weight_sum)
        tl.store(denominator_bf16_ptr + query_row, denominator)
        tl.store(final_weight_bf16_ptr + top_base + 0, final0)
        tl.store(final_weight_bf16_ptr + top_base + 1, final1)
        tl.store(final_weight_bf16_ptr + top_base + 2, final2)
        tl.store(final_weight_bf16_ptr + top_base + 3, final3)

        dim_offsets = tl.arange(0, DIM_BLOCK)
        dim_mask = dim_offsets < MEMORY_DIM
        value_base = batch_row * CAPACITY * MEMORY_DIM
        value0 = tl.load(
            values_ptr + value_base + index0 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.bfloat16)
        value1 = tl.load(
            values_ptr + value_base + index1 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.bfloat16)
        value2 = tl.load(
            values_ptr + value_base + index2 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.bfloat16)
        value3 = tl.load(
            values_ptr + value_base + index3 * MEMORY_DIM + dim_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.bfloat16)

        product0 = final0.to(tl.float32) * value0.to(tl.float32)
        product1 = final1.to(tl.float32) * value1.to(tl.float32)
        product2 = final2.to(tl.float32) * value2.to(tl.float32)
        product3 = final3.to(tl.float32) * value3.to(tl.float32)
        recalled_fp32 = product0 + product1 + product2 + product3
        recalled_bf16 = recalled_fp32.to(tl.bfloat16)

        selected_base = query_row * READ_TOP_K * MEMORY_DIM
        tl.store(
            selected_value_bf16_ptr + selected_base + 0 * MEMORY_DIM + dim_offsets,
            value0,
            mask=dim_mask,
        )
        tl.store(
            selected_value_bf16_ptr + selected_base + 1 * MEMORY_DIM + dim_offsets,
            value1,
            mask=dim_mask,
        )
        tl.store(
            selected_value_bf16_ptr + selected_base + 2 * MEMORY_DIM + dim_offsets,
            value2,
            mask=dim_mask,
        )
        tl.store(
            selected_value_bf16_ptr + selected_base + 3 * MEMORY_DIM + dim_offsets,
            value3,
            mask=dim_mask,
        )
        tl.store(
            product_fp32_ptr + selected_base + 0 * MEMORY_DIM + dim_offsets,
            product0,
            mask=dim_mask,
        )
        tl.store(
            product_fp32_ptr + selected_base + 1 * MEMORY_DIM + dim_offsets,
            product1,
            mask=dim_mask,
        )
        tl.store(
            product_fp32_ptr + selected_base + 2 * MEMORY_DIM + dim_offsets,
            product2,
            mask=dim_mask,
        )
        tl.store(
            product_fp32_ptr + selected_base + 3 * MEMORY_DIM + dim_offsets,
            product3,
            mask=dim_mask,
        )

        recalled_base = query_row * MEMORY_DIM + dim_offsets
        tl.store(recalled_fp32_ptr + recalled_base, recalled_fp32, mask=dim_mask)
        tl.store(recalled_bf16_ptr + recalled_base, recalled_bf16, mask=dim_mask)


def _run_diagnostic_mirror(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if triton is None or _repair3_checkpoint_kernel is None:
        raise RuntimeError("#439 diagnostic Triton kernel is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("#439 diagnostic requires CUDA")
    if similarity.dtype is not torch.bfloat16:
        raise TypeError("#439 diagnostic target must be bfloat16")
    batch, time, capacity = map(int, similarity.shape)
    memory_dim = int(values.size(-1))
    if (capacity, memory_dim) != (probe.CAPACITY, probe.MEMORY_DIM):
        raise RuntimeError("#439 diagnostic geometry drifted")
    if strengths.shape != (batch, capacity) or valid.shape != (batch, capacity):
        raise RuntimeError("#439 strength/valid geometry mismatch")
    if values.shape != (batch, capacity, memory_dim):
        raise RuntimeError("#439 value geometry mismatch")

    slot_shape = (batch, time, capacity)
    top_shape = (batch, time, probe.READ_TOP_K)
    selected_shape = (batch, time, probe.READ_TOP_K, memory_dim)
    recalled_shape = (batch, time, memory_dim)

    out = {
        "similarity_visible": torch.empty(slot_shape, device=similarity.device, dtype=torch.bfloat16),
        "clamped_visible": torch.empty(slot_shape, device=similarity.device, dtype=torch.bfloat16),
        "strength_bias": torch.empty(slot_shape, device=similarity.device, dtype=torch.bfloat16),
        "logits": torch.empty(slot_shape, device=similarity.device, dtype=torch.bfloat16),
        "masked_logits": torch.empty(slot_shape, device=similarity.device, dtype=torch.bfloat16),
        "top_indices": torch.empty(top_shape, device=similarity.device, dtype=torch.int32),
        "top_logits": torch.empty(top_shape, device=similarity.device, dtype=torch.bfloat16),
        "safe_logits_fp32": torch.empty(top_shape, device=similarity.device, dtype=torch.float32),
        "softmax_fp32": torch.empty(top_shape, device=similarity.device, dtype=torch.float32),
        "weight_bf16": torch.empty(top_shape, device=similarity.device, dtype=torch.bfloat16),
        "weight_valid_bf16": torch.empty(top_shape, device=similarity.device, dtype=torch.bfloat16),
        "weight_sum_bf16": torch.empty((batch, time), device=similarity.device, dtype=torch.bfloat16),
        "denominator_bf16": torch.empty((batch, time), device=similarity.device, dtype=torch.bfloat16),
        "final_weight_bf16": torch.empty(top_shape, device=similarity.device, dtype=torch.bfloat16),
        "selected_value_bf16": torch.empty(selected_shape, device=similarity.device, dtype=torch.bfloat16),
        "product_fp32": torch.empty(selected_shape, device=similarity.device, dtype=torch.float32),
        "recalled_fp32": torch.empty(recalled_shape, device=similarity.device, dtype=torch.float32),
        "recalled_bf16": torch.empty(recalled_shape, device=similarity.device, dtype=torch.bfloat16),
    }

    _repair3_checkpoint_kernel[(batch * time,)](
        similarity,
        strengths,
        valid,
        values,
        out["similarity_visible"],
        out["clamped_visible"],
        out["strength_bias"],
        out["logits"],
        out["masked_logits"],
        out["top_indices"],
        out["top_logits"],
        out["safe_logits_fp32"],
        out["softmax_fp32"],
        out["weight_bf16"],
        out["weight_valid_bf16"],
        out["weight_sum_bf16"],
        out["denominator_bf16"],
        out["final_weight_bf16"],
        out["selected_value_bf16"],
        out["product_fp32"],
        out["recalled_fp32"],
        out["recalled_bf16"],
        TIME=time,
        CAPACITY=capacity,
        MEMORY_DIM=memory_dim,
        SLOT_BLOCK=64,
        DIM_BLOCK=64,
        MIN_STRENGTH=probe.MIN_STRENGTH,
        READ_TEMPERATURE=probe.READ_TEMPERATURE,
        READ_TOP_K=probe.READ_TOP_K,
    )
    return out


def _reference_checkpoints(similarity: torch.Tensor, state) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        clamped = state.strengths.clamp(probe.MIN_STRENGTH, 1.0)
        strength_bias = torch.log(clamped)[:, None, :]
        logits = (similarity + strength_bias) / probe.READ_TEMPERATURE
        masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
        top_logits, top_indices = torch.topk(masked, k=probe.READ_TOP_K, dim=-1)
        top_valid = state.valid[:, None, :].expand(-1, similarity.size(1), -1).gather(
            -1, top_indices
        )
        safe_logits = top_logits.masked_fill(~top_valid, -1e9)
        softmax_fp32 = torch.softmax(safe_logits.float(), dim=-1)
        weight_bf16 = softmax_fp32.to(similarity.dtype)
        weight_valid = weight_bf16 * top_valid.to(weight_bf16.dtype)
        weight_sum = weight_valid.sum(dim=-1, keepdim=True)
        denominator = weight_sum.clamp_min(1e-9)
        final_weight = weight_valid / denominator
        expanded_values = state.values[:, None, :, :].expand(
            -1, similarity.size(1), -1, -1
        )
        selected_values = expanded_values.gather(
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, probe.MEMORY_DIM),
        )
        products = final_weight.unsqueeze(-1) * selected_values
        recalled = products.sum(dim=2)

    return {
        "similarity_visible": similarity,
        "clamped_visible": clamped[:, None, :].expand_as(similarity),
        "strength_bias": strength_bias.expand_as(similarity),
        "logits": logits,
        "masked_logits": masked,
        "top_indices": top_indices,
        "top_logits": top_logits,
        "top_valid": top_valid,
        "safe_logits_fp32": safe_logits.float(),
        "softmax_fp32": softmax_fp32,
        "weight_bf16": weight_bf16,
        "weight_valid_bf16": weight_valid,
        "weight_sum_bf16": weight_sum[..., 0],
        "denominator_bf16": denominator[..., 0],
        "final_weight_bf16": final_weight,
        "selected_value_bf16": selected_values,
        "product_bf16": products,
        "recalled_bf16": recalled,
    }


def _scalar_json(value: torch.Tensor) -> float | str | int | bool:
    item = value.item()
    if isinstance(item, bool):
        return bool(item)
    if isinstance(item, int):
        return int(item)
    number = float(item)
    if number == float("inf"):
        return "inf"
    if number == float("-inf"):
        return "-inf"
    if number != number:
        return "nan"
    return number


def _first_mismatch_examples(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    mismatch: torch.Tensor,
) -> list[dict[str, Any]]:
    coords = torch.nonzero(mismatch, as_tuple=False)[:MAX_EXAMPLES]
    records: list[dict[str, Any]] = []
    for coord in coords:
        index = tuple(int(x) for x in coord.tolist())
        records.append(
            {
                "index": list(index),
                "reference": _scalar_json(reference[index]),
                "candidate": _scalar_json(candidate[index]),
            }
        )
    return records


def _tensor_report(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "bit_equal": False,
            "allclose": False,
        }
    ref = reference
    cand = candidate
    equal = torch.eq(ref, cand)
    mismatch = ~equal
    finite_pair = torch.isfinite(ref) & torch.isfinite(cand)
    finite_diff = (ref.float() - cand.float()).abs()
    finite_values = finite_diff[finite_pair]
    mismatch_count = int(mismatch.sum())
    total = int(mismatch.numel())
    if finite_values.numel():
        max_abs = float(finite_values.max())
        mean_abs = float(finite_values.mean())
    else:
        max_abs = 0.0
        mean_abs = 0.0
    return {
        "shape_equal": True,
        "shape": list(ref.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(cand.dtype),
        "dtype_equal": ref.dtype == cand.dtype,
        "device_equal": ref.device == cand.device,
        "bit_equal": bool(torch.equal(ref, cand)),
        "allclose": bool(torch.allclose(ref, cand, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_count / total if total else 0.0,
        "finite_pair_count": int(finite_pair.sum()),
        "max_abs_error_finite_pairs": max_abs,
        "mean_abs_error_finite_pairs": mean_abs,
        "first_mismatches": _first_mismatch_examples(ref, cand, mismatch),
    }


def _top_set_masks(
    reference_masked_logits: torch.Tensor,
    reference_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    candidate_long = candidate_indices.to(torch.long)
    reference_long = reference_indices.to(torch.long)
    raw_set_equal = torch.all(
        torch.sort(reference_long, dim=-1).values
        == torch.sort(candidate_long, dim=-1).values,
        dim=-1,
    )
    boundary = torch.topk(
        reference_masked_logits, k=probe.READ_TOP_K + 1, dim=-1
    ).values
    cutoff = boundary[..., probe.READ_TOP_K - 1]
    fifth = boundary[..., probe.READ_TOP_K]
    distinct = cutoff != fifth
    return {
        "candidate_long": candidate_long,
        "reference_long": reference_long,
        "raw_set_equal": raw_set_equal,
        "cutoff": cutoff,
        "fifth": fifth,
        "distinct": distinct,
        "tied": ~distinct,
    }


def _selection_examples(masks: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    failing = masks["distinct"] & ~masks["raw_set_equal"]
    coords = torch.nonzero(failing, as_tuple=False)[:MAX_EXAMPLES]
    out: list[dict[str, Any]] = []
    for coord in coords:
        b, t = (int(coord[0]), int(coord[1]))
        out.append(
            {
                "batch": b,
                "time": t,
                "reference_top4": [int(x) for x in masks["reference_long"][b, t].tolist()],
                "candidate_top4": [int(x) for x in masks["candidate_long"][b, t].tolist()],
                "reference_fourth": float(masks["cutoff"][b, t]),
                "reference_fifth": float(masks["fifth"][b, t]),
                "reference_gap": float(masks["cutoff"][b, t] - masks["fifth"][b, t]),
            }
        )
    return out


def _align_candidate_top(
    reference_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a candidate top-k tensor into reference slot order where sets match."""
    reference_long = reference_indices.to(torch.long)
    candidate_long = candidate_indices.to(torch.long)
    matches = reference_long.unsqueeze(-1) == candidate_long.unsqueeze(-2)
    found = matches.any(dim=-1)
    positions = matches.to(torch.int64).argmax(dim=-1)
    gather_index = positions
    for _ in range(candidate_tensor.ndim - 3):
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(*positions.shape, *candidate_tensor.shape[3:])
    aligned = torch.gather(candidate_tensor, 2, gather_index)
    set_equal = found.all(dim=-1)
    return aligned, set_equal


def _aligned_report(
    reference_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    reference_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    aligned, set_equal = _align_candidate_top(
        reference_indices, candidate_indices, candidate_tensor
    )
    equal_queries = int(set_equal.sum())
    total_queries = int(set_equal.numel())
    if equal_queries == 0:
        return {
            "equal_selected_set_query_count": 0,
            "total_query_count": total_queries,
            "comparison_available": False,
        }
    return {
        "equal_selected_set_query_count": equal_queries,
        "total_query_count": total_queries,
        "comparison_available": True,
        "report": _tensor_report(
            reference_tensor[set_equal], aligned[set_equal], atol=atol, rtol=rtol
        ),
    }


def _dtype_map(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: str(value.dtype) for name, value in tensors.items()}


def _reference_reduction_microdiagnostics(
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    final_weight = reference["final_weight_bf16"]
    selected_values = reference["selected_value_bf16"]
    reference_products = reference["product_bf16"]
    reference_recalled = reference["recalled_bf16"]

    fp32_products = final_weight.float().unsqueeze(-1) * selected_values.float()
    repair3_fp32_product_accum = fp32_products.sum(dim=2).to(torch.bfloat16)
    rounded_product_fp32_accum = reference_products.float().sum(dim=2).to(torch.bfloat16)
    native_bf16_product_sum = reference_products.sum(dim=2)

    return {
        "reference_product_dtype": str(reference_products.dtype),
        "reference_sum_output_dtype": str(reference_recalled.dtype),
        "repair3_fp32_product_accum_vs_reference": _tensor_report(
            reference_recalled,
            repair3_fp32_product_accum,
            atol=probe.BF16_ATOL,
            rtol=probe.BF16_RTOL,
        ),
        "bf16_rounded_product_fp32_accum_vs_reference": _tensor_report(
            reference_recalled, rounded_product_fp32_accum
        ),
        "native_bf16_product_sum_recomputed_vs_reference": _tensor_report(
            reference_recalled, native_bf16_product_sum
        ),
        "fp32_product_then_bf16_vs_reference_products": _tensor_report(
            reference_products, fp32_products.to(torch.bfloat16)
        ),
    }


def run_localization() -> dict[str, Any]:
    contract = cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("#439 localization requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"#439 requires NVIDIA L4, found {device_name}")

    memory = probe.build_memory(device)
    case = reconstruct_target_case(device)
    query, keys, similarity = probe._diagnostic_tail_inputs(memory, case)
    if similarity.dtype is not torch.bfloat16:
        raise RuntimeError("#439 exact target similarity is not bfloat16")

    reference = _reference_checkpoints(similarity, case.state)
    diagnostic = _run_diagnostic_mirror(
        similarity, case.state.strengths, case.state.valid, case.state.values
    )
    production_recalled, production_indices = probe.fused_ficem_read_tail(
        similarity,
        case.state.strengths,
        case.state.valid,
        case.state.values,
        return_top_indices=True,
    )
    torch.cuda.synchronize()
    if production_indices is None:
        raise RuntimeError("#439 production repair3 diagnostic indices missing")

    selection = probe._tie_aware_top4_equivalence(
        reference["masked_logits"],
        case.state.valid,
        reference["top_indices"],
        diagnostic["top_indices"],
    )
    masks = _top_set_masks(
        reference["masked_logits"], reference["top_indices"], diagnostic["top_indices"]
    )

    checkpoint_reports: dict[str, Any] = {
        "similarity_visible": _tensor_report(
            reference["similarity_visible"], diagnostic["similarity_visible"]
        ),
        "clamped_strengths_visible": _tensor_report(
            reference["clamped_visible"], diagnostic["clamped_visible"]
        ),
        "strength_bias": _tensor_report(reference["strength_bias"], diagnostic["strength_bias"]),
        "unmasked_logits": _tensor_report(reference["logits"], diagnostic["logits"]),
        "masked_logits": _tensor_report(reference["masked_logits"], diagnostic["masked_logits"]),
        "top_logits_aligned_on_equal_sets": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["top_logits"], diagnostic["top_logits"]
        ),
        "safe_logits_fp32_aligned_on_equal_sets": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["safe_logits_fp32"], diagnostic["safe_logits_fp32"]
        ),
        "softmax_fp32_aligned_on_equal_sets": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["softmax_fp32"], diagnostic["softmax_fp32"]
        ),
        "weight_bf16_aligned_on_equal_sets": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["weight_bf16"], diagnostic["weight_bf16"]
        ),
        "weight_after_validity_aligned_on_equal_sets": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["weight_valid_bf16"], diagnostic["weight_valid_bf16"]
        ),
        "weight_sum_bf16": _tensor_report(reference["weight_sum_bf16"], diagnostic["weight_sum_bf16"]),
        "denominator_bf16": _tensor_report(reference["denominator_bf16"], diagnostic["denominator_bf16"]),
        "final_weight_bf16_aligned_on_equal_sets": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["final_weight_bf16"], diagnostic["final_weight_bf16"]
        ),
        "selected_value_bf16_aligned_on_equal_sets": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["selected_value_bf16"], diagnostic["selected_value_bf16"]
        ),
        "product_aligned_on_equal_sets_reference_bf16_vs_candidate_fp32": _aligned_report(
            reference["top_indices"], diagnostic["top_indices"], reference["product_bf16"].float(), diagnostic["product_fp32"]
        ),
        "pre_out_recalled_bf16": _tensor_report(
            reference["recalled_bf16"],
            diagnostic["recalled_bf16"],
            atol=probe.BF16_ATOL,
            rtol=probe.BF16_RTOL,
        ),
    }

    ordered_divergence_checks = (
        ("similarity_visible", checkpoint_reports["similarity_visible"]),
        ("clamped_strengths_visible", checkpoint_reports["clamped_strengths_visible"]),
        ("strength_bias", checkpoint_reports["strength_bias"]),
        ("unmasked_logits", checkpoint_reports["unmasked_logits"]),
        ("masked_logits", checkpoint_reports["masked_logits"]),
    )
    first_divergence = None
    for name, report in ordered_divergence_checks:
        if not bool(report.get("bit_equal", False)):
            first_divergence = name
            break
    if first_divergence is None and not selection["selection_semantically_equivalent"]:
        first_divergence = "top4_selection"
    if first_divergence is None:
        for name in (
            "safe_logits_fp32_aligned_on_equal_sets",
            "softmax_fp32_aligned_on_equal_sets",
            "weight_bf16_aligned_on_equal_sets",
            "weight_after_validity_aligned_on_equal_sets",
            "weight_sum_bf16",
            "denominator_bf16",
            "final_weight_bf16_aligned_on_equal_sets",
            "selected_value_bf16_aligned_on_equal_sets",
            "product_aligned_on_equal_sets_reference_bf16_vs_candidate_fp32",
            "pre_out_recalled_bf16",
        ):
            report = checkpoint_reports[name]
            nested = report.get("report", report)
            if nested.get("bit_equal") is False:
                first_divergence = name
                break

    mirror_fidelity = {
        "top_indices_bit_exact": bool(
            torch.equal(production_indices.to(torch.int32), diagnostic["top_indices"])
        ),
        "recalled_bf16": _tensor_report(
            production_recalled, diagnostic["recalled_bf16"]
        ),
    }
    diagnostic_mirror_valid = bool(
        mirror_fidelity["top_indices_bit_exact"]
        and mirror_fidelity["recalled_bf16"]["bit_equal"]
    )

    distinct_mismatch_count = int((masks["distinct"] & ~masks["raw_set_equal"]).sum())
    tied_count = int(masks["tied"].sum())

    micro = {
        "torch_log_bf16_vs_diagnostic_triton": checkpoint_reports["strength_bias"],
        "torch_bf16_add_divide_vs_diagnostic_triton": checkpoint_reports["unmasked_logits"],
        "torch_fp32_softmax_then_bf16_vs_diagnostic_triton": checkpoint_reports[
            "softmax_fp32_aligned_on_equal_sets"
        ],
        "reference_reduction_semantics": _reference_reduction_microdiagnostics(reference),
    }

    return {
        "contract": contract,
        "device": device_name,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "query_dtype": str(query.dtype),
        "normalized_key_dtype": str(keys.dtype),
        "similarity_dtype": str(similarity.dtype),
        "reference_checkpoint_dtypes": _dtype_map(reference),
        "diagnostic_checkpoint_dtypes": _dtype_map(diagnostic),
        "diagnostic_mirror_valid": diagnostic_mirror_valid,
        "production_mirror_fidelity": mirror_fidelity,
        "first_reference_vs_repair3_divergence": first_divergence,
        "selection": selection,
        "distinct_query_mismatch_count": distinct_mismatch_count,
        "distinct_query_mismatch_examples": _selection_examples(masks),
        "tied_query_count": tied_count,
        "checkpoint_reports": checkpoint_reports,
        "microdiagnostics": micro,
        "localization_only": True,
        "timing_performed": False,
        "profiling_performed": False,
        "performance_decision": None,
        "candidate_acceptance_changed": False,
        "production_backend_modified": False,
        "production_probe_modified": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
