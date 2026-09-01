from __future__ import annotations

"""Issue #452 localization-only diagnostic for the repair4 BF16 read miss.

Reconstructs only the frozen ordinal-5 `bfloat16_batch8_mixed` fixture, invokes the
unchanged #418/#433 correctness gate once, and compares PyTorch product/reduction
semantics with one isolated diagnostic Triton microkernel. No timing, profiling,
training, model/checkpoint/corpus, or candidate-acceptance change is present here.
"""

from typing import Any

import torch

from . import aera_v26_3_ficem_read_probe as probe
from .aera_hardware_core_v26 import TorchFICEMReferenceBackend
from .aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
    fused_ficem_read_tail,
)

try:  # CPU CI intentionally does not require Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


RESEARCH_ISSUE = 452
SOURCE_MAIN = "9d45b41d40c6859f4dc4ffc1b70c26e0f7768976"
SOURCE_FAILED_TRIGGER = 451
SOURCE_FAILED_ACTIONS_RUN = 33537116699
SOURCE_FAILED_JOB = 99954032841
TARGET_ROW = "bfloat16_batch8_mixed"
TARGET_ORDINAL = 5
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR4_BACKEND_GIT_BLOB = "a3a603c8a2d4b20ebcccd7663970978f4288a760"

PRIMARY_SUBGATES = (
    "selection_semantically_equivalent",
    "pre_out_recalled_close",
    "final_out_close",
    "query_and_normalized_keys_bit_exact",
    "source_unchanged",
    "finite",
    "dtype_device_shape_exact",
)


def _ordinary_rows() -> list[str]:
    return [
        probe._row_key(dtype_name, batch_size, validity_kind)
        for dtype_name in probe.DTYPE_NAMES
        for batch_size in probe.BATCH_SIZES
        for validity_kind in probe.VALIDITY_KINDS
    ]


def cpu_contract_preflight() -> dict[str, Any]:
    rows = _ordinary_rows()
    expected = [
        "float32_batch8_mixed",
        "float32_batch8_full",
        "float32_batch64_mixed",
        "float32_batch64_full",
        "bfloat16_batch8_mixed",
        "bfloat16_batch8_full",
        "bfloat16_batch64_mixed",
        "bfloat16_batch64_full",
    ]
    if rows != expected or rows[TARGET_ORDINAL - 1] != TARGET_ROW:
        raise RuntimeError("#452 frozen ordinary-row order drifted")
    if probe.DESIGN_SEED != 408_411:
        raise RuntimeError("#452 frozen design seed drifted")
    if (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) != (200, 256, 48, 50):
        raise RuntimeError("#452 geometry drifted")
    if (probe.BF16_ATOL, probe.BF16_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("#452 BF16 tolerance drifted")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_failed_trigger": SOURCE_FAILED_TRIGGER,
        "source_failed_actions_run": SOURCE_FAILED_ACTIONS_RUN,
        "source_failed_job": SOURCE_FAILED_JOB,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "design_seed": probe.DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "frozen_probe_git_blob": FROZEN_PROBE_GIT_BLOB,
        "frozen_repair4_backend_git_blob": FROZEN_REPAIR4_BACKEND_GIT_BLOB,
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
                if ordinal == TARGET_ORDINAL:
                    key = probe._row_key(dtype_name, batch_size, validity_kind)
                    if key != TARGET_ROW:
                        raise RuntimeError("#452 replay reached wrong row")
                    return case
    raise RuntimeError("#452 failed to reconstruct target")


def _error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.float()
    cand = candidate.float()
    diff = (cand - ref).abs()
    allowed = probe.BF16_ATOL + probe.BF16_RTOL * ref.abs()
    failing = diff > allowed
    total = int(diff.numel())
    mismatch = int((reference != candidate).sum()) if reference.shape == candidate.shape else total
    return {
        "bit_equal": bool(torch.equal(reference, candidate)),
        "allclose": bool(torch.allclose(reference, candidate, atol=probe.BF16_ATOL, rtol=probe.BF16_RTOL)),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "failing_element_count": int(failing.sum()),
        "failing_element_fraction": float(failing.sum()) / total if total else 0.0,
        "bit_mismatch_count": mismatch,
        "bit_mismatch_fraction": mismatch / total if total else 0.0,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "shape": list(reference.shape),
    }


def _first_mismatches(reference: torch.Tensor, candidate: torch.Tensor, limit: int = 12) -> list[dict[str, Any]]:
    ref = reference.float()
    cand = candidate.float()
    coords = torch.nonzero(ref != cand, as_tuple=False)[:limit].cpu()
    out: list[dict[str, Any]] = []
    for coord in coords:
        idx = tuple(int(v) for v in coord.tolist())
        out.append({"index": list(idx), "reference": float(ref[idx]), "candidate": float(cand[idx])})
    return out


def _reference_product_inputs(memory, case) -> dict[str, torch.Tensor]:
    _, _, similarity = probe._diagnostic_tail_inputs(memory, case)
    state = case.state
    strength_bias = torch.log(state.strengths.clamp(probe.MIN_STRENGTH, 1.0))[:, None, :]
    logits = (similarity + strength_bias) / probe.READ_TEMPERATURE
    masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
    top_logits, top_indices = torch.topk(masked, k=probe.READ_TOP_K, dim=-1)
    top_valid = state.valid[:, None, :].expand(-1, similarity.size(1), -1).gather(-1, top_indices)
    safe_logits = top_logits.masked_fill(~top_valid, -1e9)
    weights = torch.softmax(safe_logits.float(), dim=-1).to(similarity.dtype)
    weights = weights * top_valid.to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    expanded_values = state.values[:, None, :, :].expand(-1, similarity.size(1), -1, -1)
    selected_values = expanded_values.gather(
        2, top_indices.unsqueeze(-1).expand(-1, -1, -1, probe.MEMORY_DIM)
    )
    products_bf16 = weights.unsqueeze(-1) * selected_values
    native_recalled = products_bf16.sum(dim=2)
    p = products_bf16.float()
    sequential_fp32 = p[:, :, 0] + p[:, :, 1] + p[:, :, 2] + p[:, :, 3]
    pairwise_fp32 = (p[:, :, 0] + p[:, :, 1]) + (p[:, :, 2] + p[:, :, 3])
    return {
        "similarity": similarity,
        "top_indices": top_indices,
        "weights": weights.contiguous(),
        "selected_values": selected_values.contiguous(),
        "products_bf16": products_bf16.contiguous(),
        "native_recalled": native_recalled.contiguous(),
        "sequential_fp32": sequential_fp32.contiguous(),
        "pairwise_fp32": pairwise_fp32.contiguous(),
    }


_product_reduction_checkpoint_kernel = None

if triton is not None:

    @triton.jit
    def _product_reduction_checkpoint_kernel(
        weights_ptr,
        values_ptr,
        raw_products_ptr,
        rounded_products_ptr,
        sequential_ptr,
        pairwise_ptr,
        recalled_bf16_ptr,
        MEMORY_DIM: tl.constexpr,
        DIM_BLOCK: tl.constexpr,
    ):
        query_row = tl.program_id(0)
        dim = tl.arange(0, DIM_BLOCK)
        mask = dim < MEMORY_DIM
        wbase = query_row * 4
        vbase = query_row * 4 * MEMORY_DIM
        w0 = tl.load(weights_ptr + wbase + 0)
        w1 = tl.load(weights_ptr + wbase + 1)
        w2 = tl.load(weights_ptr + wbase + 2)
        w3 = tl.load(weights_ptr + wbase + 3)
        v0 = tl.load(values_ptr + vbase + 0 * MEMORY_DIM + dim, mask=mask, other=0.0)
        v1 = tl.load(values_ptr + vbase + 1 * MEMORY_DIM + dim, mask=mask, other=0.0)
        v2 = tl.load(values_ptr + vbase + 2 * MEMORY_DIM + dim, mask=mask, other=0.0)
        v3 = tl.load(values_ptr + vbase + 3 * MEMORY_DIM + dim, mask=mask, other=0.0)
        raw0 = w0.to(tl.float32) * v0.to(tl.float32)
        raw1 = w1.to(tl.float32) * v1.to(tl.float32)
        raw2 = w2.to(tl.float32) * v2.to(tl.float32)
        raw3 = w3.to(tl.float32) * v3.to(tl.float32)
        p0 = raw0.to(tl.bfloat16).to(tl.float32)
        p1 = raw1.to(tl.bfloat16).to(tl.float32)
        p2 = raw2.to(tl.bfloat16).to(tl.float32)
        p3 = raw3.to(tl.bfloat16).to(tl.float32)
        for slot in range(4):
            raw = tl.where(slot == 0, raw0, tl.where(slot == 1, raw1, tl.where(slot == 2, raw2, raw3)))
            rounded = tl.where(slot == 0, p0, tl.where(slot == 1, p1, tl.where(slot == 2, p2, p3)))
            out_base = query_row * 4 * MEMORY_DIM + slot * MEMORY_DIM
            tl.store(raw_products_ptr + out_base + dim, raw, mask=mask)
            tl.store(rounded_products_ptr + out_base + dim, rounded, mask=mask)
        sequential = p0 + p1 + p2 + p3
        pairwise = (p0 + p1) + (p2 + p3)
        obase = query_row * MEMORY_DIM
        tl.store(sequential_ptr + obase + dim, sequential, mask=mask)
        tl.store(pairwise_ptr + obase + dim, pairwise, mask=mask)
        tl.store(recalled_bf16_ptr + obase + dim, sequential, mask=mask)


def _run_product_microkernel(weights: torch.Tensor, values: torch.Tensor) -> dict[str, torch.Tensor]:
    if triton is None or _product_reduction_checkpoint_kernel is None:
        raise RuntimeError("#452 diagnostic Triton kernel unavailable")
    batch, time, topk = weights.shape
    if topk != 4 or values.shape != (batch, time, 4, probe.MEMORY_DIM):
        raise RuntimeError("#452 product input geometry drifted")
    queries = batch * time
    raw = torch.empty_like(values, dtype=torch.float32)
    rounded = torch.empty_like(values, dtype=torch.float32)
    sequential = torch.empty((batch, time, probe.MEMORY_DIM), device=values.device, dtype=torch.float32)
    pairwise = torch.empty_like(sequential)
    recalled_bf16 = torch.empty((batch, time, probe.MEMORY_DIM), device=values.device, dtype=torch.bfloat16)
    _product_reduction_checkpoint_kernel[(queries,)](
        weights,
        values,
        raw,
        rounded,
        sequential,
        pairwise,
        recalled_bf16,
        MEMORY_DIM=probe.MEMORY_DIM,
        DIM_BLOCK=64,
        num_warps=4,
    )
    torch.cuda.synchronize()
    return {
        "raw_products_fp32": raw,
        "rounded_products_fp32": rounded,
        "sequential_fp32": sequential,
        "pairwise_fp32": pairwise,
        "recalled_bf16": recalled_bf16,
    }


def run_localization() -> dict[str, Any]:
    contract = cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("#452 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"#452 requires NVIDIA L4, found {device_name}")

    memory = probe.build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = TritonFICEMReadBackend()
    case = reconstruct_target_case(device)

    primary = probe.correctness_row(memory, case, reference, candidate)
    false_subgates = [name for name in PRIMARY_SUBGATES if not bool(primary[name])]

    inputs = _reference_product_inputs(memory, case)
    with torch.no_grad():
        reference_recalled, reference_indices = probe._reference_tail(inputs["similarity"], case.state)
        production_recalled, production_indices = fused_ficem_read_tail(
            inputs["similarity"], case.state.strengths, case.state.valid, case.state.values,
            return_top_indices=True,
        )
        reference_full = probe._full_read(reference, memory, case)
        production_full = probe._full_read(candidate, memory, case)
    torch.cuda.synchronize()
    if production_indices is None:
        raise RuntimeError("#452 production indices missing")

    micro = _run_product_microkernel(inputs["weights"], inputs["selected_values"])
    products_reference_fp32 = inputs["products_bf16"].float()

    return {
        "contract": contract,
        "device": device_name,
        "target_row": TARGET_ROW,
        "primary_correctness_row": primary,
        "primary_pass": bool(primary["pass"]),
        "primary_false_subgates": false_subgates,
        "reference_vs_production_pre_out": _error_stats(reference_recalled, production_recalled),
        "reference_vs_production_final_out": _error_stats(reference_full.recalled, production_full.recalled),
        "reference_vs_production_top_indices_bit_equal": bool(torch.equal(reference_indices, production_indices.to(torch.long))),
        "product_checkpoints": {
            "reference_products_bf16_vs_micro_rounded_fp32": _error_stats(products_reference_fp32, micro["rounded_products_fp32"]),
            "first_product_mismatches": _first_mismatches(products_reference_fp32, micro["rounded_products_fp32"]),
            "reference_native_recalled_vs_micro_bf16": _error_stats(inputs["native_recalled"], micro["recalled_bf16"]),
            "first_recalled_mismatches": _first_mismatches(inputs["native_recalled"], micro["recalled_bf16"]),
            "reference_sequential_fp32_vs_micro_sequential_fp32": _error_stats(inputs["sequential_fp32"], micro["sequential_fp32"]),
            "reference_pairwise_fp32_vs_micro_pairwise_fp32": _error_stats(inputs["pairwise_fp32"], micro["pairwise_fp32"]),
            "native_bf16_sum_vs_sequential_fp32_cast_bf16": _error_stats(inputs["native_recalled"], inputs["sequential_fp32"].to(torch.bfloat16)),
            "native_bf16_sum_vs_pairwise_fp32_cast_bf16": _error_stats(inputs["native_recalled"], inputs["pairwise_fp32"].to(torch.bfloat16)),
            "micro_bf16_vs_production_repair4": _error_stats(micro["recalled_bf16"], production_recalled),
            "diagnostic_product_mirror_matches_production": bool(torch.equal(micro["recalled_bf16"], production_recalled)),
        },
        "localization_only": True,
        "timing_performed": False,
        "profiling_performed": False,
        "performance_decision": None,
        "candidate_acceptance_changed": False,
        "production_backend_modified": False,
        "production_probe_modified": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
