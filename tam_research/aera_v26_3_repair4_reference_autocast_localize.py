from __future__ import annotations

"""Issue #456 localization of the repair4 full-read/autocast discrepancy.

Reporting only. This module does not alter production FICEM execution or acceptance.
"""

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from . import aera_v26_3_ficem_read_probe as probe
from .aera_hardware_core_v24 import MIN_STRENGTH, READ_TEMPERATURE, READ_TOP_K
from .aera_hardware_core_v26 import TorchFICEMReferenceBackend
from .aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
    fused_ficem_read_tail,
)

RESEARCH_ISSUE = 456
SOURCE_MAIN = "a04e43af14b64205ee84472768cf2be850a88e75"
SOURCE_LOCALIZATION_ISSUE = 452
SOURCE_LOCALIZATION_TRIGGER = 454
SOURCE_LOCALIZATION_RUN = 33539885620
SOURCE_LOCALIZATION_JOB = 99963230232
SOURCE_DUPLICATE_TRIGGER = 455
SOURCE_DUPLICATE_RUN = 33539909378
SOURCE_DUPLICATE_JOB = 99963512537
TARGET_ROW = "bfloat16_batch8_mixed"
TARGET_ORDINAL = 5
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR4_BACKEND_GIT_BLOB = "a3a603c8a2d4b20ebcccd7663970978f4288a760"


def cpu_contract_preflight() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_localization_issue": SOURCE_LOCALIZATION_ISSUE,
        "source_localization_trigger": SOURCE_LOCALIZATION_TRIGGER,
        "source_localization_run": SOURCE_LOCALIZATION_RUN,
        "source_localization_job": SOURCE_LOCALIZATION_JOB,
        "source_duplicate_trigger": SOURCE_DUPLICATE_TRIGGER,
        "source_duplicate_run": SOURCE_DUPLICATE_RUN,
        "source_duplicate_job": SOURCE_DUPLICATE_JOB,
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
        "primary_evidence_is_frozen_correctness_row": True,
        "inside_autocast_reference_path": True,
        "outside_autocast_reference_path": True,
        "same_similarity_tail_control": True,
        "outside_projection_failure_is_reported_not_repaired": True,
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


def _ordinary_rows() -> list[str]:
    return [
        probe._row_key(dtype_name, batch_size, validity_kind)
        for dtype_name in probe.DTYPE_NAMES
        for batch_size in probe.BATCH_SIZES
        for validity_kind in probe.VALIDITY_KINDS
    ]


def _target_case(device: torch.device) -> probe.ReadCase:
    rows = _ordinary_rows()
    if rows[TARGET_ORDINAL - 1] != TARGET_ROW:
        raise RuntimeError("issue456 target ordinal drifted")
    generator = torch.Generator().manual_seed(probe.DESIGN_SEED)
    target = None
    for ordinal, row in enumerate(rows, start=1):
        dtype_name, batch_text, validity_kind = row.split("_", 2)
        batch_size = int(batch_text.removeprefix("batch"))
        case = probe.make_case(
            dtype_name=dtype_name,
            batch_size=batch_size,
            validity_kind=validity_kind,
            generator=generator,
            device=device,
        )
        if ordinal == TARGET_ORDINAL:
            target = case
            break
    if target is None or probe._row_key(
        target.dtype_name, target.batch_size, target.validity_kind
    ) != TARGET_ROW:
        raise RuntimeError("issue456 failed to reconstruct exact target")
    return target


def _autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _autocast_state() -> dict[str, Any]:
    return {
        "enabled": bool(torch.is_autocast_enabled("cuda")),
        "dtype": str(torch.get_autocast_dtype("cuda")),
    }


def _first_mismatch(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any] | None:
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [list(reference.shape), list(candidate.shape)]}
    unequal = reference != candidate
    if not bool(unequal.any()):
        return None
    first = unequal.nonzero(as_tuple=False)[0]
    coord = tuple(int(x) for x in first.tolist())
    return {
        "coordinate": list(coord),
        "reference": float(reference[coord].float()),
        "candidate": float(candidate[coord].float()),
    }


def _stats(
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
        }
    close = torch.isclose(reference, candidate, atol=atol, rtol=rtol)
    mismatch = reference != candidate
    finite_pair = torch.isfinite(reference) & torch.isfinite(candidate)
    finite_diff = torch.where(
        finite_pair,
        (reference.float() - candidate.float()).abs(),
        torch.zeros((), device=reference.device),
    )
    total = int(reference.numel())
    return {
        "shape_equal": True,
        "bit_equal": bool(torch.equal(reference, candidate)),
        "allclose": bool(torch.all(close)),
        "max_abs_error": float(finite_diff.max()) if total else 0.0,
        "mean_abs_error": float(finite_diff.mean()) if total else 0.0,
        "failing_element_count": int((~close).sum()),
        "failing_element_fraction": float((~close).sum() / total) if total else 0.0,
        "bit_mismatch_count": int(mismatch.sum()),
        "bit_mismatch_fraction": float(mismatch.sum() / total) if total else 0.0,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "shape": list(reference.shape),
        "first_mismatch": _first_mismatch(reference, candidate),
    }


def _attempt_memory_out(memory, recalled: torch.Tensor) -> dict[str, Any]:
    """Attempt the literal outside-autocast module call without changing dtypes.

    BF16 input with the unchanged FP32 parameter may be unsupported by ``nn.Linear``
    outside autocast. That is diagnostic evidence, not a reason to coerce either
    operand or modify production state.
    """
    with torch.no_grad():
        try:
            output = memory.out(recalled)
        except RuntimeError as exc:
            return {
                "supported": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "input_dtype": str(recalled.dtype),
                "weight_dtype": str(memory.out.weight.dtype),
            }
    return {
        "supported": True,
        "output": output,
        "output_dtype": str(output.dtype),
        "shape": list(output.shape),
    }


def _tail_math(
    memory,
    similarity: torch.Tensor,
    state,
    *,
    autocast_enabled: bool,
) -> dict[str, Any]:
    with torch.no_grad(), _autocast_context(autocast_enabled):
        state_at_start = _autocast_state()
        strength_bias = torch.log(
            state.strengths.clamp(MIN_STRENGTH, 1.0)
        )[:, None, :]
        logits = (similarity + strength_bias) / READ_TEMPERATURE
        masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
        top_logits, top_indices = torch.topk(masked, k=READ_TOP_K, dim=-1)
        top_valid = state.valid[:, None, :].expand(
            -1, similarity.size(1), -1
        ).gather(-1, top_indices)
        safe_logits = top_logits.masked_fill(~top_valid, -1e9)
        softmax_fp32 = torch.softmax(safe_logits.float(), dim=-1)
        weights_initial = softmax_fp32.to(similarity.dtype)
        weights_valid = weights_initial * top_valid.to(weights_initial.dtype)
        weight_sum = weights_valid.sum(dim=-1, keepdim=True)
        weights = weights_valid / weight_sum.clamp_min(1e-9)
        expanded_values = state.values[:, None, :, :].expand(
            -1, similarity.size(1), -1, -1
        )
        selected_values = expanded_values.gather(
            2,
            top_indices.unsqueeze(-1).expand(
                -1, -1, -1, memory.memory_dim
            ),
        )
        products = weights.unsqueeze(-1) * selected_values
        recalled = products.sum(dim=2)
        output = memory.out(recalled) if autocast_enabled else None
        state_at_end = _autocast_state()
    return {
        "autocast_start": state_at_start,
        "autocast_end": state_at_end,
        "strength_bias": strength_bias,
        "logits": logits,
        "masked": masked,
        "top_logits": top_logits,
        "top_indices": top_indices,
        "safe_logits": safe_logits,
        "softmax_fp32": softmax_fp32,
        "weights_initial": weights_initial,
        "weights_valid": weights_valid,
        "weight_sum": weight_sum,
        "weights": weights,
        "selected_values": selected_values,
        "products": products,
        "recalled": recalled,
        "output": output,
    }


def _full_reference_equations(memory, case: probe.ReadCase) -> dict[str, Any]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state_at_start = _autocast_state()
        _, _, query = memory.address_factors(case.identity, case.context)
        keys = F.normalize(case.state.keys, dim=-1)
        similarity = torch.einsum("btd,bsd->bts", query, keys)
        tail = _tail_math(memory, similarity, case.state, autocast_enabled=True)
        state_at_end = _autocast_state()
    tail.update(
        {
            "full_autocast_start": state_at_start,
            "full_autocast_end": state_at_end,
            "query": query,
            "keys": keys,
            "similarity": similarity,
        }
    )
    return tail


def _production_capture(memory, case: probe.ReadCase) -> dict[str, Any]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state_at_start = _autocast_state()
        _, _, query = memory.address_factors(case.identity, case.context)
        keys = F.normalize(case.state.keys, dim=-1)
        similarity = torch.einsum("btd,bsd->bts", query, keys).contiguous()
        recalled, indices = fused_ficem_read_tail(
            similarity,
            case.state.strengths.contiguous(),
            case.state.valid.contiguous(),
            case.state.values.contiguous(),
            return_top_indices=True,
        )
        output = memory.out(recalled)
        state_at_end = _autocast_state()
    if indices is None:
        raise RuntimeError("issue456 production capture missing top indices")
    return {
        "autocast_start": state_at_start,
        "autocast_end": state_at_end,
        "query": query,
        "keys": keys,
        "similarity": similarity,
        "recalled": recalled,
        "output": output,
        "top_indices": indices.to(torch.long),
    }


def run_localization() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue456 localization requires CUDA")
    if "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("issue456 localization requires NVIDIA L4")

    device = torch.device("cuda")
    memory = probe.build_memory(device)
    case = _target_case(device)
    reference = TorchFICEMReferenceBackend()
    candidate = TritonFICEMReadBackend()
    weight_before = memory.out.weight.detach().clone()
    memory_training_before = bool(memory.training)

    # Frozen primary evidence, exactly once.
    primary = probe.correctness_row(memory, case, reference, candidate)

    reference_full_backend = probe._full_read(reference, memory, case)
    production_full_backend = probe._full_read(candidate, memory, case)
    query, keys, similarity = probe._diagnostic_tail_inputs(memory, case)

    outside = _tail_math(memory, similarity, case.state, autocast_enabled=False)
    outside_reference_recalled, outside_reference_indices = probe._reference_tail(
        similarity, case.state
    )
    same_similarity_inside = _tail_math(
        memory, similarity, case.state, autocast_enabled=True
    )
    actual_reference = _full_reference_equations(memory, case)
    production = _production_capture(memory, case)

    atol, rtol = probe._tolerances(case.dtype_name)
    selection = probe._tie_aware_top4_equivalence(
        probe._reference_masked_logits(similarity, case.state),
        case.state.valid,
        actual_reference["top_indices"],
        production["top_indices"],
    )

    checkpoint_order = (
        "strength_bias",
        "logits",
        "masked",
        "top_logits",
        "safe_logits",
        "softmax_fp32",
        "weights_initial",
        "weights_valid",
        "weight_sum",
        "weights",
        "selected_values",
        "products",
        "recalled",
    )
    inside_vs_outside = {
        name: _stats(actual_reference[name], outside[name], atol=atol, rtol=rtol)
        for name in checkpoint_order
    }
    first_differing_checkpoint = next(
        (
            name
            for name in checkpoint_order
            if not inside_vs_outside[name].get("bit_equal", False)
        ),
        None,
    )

    outside_projection = _attempt_memory_out(memory, outside["recalled"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outside_recalled_inside_out = memory.out(outside["recalled"])
        actual_recalled_inside_out = memory.out(actual_reference["recalled"])
        production_recalled_inside_out = memory.out(production["recalled"])

    if outside_projection["supported"]:
        outside_projection_comparison: dict[str, Any] = {
            "outside_supported": True,
            "stats": _stats(
                outside_recalled_inside_out,
                outside_projection["output"],
                atol=atol,
                rtol=rtol,
            ),
        }
    else:
        outside_projection_comparison = {
            "outside_supported": False,
            "outside_error_type": outside_projection["error_type"],
            "outside_error": outside_projection["error"],
            "input_dtype": outside_projection["input_dtype"],
            "weight_dtype": outside_projection["weight_dtype"],
            "inside_output_dtype": str(outside_recalled_inside_out.dtype),
            "inside_output_shape": list(outside_recalled_inside_out.shape),
        }

    result = {
        "contract": cpu_contract_preflight(),
        "device": torch.cuda.get_device_name(0),
        "target_row": TARGET_ROW,
        "primary_correctness_row": primary,
        "primary_pass": bool(primary["pass"]),
        "primary_false_subgates": [
            key
            for key in (
                "selection_semantically_equivalent",
                "pre_out_recalled_close",
                "final_out_close",
                "query_and_normalized_keys_bit_exact",
                "source_unchanged",
                "finite",
                "dtype_device_shape_exact",
            )
            if not bool(primary[key])
        ],
        "autocast_states": {
            "actual_reference": {
                "start": actual_reference["full_autocast_start"],
                "tail_start": actual_reference["autocast_start"],
                "tail_end": actual_reference["autocast_end"],
                "end": actual_reference["full_autocast_end"],
            },
            "same_similarity_inside": {
                "start": same_similarity_inside["autocast_start"],
                "end": same_similarity_inside["autocast_end"],
            },
            "outside": {
                "start": outside["autocast_start"],
                "end": outside["autocast_end"],
            },
            "production": {
                "start": production["autocast_start"],
                "end": production["autocast_end"],
            },
        },
        "query_vs_reference_backend_reuse": _stats(
            reference_full_backend.projected_query, query
        ),
        "keys_vs_reference_backend_reuse": _stats(
            reference_full_backend.normalized_old_keys, keys
        ),
        "outside_tail_matches_frozen_reference_recalled": _stats(
            outside_reference_recalled, outside["recalled"], atol=0.0, rtol=0.0
        ),
        "outside_tail_matches_frozen_reference_indices": bool(
            torch.equal(outside_reference_indices, outside["top_indices"])
        ),
        "actual_reference_capture_vs_backend_final": _stats(
            reference_full_backend.recalled,
            actual_reference["output"],
            atol=atol,
            rtol=rtol,
        ),
        "production_capture_vs_backend_final": _stats(
            production_full_backend.recalled,
            production["output"],
            atol=atol,
            rtol=rtol,
        ),
        "actual_reference_vs_outside_checkpoints": inside_vs_outside,
        "first_differing_checkpoint_actual_reference_vs_outside": first_differing_checkpoint,
        "same_similarity_inside_vs_outside_pre_out": _stats(
            same_similarity_inside["recalled"],
            outside["recalled"],
            atol=atol,
            rtol=rtol,
        ),
        "actual_reference_vs_outside_pre_out": _stats(
            actual_reference["recalled"],
            outside["recalled"],
            atol=atol,
            rtol=rtol,
        ),
        "actual_reference_vs_production_pre_out": _stats(
            actual_reference["recalled"],
            production["recalled"],
            atol=atol,
            rtol=rtol,
        ),
        "outside_vs_production_pre_out": _stats(
            outside["recalled"], production["recalled"], atol=atol, rtol=rtol
        ),
        "actual_reference_vs_production_final": _stats(
            actual_reference["output"],
            production["output"],
            atol=atol,
            rtol=rtol,
        ),
        "actual_reference_vs_production_selection": selection,
        "actual_reference_top_indices_vs_production_bit_equal": bool(
            torch.equal(actual_reference["top_indices"], production["top_indices"])
        ),
        "outside_recalled_memory_out_inside_vs_outside_autocast": outside_projection_comparison,
        "same_autocast_memory_out_actual_vs_production_recalled": _stats(
            actual_recalled_inside_out,
            production_recalled_inside_out,
            atol=atol,
            rtol=rtol,
        ),
        "projection_weight_unchanged": bool(
            torch.equal(memory.out.weight.detach(), weight_before)
        ),
        "memory_training_state_unchanged": bool(
            bool(memory.training) == memory_training_before
        ),
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
    return result
