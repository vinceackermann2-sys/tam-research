from __future__ import annotations

"""Localization-only full-row diagnostic for the #438/#439 discrepancy.

This module keeps the production repair3 backend and frozen #418/#433 probe
unchanged.  It reconstructs only the original bfloat16_batch8_mixed fixture,
records the frozen ``correctness_row`` dictionary as primary evidence, and then
decomposes full-read-only gates without timing, profiling, training, or changing
candidate acceptance.
"""

import hashlib
from typing import Any

import torch

from . import aera_v26_3_ficem_read_probe as probe
from .aera_hardware_core_v26 import TorchFICEMReferenceBackend
from .aera_hardware_core_v26_3_ficem_read_triton import TritonFICEMReadBackend

RESEARCH_ISSUE = 442
SOURCE_MAIN = "58e7a2d15b7bc935eecb3ffce9097111adc8bcd7"
SOURCE_FAILED_TRIGGER = 438
SOURCE_FAILED_ACTIONS_RUN = 33510242472
SOURCE_LOCALIZATION_ISSUE = 439
SOURCE_LOCALIZATION_TRIGGER = 441
SOURCE_LOCALIZATION_ACTIONS_RUN = 33512923203
FROZEN_PROBE_GIT_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR3_BACKEND_GIT_BLOB = "b6b37f0379b280eea4e5c2b16f349951dadc4df9"
TARGET_ROW = "bfloat16_batch8_mixed"
TARGET_ORDINAL = 5

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
    if probe.DESIGN_SEED != 408_411:
        raise RuntimeError("#442 requires the frozen #411 design seed")
    if (probe.BF16_ATOL, probe.BF16_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("#442 BF16 tolerance drifted")
    rows = _ordinary_rows()
    if rows != [
        "float32_batch8_mixed",
        "float32_batch8_full",
        "float32_batch64_mixed",
        "float32_batch64_full",
        "bfloat16_batch8_mixed",
        "bfloat16_batch8_full",
        "bfloat16_batch64_mixed",
        "bfloat16_batch64_full",
    ]:
        raise RuntimeError("#442 ordinary row order drifted")
    if rows[TARGET_ORDINAL - 1] != TARGET_ROW:
        raise RuntimeError("#442 target ordinal drifted")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_failed_trigger": SOURCE_FAILED_TRIGGER,
        "source_failed_actions_run": SOURCE_FAILED_ACTIONS_RUN,
        "source_localization_issue": SOURCE_LOCALIZATION_ISSUE,
        "source_localization_trigger": SOURCE_LOCALIZATION_TRIGGER,
        "source_localization_actions_run": SOURCE_LOCALIZATION_ACTIONS_RUN,
        "frozen_probe_git_blob": FROZEN_PROBE_GIT_BLOB,
        "frozen_repair3_backend_git_blob": FROZEN_REPAIR3_BACKEND_GIT_BLOB,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "design_seed": probe.DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "bf16_atol": probe.BF16_ATOL,
        "bf16_rtol": probe.BF16_RTOL,
        "primary_evidence_is_frozen_correctness_row": True,
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
    """Replay the frozen ordinary generator through row five with no alternatives."""
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
                        raise RuntimeError("#442 replay reached the wrong target row")
                    return case
    raise RuntimeError("#442 failed to reconstruct target row")


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "numel": int(cpu.numel()),
    }


def _case_fingerprints(case) -> dict[str, dict[str, Any]]:
    return {
        "identity": _tensor_fingerprint(case.identity),
        "context": _tensor_fingerprint(case.context),
        "keys": _tensor_fingerprint(case.state.keys),
        "values": _tensor_fingerprint(case.state.values),
        "strengths": _tensor_fingerprint(case.state.strengths),
        "valid": _tensor_fingerprint(case.state.valid),
    }


def _error_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    ref = reference.float()
    cand = candidate.float()
    diff = (cand - ref).abs()
    allowed = atol + rtol * ref.abs()
    failing = diff > allowed
    fail_count = int(failing.sum())
    total = int(failing.numel())
    return {
        "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "bit_equal": bool(torch.equal(reference, candidate)),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "failing_element_count": fail_count,
        "failing_element_fraction": fail_count / total if total else 0.0,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "reference_device": str(reference.device),
        "candidate_device": str(candidate.device),
        "shape": list(reference.shape),
        "shape_equal": reference.shape == candidate.shape,
    }


def _false_subgates(correctness: dict[str, Any]) -> list[str]:
    return [name for name in PRIMARY_SUBGATES if not bool(correctness[name])]


def _decompose_full_row(
    memory,
    case,
    reference: TorchFICEMReferenceBackend,
    candidate: TritonFICEMReadBackend,
) -> dict[str, Any]:
    reference_result = probe._full_read(reference, memory, case)
    candidate_result = probe._full_read(candidate, memory, case)
    query, keys, similarity = probe._diagnostic_tail_inputs(memory, case)
    with torch.no_grad():
        reference_recalled, reference_indices = probe._reference_tail(similarity, case.state)
        candidate_recalled, candidate_indices = probe.fused_ficem_read_tail(
            similarity,
            case.state.strengths,
            case.state.valid,
            case.state.values,
            return_top_indices=True,
        )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("#442 candidate diagnostic indices missing")

    projected_query_exact = bool(
        reference_result.projected_query is not None
        and candidate_result.projected_query is not None
        and torch.equal(reference_result.projected_query, candidate_result.projected_query)
        and torch.equal(reference_result.projected_query, query)
    )
    normalized_old_keys_exact = bool(
        reference_result.normalized_old_keys is not None
        and candidate_result.normalized_old_keys is not None
        and torch.equal(
            reference_result.normalized_old_keys,
            candidate_result.normalized_old_keys,
        )
        and torch.equal(reference_result.normalized_old_keys, keys)
    )
    atol, rtol = probe.BF16_ATOL, probe.BF16_RTOL
    return {
        "projected_query_exact": projected_query_exact,
        "normalized_old_keys_exact": normalized_old_keys_exact,
        "combined_reuse_exact": projected_query_exact and normalized_old_keys_exact,
        "pre_out_recalled_error": _error_stats(
            reference_recalled, candidate_recalled, atol=atol, rtol=rtol
        ),
        "final_learned_out_error": _error_stats(
            reference_result.recalled, candidate_result.recalled, atol=atol, rtol=rtol
        ),
        "reference_candidate_top_indices_shape_equal": (
            reference_indices.shape == candidate_indices.shape
        ),
        "raw_top_indices_bit_equal": bool(
            torch.equal(reference_indices, candidate_indices.to(torch.long))
        ),
    }


def _sequence_replay(device: torch.device) -> dict[str, Any]:
    """Replay the preceding ordinary correctness rows before the target, no timing."""
    memory = probe.build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = TritonFICEMReadBackend()
    generator = torch.Generator().manual_seed(probe.DESIGN_SEED)
    preceding: list[dict[str, Any]] = []
    target_case = None
    target_correctness = None

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
                correctness = probe.correctness_row(memory, case, reference, candidate)
                if ordinal < TARGET_ORDINAL:
                    preceding.append(
                        {
                            "row": key,
                            "pass": bool(correctness["pass"]),
                            "false_subgates": _false_subgates(correctness),
                        }
                    )
                    continue
                if ordinal == TARGET_ORDINAL:
                    target_case = case
                    target_correctness = correctness
                    break
            if target_correctness is not None:
                break
        if target_correctness is not None:
            break

    if target_case is None or target_correctness is None:
        raise RuntimeError("#442 sequence replay did not reach target")
    return {
        "preceding_rows": preceding,
        "target_correctness_row": target_correctness,
        "target_false_subgates": _false_subgates(target_correctness),
        "target_fixture_fingerprints": _case_fingerprints(target_case),
    }


def run_localization() -> dict[str, Any]:
    contract = cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("#442 localization requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"#442 requires NVIDIA L4, found {device_name}")

    memory = probe.build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = TritonFICEMReadBackend()
    case = reconstruct_target_case(device)
    fixture_before = _case_fingerprints(case)

    # Primary evidence: invoke the exact frozen gate once, with no reinterpretation.
    primary = probe.correctness_row(memory, case, reference, candidate)
    primary_false = _false_subgates(primary)
    fixture_after_primary = _case_fingerprints(case)

    decomposition = _decompose_full_row(memory, case, reference, candidate)
    sequence = _sequence_replay(device)
    sequence_fingerprint_match = (
        sequence["target_fixture_fingerprints"] == fixture_before
    )

    return {
        "contract": contract,
        "device": device_name,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "primary_correctness_row": primary,
        "primary_pass": bool(primary["pass"]),
        "primary_false_subgates": primary_false,
        "fixture_fingerprints_before_primary": fixture_before,
        "fixture_fingerprints_after_primary": fixture_after_primary,
        "fixture_unchanged_by_primary": fixture_before == fixture_after_primary,
        "full_row_decomposition": decomposition,
        "sequence_replay": sequence,
        "sequence_target_fixture_matches_primary": sequence_fingerprint_match,
        "sequence_target_matches_primary_correctness": (
            sequence["target_correctness_row"] == primary
        ),
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
