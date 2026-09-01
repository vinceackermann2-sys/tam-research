from __future__ import annotations

"""Issue #460 reporting-only repair for the exhausted #459 localization harness.

The merged #456 diagnostic remains byte-for-byte frozen.  This shim substitutes only
its private reporting comparator for the duration of one localization call and
restores the original helper in ``finally``.  No tensor produced by the reporting
copies is fed back into reference, candidate, model, fixture, or acceptance paths.
"""

from typing import Any

import torch

from . import aera_v26_3_repair4_reference_autocast_localize as frozen_localize

RESEARCH_ISSUE = 460
SOURCE_DIAGNOSTIC_ISSUE = 456
SOURCE_TRIGGER = 459
SOURCE_RUN = 33546452532
SOURCE_JOB = 99985042556
SOURCE_MAIN = "d620b2a349ebb8e494b397bc534430abaeba394f"
TARGET_ROW = frozen_localize.TARGET_ROW
TARGET_ORDINAL = frozen_localize.TARGET_ORDINAL
FROZEN_LOCALIZATION_GIT_BLOB = "8ed7de14a0f29f3ac66d6228a71892fbf97e150f"
FROZEN_PROBE_GIT_BLOB = frozen_localize.FROZEN_PROBE_GIT_BLOB
FROZEN_REPAIR4_BACKEND_GIT_BLOB = frozen_localize.FROZEN_REPAIR4_BACKEND_GIT_BLOB


def cpu_contract_preflight() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_diagnostic_issue": SOURCE_DIAGNOSTIC_ISSUE,
        "source_trigger": SOURCE_TRIGGER,
        "source_run": SOURCE_RUN,
        "source_job": SOURCE_JOB,
        "source_main": SOURCE_MAIN,
        "source_failure": "torch.isclose Float did not match BFloat16",
        "source_authoritative_result_marker_emitted": False,
        "source_attempt_exhausted": True,
        "frozen_localization_git_blob": FROZEN_LOCALIZATION_GIT_BLOB,
        "frozen_probe_git_blob": FROZEN_PROBE_GIT_BLOB,
        "frozen_repair4_backend_git_blob": FROZEN_REPAIR4_BACKEND_GIT_BLOB,
        "target_row": TARGET_ROW,
        "target_ordinal": TARGET_ORDINAL,
        "design_seed": frozen_localize.probe.DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "original_global_case_order_preserved": True,
        "resampling": False,
        "rejection_sampling": False,
        "fixture_nudging": False,
        "alternate_seed": False,
        "frozen_correctness_row_invocation_unchanged": True,
        "dtype_safe_reporting_only": True,
        "numeric_reporting_dtype": "torch.float32",
        "mixed_dtype_bit_equality_not_applicable": True,
        "native_execution_tensors_unchanged": True,
        "outside_projection_control_unchanged": True,
        "localization_only": True,
        "timing_authorized": False,
        "profiling_authorized": False,
        "performance_decision_authorized": False,
        "production_backend_modified": False,
        "production_probe_modified": False,
        "source_localization_modified": False,
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


def _reporting_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Return an isolated FP32 reporting copy without mutating the native tensor."""

    return tensor.detach().to(dtype=torch.float32).clone()


def _first_reported_mismatch(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    dtype_equal: bool,
    reference_report: torch.Tensor,
    candidate_report: torch.Tensor,
) -> dict[str, Any] | None:
    if reference.shape != candidate.shape:
        return {"kind": "shape", "shape_mismatch": [list(reference.shape), list(candidate.shape)]}

    if dtype_equal:
        unequal = reference != candidate
        mismatch_kind = "native_exact"
    else:
        unequal = reference_report != candidate_report
        mismatch_kind = "numeric_fp32"

    if not bool(unequal.any()):
        return None
    first = unequal.nonzero(as_tuple=False)[0]
    coord = tuple(int(x) for x in first.tolist())
    return {
        "kind": mismatch_kind,
        "coordinate": list(coord),
        "reference": float(reference_report[coord]),
        "candidate": float(candidate_report[coord]),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }


def dtype_safe_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare native tensors without requiring their dtypes to match.

    Exact/bit fields retain the frozen #456 same-dtype semantics.  Numerical
    tolerance/error fields are computed only from detached FP32 reporting copies.
    """

    dtype_equal = reference.dtype == candidate.dtype
    device_equal = reference.device == candidate.device
    common = {
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_equal": bool(dtype_equal),
        "reference_device": str(reference.device),
        "candidate_device": str(candidate.device),
        "device_equal": bool(device_equal),
        "numeric_comparison_dtype": "torch.float32",
        "numeric_reporting_copies_only": True,
    }
    if reference.shape != candidate.shape:
        return {
            **common,
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "bit_equal": None if not dtype_equal else False,
            "bit_mismatch_count": None,
            "bit_mismatch_fraction": None,
            "allclose": False,
            "first_mismatch": {
                "kind": "shape",
                "shape_mismatch": [list(reference.shape), list(candidate.shape)],
            },
        }
    if not device_equal:
        return {
            **common,
            "shape_equal": True,
            "shape": list(reference.shape),
            "bit_equal": None,
            "bit_mismatch_count": None,
            "bit_mismatch_fraction": None,
            "allclose": False,
            "numeric_comparison_performed": False,
            "first_mismatch": {"kind": "device"},
        }

    reference_report = _reporting_fp32(reference)
    candidate_report = _reporting_fp32(candidate)
    close = torch.isclose(reference_report, candidate_report, atol=atol, rtol=rtol)
    finite_pair = torch.isfinite(reference_report) & torch.isfinite(candidate_report)
    finite_diff = torch.where(
        finite_pair,
        (reference_report - candidate_report).abs(),
        torch.zeros((), device=reference_report.device, dtype=torch.float32),
    )
    total = int(reference.numel())

    if dtype_equal:
        native_mismatch = reference != candidate
        bit_equal: bool | None = bool(torch.equal(reference, candidate))
        bit_mismatch_count: int | None = int(native_mismatch.sum())
        bit_mismatch_fraction: float | None = (
            float(native_mismatch.sum() / total) if total else 0.0
        )
    else:
        bit_equal = None
        bit_mismatch_count = None
        bit_mismatch_fraction = None

    failing = int((~close).sum())
    return {
        **common,
        "shape_equal": True,
        "shape": list(reference.shape),
        "numeric_comparison_performed": True,
        "bit_equal": bit_equal,
        "bit_mismatch_count": bit_mismatch_count,
        "bit_mismatch_fraction": bit_mismatch_fraction,
        "allclose": bool(torch.all(close)),
        "max_abs_error": float(finite_diff.max()) if total else 0.0,
        "mean_abs_error": float(finite_diff.mean()) if total else 0.0,
        "failing_element_count": failing,
        "failing_element_fraction": float(failing / total) if total else 0.0,
        "first_mismatch": _first_reported_mismatch(
            reference,
            candidate,
            dtype_equal=dtype_equal,
            reference_report=reference_report,
            candidate_report=candidate_report,
        ),
    }


def run_localization_reportfix1() -> dict[str, Any]:
    """Run the frozen #456 localization with only its reporting comparator replaced."""

    original_stats = frozen_localize._stats
    if original_stats is dtype_safe_stats:
        raise RuntimeError("issue460 reporting helper already substituted")
    frozen_localize._stats = dtype_safe_stats
    try:
        result = frozen_localize.run_localization()
    finally:
        frozen_localize._stats = original_stats

    result["reportfix1_contract"] = cpu_contract_preflight()
    result["reportfix1_applied"] = True
    result["source_diagnostic_issue"] = SOURCE_DIAGNOSTIC_ISSUE
    result["source_failed_trigger"] = SOURCE_TRIGGER
    result["source_failed_run"] = SOURCE_RUN
    result["source_failed_job"] = SOURCE_JOB
    result["source_authoritative_result_marker_emitted"] = False
    result["reporting_only_dtype_promotion"] = True
    result["native_execution_tensors_changed"] = False
    return result
