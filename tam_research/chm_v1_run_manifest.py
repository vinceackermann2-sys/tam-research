from __future__ import annotations

"""Zero-credit run-manifest validator for CHM-v1 / EIEM issue #854.

This module records and validates already-frozen evaluation/resource choices. It
contains no launcher, workflow, Modal import, GPU allocation, or paid-compute
authority. A manifest can describe an authorized future run, but this code does
not create or grant that authorization.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .chm_v1_provenance_gate import evaluate_scientific_gate_with_provenance
from .chm_v1_small_lm import SCIENTIFIC_SEEDS

LONG_MEMORY_EVALUATOR_SEED = 8_540_911
VALIDATION_SAMPLING_SEED = 8_540_912
SYSTEMS_TIMING_SEED = 8_540_913
VALIDATION_TOKENS_PER_MODEL = 524_288
VALIDATION_BATCHES = 64
VALIDATION_BATCH_SIZE = 8
SESSION_TOKENS = 1_024
CASES_PER_FAMILY = 24
TOTAL_LONG_MEMORY_CASES = 96
RARE_OVERWRITE_MEMORY_ITEMS = 512
TWO_HOP_MEMORY_ITEMS = 1_024
TIMING_BATCH1_SESSIONS = 8
TIMING_THROUGHPUT_BATCHES = 8
TIMING_BATCH_SIZE = 8
TIMING_SESSION_TOKENS = 1_024

FROZEN_RESOURCE_PLAN: dict[str, Any] = {
    "gpu_class": "NVIDIA L4",
    "cpu_cores": 4,
    "ram_gib": 8,
    "max_seconds_per_seed": 3_600,
    "max_gpu_seconds_total": 10_800,
    "scientific_seed_order": list(SCIENTIFIC_SEEDS),
    "sequential_only": True,
    "automatic_retries": False,
    "max_aggregate_billed_compute_usd": 4.0,
}

FROZEN_EVALUATION_PLAN: dict[str, Any] = {
    "long_memory_evaluator_seed": LONG_MEMORY_EVALUATOR_SEED,
    "validation_sampling_seed": VALIDATION_SAMPLING_SEED,
    "systems_timing_seed": SYSTEMS_TIMING_SEED,
    "validation_tokens_per_model": VALIDATION_TOKENS_PER_MODEL,
    "validation_batches": VALIDATION_BATCHES,
    "validation_batch_size": VALIDATION_BATCH_SIZE,
    "session_tokens": SESSION_TOKENS,
    "cases_per_family": CASES_PER_FAMILY,
    "total_long_memory_cases": TOTAL_LONG_MEMORY_CASES,
    "rare_overwrite_memory_items": RARE_OVERWRITE_MEMORY_ITEMS,
    "two_hop_memory_items": TWO_HOP_MEMORY_ITEMS,
    "timing_batch1_sessions": TIMING_BATCH1_SESSIONS,
    "timing_throughput_batches": TIMING_THROUGHPUT_BATCHES,
    "timing_batch_size": TIMING_BATCH_SIZE,
    "timing_session_tokens": TIMING_SESSION_TOKENS,
}


def frozen_run_manifest(*, authorization_ref: str) -> dict[str, Any]:
    """Return the exact preregistered manifest with an external auth reference.

    ``authorization_ref`` is documentary provenance only. Passing a string here
    does not authorize, launch, or allocate any paid resource.
    """
    if not isinstance(authorization_ref, str) or not authorization_ref.strip():
        raise ValueError("authorization_ref must be a non-empty external reference")
    return {
        "research_issue": 854,
        "evaluation_plan": dict(FROZEN_EVALUATION_PLAN),
        "frozen_resource_plan": dict(FROZEN_RESOURCE_PLAN),
        "authorization_ref": authorization_ref.strip(),
    }


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _exact_mapping(actual: Mapping[str, Any], expected: Mapping[str, Any], path: str) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(f"{path} drifted from the frozen #854 manifest")


def validate_run_manifest(value: Any) -> dict[str, Any]:
    """Validate the shared manifest without granting execution authority."""
    manifest = _mapping(value, "run_manifest")
    if manifest.get("research_issue") != 854:
        raise ValueError("run_manifest.research_issue must be 854")

    evaluation = _mapping(manifest.get("evaluation_plan"), "run_manifest.evaluation_plan")
    _exact_mapping(evaluation, FROZEN_EVALUATION_PLAN, "run_manifest.evaluation_plan")

    resources = _mapping(
        manifest.get("frozen_resource_plan"), "run_manifest.frozen_resource_plan"
    )
    _exact_mapping(resources, FROZEN_RESOURCE_PLAN, "run_manifest.frozen_resource_plan")

    authorization_ref = manifest.get("authorization_ref")
    if not isinstance(authorization_ref, str) or not authorization_ref.strip():
        raise ValueError("run_manifest.authorization_ref must be a non-empty external reference")

    # Re-check arithmetic relationships explicitly so future edits cannot retain
    # superficially plausible but internally inconsistent constants.
    if VALIDATION_BATCHES * VALIDATION_BATCH_SIZE * SESSION_TOKENS != VALIDATION_TOKENS_PER_MODEL:
        raise RuntimeError("frozen validation-token arithmetic is internally inconsistent")
    if CASES_PER_FAMILY * 4 != TOTAL_LONG_MEMORY_CASES:
        raise RuntimeError("frozen long-memory case arithmetic is internally inconsistent")
    if FROZEN_RESOURCE_PLAN["max_seconds_per_seed"] * len(SCIENTIFIC_SEEDS) != FROZEN_RESOURCE_PLAN["max_gpu_seconds_total"]:
        raise RuntimeError("frozen GPU-second arithmetic is internally inconsistent")

    return {
        "research_issue": 854,
        "evaluation_plan": dict(FROZEN_EVALUATION_PLAN),
        "frozen_resource_plan": dict(FROZEN_RESOURCE_PLAN),
        "authorization_ref": authorization_ref.strip(),
    }


def evaluate_scientific_gate_with_manifest(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate frozen manifest, then provenance, then unchanged science gates."""
    validated = validate_run_manifest(manifest)
    result = dict(evaluate_scientific_gate_with_provenance(records))
    result["run_manifest"] = validated
    return result
