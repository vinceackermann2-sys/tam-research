from __future__ import annotations

import copy

import pytest

from tam_research.chm_v1_corpus_fingerprint import (
    EXPECTED_META,
    EXPECTED_TRAIN_BYTES,
    EXPECTED_VAL_BYTES,
)
from tam_research.chm_v1_run_manifest import (
    FROZEN_EVALUATION_PLAN,
    FROZEN_RESOURCE_PLAN,
    evaluate_scientific_gate_with_manifest,
    frozen_run_manifest,
    validate_run_manifest,
)
from tam_research.chm_v1_scientific_gate import (
    REQUIRED_ABLATION_KEYS,
    REQUIRED_MEASUREMENT_KEYS,
    SCIENTIFIC_GENERATOR_VERSION,
)
from tam_research.chm_v1_small_lm import SCIENTIFIC_SEEDS

EXECUTION_SHA = "a" * 40
AUTH_REF = "https://github.com/vinceackermann2-sys/tam-research/issues/854#issuecomment-9999999999"


def _fingerprint() -> dict[str, object]:
    return {
        "logical_contract": dict(EXPECTED_META),
        "files": [
            {"name": "train.bin", "bytes": EXPECTED_TRAIN_BYTES, "sha256": "1" * 64},
            {"name": "val.bin", "bytes": EXPECTED_VAL_BYTES, "sha256": "2" * 64},
            {"name": "meta.json", "bytes": 123, "sha256": "3" * 64},
        ],
    }


def _record(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "generator_version": SCIENTIFIC_GENERATOR_VERSION,
        "execution_code_sha": EXECUTION_SHA,
        "corpus_fingerprint": _fingerprint(),
        "language": {"local_nll": 4.0, "eiem_flat_nll": 4.05},
        "families": {
            "rare_fact": {"local_accuracy": 0.20, "eiem_accuracy": 0.40},
            "overwrite": {
                "local_accuracy": 0.25,
                "eiem_accuracy": 0.45,
                "local_stale_error": 0.50,
                "eiem_stale_error": 0.20,
            },
            "two_hop": {"local_accuracy": 0.15, "eiem_accuracy": 0.35},
            "local_negative": {"local_accuracy": 0.80, "eiem_accuracy": 0.80},
        },
        "indexed": {
            "exact_match_rate": 1.0,
            "indexed_reads_1024": 20,
            "flat_reads_1024": 100,
        },
        "systems": {
            "no_nan_inf": True,
            "no_cross_session_aliasing": True,
            "no_hidden_persistent_state": True,
            "no_base_parameter_mutation": True,
        },
        "stop_conditions": {
            "oracle_or_future_leakage": False,
            "benefit_disappears_out_of_template": False,
            "index_overhead_erases_practical_advantage": False,
            "simpler_control_reproduces_frontier": False,
            "numerical_or_fairness_violation": False,
        },
        "measurements": {key: {"present": True} for key in REQUIRED_MEASUREMENT_KEYS},
        "ablations": {key: {"present": True} for key in REQUIRED_ABLATION_KEYS},
    }


def test_frozen_manifest_matches_preregistered_evaluation_and_resource_plan() -> None:
    manifest = frozen_run_manifest(authorization_ref=AUTH_REF)
    validated = validate_run_manifest(manifest)
    assert validated["evaluation_plan"] == FROZEN_EVALUATION_PLAN
    assert validated["frozen_resource_plan"] == FROZEN_RESOURCE_PLAN
    assert validated["authorization_ref"] == AUTH_REF


def test_manifest_bound_gate_preserves_existing_full_pass_math() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    result = evaluate_scientific_gate_with_manifest(
        records, frozen_run_manifest(authorization_ref=AUTH_REF)
    )
    assert result["full_pass"] is True
    assert result["classification"] == "FULL_PASS_SMALL_LM_ONLY"
    assert result["run_manifest"]["evaluation_plan"] == FROZEN_EVALUATION_PLAN


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("long_memory_evaluator_seed", 8_540_910),
        ("validation_sampling_seed", 8_540_911),
        ("systems_timing_seed", 8_540_912),
        ("validation_tokens_per_model", 524_287),
        ("cases_per_family", 23),
        ("two_hop_memory_items", 512),
        ("timing_throughput_batches", 7),
    ],
)
def test_any_evaluation_manifest_drift_refuses_classification(key: str, bad_value: object) -> None:
    manifest = frozen_run_manifest(authorization_ref=AUTH_REF)
    manifest = copy.deepcopy(manifest)
    manifest["evaluation_plan"][key] = bad_value
    with pytest.raises(ValueError, match="evaluation_plan drifted"):
        validate_run_manifest(manifest)


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("gpu_class", "NVIDIA T4"),
        ("cpu_cores", 8),
        ("ram_gib", 16),
        ("max_seconds_per_seed", 3_601),
        ("max_gpu_seconds_total", 10_801),
        ("scientific_seed_order", [8612, 8611, 8613]),
        ("sequential_only", False),
        ("automatic_retries", True),
        ("max_aggregate_billed_compute_usd", 4.01),
    ],
)
def test_any_resource_plan_drift_refuses_classification(key: str, bad_value: object) -> None:
    manifest = frozen_run_manifest(authorization_ref=AUTH_REF)
    manifest = copy.deepcopy(manifest)
    manifest["frozen_resource_plan"][key] = bad_value
    with pytest.raises(ValueError, match="frozen_resource_plan drifted"):
        validate_run_manifest(manifest)


def test_authorization_reference_is_documentary_but_mandatory() -> None:
    with pytest.raises(ValueError, match="authorization_ref"):
        frozen_run_manifest(authorization_ref="")

    manifest = frozen_run_manifest(authorization_ref=AUTH_REF)
    manifest["authorization_ref"] = "   "
    with pytest.raises(ValueError, match="authorization_ref"):
        validate_run_manifest(manifest)


def test_manifest_layer_cannot_rescue_scientific_failure() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0]["indexed"]["exact_match_rate"] = 0.99
    result = evaluate_scientific_gate_with_manifest(
        records, frozen_run_manifest(authorization_ref=AUTH_REF)
    )
    assert result["full_pass"] is False
    assert result["criteria"]["exactness"] is False
