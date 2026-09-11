from __future__ import annotations

import copy

import pytest

from tam_research.chm_v1_corpus_fingerprint import (
    EXPECTED_META,
    EXPECTED_TRAIN_BYTES,
    EXPECTED_VAL_BYTES,
)
from tam_research.chm_v1_provenance_gate import (
    evaluate_scientific_gate_with_provenance,
    validate_corpus_fingerprint_record,
    validate_execution_code_sha,
)
from tam_research.chm_v1_scientific_gate import (
    REQUIRED_ABLATION_KEYS,
    REQUIRED_MEASUREMENT_KEYS,
    SCIENTIFIC_GENERATOR_VERSION,
)
from tam_research.chm_v1_small_lm import SCIENTIFIC_SEEDS

EXECUTION_SHA = "a" * 40


def _fingerprint(train_hash: str = "1" * 64) -> dict[str, object]:
    return {
        "logical_contract": dict(EXPECTED_META),
        "files": [
            {"name": "train.bin", "bytes": EXPECTED_TRAIN_BYTES, "sha256": train_hash},
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


def test_provenance_bound_gate_preserves_existing_full_pass_math() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    result = evaluate_scientific_gate_with_provenance(records)
    assert result["full_pass"] is True
    assert result["classification"] == "FULL_PASS_SMALL_LM_ONLY"
    assert result["provenance"]["execution_code_sha"] == EXECUTION_SHA
    assert result["provenance"]["all_seed_records_match"] is True


def test_execution_sha_must_be_full_lowercase_git_sha() -> None:
    assert validate_execution_code_sha(EXECUTION_SHA) == EXECUTION_SHA
    for invalid in (None, "a" * 39, "A" * 40, "z" * 40):
        with pytest.raises(ValueError, match="40-hex"):
            validate_execution_code_sha(invalid)


def test_missing_provenance_refuses_scientific_classification() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0].pop("corpus_fingerprint")
    with pytest.raises(ValueError, match="must be a mapping"):
        evaluate_scientific_gate_with_provenance(records)


def test_cross_seed_execution_sha_disagreement_refuses_classification() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[1]["execution_code_sha"] = "b" * 40
    with pytest.raises(ValueError, match="disagree on execution_code_sha"):
        evaluate_scientific_gate_with_provenance(records)


def test_cross_seed_same_size_different_corpus_hash_refuses_classification() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[2]["corpus_fingerprint"] = _fingerprint(train_hash="4" * 64)
    with pytest.raises(ValueError, match="disagree on corpus_fingerprint"):
        evaluate_scientific_gate_with_provenance(records)


def test_fingerprint_validation_rejects_contract_size_hash_and_file_drift() -> None:
    drifted = _fingerprint()
    drifted["logical_contract"] = dict(EXPECTED_META)
    drifted["logical_contract"]["seed"] = 8101
    with pytest.raises(ValueError, match="logical contract drifted"):
        validate_corpus_fingerprint_record(drifted)

    wrong_size = _fingerprint()
    wrong_size["files"][0]["bytes"] = EXPECTED_TRAIN_BYTES - 2
    with pytest.raises(ValueError, match="train.bin bytes drifted"):
        validate_corpus_fingerprint_record(wrong_size)

    bad_hash = _fingerprint()
    bad_hash["files"][1]["sha256"] = "X" * 64
    with pytest.raises(ValueError, match="lowercase 64-hex"):
        validate_corpus_fingerprint_record(bad_hash)

    wrong_name = _fingerprint()
    wrong_name["files"][2]["name"] = "other.json"
    with pytest.raises(ValueError, match="filenames must be exactly"):
        validate_corpus_fingerprint_record(wrong_name)


def test_provenance_wrapper_cannot_rescue_existing_gate_failure() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0]["indexed"]["exact_match_rate"] = 0.99
    result = evaluate_scientific_gate_with_provenance(records)
    assert result["full_pass"] is False
    assert result["criteria"]["exactness"] is False
