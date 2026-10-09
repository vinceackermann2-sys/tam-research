"""Deterministic synthetic decisions only; zero GPU/Modal use."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tam_research.cortex_attention8_conv7_engineering_rehearsal_v1 import (
    SYNTHETIC_TAG,
    classify_synthetic_screen,
)

NAMES = (
    "fresh_transformer",
    "reduced_attention_dense_v2_attention8",
    "attention8_causal_depthwise_conv7",
)


def fixture_records():
    records = {}
    for name, nll, tps in zip(NAMES, (3.500, 3.545, 3.505), (300000.0, 319000.0, 315000.0)):
        records[name] = {
            "synthetic_tag": SYNTHETIC_TAG,
            "model_name": name,
            "seed": SYNTHETIC_TAG,
            "status": "COMPLETE",
            "optimizer_steps": 3052,
            "token_exposures": 200015872,
            "heldout_tokens": 819200,
            "heldout_stream": SYNTHETIC_TAG,
            "lr_horizon_steps": 30518,
            "nll": nll,
            "training_tokens_per_second": tps,
        }
    return records


def test_mock_pass_never_authorizes_paid_execution_or_science():
    output = classify_synthetic_screen(fixture_records())
    assert output["classification"] == "SYNTHETIC_ENGINEERING_SCREEN_PASS_NOT_SCIENTIFIC"
    assert output["would_pass_draft_screen"] is True
    assert output["quality_gate_pass"]
    assert output["parent_improvement_gate_pass"]
    assert output["systems_gate_pass"]
    assert output["scientific_evidence"] is False
    assert output["engineering_result_recorded"] is False
    for flag in ("gpu_allocated", "writes_performed", "seed_consumed",
                 "replication_authorized", "scale_250m_5b_authorized", "breakthrough_claim_allowed"):
        assert output[flag] is False


@pytest.mark.parametrize(
    ("field", "value", "expected_gate"),
    [
        ("nll", 3.520, "quality_gate_pass"),
        ("training_tokens_per_second", 305000.0, "systems_gate_pass"),
    ],
)
def test_mock_quality_and_speed_fail_closed(field, value, expected_gate):
    records = fixture_records()
    records[NAMES[2]][field] = value
    output = classify_synthetic_screen(records)
    assert output["classification"] == "SYNTHETIC_ENGINEERING_SCREEN_FAIL"
    assert output["would_pass_draft_screen"] is False
    assert output[expected_gate] is False


def test_mock_parent_improvement_gate_fail():
    records = fixture_records()
    records[NAMES[1]]["nll"] = 3.509
    output = classify_synthetic_screen(records)
    assert output["classification"] == "SYNTHETIC_ENGINEERING_SCREEN_FAIL"
    assert output["parent_improvement_gate_pass"] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("status", "OOM", "incomplete_result"),
        ("optimizer_steps", 3051, "optimizer_steps_mismatch"),
        ("token_exposures", 199950336, "token_exposures_mismatch"),
        ("heldout_tokens", 500000, "evaluation_geometry_mismatch"),
        ("lr_horizon_steps", 3052, "optimizer_horizon_mismatch"),
        ("seed", "different-seed", "seed_mismatch"),
        ("heldout_stream", "other-eval", "evaluation_stream_mismatch"),
        ("synthetic_tag", "REAL", "not_marked_synthetic"),
        ("model_name", "wrong", "model_identity_mismatch"),
        ("nll", float("nan"), "nonfinite_or_invalid_metric"),
        ("training_tokens_per_second", float("inf"), "nonfinite_or_invalid_metric"),
        ("training_tokens_per_second", 0, "nonfinite_or_invalid_metric"),
    ],
)
def test_mismatched_or_invalid_records_rejected(field, value, reason):
    records = fixture_records()
    records[NAMES[2]][field] = value
    out = classify_synthetic_screen(records)
    assert out["classification"] == "SYNTHETIC_SCREEN_REJECTED"
    assert out["reason"] == reason
    assert out["scientific_evidence"] is False
    assert out["gpu_allocated"] is False


def test_missing_and_extra_comparators_rejected():
    records = fixture_records()
    del records[NAMES[0]]
    assert classify_synthetic_screen(records)["reason"] == "missing_or_extra_comparator"
    records = fixture_records()
    records["fourth"] = deepcopy(records[NAMES[0]])
    assert classify_synthetic_screen(records)["reason"] == "missing_or_extra_comparator"


def test_no_training_or_modal_side_effects_in_classifier():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] /
              "tam_research/cortex_attention8_conv7_engineering_rehearsal_v1.py").read_text(encoding="utf-8")
    for forbidden in ("import modal", "torch.", "cuda", ".remote(", "modal run",
                      "optimizer.step(", ".backward(", ".write_text(", ".write_bytes(",
                      "requests.", "subprocess.", "h100_train_one("):
        assert forbidden not in source
