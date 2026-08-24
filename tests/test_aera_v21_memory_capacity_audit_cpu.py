import math

from aera_v21_memory_capacity_audit_cpu import (
    CAPACITY_PASS,
    CURRENT_SEPARABILITY_PASS,
    address_metrics,
    diagnose,
    payload_metrics,
    supervised_address_capacity,
    supervised_payload_capacity,
)
from aera_v21_memory_necessity_cpu import build_model


def _address(acc: float):
    return {
        "q_to_k_top1_accuracy": acc,
        "diagonal_cosine_mean": 0.5,
        "off_diagonal_cosine_mean": 0.1,
        "off_diagonal_cosine_max": 0.2,
        "q_effective_rank_1pct": 4,
        "q_singular_values": [1.0, 0.5],
    }


def _payload(acc: float):
    return {"legal_value_accuracy": acc, "legal_value_nll": 1.0}


def test_frozen_capacity_thresholds():
    assert CAPACITY_PASS == 0.95
    assert CURRENT_SEPARABILITY_PASS == 0.95


def test_diagnosis_hierarchy_is_frozen():
    current_address = _address(0.0)
    current_payload = _payload(0.0)
    assert diagnose(
        current_address,
        current_payload,
        {"final": _address(0.90)},
        {"final": _payload(1.0)},
    ) == "q_k_projection_capacity_bottleneck"
    assert diagnose(
        current_address,
        current_payload,
        {"final": _address(1.0)},
        {"final": _payload(0.90)},
    ) == "payload_projection_capacity_bottleneck"
    assert diagnose(
        current_address,
        current_payload,
        {"final": _address(1.0)},
        {"final": _payload(1.0)},
    ) == "projection_capacity_sufficient_objective_design_bottleneck"
    assert diagnose(
        _address(1.0),
        _payload(1.0),
        {"final": _address(1.0)},
        {"final": _payload(1.0)},
    ) == "capacity_and_current_code_high_reaudit_injection_task_semantics"


def test_current_metrics_are_bounded_and_finite():
    model = build_model(9701)
    address = address_metrics(model)
    payload = payload_metrics(model)
    assert 0.0 <= address["q_to_k_top1_accuracy"] <= 1.0
    assert -1.0 <= address["diagonal_cosine_mean"] <= 1.0
    assert -1.0 <= address["off_diagonal_cosine_mean"] <= 1.0
    assert -1.0 <= address["off_diagonal_cosine_max"] <= 1.0
    assert 0 <= address["q_effective_rank_1pct"] <= 12
    assert all(math.isfinite(v) for v in address["q_singular_values"])
    assert 0.0 <= payload["legal_value_accuracy"] <= 1.0
    assert math.isfinite(payload["legal_value_nll"])


def test_supervised_capacity_probes_run_on_existing_projection_parameters():
    model = build_model(9702)
    address = supervised_address_capacity(model, steps=2)
    payload = supervised_payload_capacity(model, steps=2)
    assert len(address["history"]) == 2
    assert len(payload["history"]) == 2
    assert 0.0 <= address["final"]["q_to_k_top1_accuracy"] <= 1.0
    assert 0.0 <= payload["final"]["legal_value_accuracy"] <= 1.0
    for row in address["history"] + payload["history"]:
        assert all(math.isfinite(v) for v in row.values())
