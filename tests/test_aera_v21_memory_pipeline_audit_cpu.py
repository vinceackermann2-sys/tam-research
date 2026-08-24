import math

import torch

from aera_v19_memory_necessity_cpu import EVAL_SEED, _evaluate, make_batch
from aera_v21_memory_pipeline_audit_cpu import (
    _diagnose,
    _evaluate_chunkwise,
    _force_memory_read_gate_one,
    _oracle_pair_position_one,
    _use_k_for_memory_read,
    trace_activity,
    train_deterministic_reproduction,
)
from aera_v21_memory_necessity_cpu import build_model


def test_short_reproduction_emits_finite_expected_gradient_snapshot():
    model, snapshots = train_deterministic_reproduction(steps=1)
    assert model is not None
    assert len(snapshots) == 1
    expected = {
        "pair_gate", "memory_q", "memory_k", "memory_v", "memory_out",
        "controller_memory_read", "controller_memory_write", "controller_novelty", "lm_head",
    }
    assert expected.issubset(snapshots[0])
    for key in expected:
        value = snapshots[0][key]
        assert math.isfinite(value)
        assert value >= 0.0


def test_chunkwise_normal_matches_reference_evaluator():
    model = build_model(9401)
    batch = make_batch(2, EVAL_SEED + 17)
    reference = _evaluate(model, batch, memory=True)
    chunkwise = _evaluate_chunkwise(model, batch, memory=True)
    assert abs(reference["query_accuracy"] - chunkwise["query_accuracy"]) < 1e-7
    assert abs(reference["query_nll"] - chunkwise["query_nll"]) < 1e-5


def test_intervention_contexts_restore_exact_normal_behavior():
    model = build_model(9402)
    batch = make_batch(2, EVAL_SEED + 18)
    baseline = _evaluate_chunkwise(model, batch)
    with _use_k_for_memory_read(model):
        _ = _evaluate_chunkwise(model, batch)
    with _force_memory_read_gate_one(model):
        _ = _evaluate_chunkwise(model, batch)
    with _oracle_pair_position_one(model):
        _ = _evaluate_chunkwise(model, batch)
    restored = _evaluate_chunkwise(model, batch)
    assert baseline == restored


def test_activity_trace_is_finite_and_nonnegative_for_norms():
    model = build_model(9403)
    batch = make_batch(2, EVAL_SEED + 19)
    activity = trace_activity(model, batch)
    for value in activity.values():
        assert math.isfinite(value)
    for key in (
        "memory_read_vector_norm_mean",
        "normalized_token_vector_norm_mean",
        "carried_stream_vector_norm_mean",
        "memory_matrix_norm_mean",
        "memory_matrix_norm_after_writes",
        "memory_matrix_norm_after_distractors",
        "memory_matrix_norm_final",
    ):
        assert activity[key] >= 0.0


def _evals(normal=0.05, k=0.05, gate=0.05, oracle=0.05, kgate=0.05, combined=0.05):
    return {
        "normal": {"query_accuracy": normal},
        "k_read": {"query_accuracy": k},
        "read_gate_one": {"query_accuracy": gate},
        "oracle_pair": {"query_accuracy": oracle},
        "k_read_plus_gate_one": {"query_accuracy": kgate},
        "oracle_pair_plus_k_read_plus_gate_one": {"query_accuracy": combined},
    }


def _active():
    return {"event_pair_write_strength_mean": 0.2, "memory_matrix_norm_after_writes": 1.0}


def test_diagnosis_hierarchy_is_frozen():
    assert _diagnose(_evals(k=0.20), _active()) == "q_k_alignment_bottleneck"
    assert _diagnose(_evals(gate=0.20), _active()) == "controller_memory_read_suppression_bottleneck"
    assert _diagnose(_evals(oracle=0.20), _active()) == "write_selectivity_or_interference_bottleneck"
    assert _diagnose(_evals(combined=0.20), _active()) == "combined_learning_or_control_bottleneck"
    inactive = dict(_active()); inactive["event_pair_write_strength_mean"] = 0.0
    assert _diagnose(_evals(), inactive) == "write_strength_or_matrix_activity_bottleneck"
    assert _diagnose(_evals(), _active()) == "payload_decoding_injection_or_objective_bottleneck"
