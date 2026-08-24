import math

import torch

from aera_v19_memory_necessity_cpu import EVAL_SEED, N_VALUES, make_batch
from aera_v21_memory_necessity_cpu import build_model
from aera_v21_raw_payload_audit_cpu import (
    _collect_query_logits,
    _memory_context_mode,
    _value_codebook,
    collect_raw_payload_metrics,
    diagnose,
    logit_sensitivity,
)


def test_value_codebook_has_one_finite_row_per_legal_value():
    model = build_model(9501)
    codebook = _value_codebook(model)
    assert codebook.shape == (N_VALUES, model.cfg.memory_dim)
    assert torch.isfinite(codebook).all()


def test_raw_payload_metrics_are_finite_and_bounded():
    model = build_model(9502)
    batch = make_batch(2, EVAL_SEED + 31)
    row = collect_raw_payload_metrics(model, batch)
    assert 0.0 <= row["raw_value_accuracy"] <= 1.0
    assert 0.0 <= row["decoded_value_accuracy"] <= 1.0
    for key, value in row.items():
        assert math.isfinite(value), key


def test_context_interventions_restore_exact_normal_logits():
    model = build_model(9503)
    batch = make_batch(2, EVAL_SEED + 32)
    baseline = _collect_query_logits(model, batch, mode="normal")
    _ = _collect_query_logits(model, batch, mode="removed")
    _ = _collect_query_logits(model, batch, mode="scaled")
    restored = _collect_query_logits(model, batch, mode="normal")
    torch.testing.assert_close(restored, baseline, atol=0.0, rtol=0.0)


def test_logit_sensitivity_outputs_are_finite():
    model = build_model(9504)
    batch = make_batch(2, EVAL_SEED + 33)
    row = logit_sensitivity(model, batch)
    for key, value in row.items():
        assert math.isfinite(value), key
    for key in ("normal_query_accuracy", "removed_query_accuracy", "scaled_query_accuracy"):
        assert 0.0 <= row[key] <= 1.0


def _raw(normal_raw=0.05, normal_dec=0.05, oracle_q=0.05, oracle_k=0.05):
    base = {"raw_value_accuracy": normal_raw, "decoded_value_accuracy": normal_dec}
    return {
        "normal_q": dict(base),
        "normal_k": dict(base),
        "oracle_q": {"raw_value_accuracy": oracle_q, "decoded_value_accuracy": normal_dec},
        "oracle_k": {"raw_value_accuracy": oracle_k, "decoded_value_accuracy": normal_dec},
    }


def _sens(acc=0.05):
    return {"normal_query_accuracy": acc}


def test_diagnosis_hierarchy_is_frozen():
    assert diagnose(_raw(normal_raw=0.90, normal_dec=0.10, oracle_k=0.90), _sens()) == "memory_out_or_decoder_alignment_bottleneck"
    assert diagnose(_raw(normal_raw=0.90, normal_dec=0.90, oracle_k=0.90), _sens(0.05)) == "downstream_residual_injection_or_objective_bottleneck"
    assert diagnose(_raw(normal_raw=0.05, oracle_q=0.30, oracle_k=0.30), _sens()) == "write_interference_or_selectivity_bottleneck"
    assert diagnose(_raw(normal_raw=0.05, oracle_q=0.05, oracle_k=0.05), _sens()) == "payload_representation_not_recoverably_stored"
    assert diagnose(_raw(normal_raw=0.20, oracle_q=0.25, oracle_k=0.25), _sens()) == "mixed_payload_decoder_or_objective_bottleneck"
