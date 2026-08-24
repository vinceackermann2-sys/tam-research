import math

from aera_v19_memory_necessity_cpu import EVAL_SEED, LEARNING_RATE, SEED, make_batch
from aera_v21_memory_necessity_cpu import build_model
from aera_v21_payload_optimization_audit_cpu import (
    ISOLATED_LR,
    LOCAL_CAPACITY_PASS,
    diagnose,
    gradient_snapshot,
    train_isolated_payload,
)


def test_frozen_payload_audit_constants():
    assert ISOLATED_LR == LEARNING_RATE == 4e-3
    assert LOCAL_CAPACITY_PASS == 0.95


def test_gradient_snapshot_is_finite_and_reports_clip_accounting():
    model = build_model(SEED)
    batch = make_batch(2, EVAL_SEED + 701)
    row = gradient_snapshot(model, batch)
    assert set(row["payload_gradient_norms"]) == {"query", "latent", "token"}
    assert set(row["gradient_cosines"]) == {
        "latent_vs_token",
        "query_vs_token",
        "query_vs_latent",
    }
    values = list(row["payload_gradient_norms"].values())
    values += list(row["gradient_cosines"].values())
    values += [
        row["full_objective_gradient_norm_before_clip"],
        row["implied_global_clip_multiplier"],
        row["payload_gradient_norm_before_clip"],
        row["payload_gradient_norm_after_implied_clip"],
    ]
    assert all(math.isfinite(v) for v in values)
    assert 0.0 < row["implied_global_clip_multiplier"] <= 1.0


def test_all_isolated_payload_objectives_run_without_new_parameters():
    source = build_model(SEED)
    for name in ("token", "latent", "token_plus_latent"):
        result = train_isolated_payload(source, name, steps=2)
        assert result["objective"] == name
        assert len(result["history"]) == 2
        assert 0.0 <= result["final"]["legal_value_accuracy"] <= 1.0
        assert -1.0 <= result["final"]["latent_cosine"] <= 1.0
        for row in result["history"]:
            assert all(math.isfinite(v) for v in row.values())


def test_diagnosis_hierarchy_is_frozen():
    base = {
        "reproduced_full_payload_accuracy": 0.1,
        "post_reproduction_gradients": {"gradient_cosines": {"latent_vs_token": -0.5}},
        "isolated_payload_training": {
            "token": {"final": {"legal_value_accuracy": 1.0}},
            "token_plus_latent": {"final": {"legal_value_accuracy": 0.5}},
        },
    }
    assert diagnose(base) == "latent_payload_target_conflicts_with_decoder_aligned_token_target"
    low_token = {
        **base,
        "isolated_payload_training": {
            "token": {"final": {"legal_value_accuracy": 0.5}},
            "token_plus_latent": {"final": {"legal_value_accuracy": 0.5}},
        },
    }
    assert diagnose(low_token) == "production_budget_payload_optimization_bottleneck"
    competition = {
        **base,
        "post_reproduction_gradients": {"gradient_cosines": {"latent_vs_token": 0.2}},
        "isolated_payload_training": {
            "token": {"final": {"legal_value_accuracy": 1.0}},
            "token_plus_latent": {"final": {"legal_value_accuracy": 1.0}},
        },
    }
    assert diagnose(competition) == "full_model_gradient_competition_or_clipping_bottleneck"
