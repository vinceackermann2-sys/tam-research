from __future__ import annotations

import math

import pytest
import torch

from tam_research.cpw_binding_grad_audit.audit import (
    AUDIT_SEED,
    CLASSIFICATION,
    audit,
    one_arm,
)
from tam_research.cpw_binding_v2.task import QUERY_POSITION, make_binding_batch


def test_local_sequence_has_no_gradient_path_to_256_token_source() -> None:
    result = audit(arms=("sequence_only",), seed=AUDIT_SEED, delay=256)
    row = result["results"][0]
    assert result["classification"] == CLASSIFICATION
    assert result["training_performed"] is False
    assert result["gpu_requested"] is False
    assert result["scientific_seeds_consumed"] is False
    assert row["source_value_position"] == QUERY_POSITION - 256
    assert row["source_key_position"] == QUERY_POSITION - 257
    # Dependency is impossible for the frozen 15 one-step local blocks.
    assert row["source_key_activation_grad_l2"] == pytest.approx(0.0, abs=1e-12)
    assert row["source_value_activation_grad_l2"] == pytest.approx(0.0, abs=1e-12)
    assert row["far_prefix_activation_grad_l2"] == pytest.approx(0.0, abs=1e-12)
    assert row["query_key_activation_grad_l2"] > 0.0


def test_transformer_has_finite_distant_gradient_path() -> None:
    row = audit(arms=("transformer",), seed=AUDIT_SEED, delay=256)["results"][0]
    assert math.isfinite(row["initial_query_nll"])
    assert row["source_key_activation_grad_l2"] >= 0.0
    assert row["source_value_activation_grad_l2"] >= 0.0
    assert row["far_prefix_activation_grad_l2"] > 0.0


def test_early_afm_parameter_gradients_and_memory_stats_are_finite() -> None:
    row = audit(arms=("afm_first1",), seed=AUDIT_SEED, delay=256)["results"][0]
    assert row["afm"]["stat_afm_layer_index"] == 0.0
    for field in ("key_proj", "value_proj", "out_proj", "gate_prev", "gate_cur"):
        assert math.isfinite(row["afm"][field + "_grad_l2"])
        assert row["afm"][field + "_grad_l2"] >= 0.0
    assert row["afm"]["stat_mean_write_gate"] > 0.0


def test_deterministic_input_fingerprint_and_metrics() -> None:
    a = audit(arms=("sequence_only",), seed=AUDIT_SEED)
    b = audit(arms=("sequence_only",), seed=AUDIT_SEED)
    assert a["source_sha256"] == b["source_sha256"]
    assert a["results"][0]["initial_query_nll"] == pytest.approx(
        b["results"][0]["initial_query_nll"], abs=1e-7
    )


def test_cpu_only_constraint_precedes_model_side_effect() -> None:
    tokens = torch.empty((1, 320), dtype=torch.long, device="meta")
    targets = torch.empty((1,), dtype=torch.long, device="meta")
    with pytest.raises(PermissionError, match="CPU only"):
        one_arm("afm_last1", tokens=tokens, targets=targets, delay=256)
