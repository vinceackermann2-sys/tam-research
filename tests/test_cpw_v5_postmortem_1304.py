from __future__ import annotations

import math

from scripts.cpw_v5_postmortem_1304 import (
    constructed_binding_probe,
    inclusive_current_write_probe,
    positive_key_collision_probe,
    report,
)


def test_key_initialization_diagnostic_is_finite() -> None:
    values = positive_key_collision_probe()
    assert 0.0 <= values["initial_key_offdiag_cosine"] <= 1.0
    assert -1.0 <= values["initial_centered_key_offdiag_cosine"] <= 1.0
    assert 0.0 < values["initial_write_gate_mean"] < 1.0


def test_constructed_memory_can_hold_eight_bindings() -> None:
    # Constructed oracle features are not evidence of learned memory.
    ideal = constructed_binding_probe(filler_gate=0.0)
    assert ideal["target_rank_one"] == 1.0
    assert ideal["target_margin"] > 0
    noisy = constructed_binding_probe(filler_gate=0.05)
    assert all(math.isfinite(x) for x in noisy.values())


def test_inclusive_current_write_changes_current_read_only() -> None:
    values = inclusive_current_write_probe()
    assert values["same_position_read_change"] > 0.0
    assert values["earlier_read_change"] == 0.0


def test_report_is_cpu_only_and_does_not_claim_science() -> None:
    result = report()
    assert result["cpu_only"] is True
    assert result["scientific_seeds_consumed"] is False
    assert result["trained_checkpoint_loaded"] is False
    assert result["scientific_source_issue"] == 1290
    assert "not an observed effect" in result["scientific_interpretation"]
