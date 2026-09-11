from __future__ import annotations

import re

import pytest

from tam_research.chm_v1_scientific_gate import (
    REQUIRED_ABLATION_KEYS,
    REQUIRED_MEASUREMENT_KEYS,
    SCIENTIFIC_GENERATOR_VERSION,
    build_scientific_probe_suite,
    evaluate_scientific_gate,
)
from tam_research.chm_v1_small_lm import SCIENTIFIC_SEEDS


class StableWordEncoder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str) -> list[int]:
        pieces = re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text.lower())
        out: list[int] = []
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab) + 100
            out.append(self.vocab[piece])
        return out


def _record(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "generator_version": SCIENTIFIC_GENERATOR_VERSION,
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


def test_scientific_probe_builder_binds_v3_and_full_frozen_envelope() -> None:
    suite = build_scientific_probe_suite(StableWordEncoder())
    assert len(suite) == 96
    assert {probe.generator_version for probe in suite} == {SCIENTIFIC_GENERATOR_VERSION}
    assert all(probe.used_for_training is False for probe in suite)


def test_full_pass_requires_all_three_reserved_seeds_and_all_frozen_gates() -> None:
    result = evaluate_scientific_gate([_record(seed) for seed in SCIENTIFIC_SEEDS])
    assert result["full_pass"] is True
    assert result["classification"] == "FULL_PASS_SMALL_LM_ONLY"
    assert all(result["criteria"].values())
    assert result["metrics"]["mean_long_range_absolute_gain"] == pytest.approx(0.20)
    assert result["metrics"]["sparse_read_fraction_1024"] == pytest.approx(0.20)


def test_exactness_mismatch_hard_fails_gate() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[1]["indexed"]["exact_match_rate"] = 0.999
    result = evaluate_scientific_gate(records)
    assert result["criteria"]["exactness"] is False
    assert result["full_pass"] is False


def test_two_of_three_family_win_rule_is_enforced() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    for record in records[1:]:
        record["families"]["two_hop"]["eiem_accuracy"] = 0.15
    result = evaluate_scientific_gate(records)
    assert result["metrics"]["family_seed_wins"]["two_hop"] == 1
    assert result["criteria"]["long_range_capability"] is False


def test_language_and_local_negative_thresholds_are_per_seed_as_preregistered() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0]["language"]["eiem_flat_nll"] = 4.151
    records[2]["families"]["local_negative"]["eiem_accuracy"] = 0.779
    result = evaluate_scientific_gate(records)
    assert result["criteria"]["ordinary_language"] is False
    assert result["criteria"]["local_negative_control"] is False
    assert result["full_pass"] is False


def test_sparse_gate_aggregates_exact_read_counts_not_mean_of_ratios() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0]["indexed"].update(indexed_reads_1024=1, flat_reads_1024=10)
    records[1]["indexed"].update(indexed_reads_1024=1, flat_reads_1024=10)
    records[2]["indexed"].update(indexed_reads_1024=76, flat_reads_1024=280)
    result = evaluate_scientific_gate(records)
    assert result["metrics"]["sparse_read_fraction_1024"] == pytest.approx(78 / 300)
    assert result["criteria"]["sparse_reads_1024"] is False


def test_missing_mandatory_measurement_refuses_classification() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0]["measurements"].pop("state_bytes")
    with pytest.raises(ValueError, match="missing required keys"):
        evaluate_scientific_gate(records)


def test_wrong_generator_or_seed_set_is_rejected_before_scoring() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0]["generator_version"] = "chm-v1-heldout-natural-v2"
    with pytest.raises(ValueError, match="must use"):
        evaluate_scientific_gate(records)

    duplicate = [_record(8611), _record(8611), _record(8613)]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_scientific_gate(duplicate)


def test_any_system_or_preregistered_stop_condition_blocks_pass() -> None:
    records = [_record(seed) for seed in SCIENTIFIC_SEEDS]
    records[0]["systems"]["no_cross_session_aliasing"] = False
    records[1]["stop_conditions"]["index_overhead_erases_practical_advantage"] = True
    result = evaluate_scientific_gate(records)
    assert result["criteria"]["systems_sanity"] is False
    assert result["criteria"]["no_stop_condition"] is False
    assert result["full_pass"] is False
