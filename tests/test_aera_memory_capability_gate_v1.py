from __future__ import annotations

import json
from pathlib import Path

import pytest

from tam_research import aera_memory_capability_gate_v1 as gate


def test_issue746_authority_is_cpu_fixture_only() -> None:
    p = gate.protocol_snapshot()
    assert p["research_issue"] == 746
    assert p["source_main"] == "7c6d3dd187ccb80bb93eade170dfb717549b8eeb"
    assert p["source_tree"] == "91979a36c56d36e61d5a4e7bcc798f56ef4b61b4"
    assert p["authority"] == {
        "gpu_authorized": False,
        "scientific_training_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "architecture_freeze_authorized": False,
        "scaling_authorized": False,
        "breakthrough_proven": False,
    }
    assert p["seeds"]["scientific_model_seeds_consumed"] is False


def test_synthetic_token_ranges_are_disjoint_and_in_vocab() -> None:
    ranges = [
        set(range(0, 8)),
        set(range(gate.KEY_BASE, gate.KEY_BASE + gate.KEY_COUNT)),
        set(range(gate.VALUE_BASE, gate.VALUE_BASE + gate.VALUE_COUNT)),
        set(range(gate.DISTRACTOR_KEY_BASE, gate.DISTRACTOR_KEY_BASE + gate.DISTRACTOR_KEY_COUNT)),
        set(range(gate.DISTRACTOR_VALUE_BASE, gate.DISTRACTOR_VALUE_BASE + gate.DISTRACTOR_VALUE_COUNT)),
        set(range(gate.NOISE_BASE, gate.NOISE_BASE + gate.NOISE_COUNT)),
    ]
    for i, left in enumerate(ranges):
        assert max(left) < gate.VOCAB_SIZE
        for right in ranges[i + 1 :]:
            assert left.isdisjoint(right)


def test_compositional_split_is_disjoint_and_every_symbol_appears_in_both() -> None:
    train: set[tuple[int, int]] = set()
    heldout: set[tuple[int, int]] = set()
    for ki in range(gate.KEY_COUNT):
        for vi in range(gate.VALUE_COUNT):
            pair = (gate.KEY_BASE + ki, gate.VALUE_BASE + vi)
            (heldout if gate.pair_split(*pair) == "eval" else train).add(pair)
    assert train.isdisjoint(heldout)
    assert len(train | heldout) == gate.KEY_COUNT * gate.VALUE_COUNT
    for key in range(gate.KEY_BASE, gate.KEY_BASE + gate.KEY_COUNT):
        assert any(k == key for k, _ in train)
        assert any(k == key for k, _ in heldout)
    for value in range(gate.VALUE_BASE, gate.VALUE_BASE + gate.VALUE_COUNT):
        assert any(v == value for _, v in train)
        assert any(v == value for _, v in heldout)


def _case(**kwargs):
    defaults = dict(
        split="eval",
        seed=gate.CPU_TEST_SEED,
        sample_index=11,
        retention_distance_chunks=8,
        concurrent_facts=4,
        distractor_records=16,
        correction=True,
        reset_before_query=False,
    )
    defaults.update(kwargs)
    return gate.generate_case(**defaults)


def test_generator_is_deterministic_and_exactly_chunked() -> None:
    a = _case()
    b = _case()
    assert a == b
    assert len(a.chunks) == a.retention_distance_chunks + 1
    assert all(len(chunk) == gate.CHUNK_SIZE for chunk in a.chunks)
    assert len(a.flat_tokens) == len(a.chunks) * gate.CHUNK_SIZE
    assert min(a.flat_tokens) >= 0
    assert max(a.flat_tokens) < gate.VOCAB_SIZE
    assert gate.pair_split(a.target_key, a.original_value) == "eval"
    assert gate.pair_split(a.target_key, a.latest_value) == "eval"
    assert a.chunks[a.answer_chunk_index][a.answer_token_index] == a.expected_answer


def test_correction_scores_newest_value_not_stale_value() -> None:
    case = _case(correction=True)
    assert case.stale_value == case.original_value
    assert case.latest_value != case.original_value
    assert case.expected_answer == case.latest_value
    assert case.expected_answer != case.stale_value


def test_reset_negative_control_answers_unknown() -> None:
    case = _case(reset_before_query=True)
    assert case.expected_answer == gate.UNKNOWN
    assert case.expected_answer != case.latest_value
    assert gate.RESET in case.chunks[case.retention_distance_chunks - 1]


def test_frozen_difficulty_ladders_cover_short_medium_long() -> None:
    assert gate.TRAIN_RETENTION_DISTANCES == (1, 2, 4, 8)
    assert gate.EVAL_RETENTION_DISTANCES == (2, 8, 32)
    assert gate.EVAL_CONCURRENT_FACTS == (1, 4, 8)
    assert gate.EVAL_DISTRACTOR_RECORDS == (0, 16, 32)
    for distance in gate.EVAL_RETENTION_DISTANCES:
        case = _case(retention_distance_chunks=distance, sample_index=100 + distance)
        assert len(case.chunks) == distance + 1


def test_variant_surface_is_minimal_and_isolates_memory_routing_state() -> None:
    by_name = {v.name: v for v in gate.VARIANTS}
    assert tuple(by_name) == (
        "A_backbone",
        "B_backbone_plus_memory",
        "C_backbone_plus_routing",
        "D_combined",
        "E_simple_retrieval",
        "F_backbone_plus_recurrent_state",
    )
    assert not by_name["A_backbone"].persistent_learned_memory
    assert by_name["B_backbone_plus_memory"].persistent_learned_memory
    assert by_name["C_backbone_plus_routing"].adaptive_routing
    assert by_name["D_combined"].persistent_learned_memory and by_name["D_combined"].adaptive_routing
    assert by_name["E_simple_retrieval"].simple_retrieval_cache
    assert by_name["F_backbone_plus_recurrent_state"].recurrent_state
    assert all(not v.latent_reasoning and not v.multimodality and not v.replay and not v.block_generation for v in gate.VARIANTS)


def test_budgets_seeds_and_practical_thresholds_are_frozen() -> None:
    assert gate.TOKEN_BUDGET_PER_TRAINED_VARIANT == 16_777_216
    assert gate.CHECKPOINT_INTERVAL_TOKENS == 2_097_152
    assert gate.MAX_GPU_SECONDS_PER_VARIANT_SEED == 3_600
    assert gate.SCIENTIFIC_MODEL_SEEDS == (17_641, 27_641, 37_641)
    assert gate.CPU_TEST_SEED not in gate.SCIENTIFIC_MODEL_SEEDS
    assert gate.LONG_DISTANCE_ACCURACY_GAIN_MIN == pytest.approx(0.10)
    assert gate.PAIRED_CI_LOWER_ACCURACY_GAIN_MIN == pytest.approx(0.05)
    assert gate.CORRECTION_ACCURACY_GAIN_MIN == pytest.approx(0.10)
    assert gate.STALE_VALUE_ERROR_REDUCTION_MIN == pytest.approx(0.10)
    assert gate.ROUTING_CONTROL_ACCURACY_MARGIN_MIN == pytest.approx(0.05)
    assert gate.SESSION_RESET_LEAKAGE_MAX == pytest.approx(0.01)
    assert gate.MEMORY_INFERENCE_LATENCY_MULTIPLIER_MAX == pytest.approx(2.0)
    assert gate.MEMORY_TRAIN_TIME_MULTIPLIER_MAX == pytest.approx(1.5)


def test_metric_helpers_are_paired_and_stale_reset_aware() -> None:
    assert gate.exact_match_accuracy([1, 2, 3], [1, 9, 3]) == pytest.approx(2 / 3)
    assert gate.mean_answer_nll([-0.1, -0.3]) == pytest.approx(0.2)
    assert gate.stale_value_error_rate([4, 8, 9], [4, None, 7]) == pytest.approx(0.5)
    assert gate.session_reset_leakage_rate([4, 8], [4, 9]) == pytest.approx(0.5)
    delta, lo, hi = gate.paired_bootstrap_accuracy_delta(
        [0, 0, 1, 1], [1, 1, 1, 1], seed=123, resamples=200
    )
    assert delta == pytest.approx(0.5)
    assert lo <= delta <= hi


def test_frozen_json_snapshot_matches_python_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "docs" / "aera_memory_capability_gate_v1_protocol.json"
    expected = json.loads(path.read_text())
    assert expected == gate.protocol_snapshot()
