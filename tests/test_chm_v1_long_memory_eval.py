from __future__ import annotations

import re

from tam_research.chm_v1_long_memory_eval import (
    GENERATOR_VERSION,
    LOCAL_CONTROL_MAX_DISTANCE,
    LONG_RANGE_MIN_DISTANCE,
    SPARSE_READ_GATE_MEMORY_SIZE,
    TWO_HOP_MIN_FACT_ENDPOINT_SEPARATION,
    generate_probe_suite,
    summarize_probe_predictions,
)
from tam_research.chm_v1_small_lm import LOCAL_WINDOW


class StableWordEncoder:
    """Tiny deterministic CI encoder; production supplies frozen GPT-2 BPE."""

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


def test_probe_suite_is_deterministic_and_has_strict_distance_geometry() -> None:
    encoder_a = StableWordEncoder()
    suite_a = generate_probe_suite(encoder_a, seed=991_854, cases_per_family=3)
    encoder_b = StableWordEncoder()
    suite_b = generate_probe_suite(encoder_b, seed=991_854, cases_per_family=3)

    assert len(suite_a) == 12
    assert [p.prompt_text for p in suite_a] == [p.prompt_text for p in suite_b]
    assert [p.prompt_ids for p in suite_a] == [p.prompt_ids for p in suite_b]
    assert {p.family for p in suite_a} == {
        "rare_fact", "overwrite", "two_hop", "local_negative"
    }

    for probe in suite_a:
        assert probe.generator_version == GENERATOR_VERSION
        assert probe.used_for_training is False
        assert probe.query_token == len(probe.prompt_ids) - 1
        assert probe.answer_token_id in probe.candidate_token_ids
        assert len(set(probe.candidate_token_ids)) == len(probe.candidate_token_ids)
        if probe.family == "local_negative":
            assert probe.evidence_distance <= LOCAL_CONTROL_MAX_DISTANCE
        else:
            assert probe.evidence_distance >= LONG_RANGE_MIN_DISTANCE
        if probe.family in {"rare_fact", "overwrite"}:
            assert probe.evidence_end_token < LOCAL_WINDOW
            assert LOCAL_WINDOW <= probe.query_token < 2 * LOCAL_WINDOW
        if probe.family == "two_hop":
            assert probe.first_evidence_end_token is not None
            assert (
                probe.evidence_end_token - probe.first_evidence_end_token
                >= TWO_HOP_MIN_FACT_ENDPOINT_SEPARATION
            )
            assert probe.first_evidence_end_token < LOCAL_WINDOW
            assert LOCAL_WINDOW <= probe.evidence_end_token < SPARSE_READ_GATE_MEMORY_SIZE
            assert (
                SPARSE_READ_GATE_MEMORY_SIZE
                <= probe.query_token
                < SPARSE_READ_GATE_MEMORY_SIZE + LOCAL_WINDOW
            )


def test_two_hop_relations_are_forced_into_different_local_horizons() -> None:
    suite = generate_probe_suite(StableWordEncoder(), seed=881_854, cases_per_family=4)
    two_hop = [probe for probe in suite if probe.family == "two_hop"]
    assert len(two_hop) == 4
    assert SPARSE_READ_GATE_MEMORY_SIZE == 1024
    for probe in two_hop:
        assert probe.first_evidence_end_token is not None
        # Endpoint separation is >=640 and the second fact is capped at 128
        # encoded tokens, so it must start beyond the first 512-token horizon.
        assert (
            probe.evidence_end_token - probe.first_evidence_end_token
            >= TWO_HOP_MIN_FACT_ENDPOINT_SEPARATION
        )
        assert probe.first_evidence_end_token < 512
        assert 512 <= probe.evidence_end_token < 1024
        assert 1024 <= probe.query_token < 1536
        assert probe.evidence_distance >= LONG_RANGE_MIN_DISTANCE


def test_hidden_scoring_metadata_is_not_rendered_into_model_input() -> None:
    suite = generate_probe_suite(StableWordEncoder(), seed=771_854, cases_per_family=2)
    for probe in suite:
        # case_id is evaluator bookkeeping only; the generated templates never print it.
        assert str(probe.case_id) not in probe.prompt_text
        assert "rare_fact" not in probe.prompt_text
        assert "local_negative" not in probe.prompt_text
        assert probe.used_for_training is False
        if probe.family == "overwrite":
            assert probe.stale_token_ids
            assert probe.answer_token_id not in probe.stale_token_ids


def test_summary_reports_accuracy_gain_and_stale_errors_without_changing_inputs() -> None:
    probes = generate_probe_suite(StableWordEncoder(), seed=551_854, cases_per_family=1)
    local = [probe.candidate_token_ids[0] for probe in probes]
    eiem = [probe.answer_token_id for probe in probes]
    # Deliberately make overwrite choose a stale answer to exercise that metric.
    overwrite_index = next(i for i, probe in enumerate(probes) if probe.family == "overwrite")
    eiem[overwrite_index] = probes[overwrite_index].stale_token_ids[0]

    summary = summarize_probe_predictions(probes, local, eiem)
    assert summary["rare_fact"]["eiem_accuracy"] == 1.0
    assert summary["two_hop"]["eiem_accuracy"] == 1.0
    assert summary["overwrite"]["eiem_stale_error"] == 1.0
    assert summary["rare_fact"]["minimum_evidence_distance"] >= LONG_RANGE_MIN_DISTANCE
    assert summary["local_negative"]["minimum_evidence_distance"] <= LOCAL_CONTROL_MAX_DISTANCE
