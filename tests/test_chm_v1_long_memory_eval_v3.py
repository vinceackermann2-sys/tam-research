from __future__ import annotations

from collections import Counter
import re

from tam_research.chm_v1_long_memory_eval import (
    LOCAL_CONTROL_MAX_DISTANCE,
    LONG_RANGE_MIN_DISTANCE,
    SPARSE_READ_GATE_MEMORY_SIZE,
    TWO_HOP_MIN_FACT_ENDPOINT_SEPARATION,
    _VALUES,
)
from tam_research.chm_v1_long_memory_eval_v3 import (
    GENERATOR_VERSION,
    _ENTITIES_V3,
    _LINKS_V3,
    generate_probe_suite,
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


def _reverse_vocab(encoder: StableWordEncoder) -> dict[int, str]:
    return {token_id: piece for piece, token_id in encoder.vocab.items()}


def test_v3_identifier_vocabulary_is_disjoint_from_all_probe_values() -> None:
    for identifier in _ENTITIES_V3 + _LINKS_V3:
        lower = identifier.lower()
        for value in _VALUES:
            assert value.lower() not in lower


def test_v3_24_case_scientific_envelope_is_balanced_and_has_no_query_answer_cue() -> None:
    encoder = StableWordEncoder()
    suite = generate_probe_suite(encoder, seed=8_540_911, cases_per_family=24)
    reverse = _reverse_vocab(encoder)

    assert len(suite) == 96
    assert {probe.generator_version for probe in suite} == {GENERATOR_VERSION}
    assert GENERATOR_VERSION == "chm-v1-heldout-natural-v3"

    for family in ("rare_fact", "overwrite", "two_hop", "local_negative"):
        rows = [probe for probe in suite if probe.family == family]
        assert len(rows) == 24
        counts = Counter(probe.answer_token_id for probe in rows)
        assert len(counts) == 8
        assert set(counts.values()) == {3}

        for probe in rows:
            answer_word = reverse[probe.answer_token_id]
            query_text = probe.prompt_text.rsplit("Question:", 1)[-1].lower()
            assert answer_word not in query_text
            assert probe.answer_token_id in probe.candidate_token_ids
            assert len(set(probe.candidate_token_ids)) == 8
            assert probe.used_for_training is False

            if family == "overwrite":
                assert len(probe.stale_token_ids) == 2
                assert len(set(probe.stale_token_ids)) == 2
                assert probe.answer_token_id not in probe.stale_token_ids


def test_v3_preserves_frozen_distance_and_memory_slice_geometry() -> None:
    suite = generate_probe_suite(StableWordEncoder(), seed=8_540_911, cases_per_family=8)

    for probe in suite:
        assert probe.query_token == len(probe.prompt_ids) - 1
        if probe.family == "local_negative":
            assert 0 <= probe.evidence_distance <= LOCAL_CONTROL_MAX_DISTANCE
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


def test_v3_is_deterministic_and_keeps_hidden_metadata_out_of_input() -> None:
    encoder_a = StableWordEncoder()
    encoder_b = StableWordEncoder()
    suite_a = generate_probe_suite(encoder_a, seed=8_540_911, cases_per_family=3)
    suite_b = generate_probe_suite(encoder_b, seed=8_540_911, cases_per_family=3)

    assert [probe.prompt_text for probe in suite_a] == [probe.prompt_text for probe in suite_b]
    assert [probe.prompt_ids for probe in suite_a] == [probe.prompt_ids for probe in suite_b]

    for probe in suite_a:
        assert str(probe.case_id) not in probe.prompt_text
        assert "rare_fact" not in probe.prompt_text
        assert "local_negative" not in probe.prompt_text
        assert probe.used_for_training is False
