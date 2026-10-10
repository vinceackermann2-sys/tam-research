"""Stage-2B5 #1429 candidate corpus/metrics CPU-free oracle-only preflight."""
from __future__ import annotations

import dataclasses
import random

import pytest

from tam_research.pgw_v3_stage2b5.corpus import (
    CELLS_PER_SPLIT, SPLIT_REPEATS, TOTAL_SAMPLES,
    algorithmic_predictions, candidate_corpus_manifest,
    evaluate_complete_predictions,
)


@pytest.fixture(scope="module")
def corpus():
    return candidate_corpus_manifest()


def test_complete_candidate_corpus_5760_with_cross_split_semantic_audit(corpus):
    assert TOTAL_SAMPLES == 5760
    assert corpus.total_samples == 5760
    assert corpus.total_input_tokens == 522240
    assert CELLS_PER_SPLIT == 180
    assert tuple(x.name for x in corpus.splits) == ("train", "validation", "test")
    assert tuple(x.sample_count for x in corpus.splits) == (2880, 1440, 1440)
    assert tuple(x.per_cell for x in corpus.splits) == (16, 8, 8)
    assert tuple(x.cells for x in corpus.splits) == (180, 180, 180)
    assert len(corpus.root_sha256) == 64
    assert len({r.token_sha256 for r in corpus.records}) == 5760
    assert len({r.semantic_sha256 for r in corpus.records}) == 5760
    assert len({r.sample_id for r in corpus.records}) == 5760
    assert len({r.root_sha256 for r in corpus.splits}) == 3
    for root in corpus.splits:
        assert root.present == root.missing == root.sample_count // 2
        assert len(root.root_sha256) == 64
    for row in corpus.records:
        assert 32 <= row.answer <= 64
        assert row.answer == 64 if row.missing else 32 <= row.answer <= 63
        assert row.last_position in (2, 3, 4, 5, 6)
        assert row.overwrite_count in (0, 1, 2)
        assert row.delay in (1, 2, 4)


def test_all_factorial_cell_counts_and_index_ranges(corpus):
    from collections import Counter
    for split, repeats in SPLIT_REPEATS:
        rows = [r for r in corpus.records if r.split == split]
        cells = Counter(r.cell for r in rows)
        assert len(cells) == CELLS_PER_SPLIT
        assert set(cells.values()) == {repeats}
        assert {r.index for r in rows} == set(range(repeats))
        assert sum(r.missing for r in rows) == len(rows) // 2


def test_manifest_determinism_without_global_rng_mutation(corpus):
    state = random.getstate()
    again = candidate_corpus_manifest()
    assert random.getstate() == state
    assert corpus.root_sha256 == again.root_sha256
    assert corpus.splits == again.splits
    assert corpus.records == again.records


@pytest.mark.parametrize("split,expected", (("train",2880), ("validation",1440), ("test",1440)))
def test_trivial_not_found_shortcut_has_exact_reported_failure(corpus, split, expected):
    score = evaluate_complete_predictions(
        corpus, split=split, predictions=algorithmic_predictions(
            corpus, split=split, rule="always_not_found"
        )
    )
    assert score.total == expected
    assert score.total_accuracy == score.macro_cell_accuracy == 0.5
    assert score.present_total == score.missing_total == expected // 2
    assert score.present_accuracy == 0
    assert score.missing_accuracy == 1
    assert score.missing_tp == expected // 2
    assert score.missing_fp == expected // 2
    assert score.missing_fn == score.missing_tn == 0
    assert score.missing_precision == 0.5
    assert score.missing_recall == 1
    assert len(score.cells) == 180
    assert all(x.correct == (x.total if x.cell[4] else 0) for x in score.cells)
    assert all(acc == 0.5 for _, acc in score.by_delay)
    assert all(acc == 0.5 for _, acc in score.by_overwrites)
    assert all(acc == 0.5 for _, acc in score.by_position)


@pytest.mark.parametrize("split", ("validation", "test"))
def test_independent_untrained_oracle_scores_perfect(corpus, split):
    score = evaluate_complete_predictions(
        corpus, split=split, predictions=algorithmic_predictions(
            corpus, split=split, rule="oracle"
        )
    )
    assert score.total_accuracy == 1
    assert score.present_accuracy == 1
    assert score.missing_accuracy == 1
    assert score.macro_cell_accuracy == 1
    assert score.missing_precision == score.missing_recall == 1
    assert score.missing_fp == score.missing_fn == 0
    assert all(x.total == x.correct for x in score.cells)


@pytest.mark.parametrize("rule", (
    "copy_chunk_2", "latest_global_write", "most_frequent_value",
))
def test_naive_fixed_algorithmic_shortcuts_not_perfect(corpus, rule):
    score = evaluate_complete_predictions(
        corpus, split="test", predictions=algorithmic_predictions(
            corpus, split="test", rule=rule
        )
    )
    assert 0 <= score.total_accuracy < 1
    assert score.present_accuracy < 1
    assert score.macro_cell_accuracy < 1
    assert score.present_total == score.missing_total == 720


def test_metric_panel_order_does_not_change_score(corpus):
    preds = algorithmic_predictions(corpus, split="test", rule="always_not_found")
    a = evaluate_complete_predictions(corpus, split="test", predictions=preds)
    b = evaluate_complete_predictions(corpus, split="test", predictions=reversed(preds))
    assert a == b


@pytest.mark.parametrize("failure", (
    "missing", "duplicate", "extra", "unknown_class", "bool_class", "wrong_tuple",
))
def test_metric_gate_rejects_incomplete_or_corrupted_predictions(corpus, failure):
    preds = list(algorithmic_predictions(corpus, split="test", rule="always_not_found"))
    if failure == "missing":
        preds.pop()
    elif failure == "duplicate":
        preds.append(preds[0])
    elif failure == "extra":
        preds.append(("not-in-manifest", 64))
    elif failure == "unknown_class":
        preds[0] = (preds[0][0], 160)
    elif failure == "bool_class":
        preds[0] = (preds[0][0], True)
    else:
        preds[0] = [preds[0][0], 64]
    with pytest.raises(ValueError):
        evaluate_complete_predictions(corpus, split="test", predictions=preds)


def test_strict_split_and_rule_gates(corpus):
    with pytest.raises(ValueError):
        algorithmic_predictions(corpus, split="other", rule="oracle")
    with pytest.raises(ValueError):
        algorithmic_predictions(corpus, split="test", rule="model_train")
    with pytest.raises(ValueError):
        evaluate_complete_predictions(corpus, split="train2", predictions=())
