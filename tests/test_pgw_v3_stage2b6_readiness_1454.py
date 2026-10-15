"""Stage-2B6 #1454 sealed corpus-root and no-training safety/fairness audit."""
from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch

from tam_research.pgw_v3_stage2b5.corpus import candidate_corpus_manifest
from tam_research.pgw_v3_stage2b6.readiness import (
    OPEN_SCIENCE_BLOCKERS, SEALED_ROOTS, readonly_research_gate,
    verify_sealed_corpus,
)


@pytest.fixture(scope="module")
def corpus():
    return candidate_corpus_manifest()


def test_literal_roots_equal_exact_source_bound_dataset(corpus):
    original_python_rng = random.getstate()
    report = verify_sealed_corpus(corpus)
    assert random.getstate() == original_python_rng
    assert report.valid
    assert report.roots == SEALED_ROOTS
    assert tuple(x[0] for x in report.roots) == (
        "train", "validation", "test", "all"
    )
    assert tuple(x[1] for x in report.split_samples) == (2880, 1440, 1440)
    assert report.samples == 5760 and report.input_tokens == 522240
    assert report.unique_token_fingerprints == 5760
    assert report.unique_semantic_traces == 5760
    assert report.cells_per_split == (180, 180, 180)


def test_frozen_root_is_not_dynamically_rewritten_or_accepted(corpus):
    wrong = replace(corpus, root_sha256="0" * 64)
    with pytest.raises(ValueError, match="FROZEN PGW corpus root mismatch"):
        verify_sealed_corpus(wrong)
    wrong_split = replace(
        corpus,
        splits=(replace(corpus.splits[0], root_sha256="f" * 64),)
        + corpus.splits[1:],
    )
    with pytest.raises(ValueError, match="FROZEN PGW corpus root mismatch"):
        verify_sealed_corpus(wrong_split)


def test_cardinality_collision_and_factorial_fail_closed(corpus):
    with pytest.raises(ValueError, match="cardinality"):
        verify_sealed_corpus(replace(corpus, total_samples=5759))
    with pytest.raises(ValueError, match="token count"):
        verify_sealed_corpus(replace(corpus, total_input_tokens=0))
    altered = replace(
        corpus,
        records=corpus.records[:-1] + (corpus.records[0],),
    )
    with pytest.raises(ValueError, match="collision"):
        verify_sealed_corpus(altered)
    altered_split = replace(
        corpus, splits=(replace(corpus.splits[0], cells=179),)
        + corpus.splits[1:],
    )
    with pytest.raises(ValueError, match="factorial"):
        verify_sealed_corpus(altered_split)


def test_primary_arms_all_delays_are_structurally_valid_but_science_blocked():
    py_before = random.getstate()
    torch_before = torch.get_rng_state().clone()
    report = readonly_research_gate()
    assert random.getstate() == py_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert report.corpus.valid
    assert report.corpus.roots == SEALED_ROOTS
    assert report.instantiated_parameter_delta == 7
    assert len(report.paired_derivatives) == 6
    assert {(x.arm, x.delay) for x in report.paired_derivatives} == {
        (a, d) for a in ("pgw_hybrid", "full_global") for d in (1, 2, 4)
    }
    for x in report.paired_derivatives:
        assert x.instantiated == (20354 if x.arm == "pgw_hybrid" else 20347)
        assert x.state_unchanged
        assert x.ce_predictor_nonzero == 0
        assert x.aux_predictor_nonzero > 0
        assert x.ce_nonzero > 0 and x.aux_nonzero > 0
        assert max(x.ce_nonzero, x.aux_nonzero) <= x.union_nonzero <= x.instantiated
    assert report.science_blockers == OPEN_SCIENCE_BLOCKERS
    assert len(report.science_blockers) >= 7
    assert not report.allowed_to_train
    assert not report.allowed_to_spend_modal
    assert report.decision.startswith("BLOCKED_")


def test_changed_answer_rejected(corpus):
    row = corpus.records[0]
    altered = replace(row, answer=64 if row.answer != 64 else 32)
    with pytest.raises(ValueError):
        verify_sealed_corpus(replace(corpus, records=(altered,) + corpus.records[1:]))


def test_changed_row_identifiers_rejected(corpus):
    row = corpus.records[0]
    for key in ('sample_id', 'token_sha256', 'semantic_sha256'):
        altered = replace(row, **{key: 'f' * 64})
        with pytest.raises(ValueError):
            verify_sealed_corpus(replace(corpus, records=(altered,) + corpus.records[1:]))


def test_changed_split_repeats_rejected(corpus):
    bad_split = replace(corpus.splits[0], per_cell=15)
    with pytest.raises(ValueError):
        verify_sealed_corpus(replace(corpus, splits=(bad_split,) + corpus.splits[1:]))
