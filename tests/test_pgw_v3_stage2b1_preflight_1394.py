"""PGW-v3 #1394: static fixture integrity and CPU derivative capacity only."""
from __future__ import annotations

import random

import pytest
import torch

from tam_research.pgw_v3_stage0.model import ROUTE_MODES
from tam_research.pgw_v3_stage2b1.manifest import (
    BASELINE_NAMES, CELLS_PER_SPLIT, REPEATS_PER_CELL, static_fixture_manifest,
)
from tam_research.pgw_v3_stage2b1.capacity import gradient_capacity_preflight


def test_stage2b1_exact_factorial_manifest_and_baseline_splits():
    original_rng = random.getstate()
    report = static_fixture_manifest()
    assert random.getstate() == original_rng
    assert report.per_cell == REPEATS_PER_CELL == 4
    assert report.samples == 2160
    assert report.total_input_tokens == 195840
    assert len(report.sha256) == 64
    assert CELLS_PER_SPLIT == 180
    assert tuple(s.split for s in report.per_split) == ("train", "validation", "test")
    assert len({s.sha256 for s in report.per_split}) == 3
    for split in report.per_split:
        assert split.samples == 720
        assert split.unique_cells == 180
        assert split.present == 360 and split.missing == 360
        assert len(split.sha256) == 64
        names = dict(split.baselines)
        assert tuple(names) == BASELINE_NAMES
        baseline = names["always_not_found"]
        assert baseline.correct == 360
        assert baseline.present_correct == 0
        assert baseline.missing_correct == 360
        for name, score in split.baselines:
            assert 0 <= score.correct <= 720
            assert score.correct == score.present_correct + score.missing_correct
            assert 0 <= score.present_correct <= 360
            assert 0 <= score.missing_correct <= 360
            if name != "always_not_found":
                assert score.correct < 720  # Fixed naive heuristics cannot solve all cells.


def test_static_preflight_digest_is_repeatable_and_order_independent():
    before = static_fixture_manifest(per_cell=2)
    after = static_fixture_manifest(per_cell=2)
    assert before == after
    assert before.samples == 1080
    assert len({x.sha256 for x in before.per_split}) == 3
    assert before.sha256 != static_fixture_manifest(per_cell=1).sha256


@pytest.mark.parametrize("bad", (0, -1, 17, True, 1.5))
def test_static_preflight_rejects_invalid_fixture_sizes(bad):
    with pytest.raises(ValueError):
        static_fixture_manifest(per_cell=bad)


def test_cpu_gradient_activity_preflight_preserves_weights_rng_and_route_coverage():
    original_rng = torch.get_rng_state().clone()
    reports = gradient_capacity_preflight()
    assert torch.equal(original_rng, torch.get_rng_state())
    assert len(reports) == len(ROUTE_MODES) == 6
    assert {r.mode for r in reports} == set(ROUTE_MODES)
    for r in reports:
        assert r.instantiated == 20354
        assert r.unchanged_after_backprop
        for x in (r.answer_ce, r.auxiliary):
            assert 0 <= x.nonzero_elements <= x.grad_seen_elements <= r.instantiated
            assert x.nonzero_elements == (
                x.predictor_nonzero + x.utility_nonzero
                + x.other_workspace_nonzero + x.nonworkspace_nonzero
            )
        assert max(r.answer_ce.nonzero_elements, r.auxiliary.nonzero_elements) <= r.union_nonzero <= r.instantiated
        assert r.answer_ce.nonworkspace_nonzero > 0
        assert r.answer_ce.predictor_nonzero == 0
        assert r.auxiliary.predictor_nonzero > 0
        assert r.auxiliary.nonworkspace_nonzero > 0
        assert r.auxiliary.utility_nonzero == 0
    modes = {r.mode: r for r in reports}
    assert modes["no_workspace"].answer_ce.other_workspace_nonzero == 0
    assert modes["no_workspace"].answer_ce.utility_nonzero == 0
    assert modes["hybrid"].answer_ce.other_workspace_nonzero > 0
    assert modes["hybrid"].answer_ce.utility_nonzero > 0
    assert modes["utility_only"].answer_ce.utility_nonzero > 0
    assert modes["no_workspace"].auxiliary.predictor_nonzero > 0


def test_gradient_preflight_is_deterministic_without_scientific_seeds_or_steps():
    a = gradient_capacity_preflight()
    b = gradient_capacity_preflight()
    assert a == b
    # Different gradient availability across route modes is already a parity blocker.
    assert {r.answer_ce.nonzero_elements for r in a} != {a[0].answer_ce.nonzero_elements}
