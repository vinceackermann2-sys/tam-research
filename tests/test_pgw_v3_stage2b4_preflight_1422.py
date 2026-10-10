"""Stage-2B4 #1422: frozen nine-arm CPU derivative/resource probes only."""
from __future__ import annotations

import math

import pytest
import torch

from tam_research.pgw_v3_stage2b4.preflight import (
    ARM_NAMES, DELAYS,
    _build_paired_arms, _fixture,
    cpu_derivative_timing_preflight, gradient_activity_preflight,
)
from tam_research.pgw_v3_stage2b3.position import GlobalPositionCausalReference
from tam_research.pgw_v3_stage2b2.reference import CausalReference
from tam_research.pgw_v3_stage2a.model import PGWV3Stage2A


def test_all_nine_arms_have_family_paired_initial_state_and_exact_counts():
    original_rng = torch.get_rng_state().clone()
    models = _build_paired_arms()
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert tuple(models) == ARM_NAMES
    assert len(models) == 9
    pgw_state = models["pgw_hybrid"].state_dict()
    reference_state = models["full_no_global"].state_dict()
    for name, model in models.items():
        assert isinstance(model, PGWV3Stage2A if name.startswith("pgw_") else CausalReference)
        assert model.training is False
        assert all(p.device.type == "cpu" for p in model.parameters())
        count = sum(p.numel() for p in model.parameters())
        assert count == (20354 if name.startswith("pgw_") else 20347)
        original_family = pgw_state if name.startswith("pgw_") else reference_state
        assert set(model.state_dict()) == set(original_family)
        assert all(
            torch.equal(x, original_family[key])
            for key, x in model.state_dict().items()
        )
    assert isinstance(models["full_global"], GlobalPositionCausalReference)


@pytest.mark.parametrize("delay,expected_length", ((1,80), (2,88), (4,104)))
def test_complete_nine_arm_derivative_coverage_per_delay(delay, expected_length):
    global_rng_before = torch.get_rng_state().clone()
    t, a, targets = _fixture(delay)
    assert t.shape == (2, expected_length)
    assert a.tolist() == [expected_length - 6] * 2
    assert 32 in targets.tolist()  # missing key output class index
    assert all(0 <= v <= 32 for v in targets.tolist())

    panel = gradient_activity_preflight(delays=(delay,))
    assert len(panel) == len(ARM_NAMES) == 9
    assert tuple(r.arm for r in panel) == ARM_NAMES
    assert torch.equal(torch.get_rng_state(), global_rng_before)
    for row in panel:
        assert row.delay == delay
        assert row.tokens == expected_length
        assert row.batch == 2
        assert row.instantiated == (20354 if row.arm.startswith("pgw_") else 20347)
        assert row.state_unchanged
        assert row.attention.tokens == expected_length
        assert row.attention.full_dense_pair_ops > row.attention.chunk_dense_pair_ops
        for x in (row.answer_ce, row.auxiliary):
            assert 0 <= x.nonzero_elements <= x.grad_seen_elements <= row.instantiated
            assert x.nonzero_elements == (
                x.predictor_nonzero + x.utility_nonzero +
                x.other_workspace_nonzero + x.nonworkspace_nonzero
            )
        assert row.answer_ce.predictor_nonzero == 0
        assert row.answer_ce.nonworkspace_nonzero > 0
        assert row.auxiliary.predictor_nonzero > 0
        assert row.auxiliary.nonworkspace_nonzero > 0
        assert row.auxiliary.utility_nonzero == 0
        assert max(row.answer_ce.nonzero_elements, row.auxiliary.nonzero_elements) <= row.union_nonzero <= row.instantiated

    modes = {r.arm: r for r in panel}
    assert modes["pgw_no_workspace"].answer_ce.utility_nonzero == 0
    assert modes["pgw_no_workspace"].answer_ce.other_workspace_nonzero == 0
    assert modes["pgw_hybrid"].answer_ce.other_workspace_nonzero > 0
    assert modes["pgw_utility_only"].answer_ce.utility_nonzero > 0
    for name in ("full_no_global", "full_global", "chunk_only"):
        assert modes[name].answer_ce.other_workspace_nonzero == 0
        assert modes[name].answer_ce.utility_nonzero == 0


def test_structural_gradient_panel_reproducible_without_global_rng_mutation():
    rng_before = torch.get_rng_state().clone()
    first = gradient_activity_preflight(
        arms=("pgw_hybrid", "pgw_no_workspace", "full_global"), delays=(1,)
    )
    second = gradient_activity_preflight(
        arms=("pgw_hybrid", "pgw_no_workspace", "full_global"), delays=(1,)
    )
    assert first == second
    assert torch.equal(torch.get_rng_state(), rng_before)
    # A single probe is an empirical lower bound, not mathematical active capacity.
    assert first[0].answer_ce.nonzero_elements != first[1].answer_ce.nonzero_elements


@pytest.mark.parametrize("arm", ("pgw_hybrid", "full_global"))
def test_real_cpu_forward_vs_forward_ce_backward_timing_smoke(arm):
    original_threads = torch.get_num_threads()
    original_rng = torch.get_rng_state().clone()
    report = cpu_derivative_timing_preflight(
        arm=arm, delay=1, warmups=0, repeats=2
    )
    assert report.arm == arm and report.delay == 1
    assert report.batch == 2 and report.tokens == 80
    assert report.repeats == 2 and report.warmups == 0
    assert report.requested_cpu_threads == 1
    assert report.threads_restored
    assert torch.get_num_threads() == original_threads
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert report.unchanged_after_timing
    assert report.native_peak_tensor_memory.startswith("unknown")
    assert "forward+CE+backward" in report.timings_scope
    assert report.torch_version
    assert report.python_version
    assert report.platform_name
    for v in (
        report.median_forward_ms, report.p95_forward_ms,
        report.median_forward_and_ce_backward_ms,
        report.p95_forward_and_ce_backward_ms,
    ):
        assert math.isfinite(v) and v > 0
    assert report.p95_forward_ms >= report.median_forward_ms
    assert report.p95_forward_and_ce_backward_ms >= report.median_forward_and_ce_backward_ms


@pytest.mark.parametrize("bad_arm", ("", "unknown", "pgw", "transformer"))
def test_unknown_arm_fails_closed(bad_arm):
    with pytest.raises(ValueError):
        gradient_activity_preflight(arms=(bad_arm,), delays=(1,))
    with pytest.raises(ValueError):
        cpu_derivative_timing_preflight(
            arm=bad_arm, delay=1, warmups=0, repeats=1
        )


@pytest.mark.parametrize("bad_delay", (0, 3, 5, True, 1.5))
def test_invalid_delay_fails_closed(bad_delay):
    with pytest.raises(ValueError):
        gradient_activity_preflight(arms=("full_global",), delays=(bad_delay,))
    with pytest.raises(ValueError):
        cpu_derivative_timing_preflight(
            arm="full_global", delay=bad_delay, warmups=0, repeats=1
        )


@pytest.mark.parametrize("warmups,repeats", (
    (-1,2), (6,2), (0,0), (0,21), (True,2), (0,1.1),
))
def test_unbounded_or_invalid_timing_requests_are_rejected(warmups, repeats):
    with pytest.raises(ValueError):
        cpu_derivative_timing_preflight(
            arm="full_global", delay=1, warmups=warmups, repeats=repeats
        )


def test_duplicate_or_empty_arm_panel_fails_closed():
    for arms in ((), ("pgw_hybrid", "pgw_hybrid")):
        with pytest.raises(ValueError):
            gradient_activity_preflight(arms=arms, delays=(1,))
    with pytest.raises(ValueError):
        gradient_activity_preflight(arms=("pgw_hybrid",), delays=())
