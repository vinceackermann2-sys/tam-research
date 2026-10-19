"""Original SI PCMA paid-information cost: randomized DEVELOPMENT gate tests."""
from dataclasses import replace
import math

import pytest

from tam_research.si_pcma_v1 import PublicObservation
from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_probe_randomization import (
    FrozenPublicContinuation, randomized_development_first_actions,
)
from tam_research.si_pcma_v1_randomized_probe_gate import (
    SCENARIOS, SESSIONS_PER_WORLD, WORLD_SEEDS, RANDOMIZATION_FIXTURE,
    estimate_randomized_probe_blip, frozen_development_probe_screen,
    _dev_interval,
)


def _pairs():
    configs = [OODWorldConfig(401+i, "development", SCENARIOS[i%4])
               for i in range(24)]
    rows = randomized_development_first_actions(
        configs, fixture_randomization_seed=RANDOMIZATION_FIXTURE,
        continuation=FrozenPublicContinuation(), sessions=SESSIONS_PER_WORLD)
    return rows, tuple(c.scenario for c in configs)


def test_fixed_dev_screen_uses_24_worlds_not_816_pseudoreplicates():
    assert len(WORLD_SEEDS) == 24
    report = frozen_development_probe_screen()
    assert report.status == "DESCRIPTIVE_RANDOMIZED_DEVELOPMENT_ONLY"
    assert report.worlds == 24
    assert report.randomized_factual_pairs == 816
    assert report.sessions_per_world == 8
    assert len(report.per_world) == 24
    assert len(report.per_scenario) == 4
    assert all(row.independent_worlds == 6 for row in report.per_scenario)
    assert report.learned_neural_models_evaluated == 0
    assert report.heldout_worlds_evaluated == 0
    assert report.scientific_seeds_consumed == 0
    assert not report.scientific_gate_passed
    assert not report.may_claim_probe_is_useful
    assert not report.may_claim_si_breakthrough
    assert math.isfinite(report.mean_probe_minus_uniform_guess)
    assert report.developer_95_interval[0] <= report.mean_probe_minus_uniform_guess
    assert report.developer_95_interval[1] >= report.mean_probe_minus_uniform_guess


def test_estimand_measures_actual_first_action_and_fixed_next_guess():
    rows, scenarios = _pairs()
    report = estimate_randomized_probe_blip(
        rows, scenarios, sessions_per_world=SESSIONS_PER_WORLD)
    assert report.randomized_factual_pairs == sum(r.total_factual_pairs
                                                  for r in report.per_world)
    assert report.fixed_second_action == (0, 1, 2)
    assert report.paid_probe_fee == 0.1
    assert report.first_action_probability_probe == .25
    assert report.first_action_probability_any_guess == .75
    for row in rows:
        assert row.first_observation == row.first_event.before
        assert row.first_event.action == row.assigned_first_action
        assert row.second_event.action == row.continuation_actions[
            row.second_event.before.symbol]
        assert row.second_event.time == row.first_event.time+1
        assert row.second_event.action in range(3)
        if row.assigned_first_action == 3:
            assert row.first_event.reward == 0.
            assert row.first_event.fee == .1
    assert len({(r.world_index,r.session_index,r.pair_index) for r in rows}) == len(rows)


def test_world_stratification_is_precommitted_not_action_as_independent_unit():
    report = frozen_development_probe_screen()
    assert [r.scenario for r in report.per_scenario] == list(SCENARIOS)
    assert [r.total_factual_pairs for r in report.per_world] == [
        (64 if i%4 == 3 else 24) for i in range(24)]
    assert report.randomized_factual_pairs == 18*24+6*64
    assert {r.pair_count for r in report.per_scenario} == {144,384}
    effects = [r.probe_minus_uniform_guess for r in report.per_world]
    assert report.mean_probe_minus_uniform_guess == pytest.approx(
        sum(effects)/len(effects))
    assert report.developer_95_interval == pytest.approx(
        _dev_interval(effects, 2.069))


@pytest.mark.parametrize("bad_field", ["assignment_probability","first_action",
                                       "first_fee","second_action",
                                       "event_reward"])
def test_reject_forged_randomized_outcome_and_illegal_feedback(bad_field):
    rows, scenarios = _pairs()
    arr = list(rows)
    first = arr[0]
    if bad_field == "assignment_probability":
        bad = replace(first, assignment_probability=.5)
    elif bad_field == "first_action":
        bad = replace(first, assigned_first_action=(first.assigned_first_action+1)%4)
    elif bad_field == "first_fee":
        bad = replace(first, first_event=replace(first.first_event, fee=.75))
    elif bad_field == "second_action":
        bad = replace(first, second_event=replace(first.second_event, action=3))
    else:
        bad = replace(first, first_event=replace(first.first_event, reward=2.0))
    arr[0] = bad
    with pytest.raises(ValueError):
        estimate_randomized_probe_blip(arr, scenarios,
                                       sessions_per_world=SESSIONS_PER_WORLD)


def test_reject_duplicate_or_nonchronological_log_without_new_world():
    rows, scenarios = _pairs()
    with pytest.raises(ValueError, match="duplicate|unordered"):
        estimate_randomized_probe_blip((rows[0],rows[0],*rows[1:]),
                                      scenarios,sessions_per_world=8)
    with pytest.raises(ValueError, match="time gap"):
        modified = list(rows)
        first = modified[1]
        modified[1] = replace(
            first, first_event=replace(first.first_event,time=first.first_event.time+9),
            second_event=replace(first.second_event,time=first.second_event.time+9))
        estimate_randomized_probe_blip(modified,scenarios,sessions_per_world=8)


def test_reject_missing_independent_world_and_invalid_family_plan():
    rows, scenarios = _pairs()
    with pytest.raises(ValueError):
        estimate_randomized_probe_blip(rows[:-1],scenarios,
                                       sessions_per_world=SESSIONS_PER_WORLD)
    with pytest.raises(ValueError):
        estimate_randomized_probe_blip(rows,scenarios[:-1],sessions_per_world=8)
    with pytest.raises(ValueError):
        estimate_randomized_probe_blip(rows,("heldout",)*24,sessions_per_world=8)
    with pytest.raises(ValueError):
        estimate_randomized_probe_blip(rows,scenarios,sessions_per_world=33)


def test_clustering_requires_more_than_one_world_and_no_oracle_dependency():
    with pytest.raises(ValueError):
        _dev_interval((1.0,), 2.069)
    import tam_research.si_pcma_v1_randomized_probe_gate as module
    assert not hasattr(module, "OODRuleWorld")
    assert not hasattr(module, "HELDOUT_RULES")
    assert not hasattr(module, "oracle")
    assert not hasattr(module, "counterfactual_reward")
    assert module.EXPECTED_WORLD_COUNT == 24


def test_no_scientific_claim_even_if_dev_probe_contrast_is_positive():
    report = frozen_development_probe_screen()
    assert report.scientific_gate_passed is False
    assert report.may_claim_probe_is_useful is False
    assert report.may_claim_si_breakthrough is False
    assert "fixed public continuation" in report.estimand
