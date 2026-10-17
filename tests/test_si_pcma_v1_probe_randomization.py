"""Development randomization and fixed continuation correctness checks ONLY."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_probe_randomization import (
    FrozenPublicContinuation, randomized_development_first_actions,
    descriptive_randomized_arm_values,
)


def _log():
    return randomized_development_first_actions(
        (OODWorldConfig(20, "development", "stable"),
         OODWorldConfig(21, "development", "long")),
        fixture_randomization_seed=33, sessions=1,
    )


def test_randomization_is_reproducible_and_first_arm_propensity_explicit():
    a = _log()
    assert a == _log()
    assert len(a) == 11  # 6/2 + 16/2 real two-action opportunities
    assert all(p.assignment_probability == 0.25 for p in a)
    assert all(p.assigned_first_action == p.first_event.action for p in a)
    assert all(p.second_event.time == p.first_event.time + 1 for p in a)
    assert all(p.first_event.done is False for p in a)
    assert all(p.second_event.action < 3 for p in a)
    assert all(p.continuation_actions == (0, 1, 2) for p in a)
    assert [(p.world_index, p.session_index, p.pair_index) for p in a] == (
        [(0, 0, i) for i in range(3)] +
        [(1, 0, i) for i in range(8)]
    )


def test_probe_fee_and_opportunity_cost_are_real_not_counterfactual():
    a = _log()
    assert all(p.first_event.fee == (0.1 if p.assigned_first_action == 3 else 0.0)
               for p in a)
    assert all(p.second_event.fee == 0.0 for p in a)
    assert all(p.observed_net ==
               p.first_event.reward - p.first_event.fee +
               p.second_event.reward - p.second_event.fee for p in a)


def test_heldout_logger_denied_before_any_world_construction():
    with patch("tam_research.si_pcma_v1_probe_randomization.OODRuleWorld") as world:
        with pytest.raises(ValueError, match="heldout"):
            randomized_development_first_actions(
                (OODWorldConfig(1, "development"),
                 OODWorldConfig(2, "heldout")),
                fixture_randomization_seed=3)
        world.assert_not_called()


def test_fixed_public_continuation_cannot_train_or_see_hidden_rule():
    continuation = FrozenPublicContinuation((2, 0, 1))
    from tam_research.si_pcma_v1 import PublicObservation
    assert continuation.choose_guess(PublicObservation(0, 3)) == 2
    assert continuation.choose_guess(PublicObservation(1, 3)) == 0
    with pytest.raises(ValueError):
        FrozenPublicContinuation((1, 1, 2))
    with pytest.raises(ValueError):
        continuation.choose_guess(PublicObservation(4, 3))


def test_descriptive_values_never_fill_in_unobserved_arms():
    output = descriptive_randomized_arm_values(_log())
    assert len(output) == 12
    assert sum(record.count for record in output) == 11
    assert any(row.count == 0 for row in output)
    assert all(row.observed_mean_net is None for row in output if row.count == 0)
    assert all(row.observed_mean_net is not None for row in output if row.count > 0)


def test_reject_modified_propensity_and_future_reward_assignment():
    a = list(_log())
    with pytest.raises(ValueError):
        descriptive_randomized_arm_values([replace(a[0], assignment_probability=0.5)])
    with pytest.raises(ValueError):
        descriptive_randomized_arm_values([replace(a[0],
            second_event=replace(a[0].second_event, time=100))])
    with pytest.raises(ValueError):
        descriptive_randomized_arm_values([replace(a[0], assigned_first_action=10)])


def test_descriptive_values_reject_replayed_grouping_key():
    pairs = _log()
    with pytest.raises(ValueError, match="duplicate"):
        descriptive_randomized_arm_values((pairs[0], pairs[0]))
    with pytest.raises(ValueError, match="invalid"):
        descriptive_randomized_arm_values((
            replace(pairs[0], world_index=-1),))


def test_logger_rejects_duplicate_development_world():
    with pytest.raises(ValueError, match="duplicate"):
        randomized_development_first_actions(
            (OODWorldConfig(5, "development"),
             OODWorldConfig(5, "development")),
            fixture_randomization_seed=12)


def test_logger_does_not_accept_unregistered_seed_shape_or_empty_plan():
    with pytest.raises(ValueError):
        randomized_development_first_actions((), fixture_randomization_seed=1)
    with pytest.raises(ValueError):
        randomized_development_first_actions(
            (OODWorldConfig(1, "development"),), fixture_randomization_seed=-1)
    with pytest.raises(ValueError):
        randomized_development_first_actions(
            (OODWorldConfig(1, "development"),), fixture_randomization_seed=2,
            sessions=0)
    with pytest.raises(ValueError):
        descriptive_randomized_arm_values(())
