"""Pure engineering invariants; NOT randomized scientific attempts."""
from dataclasses import replace

import pytest

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_randomized_acquisition import (
    FrozenSymbolContinuation, RandomizedAcquisitionRecord,
    collect_randomized_development_acquisition, describe_randomized_arms,
    arm_support_complete,
)


def _logs():
    return collect_randomized_development_acquisition(
        (OODWorldConfig(10, "development", "stable"),
         OODWorldConfig(20, "development", "shift"),
         OODWorldConfig(30, "development", "long")),
        draw_fixture_seed=42, sessions_per_world=2,
    )


def test_uniform_first_action_propensity_predeclared_and_real_fees():
    records = _logs()
    assert len(records) == (3 + 3 + 8)*2
    assert all(r.first_propensity == 0.25 for r in records)
    assert all(r.first_event.action == r.first_action and r.second_event.action < 3
               for r in records)
    assert all(r.first_event.fee == (0.1 if r.first_action == 3 else 0.0)
               for r in records)
    assert all(r.second_event.fee == 0.0 for r in records)
    assert all(r.realized_two_action_net ==
               r.first_event.reward - r.first_event.fee +
               r.second_event.reward - r.second_event.fee for r in records)


def test_fixed_continuation_sees_only_actual_public_observation_and_first_feedback():
    seen = []
    def public_continuation(observation, event):
        seen.append((observation, event))
        assert type(observation) is PublicObservation
        assert type(event) is FactualTransition
        assert observation == event.after
        return (observation.symbol + 1) % 3
    records = collect_randomized_development_acquisition(
        (OODWorldConfig(8, "development"),), draw_fixture_seed=2,
        sessions_per_world=1, continuation=public_continuation)
    assert len(records) == len(seen) == 3
    assert all(x.second_event.action == (x.first_event.after.symbol + 1) % 3
               for x in records)


def test_holdout_configs_rejected_before_any_continuation_executes():
    calls = []
    def continuation(*args):
        calls.append(args)
        return 0
    with pytest.raises(ValueError, match="heldout"):
        collect_randomized_development_acquisition(
            (OODWorldConfig(8, "development"),
             OODWorldConfig(9, "heldout")),
            draw_fixture_seed=4, continuation=continuation)
    assert calls == []


def test_two_action_targets_never_link_to_next_session_or_other_world():
    records = _logs()
    assert all(r.second_event.time == r.first_event.time + 1 for r in records)
    assert all(r.second_event.before == r.first_event.after for r in records)
    assert len({(r.world_id,r.first_event.time) for r in records}) == len(records)
    assert all(not r.first_event.done for r in records)
    assert all(r.second_event.done ==
               (r.second_event.after.steps_remaining == 0) for r in records)


def test_factual_record_cannot_forge_unpurchased_return_or_propensity():
    original = _logs()[0]
    with pytest.raises(ValueError):
        replace(original, first_propensity=0.5)
    with pytest.raises(ValueError):
        replace(original, first_action=(original.first_action+1)%4)
    with pytest.raises(ValueError):
        replace(original, realized_two_action_net=original.realized_two_action_net+1)
    with pytest.raises(ValueError):
        replace(original, second_event=replace(original.second_event, time=original.first_event.time))


def test_factual_arm_description_counts_and_weighted_sums():
    records = _logs()
    summary = describe_randomized_arms(records)
    assert len(summary) == 4
    assert sum(x.factual_count for x in summary) == len(records)
    assert sum(x.effective_weight_sum for x in summary) == 4*len(records)
    for action, arm in enumerate(summary):
        assert arm.action == action
        chosen = [r for r in records if r.first_action == action]
        assert arm.factual_count == len(chosen)
        if chosen:
            assert arm.sample_mean == pytest.approx(
                sum(r.realized_two_action_net for r in chosen)/len(chosen))
            assert arm.importance_weighted_total == pytest.approx(
                4*sum(r.realized_two_action_net for r in chosen))
        else:
            assert arm.sample_mean is None


def test_each_symbol_arm_coverage_is_diagnostic_not_imputed():
    records = _logs()
    for symbol in range(3):
        arms = describe_randomized_arms(records, symbol=symbol)
        assert sum(a.factual_count for a in arms) == sum(
            r.before.symbol == symbol for r in records)
    assert not arm_support_complete(records, min_each=10000)
    assert not arm_support_complete((), min_each=1)


def test_reject_invalid_collection_and_continuation_contract():
    with pytest.raises(ValueError):
        collect_randomized_development_acquisition((), draw_fixture_seed=1)
    with pytest.raises(ValueError):
        collect_randomized_development_acquisition(
            (OODWorldConfig(1, "development"),), draw_fixture_seed=-1)
    with pytest.raises(ValueError):
        collect_randomized_development_acquisition(
            (OODWorldConfig(1, "development"),), draw_fixture_seed=1,
            sessions_per_world=0)
    with pytest.raises(ValueError):
        collect_randomized_development_acquisition(
            (OODWorldConfig(1, "development"), OODWorldConfig(1, "development")),
            draw_fixture_seed=1)
    with pytest.raises(ValueError, match="invalid guess"):
        collect_randomized_development_acquisition(
            (OODWorldConfig(1, "development"),), draw_fixture_seed=1,
            continuation=lambda o,e: 3)
    with pytest.raises(ValueError):
        FrozenSymbolContinuation((0,1,3))


def test_predeclared_scope_protects_empty_statistics_or_invalid_threshold():
    with pytest.raises(ValueError):
        describe_randomized_arms(())
    with pytest.raises(ValueError):
        describe_randomized_arms(_logs(), symbol=3)
    with pytest.raises(ValueError):
        arm_support_complete(_logs(), min_each=0)


def test_no_simulator_rule_access_is_imported_by_logger_public_code():
    import tam_research.si_pcma_v1_randomized_acquisition as m
    assert not hasattr(m, "DEVELOPMENT_RULES")
    assert not hasattr(m, "HELDOUT_RULES")
    assert hasattr(m, "OODRuleWorld")  # evaluator only; never passed to continuation
