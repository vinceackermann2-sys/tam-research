"""Engineering tests for two-step factual information acquisition; no results."""
from dataclasses import replace

import pytest

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODWorldConfig, OODRuleWorld, run_ood_sessions
from tam_research.si_pcma_v1_probe_value import (
    FactualTwoStepOutcome, OnlineFactualTwoStepPolicy, factual_two_step_outcomes,
)


def _event(before, action, after, reward, time, done=False):
    return FactualTransition(
        PublicObservation(before, 4), action, PublicObservation(after, 3),
        float(reward), 0.1 if action == 3 else 0.0, time,
        f"fixture-{time}", done,
    )


def test_two_step_outcomes_are_exactly_chosen_factual_pairs():
    # Probe sacrifices the first reward; the paid fee is subtracted.
    probe = _event(0, 3, 0, 0, 1)
    observed_guess = _event(0, 2, 1, 1, 2, done=True)
    pair = factual_two_step_outcomes(((probe, observed_guess),))
    assert len(pair) == 1
    assert pair[0] == FactualTwoStepOutcome(
        probe.before, 3, 2, 1, 2, pytest.approx(.9))
    # Two actual successful guesses produce a different, two-action return.
    first_guess = _event(0, 1, 1, 1, 3)
    second_guess = _event(1, 0, 2, 1, 4, done=True)
    assert factual_two_step_outcomes(((first_guess, second_guess),))[0].observed_total_net == 2.0


def test_no_false_cross_session_pairs_or_odd_horizon_completion():
    sessions = (
        (_event(0, 1, 1, 1, 1), _event(1, 1, 2, 0, 2), _event(2, 1, 0, 0, 3, True)),
        (_event(0, 0, 1, 0, 4), _event(1, 2, 2, 1, 5, True)),
    )
    samples = factual_two_step_outcomes(sessions)
    assert len(samples) == 2
    assert [(x.first_time, x.second_time) for x in samples] == [(1, 2), (4, 5)]
    assert not any(x.first_time == 3 or x.second_time == 4 for x in samples)


def test_session_parser_rejects_bad_or_replayed_outcomes():
    normal = (_event(0, 0, 1, 1, 1), _event(1, 1, 2, 0, 2, True))
    with pytest.raises(ValueError):
        factual_two_step_outcomes((normal, normal))
    with pytest.raises(ValueError):
        factual_two_step_outcomes(((normal[0],),))
    with pytest.raises(ValueError):
        factual_two_step_outcomes(((replace(normal[0], fee=.1), normal[1]),))
    with pytest.raises(ValueError):
        factual_two_step_outcomes(((replace(normal[0], done=True), normal[1]),))


def test_learner_does_not_update_first_stage_value_until_both_actions_observed():
    actor = OnlineFactualTwoStepPolicy(exploration=0, allow_probe=False)
    world = OODRuleWorld(OODWorldConfig(6, "development"))
    first = actor.choose(world.observe())
    assert first in range(3)
    actor.observe(world.step(first))
    assert actor.paired_samples == 0
    assert actor.observed_first_count(0, first) == 0
    second = actor.choose(world.observe())
    assert second in range(3)
    actor.observe(world.step(second))
    assert actor.paired_samples == 1
    assert sum(n for n, _ in actor._first_values.values()) == 1


def test_probe_decision_requires_observed_purchase_evidence_and_margin():
    actor = OnlineFactualTwoStepPolicy(exploration=0, minimum_probe_samples=2)
    actor._first_values[(1, 3)] = (1, 1.0)
    assert actor.choose(PublicObservation(1, 6)) != 3
    # Finish the first decision in a real world so the pending state is valid.
    world = OODRuleWorld(OODWorldConfig(7, "development"))
    # A new independent policy can receive a preloaded factual score fixture.
    actor = OnlineFactualTwoStepPolicy(exploration=0, minimum_probe_samples=2)
    actor._first_values[(1, 3)] = (10, 10.0)
    assert actor.choose(PublicObservation(1, 6)) == 3
    assert actor._unresolved[1] == 3


def test_no_probe_only_control_never_buys_and_never_probes_last_turn():
    a = OnlineFactualTwoStepPolicy(allow_probe=False, exploration=1.0, fixture_seed=4)
    result = run_ood_sessions(a, OODWorldConfig(12, "development"), sessions=2)
    assert all(event.action < 3 for part in result for event in part)
    assert a.paired_samples == 6
    b = OnlineFactualTwoStepPolicy(allow_probe=True, exploration=1.0)
    assert b.choose(PublicObservation(1, 1)) < 3


def test_exploratory_probe_is_real_with_correct_fee_and_two_step_count():
    actor = OnlineFactualTwoStepPolicy(
        allow_probe=True, exploration=1.0, fixture_seed=21)
    traces = run_ood_sessions(actor, OODWorldConfig(14, "development"), sessions=3)
    items = [event for session in traces for event in session]
    assert len(items) == 18 and actor.paired_samples == 9
    assert actor.observed_steps == 18
    assert all(e.fee == (0.1 if e.action == 3 else 0.0) for e in items)
    assert all(e.action in range(3) for session in traces
               for e in session[1::2])


def test_adaptation_persists_across_sessions_and_reset_deletes_values():
    actor = OnlineFactualTwoStepPolicy(
        allow_probe=False, exploration=0, fixture_seed=2)
    run_ood_sessions(actor, OODWorldConfig(15, "development"), sessions=2)
    assert actor.paired_samples == 6
    assert actor.observed_steps == 12
    actor.restart_session()
    assert actor.paired_samples == 6
    actor.restart_session(erase_persistent=True)
    assert actor.paired_samples == 0 and not actor._first_values and not actor._guess_values


def test_rejects_wrong_or_unresolved_feedback_without_updating_samples():
    actor = OnlineFactualTwoStepPolicy(exploration=0, allow_probe=False)
    world = OODRuleWorld(OODWorldConfig(17, "development"))
    before = world.observe()
    action = actor.choose(before)
    with pytest.raises(RuntimeError):
        actor.choose(before)
    event = world.step(action)
    with pytest.raises(ValueError):
        actor.observe(replace(event, action=(action+1)%4))
    with pytest.raises(ValueError):
        actor.observe(replace(event, fee=1.0))
    assert actor.paired_samples == 0 and actor.observed_steps == 0
    actor.observe(event)
    with pytest.raises(RuntimeError):
        actor.restart_session()


def test_policy_module_never_imports_latent_rule_or_world():
    # The policy logic must be implementable using ONLY public types/records.
    import tam_research.si_pcma_v1_probe_value as module
    assert not hasattr(module, "OODRuleWorld")
    assert not hasattr(module, "DEVELOPMENT_RULES")
    assert not hasattr(module, "HELDOUT_RULES")


@pytest.mark.parametrize("kwargs", [
    {"exploration": -0.01},
    {"exploration": 1.01},
    {"pseudocount": 0},
    {"minimum_probe_samples": -1},
    {"probe_margin": -0.1},
])
def test_invalid_purchase_config_rejected(kwargs):
    with pytest.raises(ValueError):
        OnlineFactualTwoStepPolicy(**kwargs)
