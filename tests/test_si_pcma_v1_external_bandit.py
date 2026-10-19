"""Published-bandit causal-information contract; no scientific trials."""
from dataclasses import fields
from unittest.mock import patch

import pytest

from tam_research.si_pcma_v1_external_bandit import (
    ARMS, HORIZON, PACKAGE_VERSION, FIXTURE_EPISODES, METHODS,
    FactualOnlineBanditPolicy, PublicBanditFeedback,
    published_bandit_development_screen, run_development_bandit_episode,
)


def test_public_choice_requires_one_executed_feedback_event():
    p = FactualOnlineBanditPolicy("ucb1", fixture_rng_seed=2)
    arm = p.choose(0)
    assert arm == 0
    with pytest.raises(RuntimeError, match="previous"):
        p.choose(1)
    with pytest.raises(ValueError):
        p.observe(PublicBanditFeedback(arm+1, 1, 1/HORIZON, 1))
    assert sum(p.counts) == 0 and p._pending == arm
    p.observe(PublicBanditFeedback(arm, 1, 1/HORIZON, 1))
    assert sum(p.counts) == 1 and p.wins[arm] == 1
    assert p.choose(1) == 1


def test_private_probabilities_never_part_of_policy_data():
    p = FactualOnlineBanditPolicy("beta_thompson", fixture_rng_seed=3)
    assert not hasattr(p, "bandits")
    assert not hasattr(p, "get_state")
    assert not hasattr(p, "oracle")
    assert [f.name for f in fields(PublicBanditFeedback)] == [
        "action", "observed_payout", "reward", "step"]
    assert p.choose(0) in range(ARMS)
    # Reject data-shaped stubs other than real public feedback.
    with pytest.raises(ValueError):
        p.observe({"bandits": [1.0] * ARMS})


def test_reward_and_payout_consistency_prevents_fabricated_outcomes():
    p = FactualOnlineBanditPolicy("fixed")
    assert p.choose(0) == 0
    bad = (
        PublicBanditFeedback(0, 1, -1/HORIZON, 1),
        PublicBanditFeedback(0, 1, 1/HORIZON, 2),
        PublicBanditFeedback(0, 2, 1/HORIZON, 1),
        PublicBanditFeedback(0, 0, float("nan"), 1),
        PublicBanditFeedback(0, 0, -1/HORIZON, 0),
    )
    for event in bad:
        with pytest.raises(ValueError):
            p.observe(event)
    assert p._time == 0 and not p.history
    p.observe(PublicBanditFeedback(0, 0, -1/HORIZON, 1))
    assert p._time == 1 and p.counts[0] == 1 and p.wins[0] == 0


def test_ucb_untried_exploration_and_boundaries():
    p = FactualOnlineBanditPolicy("ucb1")
    for i in range(ARMS):
        arm = p.choose(0)
        assert arm == i
        p.observe(PublicBanditFeedback(arm, i % 2,
                                      (2*(i % 2)-1)/HORIZON, i+1))
    assert sum(p.counts) == ARMS and p.wins == [i % 2 for i in range(ARMS)]
    assert p.choose(0) in range(ARMS)


def test_beta_thompson_internal_rng_repeatable_independent_of_world():
    a = FactualOnlineBanditPolicy("beta_thompson", fixture_rng_seed=44)
    b = FactualOnlineBanditPolicy("beta_thompson", fixture_rng_seed=44)
    actions = []
    for step in range(1, 21):
        left, right = a.choose(0), b.choose(0)
        assert left == right
        actions.append(left)
        event = PublicBanditFeedback(left, step % 2,
                                     (2*(step % 2)-1)/HORIZON, step)
        a.observe(event)
        b.observe(event)
    assert len(set(actions)) > 1


def test_actual_unmodified_published_world_one_episode_all_methods():
    for method in METHODS:
        record = run_development_bandit_episode(method, FIXTURE_EPISODES[0])
        assert record.method == method
        assert record.episode_id == FIXTURE_EPISODES[0]
        assert record.feedback_count == record.actions == HORIZON == 200
        assert -1.0000001 <= record.total_normalized_return <= 1.0000001
        if method == "fixed":
            assert record.distinct_arms_sampled == 1
        elif method == "round_robin":
            assert record.distinct_arms_sampled == ARMS


def test_fixed_development_report_never_promotes_scientific_seeds():
    report = published_bandit_development_screen()
    assert report.status == "NONCONFIRMATORY_PUBLISHED_DEVELOPMENT_SCREEN"
    assert report.package_version == PACKAGE_VERSION == "1.0.7"
    assert report.training_models_evaluated == 0
    assert report.science_seeds_reserved == 0
    assert report.heldout_worlds_evaluated == 0
    assert report.episode_count_per_control == len(FIXTURE_EPISODES) == 8
    assert report.action_opportunities_per_episode == HORIZON
    assert not report.may_claim_architecture_gain
    assert report.original_v26_verdict == "NO_GO"
    assert report.pcma_verdict == "UNKNOWN"
    assert {x.method for x in report.summaries} == set(METHODS)
    assert all(x.episodes == 8 and -1 <= x.mean_episode_return <= 1
               for x in report.summaries)


@pytest.mark.parametrize("bad", [39, 40, 49, -1, 41.0, True])
def test_unknown_or_noninteger_episode_ids_denied(bad):
    with pytest.raises(ValueError):
        run_development_bandit_episode("ucb1", bad)


def test_unsupported_methods_and_bad_public_inputs_denied():
    with pytest.raises(ValueError):
        FactualOnlineBanditPolicy("oracle")
    p = FactualOnlineBanditPolicy("ucb1")
    for bad in (3, -1, True, 0.0):
        with pytest.raises(ValueError):
            p.choose(bad)


def test_independent_published_world_code_does_not_enter_controller():
    import tam_research.si_pcma_v1_external_bandit as module
    assert not hasattr(module.FactualOnlineBanditPolicy, "get_state")
    assert not hasattr(module.FactualOnlineBanditPolicy, "bandits")
    assert not hasattr(module, "optimal_reward")
    assert not hasattr(module, "counterfactual")


def test_published_distribution_is_not_implicitly_independent_action_trials():
    # Replication/uncertainty units are WORLDS/EPISODES, not their 200
    # successive correlated chosen-action outcomes. No GO gate asserted.
    report = published_bandit_development_screen()
    assert all(x.episodes == 8 for x in report.summaries)
    assert all(x.min_episode_return <= x.mean_episode_return <=
               x.max_episode_return for x in report.summaries)
