"""Contract tests for NEURAL two-step PCMA. No scientific attempts/results."""
from dataclasses import replace
import math

import pytest
import torch

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODWorldConfig, OODRuleWorld, run_ood_sessions
from tam_research.si_pcma_v1_probe_randomization import (
    randomized_development_first_actions,
)
from tam_research.si_pcma_v1_factual_learning import (
    FactualRewardPredictor, causal_tokens,
)
from tam_research.si_pcma_v1_fair_comparison import PCMAGRURewardPredictor
from tam_research.si_pcma_v1_neural_probe import (
    randomized_pairs_to_causal_samples, TwoStepPCMAState,
    FrozenNeuralTwoStepPolicy, fit_development_two_step_predictors,
)


def _pairs():
    return randomized_development_first_actions(
        (OODWorldConfig(19, "development", "stable"),
         OODWorldConfig(20, "development", "long")),
        fixture_randomization_seed=22, sessions=2,
    )


def test_two_step_training_targets_are_only_factual_observed_return():
    pairs = _pairs()
    samples = randomized_pairs_to_causal_samples(pairs)
    assert len(samples) == len(pairs) == 22
    for sample, pair in zip(samples, pairs):
        assert sample.observation == pair.first_observation
        assert sample.action == pair.assigned_first_action
        assert sample.logical_time == pair.first_event.time
        assert sample.observed_net_reward == pytest.approx(
            pair.first_event.reward - pair.first_event.fee +
            pair.second_event.reward - pair.second_event.fee)
        assert all(row.time < sample.logical_time for row in sample.prior)
        # The current two-step target can change without touching the query.
        altered = replace(sample, observed_net_reward=12345.0)
        a = causal_tokens(sample.observation, sample.action, sample.prior,
                          logical_time=sample.logical_time)
        b = causal_tokens(altered.observation, altered.action, altered.prior,
                          logical_time=altered.logical_time)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert a[0][0, -1, 2].item() == 0.0


def test_public_prior_resets_at_every_session_and_world_boundary():
    samples = randomized_pairs_to_causal_samples(_pairs())
    # 6-step worlds yield 3 pairs per session; 16-step worlds 8.
    for i in (0, 3, 6, 14):
        assert samples[i].prior == ()
    assert len(samples[1].prior) == 2
    assert len(samples[2].prior) == 4
    assert all(p.action in range(4) for s in samples for p in s.prior)


def test_training_data_rejects_duplicate_group_and_changed_continuation():
    pairs = list(_pairs())
    with pytest.raises(ValueError, match="duplicate"):
        randomized_pairs_to_causal_samples((pairs[0], pairs[0]))
    with pytest.raises(ValueError, match="continuation"):
        randomized_pairs_to_causal_samples((
            replace(pairs[0], continuation_actions=(1, 2, 0)),))


def test_training_data_rejects_reversed_order_and_forged_second_feedback():
    pairs = list(_pairs())
    with pytest.raises(ValueError):
        randomized_pairs_to_causal_samples((pairs[1], pairs[0]))
    with pytest.raises(ValueError):
        randomized_pairs_to_causal_samples((
            replace(pairs[0], second_event=replace(
                pairs[0].second_event, time=pairs[0].second_event.time + 1)),))


def test_pair_memory_only_updates_after_two_verified_executions():
    world = OODRuleWorld(OODWorldConfig(31, "development"))
    state = TwoStepPCMAState()
    f = world.step(3)
    assert len(state.memory) == 0 and state.total_updates == 0
    a = world.step(0)
    state.record(f, a)
    assert len(state.memory) == 1 and state.total_updates == 1
    assert state.memory[0].first_action == 3
    assert state.memory[0].observed_net == pytest.approx(-.1 + a.reward)
    with pytest.raises(ValueError):
        state.record(f, a)
    assert len(state.memory) == 1
    assert state.value(f.before, 3) == pytest.approx(
        (state.memory[0].observed_net + state._predict_adapter(f.before, 3)) / 2
    )


def test_adapter_probe_action_is_not_skipped_and_update_budget_is_bounded():
    world = OODRuleWorld(OODWorldConfig(12, "development", "long"))
    state = TwoStepPCMAState(update_limit=1)
    first, second = world.step(3), world.step(1)
    state.record(first, second)
    assert state.total_updates == 1
    for i in range(3):
        f, s = world.step(3), world.step(0)
        state.record(f, s)
    assert len(state.memory) == 4
    assert state.total_updates == 1 and state.session_updates == 1
    state.restart_session()
    assert state.session_updates == 0
    assert len(state.memory) == 4
    state.restart_session(erase_persistent=True)
    assert state.total_updates == 0 and not state.memory


def test_independent_memory_and_adapter_ablation():
    world = OODRuleWorld(OODWorldConfig(44, "development"))
    first, second = world.step(3), world.step(1)
    memory_only = TwoStepPCMAState(adaptation_enabled=False)
    memory_only.record(first, second)
    assert len(memory_only.memory) == 1 and memory_only.total_updates == 0
    adapter_only = TwoStepPCMAState(memory_enabled=False)
    adapter_only.record(first, second)
    assert not adapter_only.memory and adapter_only.total_updates == 1
    neither = TwoStepPCMAState(memory_enabled=False, adaptation_enabled=False)
    neither.record(first, second)
    assert neither.value(first.before, 3) == 0.0


def test_both_neural_models_and_gru_receive_identical_two_step_causal_tensors():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    samples = randomized_pairs_to_causal_samples(_pairs())
    sample = samples[1]
    obs, act, mode = causal_tokens(sample.observation, sample.action, sample.prior,
                                   logical_time=sample.logical_time)
    assert obs.ndim == 3 and act.shape[-1] == 4
    for model in (FactualRewardPredictor("hybrid"),
                  FactualRewardPredictor("transformer"),
                  PCMAGRURewardPredictor()):
        pred = model(obs, act, mode)
        assert pred.shape == (1,) and torch.isfinite(pred).all()


def test_neural_two_step_policy_factual_timing_without_any_scientific_scores():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    for kind in ("hybrid", "transformer", "gru"):
        model = (PCMAGRURewardPredictor() if kind == "gru"
                 else FactualRewardPredictor(kind))
        actor = FrozenNeuralTwoStepPolicy(model)
        world = OODRuleWorld(OODWorldConfig(42, "development"))
        first_action = actor.choose(world.observe())
        with pytest.raises(RuntimeError):
            actor.choose(world.observe())
        e1 = world.step(first_action)
        actor.observe(e1)
        assert len(actor.state.memory) == 0
        second_action = actor.choose(world.observe())
        assert second_action == actor.continuation.choose_guess(world.observe())
        e2 = world.step(second_action)
        actor.observe(e2)
        assert len(actor.state.memory) == 1
        assert actor.state.total_updates == 1
        assert all(not p.requires_grad for p in actor.predictor.parameters())


def test_no_probe_ablated_neural_policy_uses_only_guesses():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    actor = FrozenNeuralTwoStepPolicy(FactualRewardPredictor("transformer"),
                                       allow_probe=False)
    records = run_ood_sessions(actor, OODWorldConfig(55, "development"),
                                sessions=2)
    assert len(records) == 2
    assert all(e.action in range(3) for part in records for e in part)
    assert len(actor.state.memory) == 6
    actor.restart_session()
    assert len(actor.state.memory) == 6
    actor.restart_session(erase_persistent=True)
    assert len(actor.state.memory) == 0


def test_training_entry_enforces_development_boundary_before_models():
    with pytest.raises(ValueError, match="heldout"):
        fit_development_two_step_predictors(
            (OODWorldConfig(9, "development"),
             OODWorldConfig(9, "heldout")),
            fixture_randomization_seed=1,
            fixture_initialization_seed=2,
            updates=1)
    with pytest.raises(ValueError):
        fit_development_two_step_predictors(
            (OODWorldConfig(1, "development"),),
            fixture_randomization_seed=1,
            fixture_initialization_seed=2,
            updates=33)


def test_micro_fit_same_factual_examples_and_budget_all_three_backbones():
    torch.set_num_threads(2)
    models, samples = fit_development_two_step_predictors(
        (OODWorldConfig(7, "development"),),
        fixture_randomization_seed=10,
        fixture_initialization_seed=11,
        updates=1, sessions=1,
    )
    assert set(models) == {"hybrid", "transformer", "gru"}
    assert len(samples) == 3
    assert all(not model.training for model in models.values())
    for kind, model in models.items():
        actor = FrozenNeuralTwoStepPolicy(model, pcma_weight=0.5)
        assert actor.choose(PublicObservation(2, 6)) in range(4)


def test_factual_pair_replay_rejected_independently_of_memory_eviction():
    world = OODRuleWorld(OODWorldConfig(55, "development"))
    state = TwoStepPCMAState(capacity=1)
    f1, s1 = world.step(0), world.step(1)
    f2, s2 = world.step(2), world.step(0)
    state.record(f1, s1)
    state.record(f2, s2)
    assert len(state.memory) == 1 and state.memory[0].first_time == f2.time
    with pytest.raises(ValueError):
        state.record(f1, s1)
