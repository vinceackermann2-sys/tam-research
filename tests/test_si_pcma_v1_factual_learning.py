"""PCMA-v1 factual learning tests: correctness, not research evidence."""
from dataclasses import replace
import pytest
import torch

from tam_research.si_pcma_v1 import (
    WorldConfig, PublicObservation, FactualTransition,
)
from tam_research.si_pcma_v1_factual_learning import (
    causal_samples, causal_tokens, FactualRewardPredictor,
    fit_factual_predictor, FrozenNeuralFactualPolicy, ExactMatchValuePolicy,
    OnlineTabularTDPolicy, collect_factual_rollouts, run_policy_sessions,
)


def _records():
    return collect_factual_rollouts(
        [WorldConfig(seed=17, max_steps=4, shift_after=3),
         WorldConfig(seed=22, max_steps=4, family="shift", shift_after=3)],
        exploration_seed=99, sessions=2
    )


def test_factual_rollouts_are_reproducible_and_single_action_only():
    rows = _records()
    assert rows == _records()
    assert len(rows) == 16
    assert len({e.provenance for e in rows}) == len(rows)
    assert all(e.action in range(4) and e.time > 0 for e in rows)


def test_training_sample_has_only_strictly_prior_in_session_events():
    rows = _records()
    samples = causal_samples(rows)
    assert len(samples) == len(rows)
    assert samples[0].prior == ()
    assert samples[4].prior == ()
    assert samples[8].prior == ()
    assert samples[-1].logical_time == rows[-1].time
    assert all(all(x.time < sample.logical_time for x in sample.prior)
               for sample in samples)
    assert samples[1].prior == (rows[0],)


def test_current_reward_and_latent_rule_cannot_enter_query_tokens():
    first = causal_samples(_records())[1]
    changed = replace(first, observed_net_reward=1e9)
    args = (first.observation, first.action, first.prior)
    original = causal_tokens(*args, logical_time=first.logical_time)
    alternative = causal_tokens(changed.observation, changed.action, changed.prior,
                                logical_time=changed.logical_time)
    assert all(torch.equal(a, b) for a, b in zip(original, alternative))
    obs, actions, modality = original
    assert obs.shape == (1, 2, 3)
    assert obs[0, -1, 2].item() == 0
    assert modality[0, -1].item() == 1
    assert actions[0, -1, first.action].item() == 1


def test_rejects_future_label_as_context():
    row = _records()[0]
    with pytest.raises(ValueError):
        causal_tokens(row.before, row.action, [row], logical_time=row.time)


def test_causal_context_limit_and_visible_action_only():
    rows = _records()[:4]
    obs, a, mode = causal_tokens(PublicObservation(1, 4), 2, rows,
                                 logical_time=100, max_context=2)
    assert obs.shape == (1, 2, 3)
    assert mode.tolist() == [[0, 1]]
    assert a.shape == (1, 2, 4)
    with pytest.raises(ValueError):
        causal_tokens(PublicObservation(1, 4), 2, rows, max_context=65)


def test_online_classical_models_observe_only_factually_executed_action():
    rows = _records()
    e = rows[0]
    lookup = ExactMatchValuePolicy()
    td = OnlineTabularTDPolicy()
    assert lookup.choose(e.before) in range(4)
    assert td.choose(e.before) in range(4)
    lookup.observe(e)
    td.observe(e)
    assert lookup.memory.retrieve(e.before, e.action) == (e,)
    assert set(td.q) == {(e.before.symbol, e.action)}
    lookup.restart_session()
    td.restart_session()
    assert lookup.memory.items == (e,) and td.q
    lookup.restart_session(erase_persistent=True)
    td.restart_session(erase_persistent=True)
    assert lookup.memory.items == () and td.q == {}


def test_session_runner_has_factual_time_and_no_policy_oracle():
    config = WorldConfig(seed=19, family="shift", max_steps=4, shift_after=3)
    result = run_policy_sessions(OnlineTabularTDPolicy(), config, sessions=3)
    assert [e.time for block in result for e in block] == list(range(1, 13))
    assert len(result) == 3
    assert all(block[-1].done for block in result)
    assert all(len(block) == 4 for block in result)


def test_training_inputs_same_for_matched_neural_models_and_grads_finite():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    rows = causal_samples(_records()[:3])
    for arch in ("hybrid", "transformer"):
        learner = FactualRewardPredictor(arch)
        obs, act, mode = causal_tokens(rows[0].observation, rows[0].action, ())
        predicted = learner(obs, act, mode)
        assert predicted.shape == (1,)
        assert torch.isfinite(predicted).all()
        losses = fit_factual_predictor(learner, rows, updates=2)
        assert len(losses) == 2 and all(x >= 0 for x in losses)
        actor = FrozenNeuralFactualPolicy(learner)
        observed = actor.choose(rows[0].observation)
        assert observed in range(4)
        assert all(not p.requires_grad for p in actor.predictor.parameters())


def test_neural_state_lifecycle_separate_from_frozen_weights():
    torch.manual_seed(0)
    actor = FrozenNeuralFactualPolicy(
        FactualRewardPredictor("transformer"), pcma_weight=0.5
    )
    event = _records()[0]
    before = tuple(p.detach().clone() for p in actor.predictor.parameters())
    actor.observe(event)
    assert actor.state.memory.items == (event,)
    assert actor.history == [event]
    assert actor.state.adapter.total_updates == int(event.action != 3)
    actor.restart_session()
    assert actor.history == [] and actor.state.memory.items == (event,)
    actor.restart_session(erase_persistent=True)
    assert actor.state.memory.items == ()
    assert all(torch.equal(x, p) for x, p in zip(before, actor.predictor.parameters()))


def test_full_policy_ablation_disables_online_learning():
    actor = FrozenNeuralFactualPolicy(
        FactualRewardPredictor("transformer"),
        memory_enabled=False, adaptation_enabled=False, pcma_weight=0.0,
    )
    actor.observe(_records()[0])
    assert actor.state.memory.items == ()
    assert actor.state.adapter.total_updates == 0


def test_reject_invalid_factual_fit_and_control_settings():
    records = causal_samples(_records())
    with pytest.raises(ValueError):
        fit_factual_predictor(FactualRewardPredictor("transformer"), records, updates=0)
    with pytest.raises(ValueError):
        fit_factual_predictor(FactualRewardPredictor("transformer"), ())
    with pytest.raises(ValueError):
        FrozenNeuralFactualPolicy(FactualRewardPredictor("hybrid"), pcma_weight=1.5)
    with pytest.raises(ValueError):
        OnlineTabularTDPolicy(alpha=0.0)
    with pytest.raises(ValueError):
        run_policy_sessions(ExactMatchValuePolicy(), WorldConfig(1), sessions=0)
