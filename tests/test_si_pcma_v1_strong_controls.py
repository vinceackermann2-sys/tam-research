"""Development-only contract tests for cheap factual two-action controls."""
from dataclasses import replace

import pytest
import torch

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_factual_learning import CausalRewardSample
from tam_research.si_pcma_v1_fair_comparison import MeasuredFactualPolicy, account_run
from tam_research.si_pcma_v1_ood_world import OODWorldConfig, OODRuleWorld
from tam_research.si_pcma_v1_probe_randomization import (
    FrozenPublicContinuation, randomized_development_first_actions,
)
from tam_research.si_pcma_v1_neural_probe import (
    randomized_pairs_to_causal_samples, fit_development_two_step_predictors,
)
from tam_research.si_pcma_v1_strong_controls import (
    ClassicalFactualTwoStepPredictor, FrozenClassicalTwoStepPolicy,
    fit_classical_two_step, fit_development_classical_two_step,
    public_two_step_features,
)
from tam_research.si_pcma_v1_world_ipc import run_ipc_development_sessions


def _samples():
    pair = randomized_development_first_actions(
        (OODWorldConfig(7, "development", "stable"),
         OODWorldConfig(8, "development", "long")),
        fixture_randomization_seed=17, sessions=2)
    return randomized_pairs_to_causal_samples(pair)


def test_all_cheap_backbones_share_one_factual_development_data_source():
    samples = _samples()
    assert len(samples) == 22
    for kind in ("constant", "symbol", "ridge"):
        fitted = fit_classical_two_step(kind, samples)
        assert fitted.training_rows == 22
        assert fitted.kind == kind
        assert all(
            fitted.predict(example.observation, example.action, example.prior)
            == pytest.approx(fitted.predict(
                example.observation, example.action, example.prior))
            for example in samples)
    assert fit_classical_two_step("ridge", samples).coefficients == (
        fit_classical_two_step("ridge", samples).coefficients)
    assert len(fit_classical_two_step("ridge", samples).coefficients) == len(
        public_two_step_features(PublicObservation(1, 6), 3, ()))


def test_ridge_training_and_neural_training_use_identical_causal_pair_records():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    configs = (OODWorldConfig(5, "development"),)
    cheap, cheap_samples = fit_development_classical_two_step(
        configs, fixture_randomization_seed=21, sessions=1)
    neural, neural_samples = fit_development_two_step_predictors(
        configs, fixture_randomization_seed=21,
        fixture_initialization_seed=22, sessions=1, updates=1)
    assert cheap_samples == neural_samples
    assert set(cheap) == {"constant", "symbol", "ridge"}
    assert set(neural) == {"hybrid", "transformer", "gru"}


def test_real_two_step_target_is_a_label_never_a_public_input():
    samples = _samples()
    before = samples[2]
    changed = replace(before, observed_net_reward=999)
    for kind in ("constant", "symbol", "ridge"):
        p = fit_classical_two_step(kind, samples)
        assert p.predict(before.observation, before.action, before.prior) == (
            p.predict(changed.observation, changed.action, changed.prior))
    a = public_two_step_features(before.observation, before.action,
                                 before.prior, logical_time=before.logical_time)
    b = public_two_step_features(changed.observation, changed.action,
                                 changed.prior, logical_time=changed.logical_time)
    assert a == b


def test_last_seven_factual_transitions_only_and_future_feedback_rejected():
    before = PublicObservation(1, 6)
    history = [
        FactualTransition(PublicObservation(i % 3, 6), i % 3,
                          PublicObservation((i+1)%3, 5), float(i % 2), 0.0,
                          i+1, f"fixture-{i+1}", False)
        for i in range(10)
    ]
    # Training features see exactly the same latest-7 limit as causal tokens.
    a = public_two_step_features(before, 2, history, logical_time=11)
    b = public_two_step_features(before, 2, history[-7:], logical_time=11)
    assert a == b
    with pytest.raises(ValueError, match="future"):
        public_two_step_features(before, 2, history, logical_time=10)
    with pytest.raises(ValueError):
        public_two_step_features(before, 4, ())


def test_simple_constant_and_symbol_lookup_learn_only_observed_arms():
    obs0, obs1 = PublicObservation(0, 6), PublicObservation(1, 6)
    inputs = (
        CausalRewardSample(obs0, 0, (), 2.0, 1),
        CausalRewardSample(obs0, 0, (), 2.0, 2),
        CausalRewardSample(obs1, 0, (), 0.0, 3),
    )
    const = fit_classical_two_step("constant", inputs)
    symbol = fit_classical_two_step("symbol", inputs)
    assert const.predict(obs0, 0, ()) == const.predict(obs1, 0, ())
    assert symbol.predict(obs0, 0, ()) > symbol.predict(obs1, 0, ())
    # Unobserved arm stays at its explicit smoothing prior, never uses an
    # oracle or fabricated reward.
    assert const.predict(obs0, 3, ()) == pytest.approx(1 / 3 - 0.1)
    assert symbol.predict(obs0, 3, ()) == pytest.approx(1 / 3 - 0.1)


def test_ridge_honors_public_symbol_recent_chosen_action_and_horizon():
    a = public_two_step_features(PublicObservation(0, 6), 2, ())
    b = public_two_step_features(PublicObservation(1, 6), 2, ())
    c = public_two_step_features(PublicObservation(1, 16), 2, ())
    d = public_two_step_features(PublicObservation(1, 6), 3, ())
    assert a != b and b != c and b != d
    assert all(isinstance(x, float) for x in a)
    assert not any(float("inf") == x for x in a)


def test_all_classical_controls_work_across_real_subprocess_public_feedback():
    controls, samples = fit_development_classical_two_step(
        (OODWorldConfig(1, "development"),),
        fixture_randomization_seed=16, sessions=1)
    for kind, predictor in controls.items():
        actor = MeasuredFactualPolicy(
            FrozenClassicalTwoStepPolicy(predictor))
        traces = run_ipc_development_sessions(
            actor, OODWorldConfig(2, "development", "shift"),
            sessions=2)
        account = account_run(kind, actor, traces)
        assert account.actions == 12 and account.world_count == 1
        assert account.memory_entries == 6 and account.adapter_updates == 6
        assert account.param_count is None
        assert account.net_reward == pytest.approx(
            account.gross_reward - account.total_fee)
        assert account.probes == sum(
            e.action == 3 for group in traces for e in group)


def test_same_online_pcma_ablation_and_fixed_continuation_as_neural():
    control = fit_classical_two_step("symbol", _samples())
    actor = FrozenClassicalTwoStepPolicy(control,
        memory_enabled=False, adaptation_enabled=False, allow_probe=False)
    traces = run_ipc_development_sessions(
        actor, OODWorldConfig(3, "development"), sessions=2)
    assert not actor.state.memory
    assert actor.state.total_updates == 0
    assert all(e.action != 3 for group in traces for e in group)
    assert all(e.action == actor.continuation.choose_guess(e.before)
               for group in traces for e in group[1::2])
    actor.restart_session(erase_persistent=True)
    assert not actor.state.memory


def test_first_action_and_feedback_have_strict_factual_phase():
    model = fit_classical_two_step("constant", _samples())
    actor = FrozenClassicalTwoStepPolicy(model)
    world = OODRuleWorld(OODWorldConfig(4, "development"))
    first = actor.choose(world.observe())
    with pytest.raises(RuntimeError):
        actor.choose(world.observe())
    event = world.step(first)
    with pytest.raises(ValueError):
        actor.observe(replace(event, action=(first+1)%4))
    assert actor._pending is not None and actor.state.total_updates == 0
    actor.observe(event)
    assert actor.state.total_updates == 0
    second = actor.choose(world.observe())
    assert second == actor.continuation.choose_guess(world.observe())
    actor.observe(world.step(second))
    assert actor.state.total_updates == 1 and len(actor.state.memory) == 1


def test_heldout_classical_training_rejected_before_randomization_world():
    from unittest.mock import patch
    with patch("tam_research.si_pcma_v1_strong_controls.randomized_development_first_actions") as logger:
        with pytest.raises(ValueError, match="heldout"):
            fit_development_classical_two_step(
                (OODWorldConfig(11, "development"),
                 OODWorldConfig(12, "heldout")),
                fixture_randomization_seed=4)
        logger.assert_not_called()


def test_reject_bad_training_data_and_invalid_budgets():
    sample = _samples()[0]
    with pytest.raises(ValueError):
        fit_classical_two_step("unknown", [sample])
    with pytest.raises(ValueError):
        fit_classical_two_step("ridge", [replace(
            sample, observed_net_reward=float("nan"))])
    with pytest.raises(ValueError):
        fit_classical_two_step("constant", [sample], smoothing=0)
    with pytest.raises(ValueError):
        fit_classical_two_step("ridge", [sample], ridge_penalty=-1)
    with pytest.raises(ValueError):
        fit_classical_two_step("ridge", [])
    with pytest.raises(ValueError):
        FrozenClassicalTwoStepPolicy(fit_classical_two_step("constant", [sample]),
                                     pcma_weight=-0.1)


def test_strong_classical_controls_must_not_import_privileged_ood_rule_table():
    import tam_research.si_pcma_v1_strong_controls as controls
    assert not hasattr(controls, "OODRuleWorld")
    assert not hasattr(controls, "DEVELOPMENT_RULES")
    assert not hasattr(controls, "HELDOUT_RULES")
    assert not hasattr(controls, "oracle_value")
