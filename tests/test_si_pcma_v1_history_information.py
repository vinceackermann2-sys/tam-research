"""Factual cheap-history predictive-information gates; development ONLY."""
from dataclasses import replace
import math
from unittest.mock import patch

import pytest

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_factual_learning import CausalRewardSample
from tam_research.si_pcma_v1_history_information import (
    FrozenLastFeedbackTwoStepPolicy,
    development_history_information_audit,
    fit_last_feedback_backoff,
)
from tam_research.si_pcma_v1_ood_world import OODWorldConfig, OODRuleWorld
from tam_research.si_pcma_v1_fair_comparison import (
    MeasuredFactualPolicy, account_run,
)
from tam_research.si_pcma_v1_world_ipc import run_ipc_development_sessions
from tam_research.si_pcma_v1_strong_controls import fit_classical_two_step


def _feedback(time, action, reward, symbol=0):
    before = PublicObservation(symbol, 6)
    after = PublicObservation((symbol+1)%3, 5)
    return FactualTransition(
        before, action, after, float(reward),
        0.1 if action == 3 else 0.0, time,
        f"developer-{time}", False)


def test_same_symbol_and_action_with_distinct_factual_history_can_predict_differently():
    obs = PublicObservation(0, 4)
    cold = _feedback(1, 0, 0)
    warm = _feedback(1, 1, 1)
    samples = (
        CausalRewardSample(obs, 2, (cold,), 0.0, 2),
        CausalRewardSample(obs, 2, (warm,), 2.0, 2),
        CausalRewardSample(obs, 2, (cold,), 0.0, 3),
        CausalRewardSample(obs, 2, (warm,), 2.0, 3),
    )
    symbol = fit_classical_two_step("symbol", samples)
    recent = fit_last_feedback_backoff(samples, pseudocount=0.1)
    assert symbol.predict(obs, 2, (cold,)) == symbol.predict(obs, 2, (warm,))
    assert recent.predict(obs, 2, (cold,)) < recent.predict(obs, 2, (warm,))
    assert recent.predict(obs, 2, (cold,)) < symbol.predict(obs, 2, ())
    assert recent.predict(obs, 2, (warm,)) > symbol.predict(obs, 2, ())


def test_query_cannot_recover_its_own_two_step_target_through_public_history():
    obs = PublicObservation(1, 6)
    before = _feedback(3, 0, 1)
    sample = CausalRewardSample(obs, 3, (before,), -.1, 4)
    fit = fit_last_feedback_backoff((sample,))
    assert fit.training_rows == 1
    # At prediction time there is *no target argument*.
    original = fit.predict(obs, 3, sample.prior)
    changed = replace(sample, observed_net_reward=2.0)
    assert original == fit.predict(changed.observation, changed.action, changed.prior)
    assert fit.predict(obs, 3, ()) != original


def test_regression_refuses_future_feedback_and_invalid_targets():
    obs = PublicObservation(0, 6)
    future = _feedback(5, 2, 1)
    with pytest.raises(ValueError, match="future"):
        fit_last_feedback_backoff((CausalRewardSample(
            obs, 1, (future,), 0.0, 5),))
    with pytest.raises(ValueError):
        fit_last_feedback_backoff((CausalRewardSample(
            obs, 1, (), float("nan"), 5),))
    with pytest.raises(ValueError):
        fit_last_feedback_backoff((CausalRewardSample(
            obs, 1, (), 0.0, 5),), pseudocount=0)


def test_development_world_partitions_are_strictly_disjoint_before_any_collection():
    train = (OODWorldConfig(11, "development"),)
    eval_worlds = (OODWorldConfig(12, "heldout"),)
    with patch("tam_research.si_pcma_v1_history_information.randomized_history_development_pairs") as collect:
        with pytest.raises(ValueError, match="development"):
            development_history_information_audit(train, eval_worlds)
        with pytest.raises(ValueError, match="overlapping"):
            development_history_information_audit(train, train)
        collect.assert_not_called()


def test_no_duplicate_development_seed_even_when_scenario_differs():
    a = (OODWorldConfig(5, "development", "stable"),)
    b = (OODWorldConfig(5, "development", "long"),)
    with pytest.raises(ValueError, match="overlapping"):
        development_history_information_audit(a, b)
    with pytest.raises(ValueError, match="duplicated"):
        development_history_information_audit((a[0],a[0]), b)


def test_no_world_level_pseudoreplication_or_breakthrough_gates():
    report = development_history_information_audit(
        (OODWorldConfig(71, "development", "stable"),
         OODWorldConfig(72, "development", "long")),
        (OODWorldConfig(81, "development", "shift"),
         OODWorldConfig(82, "development", "noisy")),
        training_randomization_fixture=101,
        validation_randomization_fixture=102, sessions=2,
    )
    assert report.status == "DESCRIPTIVE_DEVELOPMENT_ONLY"
    assert report.train_worlds == report.validation_worlds == 2
    assert report.train_factual_pairs == report.validation_factual_pairs == 22
    assert len(report.per_world) == 2
    assert report.per_world[0].factual_pairs == 6
    assert report.per_world[1].factual_pairs == 6
    assert report.per_world[0].scenario == "shift"
    assert report.per_world[1].scenario == "noisy"
    assert not report.passed_scientific_gate
    assert not report.may_evaluate_heldout
    assert not report.may_claim_architecture_gain
    for row in report.per_world:
        assert all(math.isfinite(v) and v >= 0 for v in (
            row.constant_mse, row.symbol_mse,
            row.public_ridge_mse, row.last_feedback_mse,
        ))
    assert report.mean_last_feedback_mse-report.mean_symbol_mse == pytest.approx(
        report.last_feedback_minus_symbol_mse)


def test_same_fixture_generates_identical_diagnostic_not_optimization_metric():
    train = (OODWorldConfig(32, "development"),)
    evaluate = (OODWorldConfig(33, "development", "long"),)
    a = development_history_information_audit(
        train, evaluate, sessions=1)
    b = development_history_information_audit(
        train, evaluate, sessions=1)
    assert a == b
    assert a.train_factual_pairs == 3
    assert a.validation_factual_pairs == 8
    assert a.per_world[0].factual_pairs == 8


def test_shared_pcma_history_backoff_is_executable_through_public_worker():
    sample = CausalRewardSample(
        PublicObservation(0, 6), 0, (), 0.0, 1)
    forecast = fit_last_feedback_backoff((sample,))
    for memory_enabled, adaptation_enabled, allow_probe in (
        (True, True, True), (False, False, False),
    ):
        policy = FrozenLastFeedbackTwoStepPolicy(
            forecast, memory_enabled=memory_enabled,
            adaptation_enabled=adaptation_enabled, allow_probe=allow_probe)
        measured = MeasuredFactualPolicy(policy)
        traces = run_ipc_development_sessions(
            measured, OODWorldConfig(37, "development", "long"),
            world_kind="history", sessions=2)
        accounting = account_run("history cheap control", measured, traces)
        assert accounting.actions == 32
        assert accounting.probes == sum(
            event.action == 3 for session in traces for event in session)
        assert accounting.total_fee == pytest.approx(.1*accounting.probes)
        assert accounting.memory_entries == (16 if memory_enabled else 0)
        assert accounting.adapter_updates == (16 if adaptation_enabled else 0)
        assert not allow_probe or not any(
            event.action == 3 for session in traces for event in session)


def test_history_backoff_does_not_use_private_world_truth_or_unexecuted_actions():
    import tam_research.si_pcma_v1_history_information as module
    assert not hasattr(module, "HistoryNecessaryDevelopmentWorld")
    assert not hasattr(module, "HELDOUT_RULES")
    assert not hasattr(module, "oracle")
    assert not hasattr(module, "counterfactual_reward")


def test_history_backoff_rejects_nonfactual_provenance_and_negative_smoothing():
    with pytest.raises(ValueError):
        fit_last_feedback_backoff(())
    with pytest.raises(ValueError):
        fit_last_feedback_backoff((
            CausalRewardSample(PublicObservation(0, 6), 1, (),
                               10.0, 1),))
    with pytest.raises(ValueError):
        fit_last_feedback_backoff((
            CausalRewardSample(PublicObservation(0, 6), 1, (),
                               0.0, 1),), pseudocount=-1.0)
