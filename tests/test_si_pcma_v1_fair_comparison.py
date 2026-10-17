"""Engineering-only equal-information/measurement contract tests.

Never use CI fixtures as scientific seed reservations, statistical claims,
threshold tuning, or evidence of a PCMA advantage.
"""
from dataclasses import replace

import pytest
import torch

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODWorldConfig, OODRuleWorld, run_ood_sessions
from tam_research.si_pcma_v1_ood_controls import (
    FixedGuessControl, SlidingWindowFactualControl, SurpriseResetFactualControl,
)
from tam_research.si_pcma_v1_ood_development import collect_development_factual_data
from tam_research.si_pcma_v1_factual_learning import (
    FactualRewardPredictor, FrozenNeuralFactualPolicy, causal_tokens,
    fit_factual_predictor,
)
from tam_research.si_pcma_v1_fair_comparison import (
    PCMAGRURewardPredictor, MeasuredFactualPolicy, account_run,
    count_trainable_and_frozen_parameters, development_comparison_contract,
    build_frozen_gru_policy,
)


def test_gru_and_original_backbones_share_exact_public_feature_schema():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    obs, act, modality = causal_tokens(PublicObservation(2, 6), 1, ())
    for model in (PCMAGRURewardPredictor(),
                  FactualRewardPredictor("hybrid"),
                  FactualRewardPredictor("transformer")):
        outcome = model(obs, act, modality)
        assert outcome.shape == (1,)
        assert torch.isfinite(outcome).all()
    assert obs.shape == (1, 1, 3)
    assert act.shape == (1, 1, 4)
    assert modality.tolist() == [[1]]


def test_gru_development_factual_target_gradients_only():
    torch.manual_seed(0)
    dataset = collect_development_factual_data(
        [OODWorldConfig(19, "development")], exploration_fixture_seed=4,
        sessions=1,
    )
    gru = PCMAGRURewardPredictor()
    losses = fit_factual_predictor(gru, dataset.samples, updates=1)
    assert len(losses) == 1 and losses[0] >= 0
    frozen = FrozenNeuralFactualPolicy(gru)
    assert frozen.choose(PublicObservation(1, 6)) in range(4)
    assert all(not p.requires_grad for p in frozen.predictor.parameters())


def test_gru_unmatched_parameter_count_reported_not_treated_as_parity():
    gru = build_frozen_gru_policy()
    transformer = FrozenNeuralFactualPolicy(FactualRewardPredictor("transformer"))
    a = count_trainable_and_frozen_parameters(gru)
    b = count_trainable_and_frozen_parameters(transformer)
    assert isinstance(a, int) and isinstance(b, int)
    assert a > 0 and b > 0 and a != b
    assert gru.predictor.kind == "gru_unmatched"


def test_action_boundary_delivers_only_actual_chosen_feedback():
    actor = MeasuredFactualPolicy(FixedGuessControl(1))
    world = OODRuleWorld(OODWorldConfig(3, "development"))
    obs = world.observe()
    assert actor.choose(obs) == 1
    with pytest.raises(RuntimeError):
        actor.choose(obs)
    with pytest.raises(RuntimeError):
        actor.restart_session()
    event = world.step(1)
    actor.observe(event)
    assert actor.calls == actor.feedback_events == 1
    assert actor.choose_ns >= 0 and actor.feedback_ns >= 0
    with pytest.raises(RuntimeError):
        actor.observe(event)


def test_bad_policy_action_and_forged_feedback_are_rejected():
    class BadPolicy(FixedGuessControl):
        def choose(self, observation):
            return True
    with pytest.raises(ValueError):
        MeasuredFactualPolicy(BadPolicy()).choose(PublicObservation(0, 6))
    good = MeasuredFactualPolicy(FixedGuessControl(0))
    world = OODRuleWorld(OODWorldConfig(4, "development"))
    good.choose(world.observe())
    factual = world.step(0)
    with pytest.raises(ValueError):
        good.observe(replace(factual, action=1))
    with pytest.raises(ValueError):
        good.observe(replace(factual, fee=1.0))
    assert good.feedback_events == 0
    good.observe(factual)
    with pytest.raises(RuntimeError):
        good.observe(factual)


def test_development_only_comparison_blocks_heldout_before_factory_is_called():
    invoked = []
    def factory():
        invoked.append(True)
        return FixedGuessControl(0)
    with pytest.raises(ValueError, match="heldout"):
        development_comparison_contract(
            {"fixed": factory}, (OODWorldConfig(12, "heldout"),)
        )
    assert invoked == []
    with pytest.raises(ValueError, match="duplicate"):
        development_comparison_contract(
            {"fixed": factory},
            (OODWorldConfig(3, "development"),
             OODWorldConfig(3, "development")),
        )
    assert invoked == []


def test_equal_budget_for_independent_classical_policy_runs():
    configurations = (
        OODWorldConfig(1, "development", "stable"),
        OODWorldConfig(2, "development", "shift"),
    )
    results = development_comparison_contract({
        "fixed": lambda: FixedGuessControl(0),
        "window": lambda: SlidingWindowFactualControl(
            exploration=0.0, probe_period=3),
        "surprise": lambda: SurpriseResetFactualControl(),
    }, configurations, sessions=2)
    assert set(results) == {"fixed", "window", "surprise"}
    assert all(x.world_count == 2 and x.actions == 24 for x in results.values())
    assert results["fixed"].probes == 0
    assert results["window"].probes > 0
    for record in results.values():
        assert record.net_reward == pytest.approx(
            record.gross_reward - record.total_fee)
        assert record.net_per_action == pytest.approx(
            record.net_reward / record.actions)
        assert record.decision_ns >= 0 and record.feedback_ns >= 0


def test_memory_adapter_metrics_use_only_explicit_online_updates():
    torch.manual_seed(0)
    actor = MeasuredFactualPolicy(build_frozen_gru_policy())
    trace = run_ood_sessions(
        actor, OODWorldConfig(2, "development"), sessions=2)
    summary = account_run("gru", actor, trace)
    assert summary.actions == 12
    assert summary.param_count is not None
    assert summary.memory_entries == 12
    assert summary.adapter_updates is not None
    assert 0 <= summary.adapter_updates <= 12


def test_erasing_between_sessions_stops_persistent_memory_accumulation():
    torch.manual_seed(0)
    actor = MeasuredFactualPolicy(build_frozen_gru_policy())
    trace = run_ood_sessions(
        actor, OODWorldConfig(3, "development"), sessions=2,
        erase_persistent=True,
    )
    summary = account_run("gru-reset", actor, trace)
    assert summary.actions == 12
    assert summary.memory_entries == 6


def test_trace_contract_rejects_missing_or_duplicate_events():
    actor = MeasuredFactualPolicy(FixedGuessControl(0))
    trace = run_ood_sessions(actor, OODWorldConfig(1, "development"),
                             sessions=1)
    with pytest.raises(ValueError, match="nonterminal"):
        account_run("bad", actor, (trace[0][:-1],))
    with pytest.raises(ValueError, match="replayed"):
        account_run("bad", actor, (trace[0], trace[0]))


def test_frozen_models_are_not_used_for_unapproved_heldout_tuning():
    with pytest.raises(ValueError, match="heldout"):
        development_comparison_contract(
            {"untrained": lambda: build_frozen_gru_policy()},
            (OODWorldConfig(1, "development"),
             OODWorldConfig(2, "heldout"))
        )
