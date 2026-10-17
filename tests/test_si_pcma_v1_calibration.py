"""Factual-feedback calibration gate checks; no promotion metrics."""
import pytest
import torch

from tam_research.si_pcma_v1 import WorldConfig, PersistentWorld
from tam_research.si_pcma_v1_factual_learning import (
    FactualRewardPredictor, run_policy_sessions,
)
from tam_research.si_pcma_v1_calibration import (
    FeedbackReliabilityGate, FeedbackCalibratedPCMAPolicy,
)


def test_gate_rewards_previously_better_factual_predictor():
    g = FeedbackReliabilityGate(decay=1.0)
    assert g.weights() == (0.5, 0.5)
    g.observe(1.0, 0.0, 1.0)
    neural, memory = g.weights()
    assert neural > 0.95 and memory < 0.05 and g.count == 1
    assert sum(g.weights()) == pytest.approx(1.0)


def test_gate_has_no_update_without_returned_factual_feedback():
    torch.manual_seed(0)
    policy = FeedbackCalibratedPCMAPolicy(FactualRewardPredictor("transformer"))
    world = PersistentWorld(WorldConfig(5, max_steps=4))
    chosen = policy.choose(world.public_observation())
    assert chosen in range(4)
    assert policy.gate.count == 0
    with pytest.raises(RuntimeError):
        policy.choose(world.public_observation())
    event = world.step(chosen)
    policy.observe(event)
    assert policy.gate.count == 1


def test_gate_denies_mismatched_feedback_and_does_not_store_it():
    torch.manual_seed(0)
    policy = FeedbackCalibratedPCMAPolicy(FactualRewardPredictor("transformer"))
    world = PersistentWorld(WorldConfig(6, max_steps=4))
    chosen = policy.choose(world.public_observation())
    wrong = world.step((chosen + 1) % 4)
    with pytest.raises(ValueError):
        policy.observe(wrong)
    assert policy.gate.count == 0
    assert not policy.state.memory.items


def test_gate_persists_or_resets_independently_of_session_window():
    torch.manual_seed(0)
    policy = FeedbackCalibratedPCMAPolicy(FactualRewardPredictor("transformer"))
    out = run_policy_sessions(policy, WorldConfig(9, max_steps=4), sessions=2)
    assert sum(map(len, out)) == 8
    assert policy.gate.count == 8
    policy.restart_session()
    assert policy.gate.count == 8 and policy.history == []
    policy.restart_session(erase_persistent=True)
    assert policy.gate.count == 0 and policy.state.memory.items == ()


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):
        FeedbackReliabilityGate(decay=0)
    with pytest.raises(ValueError):
        FeedbackReliabilityGate(count=2)
    with pytest.raises(ValueError):
        FeedbackReliabilityGate().observe(float("nan"), 0.0, 1.0)
