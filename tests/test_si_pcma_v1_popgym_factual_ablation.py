"""Contract tests for honest online PUBLIC factual PCMA port — no science."""
import math

import pytest
import torch

from tam_research.si_pcma_v1_popgym_factual_ablation import (
    ABLATIONS, FEATURE_WIDTH, MAX_ADAPTER_UPDATES, MAX_FACTUAL_MEMORY,
    PublicFactualRank2State, _run_one,
)
from tam_research.si_pcma_v1_popgym_neural_pilot import (
    OriginalWorldHeadMemoryReadout, PUBLIC_EPISODE_STEPS,
    collect_public_sequences,
)
from tam_research.si_pcma_v1_published_memory import TASKS


def test_history_adapter_is_rank_two_bounded_public_feedback_only():
    state = PublicFactualRank2State(
        memory_enabled=True, adaptation_enabled=True)
    assert len(state.u) == len(state.v) == 2
    assert all(len(row) == FEATURE_WIDTH for row in state.u)
    assert state.memory.maxlen == MAX_FACTUAL_MEMORY
    assert state.updates == 0
    assert state.correction("RepeatFirstEasy", 1, 2, 0) == 0
    state.record("RepeatFirstEasy", 1, 2, 1 / 51, 0)
    assert state.updates == 1
    assert state.observations == 1
    assert len(state.memory) == 1
    assert state.correction("RepeatFirstEasy", 1, 2, 1) != 0
    assert not hasattr(state, "hidden_state") and not hasattr(state, "oracle")
    assert not hasattr(state, "unobserved_reward")


def test_online_factual_adapter_uses_only_past_executed_actions():
    state = PublicFactualRank2State(
        memory_enabled=True, adaptation_enabled=False)
    state.record("RepeatFirstEasy", 3, 2, 1 / 51, 0)
    assert state.correction("RepeatFirstEasy", 3, 2, 1) > 0
    assert state.correction("RepeatFirstEasy", 3, 1, 1) == 0
    assert state.correction("RepeatFirstEasy", 2, 2, 1) == 0
    assert state.correction("RepeatPreviousEasy", 3, 2, 1) == 0
    assert state.correction("RepeatFirstEasy", 3, 2, 0) == 0
    with pytest.raises(ValueError):
        state.record("RepeatFirstEasy", 3, 2, 1 / 51, 0)


def test_external_repeat_previous_k4_reward_scale_is_correct():
    from tam_research.si_pcma_v1_popgym_factual_ablation import (
        ACTUAL_REWARD_DENOMINATOR,)
    assert ACTUAL_REWARD_DENOMINATOR == {
        "RepeatFirstEasy": 51,
        "RepeatPreviousEasy": 48,
        "CountRecallEasy": 51,
    }
    state = PublicFactualRank2State(
        memory_enabled=True, adaptation_enabled=True)
    state.record("RepeatPreviousEasy", 3, 2, 1/48, 0)
    assert state.observations == 1
    assert state.correction("RepeatPreviousEasy", 3, 2, 1) > 0


def test_zero_state_and_independent_episode_resets_are_exact():
    no_pcma = PublicFactualRank2State(
        memory_enabled=False, adaptation_enabled=False)
    memory_only = PublicFactualRank2State(
        memory_enabled=True, adaptation_enabled=False)
    adapter_only = PublicFactualRank2State(
        memory_enabled=False, adaptation_enabled=True)
    for state in (no_pcma, memory_only, adapter_only):
        state.record("CountRecallEasy", (1, 0), 2, -1/51, 0)
    assert no_pcma.correction("CountRecallEasy", (1, 0), 2, 1) == 0
    assert len(no_pcma.memory) == 0 and no_pcma.updates == 0
    assert len(memory_only.memory) == 1 and memory_only.updates == 0
    assert len(adapter_only.memory) == 0 and adapter_only.updates == 1
    fresh = PublicFactualRank2State(
        memory_enabled=True, adaptation_enabled=True)
    assert fresh.observations == 0 and not fresh.memory and fresh.updates == 0


def test_factual_update_limit_and_memory_capacity_are_held():
    state = PublicFactualRank2State(
        memory_enabled=True, adaptation_enabled=True)
    for t in range(PUBLIC_EPISODE_STEPS):
        state.record("RepeatFirstEasy", t % 4, t % 4, 1/51, t)
    assert state.observations == PUBLIC_EPISODE_STEPS
    assert len(state.memory) == PUBLIC_EPISODE_STEPS
    assert state.updates == MAX_ADAPTER_UPDATES
    assert all(math.isfinite(x) and abs(x) <= 1
               for row in state.u for x in row)
    assert all(math.isfinite(x) and abs(x) <= 1 for x in state.v)


def test_invalid_public_features_and_unexecuted_feedback_fail_closed():
    state = PublicFactualRank2State(
        memory_enabled=True, adaptation_enabled=True)
    with pytest.raises(ValueError):
        state.correction("unknown", 0, 0, 0)
    with pytest.raises(ValueError):
        state.correction("RepeatFirstEasy", 0, 27, 0)
    with pytest.raises(ValueError):
        state.correction("CountRecallEasy", (0, 2), 3, 0)
    with pytest.raises(ValueError):
        state.correction("RepeatFirstEasy", 0, 1, PUBLIC_EPISODE_STEPS)
    with pytest.raises(ValueError):
        state.record("RepeatFirstEasy", 0, 1, float("nan"), 0)
    with pytest.raises(ValueError):
        state.record("RepeatFirstEasy", 0, 1, 100, 0)
    assert state.observations == 0 and not state.memory


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_all_four_state_ablations_execute_real_public_rewards(ablation):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = OriginalWorldHeadMemoryReadout("transformer").eval()
    seq = collect_public_sequences("RepeatFirstEasy", (201,))[0]
    memory = ablation in ("memory_only", "both")
    adapter = ablation in ("adapter_only", "both")
    score, accuracy, actions, memory_records, updates = _run_one(
        model, seq, memory_enabled=memory, adaptation_enabled=adapter)
    assert -1.01 <= score <= 1.01
    assert 0 <= accuracy <= 1
    assert actions == PUBLIC_EPISODE_STEPS
    assert memory_records == (PUBLIC_EPISODE_STEPS if memory else 0)
    assert updates == (MAX_ADAPTER_UPDATES if adapter else 0)


def test_original_backbones_share_unmodified_interface_and_external_action_space():
    torch.set_num_threads(2)
    for kind in ("hybrid", "transformer"):
        torch.manual_seed(12)
        model = OriginalWorldHeadMemoryReadout(kind).eval()
        assert type(model.backbone).__name__ in (
            "OriginalSIHybrid", "TransformerControl")
        for task in TASKS:
            seq = collect_public_sequences(task, (201,))[0]
            score, accuracy, actions, _, _ = _run_one(
                model, seq, memory_enabled=False, adaptation_enabled=False)
            assert actions == PUBLIC_EPISODE_STEPS
            assert -1.01 <= score <= 1.01
            assert 0 <= accuracy <= 1


def test_backbone_never_receives_rewards_as_input():
    import tam_research.si_pcma_v1_popgym_factual_ablation as module
    assert not hasattr(module, "oracle_reward")
    assert not hasattr(module, "HELDOUT_RULES")
    assert not hasattr(module, "get_state")
    assert not hasattr(module, "counterfactual")
