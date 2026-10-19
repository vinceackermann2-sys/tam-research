"""Development-only POPGym hybrid/Transformer pilot contracts; no science."""
from dataclasses import replace
import math

import pytest
import torch

from tam_research.si_original_hybrid_v0 import (
    SIConfig, OriginalSIHybrid, TransformerControl,
)
from tam_research.si_pcma_v1_popgym_neural_pilot import (
    ACTION_DIM, INITIALIZATION_FIXTURE, PUBLIC_EPISODE_STEPS,
    TRAINING_UPDATES, TRAIN_DEVELOPMENT_EPISODES,
    VALIDATION_DEVELOPMENT_EPISODES,
    OriginalWorldHeadMemoryReadout,
    _mask_class_logits, _same_public_trajectory_real_reward,
    _sequence_tensors, collect_public_sequences, run_bounded_development_training,
)
from tam_research.si_pcma_v1_published_memory import TASKS


def _rows(episode=101):
    return tuple(collect_public_sequences(task, (episode,))[0] for task in TASKS)


def test_exact_original_backbones_are_used_without_source_rewrites():
    torch.manual_seed(INITIALIZATION_FIXTURE)
    hybrid = OriginalWorldHeadMemoryReadout("hybrid")
    torch.manual_seed(INITIALIZATION_FIXTURE)
    transformer = OriginalWorldHeadMemoryReadout("transformer")
    assert type(hybrid.backbone) is OriginalSIHybrid
    assert type(transformer.backbone) is TransformerControl
    assert type(hybrid.backbone.config) is SIConfig
    assert type(transformer.backbone.config) is SIConfig
    assert hybrid.readout.in_features == transformer.readout.in_features == 3
    assert hybrid.readout.out_features == transformer.readout.out_features == ACTION_DIM


def test_training_and_validation_development_episodes_never_overlap():
    assert not set(TRAIN_DEVELOPMENT_EPISODES) & set(VALIDATION_DEVELOPMENT_EPISODES)
    assert len(TRAIN_DEVELOPMENT_EPISODES) == 12
    assert len(VALIDATION_DEVELOPMENT_EPISODES) == 8
    assert TRAINING_UPDATES == 48
    for invalid in ((99999,), (201, 201), (True,), (1.1,), ()):
        with pytest.raises(ValueError):
            collect_public_sequences("RepeatFirstEasy", invalid)


def test_only_public_inputs_and_history_derived_expert_targets_are_recorded():
    rows = _rows()
    for row in rows:
        assert row.task in TASKS
        assert len(row.inputs) == len(row.expert_actions) == PUBLIC_EPISODE_STEPS
        assert all(type(x) is int or
                   (type(x) is tuple and len(x) == 2) for x in row.inputs)
        assert all(0 <= x < row.action_count for x in row.expert_actions)
        assert all(type(x) is bool for x in row.valid_steps)
    assert all(action == rows[0].inputs[0] for action in rows[0].expert_actions)
    assert rows[1].valid_steps[:3] == (False, False, False)
    assert rows[1].valid_steps[3]


def test_identical_public_tensor_and_zero_policy_actions_for_both_architectures():
    rows = _rows()
    observations, past_actions, modes, targets, valid, classes = _sequence_tensors(rows)
    assert observations.shape == (3, PUBLIC_EPISODE_STEPS, 3)
    assert past_actions.shape == (3, PUBLIC_EPISODE_STEPS, 4)
    assert torch.count_nonzero(past_actions) == 0  # no expert-action shortcut
    assert set(modes.unique().tolist()) == {0, 1, 2}
    assert targets.shape == valid.shape == (3, PUBLIC_EPISODE_STEPS)
    assert classes.shape == (3, ACTION_DIM)
    assert classes.sum(dim=1).tolist() == [4, 4, 27]


@pytest.mark.parametrize("kind", ["hybrid", "transformer"])
def test_neural_causality_prefix_invariance_and_legal_action_mask(kind):
    torch.set_num_threads(2)
    rows = _rows()
    obs, actions, mode, _, _, classes = _sequence_tensors(rows)
    torch.manual_seed(1)
    model = OriginalWorldHeadMemoryReadout(kind).eval()
    with torch.no_grad():
        all_tokens = _mask_class_logits(model(obs, actions, mode), classes)
        prefix = _mask_class_logits(
            model(obs[:, :9], actions[:, :9], mode[:, :9]), classes)
    assert all_tokens.shape == (3, PUBLIC_EPISODE_STEPS, ACTION_DIM)
    assert torch.allclose(all_tokens[:, :9, :], prefix, atol=1e-4, rtol=1e-4)
    for k in range(2):
        assert all_tokens[k, :, 4:].max() <= -1e5
        assert torch.all(all_tokens[k].argmax(dim=-1) < 4)


def test_real_reward_comes_from_executed_actions_and_action_independent_public_trace():
    for task in TASKS:
        row = collect_public_sequences(task, (101,))[0]
        expert = _same_public_trajectory_real_reward(row, row.expert_actions)
        zero = _same_public_trajectory_real_reward(
            row, (0,) * PUBLIC_EPISODE_STEPS)
        assert expert == pytest.approx(1.0, abs=1e-6)
        assert -1.01 <= zero <= 1.01
        assert expert >= zero


def test_public_trajectory_action_dependence_is_rejected_before_scoring():
    row = collect_public_sequences("RepeatFirstEasy", (101,))[0]
    forged = replace(row, inputs=(3 if row.inputs[0] != 3 else 2,) + row.inputs[1:])
    with pytest.raises(RuntimeError, match="action-dependent observation"):
        _same_public_trajectory_real_reward(
            forged, (0,) * PUBLIC_EPISODE_STEPS)


def test_training_plan_cannot_change_optimizer_budget_or_use_heldout_ids():
    # Validate ALL experiment inputs before any model is trained or any env
    # is constructed; the fixed fixture cannot be reinterpreted as approval.
    for changed in (
        {"updates": 49},
        {"updates": 0},
        {"training_ids": VALIDATION_DEVELOPMENT_EPISODES},
        {"validation_ids": TRAIN_DEVELOPMENT_EPISODES},
    ):
        with pytest.raises(ValueError, match="frozen development"):
            run_bounded_development_training(**changed)


def test_incorrect_shape_and_task_head_violations_fail_closed():
    rows = _rows()
    obs, actions, mode, targets, valid, classes = _sequence_tensors(rows)
    with pytest.raises(ValueError):
        _mask_class_logits(torch.zeros(3, 9, 4), classes)
    with pytest.raises(ValueError):
        _mask_class_logits(torch.zeros(3, 9, ACTION_DIM),
                           torch.ones(2, ACTION_DIM, dtype=torch.bool))
    with pytest.raises(ValueError):
        _sequence_tensors((rows[0], rows[0]))


def test_public_model_reports_no_pcma_science_authority():
    import tam_research.si_pcma_v1_popgym_neural_pilot as module
    assert not hasattr(module, "HELDOUT_RULES")
    assert not hasattr(module, "oracle_reward")
    assert not hasattr(module, "get_state")
    assert module.TRAINING_UPDATES < 100
