"""Published POMDP memory task sanity checks; no neural science."""
from dataclasses import fields

import pytest

from tam_research.si_pcma_v1_published_memory import (
    DEVELOPER_EPISODE_IDS, POPGYM_VERSION, TASKS, PublicMemoryPolicy,
    _task_observation, published_memory_development_diagnostic,
    run_public_episode,
)


def test_public_memory_has_clear_memory_necessity_without_hidden_state():
    current_only = PublicMemoryPolicy("RepeatFirstEasy", memory_enabled=False)
    memory = PublicMemoryPolicy("RepeatFirstEasy", memory_enabled=True)
    for value, expected in ((2, 2), (1, 2), (3, 2), (0, 2)):
        assert memory.choose(value) == expected
        assert current_only.choose(value) == value


def test_repeat_previous_four_step_ring_keeps_correct_old_public_value():
    agent = PublicMemoryPolicy("RepeatPreviousEasy", memory_enabled=True)
    actions = [agent.choose(x) for x in (2, 0, 3, 1, 1, 2)]
    assert actions == [0, 0, 0, 2, 0, 3]
    no_memory = PublicMemoryPolicy("RepeatPreviousEasy",
                                   memory_enabled=False)
    assert [no_memory.choose(x) for x in (2, 0, 3, 1)] == [2, 0, 3, 1]


def test_count_recall_updates_only_from_visible_dealt_cards():
    memory = PublicMemoryPolicy("CountRecallEasy", memory_enabled=True)
    no_memory = PublicMemoryPolicy("CountRecallEasy", memory_enabled=False)
    records = ((1, 1), (0, 1), (1, 1), (1, 0), (0, 0))
    actions = [memory.choose(x) for x in records]
    assert actions == [1, 1, 2, 1, 2]
    assert [no_memory.choose(x) for x in records] == [1, 0, 1, 0, 1]


def test_published_public_view_has_no_hidden_world_or_info_fields():
    assert _task_observation("RepeatFirstEasy", 3) == 3
    assert _task_observation("CountRecallEasy", [1, 0]) == (1, 0)
    with pytest.raises(ValueError):
        _task_observation("CountRecallEasy", [1, 2])
    with pytest.raises(ValueError):
        _task_observation("RepeatPreviousEasy", 9)
    with pytest.raises(ValueError):
        PublicMemoryPolicy("unknown", memory_enabled=True)


def test_real_external_world_episodes_have_same_action_budget():
    for task in TASKS:
        a = run_public_episode(task, 21, memory_enabled=False)
        b = run_public_episode(task, 21, memory_enabled=True)
        assert a.task == b.task == task
        assert a.episode_id == b.episode_id == 21
        assert a.decisions == b.decisions > 0
        assert b.total_return >= a.total_return
        assert b.return_finite
        assert 0.98 <= b.total_return <= 1.001


def test_fixed_published_developer_screen_never_evaluates_pcma_or_science():
    result = published_memory_development_diagnostic()
    assert result.status == "DEVELOPMENT_POSITIVE_CONTROL_ONLY"
    assert result.package_version == POPGYM_VERSION == "1.0.7"
    assert result.trainable_neural_models_evaluated == 0
    assert result.heldout_worlds_evaluated == 0
    assert result.scientific_seeds_reserved == 0
    assert result.original_si_v26 == "NO_GO"
    assert result.pcma_architecture_verdict == "UNKNOWN"
    assert result.may_claim_pcma_advantage is False
    assert len(result.tasks) == 3
    assert all(row.memory_positive_control_pass for row in result.tasks)
    assert all(row.episodes == len(DEVELOPER_EPISODE_IDS) for row in result.tasks)
    assert len({row.task for row in result.tasks}) == 3


@pytest.mark.parametrize("bad", [-1, 0, 20, 29, True, 21.0])
def test_episode_id_must_be_predeclared_developer_fixture(bad):
    with pytest.raises(ValueError):
        run_public_episode("RepeatFirstEasy", bad, memory_enabled=True)


def test_no_policy_can_consume_hidden_env_info_or_state():
    from tam_research import si_pcma_v1_published_memory as module
    assert not hasattr(module.PublicMemoryPolicy, "get_state")
    assert not hasattr(module.PublicMemoryPolicy, "get_hidden_state")
    assert not hasattr(module.PublicMemoryPolicy, "peek")
    assert [f.name for f in fields(module.EpisodeRecord)] == [
        "task", "episode_id", "memory_enabled", "decisions",
        "total_return", "return_finite",
    ]
