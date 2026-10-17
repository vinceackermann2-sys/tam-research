"""Engineering-only behavior and leakage tests, NOT confirmation experiments."""
from dataclasses import fields
import pytest
import torch

from tam_research.si_pcma_v1 import (
    WorldConfig, PersistentWorld, PublicObservation, FactualTransition,
    FactualMemory, ReversibleLowRankRewardAdapter, PCMAState, SharedBackboneInterface,
)


def _run(seed=5, family="stable", sessions=2, max_steps=4):
    world = PersistentWorld(WorldConfig(seed=seed, family=family, max_steps=max_steps,
                                        shift_after=5))
    results = []
    for s in range(sessions):
        if s:
            world.restart_session()
        for i in range(max_steps):
            results.append(world.step(i % 4))
    return results


def test_replay_world_is_deterministic_without_oracle_access():
    assert _run() == _run()
    assert _run(5) != _run(7)
    assert [f.name for f in fields(FactualTransition)] == [
        "before", "action", "after", "reward", "fee", "time", "provenance", "done"]
    assert [f.name for f in fields(PublicObservation)] == ["symbol", "steps_remaining"]


def test_one_true_feedback_per_executed_action_and_no_counterfactual_api():
    w = PersistentWorld(WorldConfig(8, max_steps=4))
    assert not hasattr(w, "counterfactual")
    event = w.step(3)
    assert event.action == 3 and event.fee == pytest.approx(.10)
    assert event.reward == 0 and event.time == 1
    with pytest.raises(ValueError):
        w.step(9)
    for i in range(3):
        w.step(i)
    with pytest.raises(RuntimeError):
        w.step(0)
    assert w.restart_session().steps_remaining == 4


def test_restart_does_not_reset_the_environment_clock():
    rows = _run(5, "shift", sessions=3)
    assert [r.time for r in rows] == list(range(1, 13))
    assert sum(r.done for r in rows) == 3


def test_memory_capacity_action_conditioning_and_provenance():
    rows = _run(4, sessions=3)
    store = FactualMemory(capacity=5)
    for row in rows:
        store.record(row)
    assert len(store.items) == 5
    assert [x.time for x in store.items] == list(range(8, 13))
    for action in range(4):
        results = store.retrieve(PublicObservation(0, 4), action)
        assert all(x.action == action and x.before.symbol == 0 for x in results)
    with pytest.raises(ValueError):
        store.record(rows[-1])
    with pytest.raises(ValueError):
        store.record(rows[0])


def test_memory_only_observes_chosen_actions():
    w = PersistentWorld(WorldConfig(3, max_steps=4))
    before = w.public_observation()
    event = w.step(1)
    state = PCMAState()
    state.observe(event)
    assert state.memory.retrieve(before, 1) == (event,)
    assert state.memory.retrieve(before, 2) == ()
    assert len(state.window) == 1


def test_online_adaptation_is_factual_bounded_and_reversible():
    row = next(r for r in _run(2) if r.action != 3)
    adapter = ReversibleLowRankRewardAdapter(max_updates_per_session=1)
    baseline = adapter.snapshot()
    assert adapter.update(row)
    assert not adapter.update(row)
    assert adapter.total_updates == 1
    adapter.restore(baseline)
    assert adapter.snapshot() == baseline
    adapter.restart_session()
    assert adapter.updates == 0


def test_probe_does_not_train_reward_predictor():
    row = _run(8)[-1]
    assert row.action == 3
    adapter = ReversibleLowRankRewardAdapter()
    assert not adapter.update(row)
    assert adapter.total_updates == 0


def test_persistence_independent_ablations_and_forced_reset():
    row = _run(2)[0]
    state = PCMAState()
    state.observe(row)
    snap = state.adapter.snapshot()
    state.restart_session()
    assert not state.window
    assert state.memory.items == (row,)
    assert state.adapter.total_updates == snap[-1]
    state.restart_session(erase_persistent=True)
    assert state.memory.items == () and state.adapter.total_updates == 0
    mem_only = PCMAState(adaptation_enabled=False)
    mem_only.observe(row)
    assert len(mem_only.memory.items) == 1 and mem_only.adapter.total_updates == 0
    adapter_only = PCMAState(memory_enabled=False)
    adapter_only.observe(row)
    assert not adapter_only.memory.items and adapter_only.adapter.total_updates == 1


def test_same_public_inputs_to_both_neural_backbones():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    obs = PublicObservation(2, 4)
    state = PCMAState()
    for kind in ("hybrid", "transformer"):
        wrapper = SharedBackboneInterface(kind, state)
        pred = wrapper.forecast(obs, 1)
        assert pred.shape == (3,) and torch.isfinite(pred).all()
        assert wrapper.state is state


def test_invalid_config_and_inputs():
    with pytest.raises(ValueError):
        WorldConfig(1, max_steps=2)
    with pytest.raises(ValueError):
        WorldConfig(1, family="invalid")
    with pytest.raises(ValueError):
        FactualMemory(0)
    with pytest.raises(ValueError):
        ReversibleLowRankRewardAdapter(rank=0)
    with pytest.raises(ValueError):
        ReversibleLowRankRewardAdapter().predict(PublicObservation(4, 1), 0)
