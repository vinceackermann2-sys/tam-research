"""Contract tests only for separated development world (never heldout science)."""
from dataclasses import fields
from unittest.mock import patch

import pytest

from tam_research.si_pcma_v1 import PublicObservation, FactualTransition
from tam_research.si_pcma_v1_ood_world import (
    OODWorldConfig, OODRuleWorld, run_ood_sessions,
)
from tam_research.si_pcma_v1_ood_controls import (
    FixedGuessControl, SlidingWindowFactualControl,
)
from tam_research.si_pcma_v1_world_ipc import (
    DevelopmentWorldIPC, _public_observation, _factual_transition,
    run_ipc_development_sessions,
)


def test_same_development_trace_as_original_inprocess_reference():
    # This compares *correctness*, NEVER scores or scientific promotion.
    for scenario in ("stable", "shift", "noisy", "long"):
        cfg = OODWorldConfig(10, "development", scenario)
        original = run_ood_sessions(FixedGuessControl(1), cfg, sessions=2)
        isolated = run_ipc_development_sessions(
            FixedGuessControl(1), cfg, sessions=2)
        assert isolated == original
        assert [e.time for group in isolated for e in group] == list(
            range(1, 2 * cfg.session_horizon + 1))


def test_context_probe_and_actual_chosen_feedback_match_reference():
    cfg = OODWorldConfig(11, "development", "noisy")
    reference = run_ood_sessions(
        SlidingWindowFactualControl(window=4, probe_period=3,
                                    exploration=0.0), cfg, sessions=2)
    isolated = run_ipc_development_sessions(
        SlidingWindowFactualControl(window=4, probe_period=3,
                                    exploration=0.0), cfg, sessions=2)
    assert isolated == reference
    assert all(e.fee == (0.1 if e.action == 3 else 0.0)
               for group in isolated for e in group)
    assert sum(e.action == 3 for group in isolated for e in group) == 4


def test_parent_policy_never_receives_private_world_or_child_handle():
    class SpyPolicy(FixedGuessControl):
        def __init__(self):
            super().__init__()
            self.received = []
        def choose(self, observation):
            self.received.append(observation)
            return super().choose(observation)
        def observe(self, event):
            self.received.append(event)
            return super().observe(event)
    p = SpyPolicy()
    run_ipc_development_sessions(
        p, OODWorldConfig(12, "development"), sessions=2)
    assert len(p.received) == 24
    assert {type(x) for x in p.received} == {PublicObservation, FactualTransition}
    assert all(not hasattr(x, "_rule") and not hasattr(x, "_context")
               and not hasattr(x, "seed") and not hasattr(x, "_rng")
               for x in p.received)


def test_ipc_runner_heldout_denied_before_worker_spawn():
    cfg = OODWorldConfig(7, "heldout")
    with patch("tam_research.si_pcma_v1_world_ipc.subprocess.Popen") as launch:
        with pytest.raises(ValueError, match="heldout"):
            run_ipc_development_sessions(FixedGuessControl(), cfg)
        with pytest.raises(ValueError, match="heldout"):
            DevelopmentWorldIPC(cfg)
        launch.assert_not_called()


def test_dev_client_rejects_invalid_action_and_invalid_restart():
    cfg = OODWorldConfig(9, "development")
    with DevelopmentWorldIPC(cfg) as client:
        assert client.observation.steps_remaining == 6
        with pytest.raises(ValueError):
            client.step(True)
        with pytest.raises(ValueError):
            client.step(9)
        with pytest.raises(RuntimeError):
            client.restart_session()
        for _ in range(6):
            event = client.step(0)
        assert event.done
        with pytest.raises(RuntimeError):
            client.step(0)
        restarted = client.restart_session()
        assert restarted.steps_remaining == 6
        assert client.step(2).time == 7


def test_child_disallows_private_queries_and_mutation_with_fixed_errors():
    cfg = OODWorldConfig(6, "development")
    with DevelopmentWorldIPC(cfg) as client:
        for request in (
            {"op": "get_hidden_rule"}, {"op": "peek"}, {"op": "oracle"},
            {"op": "reset_hidden_world"}, {"op": "step", "action": 0,
                                          "request_hidden": True},
            {"op": "start"},
        ):
            with pytest.raises(ValueError, match="rejected"):
                client._exchange(request)
        assert client.step(1).time == 1


def test_worker_prohibits_heldout_even_if_called_without_parent_runner():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-u", "-m",
         "tam_research.si_pcma_v1_world_worker",
         "--split", "heldout", "--seed", "3"],
        input="", capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "heldout" not in result.stdout


def test_explicit_schema_rejects_unexpected_extra_private_fields():
    with pytest.raises(ValueError):
        _public_observation({"symbol": 0, "steps_remaining": 6,
                             "hidden_context": 2})
    with pytest.raises(ValueError):
        _public_observation({"symbol": 0, "steps_remaining": True})
    with pytest.raises(ValueError):
        _factual_transition({"rule": (0, 1, 2)})
    with pytest.raises(ValueError):
        _factual_transition({
            "before": {"symbol": 0, "steps_remaining": 6},
            "after": {"symbol": 1, "steps_remaining": 5},
            "action": 3, "reward": 1.0, "fee": 0.1,
            "time": 1, "provenance": "forge", "done": False,
        })


def test_session_budget_and_cleanup_on_policy_exception():
    class AbortingPolicy(FixedGuessControl):
        def __init__(self):
            super().__init__()
            self.count = 0
        def choose(self, observation):
            self.count += 1
            if self.count == 3:
                raise RuntimeError("controlled fixture abort")
            return super().choose(observation)
    with pytest.raises(RuntimeError, match="controlled fixture abort"):
        run_ipc_development_sessions(
            AbortingPolicy(), OODWorldConfig(3, "development"), sessions=2)
    with pytest.raises(ValueError):
        run_ipc_development_sessions(
            FixedGuessControl(), OODWorldConfig(4, "development"), sessions=0)


def test_reward_and_model_seed_do_not_get_packed_into_observation():
    p = FixedGuessControl()
    traces = run_ipc_development_sessions(
        p, OODWorldConfig(13, "development"), sessions=1)
    for e in traces[0]:
        assert [f.name for f in fields(e.before)] == ["symbol", "steps_remaining"]
        assert [f.name for f in fields(e)] == [
            "before", "action", "after", "reward", "fee",
            "time", "provenance", "done",
        ]
