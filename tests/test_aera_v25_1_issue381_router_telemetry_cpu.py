from __future__ import annotations

import inspect

import torch

from tests.test_aera_v25_1_execution_equivalent_cpu import _models
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research.aera_hardware_core_v25_1 import (
    ExecutionEquivalentStageRouteGate,
    execution_equivalent_v25_1_protocol,
)
from tam_research.aera_hardware_core_v8 import StageRouteGate

RTOL = 1e-6
ATOL = 1e-6


def _assert_state_close(old, new):
    assert len(old.stages) == len(new.stages)
    for old_stage, new_stage in zip(old.stages, new.stages):
        torch.testing.assert_close(old_stage.stream, new_stage.stream, rtol=RTOL, atol=ATOL)
        assert isinstance(old_stage.memory, ContextualEpisodicMemoryState)
        assert isinstance(new_stage.memory, ContextualEpisodicMemoryState)
        torch.testing.assert_close(old_stage.memory.keys, new_stage.memory.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(old_stage.memory.values, new_stage.memory.values, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            old_stage.memory.strengths,
            new_stage.memory.strengths,
            rtol=RTOL,
            atol=ATOL,
        )
        assert torch.equal(old_stage.memory.valid, new_stage.memory.valid)


def test_issue381_router_wrapper_preserves_state_dict_schema_and_values():
    baseline, candidate = _models(138401)
    assert list(baseline.state_dict()) == list(candidate.state_dict())
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0.0, atol=0.0)
    assert all(
        isinstance(router, ExecutionEquivalentStageRouteGate)
        for router in candidate.stage_routers
    )
    assert all(isinstance(router, StageRouteGate) for router in candidate.stage_routers)


def test_issue381_router_gate_logits_and_stats_are_exact_for_all_modes():
    torch.manual_seed(138402)
    original = StageRouteGate(12)
    source = StageRouteGate(12)
    source.load_state_dict(original.state_dict(), strict=True)
    candidate = ExecutionEquivalentStageRouteGate(source)
    first_event = torch.randn(7, 12)
    stream = torch.randn(7, 12)

    for mode in ("soft", "straight_through", "hard_sparse"):
        old_gate, old_logits = original(first_event, stream, mode=mode)
        new_gate, new_logits = candidate(first_event, stream, mode=mode)
        torch.testing.assert_close(old_logits, new_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(old_gate, new_gate, rtol=0.0, atol=0.0)
        assert torch.equal(original.last_probability, candidate.last_probability)
        assert torch.equal(original.last_hard_gate, candidate.last_hard_gate)
        assert original.stats() == candidate.stats()
        assert candidate.last_probability is not None
        assert candidate.last_hard_gate is not None
        assert candidate.last_probability.device == first_event.device
        assert candidate.last_hard_gate.device == first_event.device


def test_issue381_router_wrapper_preserves_straight_through_gradients():
    torch.manual_seed(138403)
    original = StageRouteGate(10)
    source = StageRouteGate(10)
    source.load_state_dict(original.state_dict(), strict=True)
    candidate = ExecutionEquivalentStageRouteGate(source)
    first_old = torch.randn(6, 10, requires_grad=True)
    stream_old = torch.randn(6, 10, requires_grad=True)
    first_new = first_old.detach().clone().requires_grad_(True)
    stream_new = stream_old.detach().clone().requires_grad_(True)

    old_gate, old_logits = original(first_old, stream_old, mode="straight_through")
    new_gate, new_logits = candidate(first_new, stream_new, mode="straight_through")
    old_loss = old_gate.square().mean() + 0.17 * old_logits.square().mean()
    new_loss = new_gate.square().mean() + 0.17 * new_logits.square().mean()
    old_loss.backward()
    new_loss.backward()

    torch.testing.assert_close(old_loss, new_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        original.proj.weight.grad, candidate.proj.weight.grad, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original.proj.bias.grad, candidate.proj.bias.grad, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(first_old.grad, first_new.grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stream_old.grad, stream_new.grad, rtol=0.0, atol=0.0)


def test_issue381_full_model_hard_sparse_routes_logits_and_state_remain_equivalent():
    baseline, candidate = _models(138404)
    tokens = torch.randint(0, baseline.cfg.vocab_size, (4, baseline.cfg.chunk_size * 2))
    with torch.no_grad():
        old = baseline(tokens, hard=True, route_mode="hard_sparse", update_memory=True)
        new = candidate(tokens, hard=True, route_mode="hard_sparse", update_memory=True)

    torch.testing.assert_close(old["logits"], new["logits"], rtol=RTOL, atol=ATOL)
    _assert_state_close(old["state"], new["state"])
    assert len(old["stage_routes"]) == len(new["stage_routes"])
    for old_chunk, new_chunk in zip(old["stage_routes"], new["stage_routes"]):
        assert len(old_chunk) == len(new_chunk)
        for old_route, new_route in zip(old_chunk, new_chunk):
            torch.testing.assert_close(
                old_route["stage_route_probability"],
                new_route["stage_route_probability"],
                rtol=0.0,
                atol=0.0,
            )
            assert torch.equal(
                old_route["stage_route_gate"], new_route["stage_route_gate"]
            )
            assert old_route["executed_fraction"] == new_route["executed_fraction"]


def test_issue381_v25_1_router_forward_has_no_eager_host_copy_and_protocol_is_frozen():
    source = inspect.getsource(ExecutionEquivalentStageRouteGate.forward)
    assert ".cpu(" not in source
    assert "prob.detach()" in source
    assert "hard.detach()" in source
    protocol = execution_equivalent_v25_1_protocol()
    assert protocol["router_gate_math_changed"] is False
    assert protocol["router_telemetry_forward_host_copy"] is False
    assert protocol["router_state_dict_changed"] is False
    assert protocol["routing_policy_changed"] is False
    assert protocol["state_dict_schema_changed"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["100m_authorized"] is False
