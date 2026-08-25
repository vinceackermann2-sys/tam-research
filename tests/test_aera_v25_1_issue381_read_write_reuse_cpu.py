from __future__ import annotations

import ast
import inspect
import textwrap

import torch

from aera_v19_memory_necessity_cpu import diagnostic_config
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research.aera_hardware_core_v25 import HardwareAwareAERATextLMV25
from tam_research.aera_hardware_core_v25_1 import (
    ExecutionEquivalentFactorizedIdentityContextMemory,
    HardwareAwareAERATextLMV251,
    execution_equivalent_v25_1_protocol,
)

RTOL = 1e-6
ATOL = 1e-6


def _models(seed: int):
    cfg = diagnostic_config()
    torch.manual_seed(seed)
    baseline = HardwareAwareAERATextLMV25(cfg)
    torch.manual_seed(seed + 1)
    candidate = HardwareAwareAERATextLMV251(cfg)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    candidate.eval()
    return baseline, candidate


def _assert_state_close(old, new):
    torch.testing.assert_close(old.stream, new.stream, rtol=RTOL, atol=ATOL)
    assert isinstance(old.memory, ContextualEpisodicMemoryState)
    assert isinstance(new.memory, ContextualEpisodicMemoryState)
    torch.testing.assert_close(old.memory.keys, new.memory.keys, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(old.memory.values, new.memory.values, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        old.memory.strengths, new.memory.strengths, rtol=RTOL, atol=ATOL
    )
    assert torch.equal(old.memory.valid, new.memory.valid)


def test_issue381_nonempty_second_chunk_reuses_read_projection_and_prior_key_norm():
    baseline, candidate = _models(138301)
    old_stage = baseline.stages[0]
    new_stage = candidate.stages[0]
    assert isinstance(new_stage.memory, ExecutionEquivalentFactorizedIdentityContextMemory)
    g = torch.Generator().manual_seed(138302)
    first = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)
    second = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)

    with torch.no_grad():
        old0, old_state0, _ = old_stage.forward_chunk(
            first, None, hard=True, update_memory=True
        )
        new0, new_state0, _ = new_stage.forward_chunk(
            first, None, hard=True, update_memory=True
        )
    torch.testing.assert_close(old0, new0, rtol=RTOL, atol=ATOL)
    _assert_state_close(old_state0, new_state0)
    assert new_stage.last_reused_read_key_update_calls == 0
    assert new_stage.memory.projected_update_reuse_calls == 0

    with torch.no_grad():
        old1, old_state1, _ = old_stage.forward_chunk(
            second, old_state0, hard=True, update_memory=True
        )
        new1, new_state1, _ = new_stage.forward_chunk(
            second, new_state0, hard=True, update_memory=True
        )
    torch.testing.assert_close(old1, new1, rtol=RTOL, atol=ATOL)
    _assert_state_close(old_state1, new_state1)
    assert torch.equal(old_stage.last_selected_indices, new_stage.last_selected_indices)
    assert old_stage.last_selected_count == new_stage.last_selected_count
    assert old_stage.last_vectorized_update_calls == new_stage.last_vectorized_update_calls == 1
    assert new_stage.last_reused_read_key_update_calls == 1
    assert new_stage.memory.projected_update_reuse_calls == 1
    assert new_stage._runtime_factor_cache is None


def test_issue381_reuse_path_preserves_differentiable_two_chunk_gradients():
    baseline, candidate = _models(138303)
    baseline.set_memory_pretraining_mode(True)
    candidate.set_memory_pretraining_mode(True)
    old_stage = baseline.stages[0]
    new_stage = candidate.stages[0]
    g = torch.Generator().manual_seed(138304)
    first = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)
    second = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)

    def run(stage):
        stage.zero_grad(set_to_none=True)
        out0, state0, _ = stage.forward_chunk(
            first.clone().requires_grad_(True), None, hard=False, update_memory=True
        )
        out1, state1, _ = stage.forward_chunk(
            second.clone().requires_grad_(True), state0, hard=False, update_memory=True
        )
        loss = (
            out0.float().square().mean()
            + out1.float().square().mean()
            + 0.01 * state1.memory.values.float().square().mean()
            + 0.01 * state1.memory.strengths.float().mean()
        )
        loss.backward()
        return loss.detach(), {
            "identity_proj": stage.memory.identity_proj.weight.grad,
            "context_proj": stage.memory.context_proj.weight.grad,
            "v": stage.memory.v.weight.grad,
            "out": stage.memory.out.weight.grad,
            "pair_write_gate": stage.pair_write_gate.weight.grad,
        }

    old_loss, old_grads = run(old_stage)
    new_loss, new_grads = run(new_stage)
    torch.testing.assert_close(old_loss, new_loss, rtol=2e-5, atol=2e-6)
    for name, old_grad in old_grads.items():
        new_grad = new_grads[name]
        assert old_grad is not None and new_grad is not None, name
        torch.testing.assert_close(old_grad, new_grad, rtol=2e-5, atol=2e-6)
    assert new_stage.last_reused_read_key_update_calls == 1
    assert new_stage.memory.projected_update_reuse_calls == 1
    assert new_stage._runtime_factor_cache is None


def test_issue381_projected_update_remains_single_vectorized_no_python_loop_path():
    source = inspect.getsource(
        ExecutionEquivalentFactorizedIdentityContextMemory._vectorized_update_from_projected
    )
    tree = ast.parse(textwrap.dedent(source))
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
    assert 'torch.einsum("bkd,bjd->bkj"' in source
    assert 'torch.einsum("bkd,bsd->bks"' in source
    assert "_gather_slots" in source


def test_issue381_protocol_marks_only_ephemeral_execution_reuse():
    protocol = execution_equivalent_v25_1_protocol()
    assert protocol["projected_read_query_reused_for_write"] is True
    assert protocol["normalized_prior_keys_reused_for_write"] is True
    assert protocol["known_empty_write_reuse_fallback"] == "existing v25.1 update path"
    assert protocol["runtime_factor_cache_persistent"] is False
    assert protocol["learned_equations_changed"] is False
    assert protocol["learned_parameter_count_changed"] is False
    assert protocol["state_dict_schema_changed"] is False
    assert protocol["routing_policy_changed"] is False
    assert protocol["write_budget_changed"] is False
    assert protocol["real_language_selected_writes"] == 16
    assert protocol["vectorized_update_calls_per_completed_stage_chunk"] == 1
    assert protocol["state_bytes_real_language_four_stage_memory_dim50"] == 77_760
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["100m_authorized"] is False
