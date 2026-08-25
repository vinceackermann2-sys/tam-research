from __future__ import annotations

import inspect

import torch

from aera_v19_memory_necessity_cpu import EVAL_SEED, _force_all_stages_run, diagnostic_config, make_batch
from tam_research.aera_hardware_core_v25_1_compact import HardwareAwareAERATextLMV251StableCompact
from tam_research.aera_hardware_core_v25_1_nohost import (
    ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner,
    ExecutionEquivalentNoHostNativeGroupedMMSparseExpertBank,
    HardwareAwareAERATextLMV251NoHostTelemetry,
    no_host_adaptive_telemetry_v25_1_protocol,
)

RTOL = 1e-6
ATOL = 1e-6


def _models(seed: int):
    cfg = diagnostic_config()
    torch.manual_seed(seed)
    baseline = HardwareAwareAERATextLMV251StableCompact(cfg)
    torch.manual_seed(seed + 113)
    candidate = HardwareAwareAERATextLMV251NoHostTelemetry(cfg)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    candidate.eval()
    return baseline, candidate


def _assert_state_close(old, new):
    assert len(old.stages) == len(new.stages)
    for old_stage, new_stage in zip(old.stages, new.stages):
        torch.testing.assert_close(old_stage.stream, new_stage.stream, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(old_stage.memory.keys, new_stage.memory.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(old_stage.memory.values, new_stage.memory.values, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            old_stage.memory.strengths, new_stage.memory.strengths, rtol=RTOL, atol=ATOL
        )
        assert torch.equal(old_stage.memory.valid, new_stage.memory.valid)


def test_issue381_nohost_state_dict_is_exact_and_wrappers_are_active():
    baseline, candidate = _models(138601)
    assert list(baseline.state_dict()) == list(candidate.state_dict())
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0.0, atol=0.0)
    assert all(
        isinstance(stage.experts, ExecutionEquivalentNoHostNativeGroupedMMSparseExpertBank)
        for stage in candidate.stages
    )
    assert all(
        isinstance(stage.reasoner, ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner)
        for stage in candidate.stages
    )


def test_issue381_hard_expert_cpu_fallback_output_telemetry_and_stats_match():
    baseline, candidate = _models(138602)
    old_expert = baseline.stages[0].experts
    new_expert = candidate.stages[0].experts
    b, t, d = 7, baseline.cfg.chunk_size, baseline.cfg.d_model
    g = torch.Generator().manual_seed(138603)
    x = torch.randn(b, t, d, generator=g)
    expert_logits = torch.randn(b, baseline.cfg.n_experts, generator=g)
    count_logits = torch.randn(b, 2, generator=g)
    # Exercise both top-1 and top-2 decisions deterministically.
    count_logits[: b // 2, 0] += 8.0
    count_logits[b // 2 :, 1] += 8.0

    old = old_expert(x, expert_logits, count_logits, hard=True)
    new = new_expert(x, expert_logits, count_logits, hard=True)
    torch.testing.assert_close(old, new, rtol=RTOL, atol=ATOL)
    assert torch.equal(old_expert.last_counts, new_expert.last_counts)
    torch.testing.assert_close(
        old_expert.last_route_probs, new_expert.last_route_probs, rtol=0.0, atol=0.0
    )
    assert new_expert.last_counts.device == x.device
    assert new_expert.last_route_probs.device == x.device
    assert old_expert.stats() == new_expert.stats()


def test_issue381_soft_expert_output_and_gradients_remain_exact_inherited_path():
    baseline, candidate = _models(138604)
    old_expert = baseline.stages[0].experts
    new_expert = candidate.stages[0].experts
    g = torch.Generator().manual_seed(138605)
    x0 = torch.randn(4, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)
    e0 = torch.randn(4, baseline.cfg.n_experts, generator=g)
    c0 = torch.randn(4, 2, generator=g)

    def run(expert):
        expert.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        e = e0.clone().requires_grad_(True)
        c = c0.clone().requires_grad_(True)
        out = expert(x, e, c, hard=False)
        loss = out.float().square().mean()
        loss.backward()
        return out.detach(), x.grad, e.grad, c.grad, expert.w1.grad, expert.w2.grad

    old = run(old_expert)
    new = run(new_expert)
    for old_tensor, new_tensor in zip(old, new):
        assert old_tensor is not None and new_tensor is not None
        torch.testing.assert_close(old_tensor, new_tensor, rtol=RTOL, atol=ATOL)


def test_issue381_hard_and_soft_reasoner_outputs_decisions_stats_and_gradients_match():
    baseline, candidate = _models(138606)
    old_reasoner = baseline.stages[0].reasoner
    new_reasoner = candidate.stages[0].reasoner
    b, d, steps = 8, baseline.cfg.d_model, baseline.cfg.max_reason_steps
    g = torch.Generator().manual_seed(138607)
    summary = torch.randn(b, d, generator=g)
    depth_logits = torch.full((b, steps), -7.0)
    chosen = torch.arange(b) % steps
    depth_logits.scatter_(1, chosen[:, None], 7.0)

    old_hard = old_reasoner(summary, depth_logits, hard=True)
    new_hard = new_reasoner(summary, depth_logits, hard=True)
    torch.testing.assert_close(old_hard, new_hard, rtol=RTOL, atol=ATOL)
    assert torch.equal(old_reasoner.last_steps, new_reasoner.last_steps)
    torch.testing.assert_close(
        old_reasoner.last_expected, new_reasoner.last_expected, rtol=0.0, atol=0.0
    )
    assert new_reasoner.last_steps.device == summary.device
    assert new_reasoner.last_expected.device == summary.device
    assert old_reasoner.stats() == new_reasoner.stats()

    def run_soft(reasoner):
        reasoner.zero_grad(set_to_none=True)
        s = summary.clone().requires_grad_(True)
        logits = torch.randn(b, steps, generator=g).requires_grad_(True)
        out = reasoner(s, logits, hard=False)
        loss = out.float().square().mean()
        loss.backward()
        return out.detach(), s.grad, logits.grad, reasoner.cell.weight_ih.grad, reasoner.cell.weight_hh.grad

    # Reset generator so the two calls receive identical soft depth logits.
    g.manual_seed(138608)
    old_soft = run_soft(old_reasoner)
    g.manual_seed(138608)
    new_soft = run_soft(new_reasoner)
    for old_tensor, new_tensor in zip(old_soft, new_soft):
        assert old_tensor is not None and new_tensor is not None
        torch.testing.assert_close(old_tensor, new_tensor, rtol=RTOL, atol=ATOL)


def test_issue381_nohost_full_model_hard_sparse_memory_on_and_off_match_compact_candidate():
    for update_memory in (False, True):
        baseline, candidate = _models(138609 + int(update_memory))
        _force_all_stages_run(baseline)
        _force_all_stages_run(candidate)
        tokens = make_batch(3, EVAL_SEED + 138610 + int(update_memory)).tokens[
            :, : 2 * baseline.cfg.chunk_size
        ]
        with torch.no_grad():
            old = baseline(
                tokens,
                hard=True,
                route_mode="hard_sparse",
                update_memory=update_memory,
            )
            new = candidate(
                tokens,
                hard=True,
                route_mode="hard_sparse",
                update_memory=update_memory,
            )
        torch.testing.assert_close(old["logits"], new["logits"], rtol=RTOL, atol=ATOL)
        _assert_state_close(old["state"], new["state"])
        for old_stage, new_stage in zip(baseline.stages, candidate.stages):
            assert old_stage.last_candidate_count == new_stage.last_candidate_count
            assert old_stage.last_selected_count == new_stage.last_selected_count
            assert old_stage.last_vectorized_update_calls == new_stage.last_vectorized_update_calls
            if old_stage.last_selected_indices is None:
                assert new_stage.last_selected_indices is None
            else:
                assert torch.equal(old_stage.last_selected_indices, new_stage.last_selected_indices)
            assert torch.equal(old_stage.experts.last_counts, new_stage.experts.last_counts)
            assert torch.equal(old_stage.reasoner.last_steps, new_stage.reasoner.last_steps)
        for old_chunk, new_chunk in zip(old["stage_routes"], new["stage_routes"]):
            for old_route, new_route in zip(old_chunk, new_chunk):
                assert torch.equal(old_route["stage_route_gate"], new_route["stage_route_gate"])
                torch.testing.assert_close(
                    old_route["stage_route_probability"],
                    new_route["stage_route_probability"],
                    rtol=0.0,
                    atol=0.0,
                )
                assert old_route["executed_fraction"] == new_route["executed_fraction"]


def test_issue381_hard_adaptive_execution_methods_have_no_eager_cpu_copy_and_protocol_is_frozen():
    expert_bmm = inspect.getsource(
        ExecutionEquivalentNoHostNativeGroupedMMSparseExpertBank._bmm_hard_forward
    )
    expert_native = inspect.getsource(
        ExecutionEquivalentNoHostNativeGroupedMMSparseExpertBank._native_hard_forward
    )
    reasoner = inspect.getsource(ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner.forward)
    assert ".cpu(" not in expert_bmm
    assert ".cpu(" not in expert_native
    assert ".cpu(" not in reasoner

    protocol = no_host_adaptive_telemetry_v25_1_protocol()
    assert protocol["expert_hard_math_changed"] is False
    assert protocol["reasoner_math_changed"] is False
    assert protocol["expert_telemetry_forward_host_copy"] is False
    assert protocol["reasoner_telemetry_forward_host_copy"] is False
    assert protocol["physical_expert_sparsity_changed"] is False
    assert protocol["physical_reasoning_sparsity_changed"] is False
    assert protocol["learned_parameter_count_changed"] is False
    assert protocol["state_dict_schema_changed"] is False
    assert protocol["routing_policy_changed"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["100m_authorized"] is False
