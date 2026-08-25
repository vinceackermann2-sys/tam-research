import torch

from aera_v19_memory_necessity_cpu import (
    EVAL_SEED,
    _force_all_stages_run,
    diagnostic_config,
    make_batch,
)
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research.aera_hardware_core_v25 import HardwareAwareAERATextLMV25
from tam_research.aera_hardware_core_v25_1 import (
    ExecutionEquivalentFICEMStage,
    HardwareAwareAERATextLMV251,
    execution_equivalent_v25_1_protocol,
)

FORWARD_RTOL = 1e-5
FORWARD_ATOL = 1e-6
GRAD_RTOL = 2e-5
GRAD_ATOL = 2e-6


def _models(seed: int, *, chunk_size: int | None = None):
    cfg = diagnostic_config()
    if chunk_size is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "chunk_size": chunk_size})
    torch.manual_seed(seed)
    baseline = HardwareAwareAERATextLMV25(cfg)
    torch.manual_seed(seed + 91)
    candidate = HardwareAwareAERATextLMV251(cfg)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    candidate.eval()
    return baseline, candidate


def _assert_epi_close(a: ContextualEpisodicMemoryState, b: ContextualEpisodicMemoryState):
    torch.testing.assert_close(a.keys, b.keys, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    torch.testing.assert_close(a.values, b.values, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    torch.testing.assert_close(a.strengths, b.strengths, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    assert torch.equal(a.valid, b.valid)


def _assert_model_state_close(a, b):
    assert len(a.stages) == len(b.stages)
    for old, new in zip(a.stages, b.stages):
        torch.testing.assert_close(old.stream, new.stream, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
        assert isinstance(old.memory, ContextualEpisodicMemoryState)
        assert isinstance(new.memory, ContextualEpisodicMemoryState)
        _assert_epi_close(old.memory, new.memory)


def _assert_route_metadata_equal(old, new):
    old_routes = old["stage_routes"]
    new_routes = new["stage_routes"]
    assert len(old_routes) == len(new_routes)
    for old_chunk, new_chunk in zip(old_routes, new_routes):
        assert len(old_chunk) == len(new_chunk)
        for old_stage, new_stage in zip(old_chunk, new_chunk):
            torch.testing.assert_close(
                old_stage["stage_route_gate"],
                new_stage["stage_route_gate"],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                old_stage["stage_route_probability"],
                new_stage["stage_route_probability"],
                rtol=0.0,
                atol=0.0,
            )
            assert old_stage["executed_fraction"] == new_stage["executed_fraction"]


def test_v25_1_state_dict_schema_parameters_and_values_are_exact_v25():
    baseline, candidate = _models(13001)
    old = baseline.state_dict()
    new = candidate.state_dict()
    assert old.keys() == new.keys()
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )
    for key, value in old.items():
        assert torch.equal(value, new[key]), key
    assert all(isinstance(stage, ExecutionEquivalentFICEMStage) for stage in candidate.stages)
    assert not any("runtime_factor_cache" in key for key in new)


def test_v25_1_hard_sparse_empty_memory_matches_v25_and_direct_dispatches_foundation():
    baseline, candidate = _models(13002)
    tokens = make_batch(3, EVAL_SEED + 13002).tokens[:, : 2 * baseline.cfg.chunk_size]
    with torch.no_grad():
        old = baseline(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
            return_block_logits=False,
        )
        new = candidate(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
            return_block_logits=False,
        )
    torch.testing.assert_close(old["logits"], new["logits"], rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    _assert_model_state_close(old["state"], new["state"])
    _assert_route_metadata_equal(old, new)
    assert candidate.foundation_direct_dispatch_calls == 2
    assert candidate.stages[0].memory.empty_read_fastpath_calls >= 1
    assert all(stage._runtime_factor_cache is None for stage in candidate.stages)


def test_v25_1_two_chunk_memory_write_state_selection_and_base_nonmutation_match_v25():
    baseline, candidate = _models(13003)
    _force_all_stages_run(baseline)
    _force_all_stages_run(candidate)
    tokens = make_batch(2, EVAL_SEED + 13003).tokens[:, : 2 * baseline.cfg.chunk_size]
    baseline_versions = tuple(p._version for p in baseline.parameters())
    candidate_versions = tuple(p._version for p in candidate.parameters())
    with torch.no_grad():
        old = baseline(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )
        new = candidate(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )
    torch.testing.assert_close(old["logits"], new["logits"], rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    _assert_model_state_close(old["state"], new["state"])
    _assert_route_metadata_equal(old, new)
    assert tuple(p._version for p in baseline.parameters()) == baseline_versions
    assert tuple(p._version for p in candidate.parameters()) == candidate_versions
    for old_stage, new_stage in zip(baseline.stages, candidate.stages):
        assert old_stage.last_candidate_count == new_stage.last_candidate_count == baseline.cfg.chunk_size - 1
        assert old_stage.last_selected_count == new_stage.last_selected_count == 2
        assert old_stage.last_vectorized_update_calls == new_stage.last_vectorized_update_calls == 1
        assert torch.equal(old_stage.last_selected_indices, new_stage.last_selected_indices)
        assert new_stage._runtime_factor_cache is None


def test_v25_1_empty_read_is_exact_zero_and_mixed_valid_read_uses_v25_path():
    baseline, candidate = _models(13004)
    old_memory = baseline.stages[0].memory
    new_memory = candidate.stages[0].memory
    b, t, d = 2, baseline.cfg.chunk_size, baseline.cfg.d_model
    g = torch.Generator().manual_seed(13004)
    identity = torch.randn(b, t, d, generator=g)
    context = torch.randn(b, t, d, generator=g)
    empty = old_memory.empty_state(b, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        old_empty = old_memory.read(identity, context, empty)
        new_empty = new_memory.read(identity, context, empty)
    assert torch.count_nonzero(new_empty) == 0
    assert torch.equal(old_empty, new_empty)
    assert new_empty.shape == old_empty.shape
    assert new_empty.dtype == old_empty.dtype
    assert new_empty.device == old_empty.device
    assert new_memory.empty_read_fastpath_calls == 1

    # Build a common mixed-valid state with one example populated and one empty.
    address_identity = torch.randn(1, 2, d, generator=g)
    address_context = torch.randn(1, 2, d, generator=g)
    payload = torch.randn(1, 2, d, generator=g)
    strength = torch.ones(1, 2, 1)
    one = old_memory.empty_state(1, torch.device("cpu"), torch.float32)
    one = old_memory.update_block(address_identity, address_context, payload, strength, one)
    mixed = ContextualEpisodicMemoryState(
        keys=torch.cat((one.keys, empty.keys[1:]), dim=0),
        values=torch.cat((one.values, empty.values[1:]), dim=0),
        strengths=torch.cat((one.strengths, empty.strengths[1:]), dim=0),
        valid=torch.cat((one.valid, empty.valid[1:]), dim=0),
    )
    with torch.no_grad():
        old_mixed = old_memory.read(identity, context, mixed)
        new_mixed = new_memory.read(identity, context, mixed)
    torch.testing.assert_close(old_mixed, new_mixed, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    # The batch contains a valid slot, so the exact all-empty shortcut must not fire.
    assert new_memory.empty_read_fastpath_calls == 1


def test_v25_1_differentiable_two_chunk_loss_and_ficem_gradients_match_v25():
    baseline, candidate = _models(13005)
    baseline.set_memory_pretraining_mode(True)
    candidate.set_memory_pretraining_mode(True)
    old_stage = baseline.stages[0]
    new_stage = candidate.stages[0]
    g = torch.Generator().manual_seed(13006)
    events0 = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)
    events1 = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)

    def run(stage, e0, e1):
        stage.zero_grad(set_to_none=True)
        out0, state0, _ = stage.forward_chunk(
            e0.clone().requires_grad_(True), None, hard=False, update_memory=True
        )
        out1, state1, _ = stage.forward_chunk(
            e1.clone().requires_grad_(True), state0, hard=False, update_memory=True
        )
        loss = (
            out0.float().square().mean()
            + out1.float().square().mean()
            + 0.01 * state1.memory.values.float().square().mean()
            + 0.01 * state1.memory.strengths.float().mean()
        )
        loss.backward()
        grads = {
            "identity_proj": stage.memory.identity_proj.weight.grad,
            "context_proj": stage.memory.context_proj.weight.grad,
            "v": stage.memory.v.weight.grad,
            "out": stage.memory.out.weight.grad,
            "pair_write_gate": stage.pair_write_gate.weight.grad,
        }
        return loss.detach(), out0.detach(), out1.detach(), state1.detach(), grads

    old_loss, old0, old1, old_state, old_grads = run(old_stage, events0, events1)
    new_loss, new0, new1, new_state, new_grads = run(new_stage, events0, events1)
    torch.testing.assert_close(old_loss, new_loss, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    torch.testing.assert_close(old0, new0, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    torch.testing.assert_close(old1, new1, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    _assert_epi_close(old_state.memory, new_state.memory)
    for name in old_grads:
        old_grad = old_grads[name]
        new_grad = new_grads[name]
        assert old_grad is not None and new_grad is not None, name
        assert torch.isfinite(old_grad).all() and torch.isfinite(new_grad).all(), name
        assert float(old_grad.abs().sum()) > 0.0 and float(new_grad.abs().sum()) > 0.0, name
        torch.testing.assert_close(old_grad, new_grad, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    assert new_stage._runtime_factor_cache is None


def test_v25_1_production_shape_stage_keeps_exact_16_of_255_one_update_and_state():
    baseline, candidate = _models(13007, chunk_size=256)
    old_stage = baseline.stages[0]
    new_stage = candidate.stages[0]
    old_stage.memory.set_differentiable_pretraining(False)
    new_stage.memory.set_differentiable_pretraining(False)
    g = torch.Generator().manual_seed(13008)
    events = torch.randn(1, 256, baseline.cfg.d_model, generator=g)
    with torch.no_grad():
        old_out, old_state, _ = old_stage.forward_chunk(
            events, None, hard=True, update_memory=True
        )
        new_out, new_state, _ = new_stage.forward_chunk(
            events, None, hard=True, update_memory=True
        )
    torch.testing.assert_close(old_out, new_out, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    _assert_epi_close(old_state.memory, new_state.memory)
    torch.testing.assert_close(old_state.stream, new_state.stream, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    assert old_stage.last_candidate_count == new_stage.last_candidate_count == 255
    assert old_stage.last_selected_count == new_stage.last_selected_count == 16
    assert old_stage.last_vectorized_update_calls == new_stage.last_vectorized_update_calls == 1
    assert torch.equal(old_stage.last_selected_indices, new_stage.last_selected_indices)
    assert new_stage._runtime_factor_cache is None


def test_v25_1_protocol_freezes_no_scientific_or_scale_authorization():
    protocol = execution_equivalent_v25_1_protocol()
    assert protocol["version"] == "aera-v25.1-execution-equivalent-runtime"
    assert protocol["research_issue"] == 380
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
