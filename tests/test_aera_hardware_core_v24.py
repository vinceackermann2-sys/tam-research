import ast
import inspect

import torch

from aera_v19_memory_necessity_cpu import (
    CHUNK_SIZE,
    EVAL_SEED,
    _force_all_stages_run,
    diagnostic_config,
    make_batch,
)
from tam_research.aera_hardware_core_v23 import HardwareAwareAERATextLMV23
from tam_research.aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    HardwareAwareAERATextLMV24,
    VectorizedContextualEpisodicMemory,
    causal_contextualize,
    episodic_state_bytes_per_session,
    vectorized_contextual_episodic_protocol,
)


def test_v24_protocol_matches_frozen_issue347_constants():
    p = vectorized_contextual_episodic_protocol()
    assert p["context_window_previous_events"] == 8
    assert p["capacity_slots_per_stage"] == 48
    assert p["duplicate_similarity_threshold"] == 0.95
    assert p["read_top_k"] == 4
    assert p["read_temperature"] == 0.10
    assert p["controlled_selected_writes"] == 2
    assert p["real_language_selected_writes"] == 16
    assert p["write_budget_changed_from_v23"] is False
    assert p["sequential_delta_recurrence"] is False
    assert p["inverse_covariance_state"] is False
    assert p["extra_learned_parameters"] == 0
    assert p["gpu_authorized"] is False


def test_causal_contextualize_cannot_see_future_events():
    g = torch.Generator().manual_seed(9500)
    h = torch.randn(2, 17, 12, generator=g)
    perturbed = h.clone()
    perturbed[:, 9:] = torch.randn(2, 8, 12, generator=g) * 100.0
    a = causal_contextualize(h)
    b = causal_contextualize(perturbed)
    torch.testing.assert_close(a[:, :9], b[:, :9], atol=0.0, rtol=0.0)
    # t=0 has no prior context.
    torch.testing.assert_close(a[:, 0], h[:, 0], atol=0.0, rtol=0.0)


def test_v24_has_exact_v23_learned_parameter_layout_and_values():
    torch.manual_seed(9501)
    v23 = HardwareAwareAERATextLMV23(diagnostic_config())
    _force_all_stages_run(v23)
    torch.manual_seed(9501)
    v24 = HardwareAwareAERATextLMV24(diagnostic_config())
    _force_all_stages_run(v24)
    assert sum(p.numel() for p in v23.parameters()) == sum(p.numel() for p in v24.parameters())
    assert v23.state_dict().keys() == v24.state_dict().keys()
    for key, value in v23.state_dict().items():
        torch.testing.assert_close(v24.state_dict()[key], value, atol=0.0, rtol=0.0)


def test_v24_empty_memory_logits_and_stream_are_bit_exact_v23():
    torch.manual_seed(9502)
    v23 = HardwareAwareAERATextLMV23(diagnostic_config())
    _force_all_stages_run(v23)
    torch.manual_seed(9502)
    v24 = HardwareAwareAERATextLMV24(diagnostic_config())
    _force_all_stages_run(v24)
    tokens = make_batch(2, EVAL_SEED + 9502).tokens[:, : 2 * CHUNK_SIZE]
    out23 = v23(
        tokens,
        hard=True,
        route_mode="hard_sparse",
        update_memory=False,
        return_block_logits=False,
    )
    out24 = v24(
        tokens,
        hard=True,
        route_mode="hard_sparse",
        update_memory=False,
        return_block_logits=False,
    )
    torch.testing.assert_close(out24["logits"], out23["logits"], atol=0.0, rtol=0.0)
    for old, new in zip(out23["state"].stages, out24["state"].stages):
        torch.testing.assert_close(new.stream, old.stream, atol=0.0, rtol=0.0)
        assert isinstance(new.memory, ContextualEpisodicMemoryState)
        assert not bool(new.memory.valid.any())


def test_real_language_shaped_v24_state_is_within_v23_80000_byte_budget():
    exact = episodic_state_bytes_per_session(n_stages=4, memory_dim=50)
    assert exact == 77_760
    assert exact <= 80_000


def test_vectorized_update_contains_no_python_candidate_loop():
    source = inspect.getsource(VectorizedContextualEpisodicMemory._vectorized_update)
    tree = ast.parse(inspect.cleandoc(source))
    assert not any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))


def test_newer_duplicate_context_invalidates_stale_slot_in_one_block_update():
    torch.manual_seed(9503)
    model = HardwareAwareAERATextLMV24(diagnostic_config())
    memory = model.stages[0].memory
    memory.set_differentiable_pretraining(False)
    d_model = model.cfg.d_model
    address = torch.randn(1, 1, d_model)
    old_payload = torch.randn(1, 1, d_model)
    new_payload = torch.randn(1, 1, d_model)
    strength = torch.ones(1, 1, 1)
    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    first = memory.update_block(address, old_payload, strength, state)
    assert int(first.valid.sum()) == 1
    second = memory.update_block(address, new_payload, strength, first)
    assert int(second.valid.sum()) == 1
    expected = torch.tanh(memory.v(new_payload))[0, 0]
    newest = second.values[0, second.valid[0]][0]
    torch.testing.assert_close(newest, expected, atol=1e-6, rtol=1e-6)


def test_empty_states_are_isolated_and_deployment_update_is_detached():
    torch.manual_seed(9504)
    model = HardwareAwareAERATextLMV24(diagnostic_config())
    memory = model.stages[0].memory
    a = memory.empty_state(1, torch.device("cpu"), torch.float32)
    b = memory.empty_state(1, torch.device("cpu"), torch.float32)
    assert a.keys.data_ptr() != b.keys.data_ptr()
    assert a.values.data_ptr() != b.values.data_ptr()
    address = torch.randn(1, 2, model.cfg.d_model)
    payload = torch.randn_like(address)
    strength = torch.ones(1, 2, 1)
    versions = tuple(p._version for p in memory.parameters())
    updated = memory.update_block(address, payload, strength, a)
    assert tuple(p._version for p in memory.parameters()) == versions
    assert updated.keys.grad_fn is None
    assert updated.values.grad_fn is None
    assert not bool(b.valid.any())


def test_differentiable_two_chunk_stage_path_trains_selector_and_qkvout():
    torch.manual_seed(9505)
    model = HardwareAwareAERATextLMV24(diagnostic_config())
    _force_all_stages_run(model)
    model.set_memory_pretraining_mode(True)
    stage = model.stages[0]
    events0 = torch.randn(2, CHUNK_SIZE, model.cfg.d_model, requires_grad=True)
    events1 = torch.randn(2, CHUNK_SIZE, model.cfg.d_model, requires_grad=True)
    out0, state0, _ = stage.forward_chunk(events0, None, hard=False, update_memory=True)
    assert stage.last_candidate_count == 5
    assert stage.last_selected_count == 2
    assert stage.last_vectorized_update_calls == 1
    out1, state1, _ = stage.forward_chunk(events1, state0, hard=False, update_memory=True)
    loss = (
        out0.float().square().mean()
        + out1.float().square().mean()
        + state1.memory.strengths.float().sum() * 0.01
    )
    loss.backward()
    assert stage.pair_write_gate.weight.grad is not None
    assert float(stage.pair_write_gate.weight.grad.abs().sum()) > 0.0
    for parameter in (
        stage.memory.q.weight,
        stage.memory.k.weight,
        stage.memory.v.weight,
        stage.memory.out.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_production_geometry_executes_exact_sixteen_writes_one_update_call():
    torch.manual_seed(9506)
    cfg = diagnostic_config()
    cfg = type(cfg)(**{**cfg.__dict__, "chunk_size": 256})
    model = HardwareAwareAERATextLMV24(cfg)
    _force_all_stages_run(model)
    stage = model.stages[0]
    stage.memory.set_differentiable_pretraining(False)
    events = torch.randn(1, 256, cfg.d_model)
    _, state, _ = stage.forward_chunk(events, None, hard=True, update_memory=True)
    assert stage.last_candidate_count == 255
    assert stage.last_selected_count == 16
    assert stage.last_vectorized_update_calls == 1
    assert isinstance(state.memory, ContextualEpisodicMemoryState)
    assert state.memory.keys.shape[1] == 48
