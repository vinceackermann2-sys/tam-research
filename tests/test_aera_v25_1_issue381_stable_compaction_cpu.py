from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import EVAL_SEED, _force_all_stages_run, diagnostic_config, make_batch
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research.aera_hardware_core_v25_1 import HardwareAwareAERATextLMV251
from tam_research.aera_hardware_core_v25_1_compact import (
    HardwareAwareAERATextLMV251StableCompact,
    StableCompactExecutionEquivalentFactorizedIdentityContextMemory,
    stable_compaction_v25_1_protocol,
)

RTOL = 1e-6
ATOL = 1e-6
PRODUCTION_K = 16
PRODUCTION_CAPACITY = 48


def _models(seed: int):
    cfg = diagnostic_config()
    torch.manual_seed(seed)
    baseline = HardwareAwareAERATextLMV251(cfg)
    torch.manual_seed(seed + 97)
    candidate = HardwareAwareAERATextLMV251StableCompact(cfg)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    candidate.eval()
    return baseline, candidate


def _assert_state_close(old, new):
    assert len(old.stages) == len(new.stages)
    for old_stage, new_stage in zip(old.stages, new.stages):
        torch.testing.assert_close(old_stage.stream, new_stage.stream, rtol=RTOL, atol=ATOL)
        assert isinstance(old_stage.memory, ContextualEpisodicMemoryState)
        assert isinstance(new_stage.memory, ContextualEpisodicMemoryState)
        torch.testing.assert_close(old_stage.memory.keys, new_stage.memory.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(old_stage.memory.values, new_stage.memory.values, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(
            old_stage.memory.strengths, new_stage.memory.strengths, rtol=RTOL, atol=ATOL
        )
        assert torch.equal(old_stage.memory.valid, new_stage.memory.valid)


def _reference_tail(
    new_keys,
    new_values,
    new_strengths,
    new_valid,
    old_keys,
    old_values,
    old_strengths,
    keep_old,
    *,
    capacity: int,
):
    all_keys = torch.cat((new_keys, old_keys), dim=1)
    all_values = torch.cat((new_values, old_values), dim=1)
    all_strengths = torch.cat((new_strengths, old_strengths), dim=1)
    all_valid = torch.cat((new_valid, keep_old), dim=1)
    total = all_valid.size(1)
    slot_position = torch.arange(total, device=all_valid.device, dtype=torch.float32)
    priority = all_valid.float() * 2.0 - slot_position[None, :] * 1e-6
    keep_indices = torch.topk(
        priority, k=capacity, dim=1, largest=True, sorted=True
    ).indices

    def gather(x):
        index = keep_indices
        while index.ndim < x.ndim:
            index = index.unsqueeze(-1)
        return x.gather(1, index.expand(*keep_indices.shape, *x.shape[2:]))

    return ContextualEpisodicMemoryState(
        keys=gather(all_keys),
        values=gather(all_values),
        strengths=gather(all_strengths),
        valid=gather(all_valid),
    )


def test_issue381_stable_compaction_matches_authoritative_tail_for_adversarial_validity():
    _, model = _models(138501)
    memory = model.stages[0].memory
    assert isinstance(memory, StableCompactExecutionEquivalentFactorizedIdentityContextMemory)
    g = torch.Generator().manual_seed(138502)
    batch, feature = 4, 11
    new_keys = torch.randn(batch, PRODUCTION_K, feature, generator=g)
    new_values = torch.randn(batch, PRODUCTION_K, feature, generator=g)
    new_strengths = torch.rand(batch, PRODUCTION_K, generator=g)
    old_keys = torch.randn(batch, PRODUCTION_CAPACITY, feature, generator=g)
    old_values = torch.randn(batch, PRODUCTION_CAPACITY, feature, generator=g)
    old_strengths = torch.rand(batch, PRODUCTION_CAPACITY, generator=g)

    patterns = [
        (
            torch.zeros(batch, PRODUCTION_K, dtype=torch.bool),
            torch.zeros(batch, PRODUCTION_CAPACITY, dtype=torch.bool),
        ),
        (
            torch.ones(batch, PRODUCTION_K, dtype=torch.bool),
            torch.ones(batch, PRODUCTION_CAPACITY, dtype=torch.bool),
        ),
        (
            (torch.arange(PRODUCTION_K)[None, :] % 2 == 0).expand(batch, -1),
            (torch.arange(PRODUCTION_CAPACITY)[None, :] % 3 == 1).expand(batch, -1),
        ),
        (
            torch.rand(batch, PRODUCTION_K, generator=g) > 0.55,
            torch.rand(batch, PRODUCTION_CAPACITY, generator=g) > 0.35,
        ),
    ]
    for new_valid, keep_old in patterns:
        reference = _reference_tail(
            new_keys,
            new_values,
            new_strengths,
            new_valid,
            old_keys,
            old_values,
            old_strengths,
            keep_old,
            capacity=PRODUCTION_CAPACITY,
        )
        compact = memory._stable_compact_state(
            new_keys,
            new_values,
            new_strengths,
            new_valid,
            old_keys,
            old_values,
            old_strengths,
            keep_old,
        )
        # Invalid-slot payloads are deliberately random and must also survive in
        # exactly the same stable order; comparing only valid slots is insufficient.
        torch.testing.assert_close(reference.keys, compact.keys, rtol=0.0, atol=0.0)
        torch.testing.assert_close(reference.values, compact.values, rtol=0.0, atol=0.0)
        torch.testing.assert_close(reference.strengths, compact.strengths, rtol=0.0, atol=0.0)
        assert torch.equal(reference.valid, compact.valid)


def test_issue381_duplicate_heavy_projected_update_matches_current_v25_1():
    baseline, candidate = _models(138503)
    old_memory = baseline.stages[0].memory
    new_memory = candidate.stages[0].memory
    b, k = 3, PRODUCTION_K
    c, m, d = old_memory.capacity, old_memory.memory_dim, baseline.cfg.d_model
    g = torch.Generator().manual_seed(138504)

    old_keys = F.normalize(torch.randn(b, c, m, generator=g), dim=-1)
    old_values = torch.randn(b, c, m, generator=g)
    old_strengths = torch.rand(b, c, generator=g).clamp_min(0.05)
    old_valid = torch.rand(b, c, generator=g) > 0.30
    state = ContextualEpisodicMemoryState(old_keys, old_values, old_strengths, old_valid)

    projected = F.normalize(torch.randn(b, k, m, generator=g), dim=-1)
    # Force exact incoming duplicates and duplicates against prior state.
    projected[:, 1] = projected[:, 0]
    projected[:, 5] = projected[:, 4]
    projected[:, 9] = old_keys[:, 2]
    projected[:, 13] = old_keys[:, 7]
    payload = torch.randn(b, k, d, generator=g)
    strength = torch.rand(b, k, 1, generator=g)
    normalized_old = F.normalize(old_keys, dim=-1)

    old = old_memory._vectorized_update_from_projected(
        projected, normalized_old, payload, strength, state, detach_inputs=False
    )
    new = new_memory._vectorized_update_from_projected(
        projected, normalized_old, payload, strength, state, detach_inputs=False
    )
    torch.testing.assert_close(old.keys, new.keys, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(old.values, new.values, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(old.strengths, new.strengths, rtol=RTOL, atol=ATOL)
    assert torch.equal(old.valid, new.valid)


def test_issue381_stable_compaction_preserves_state_dict_and_full_model_memory_transition():
    baseline, candidate = _models(138505)
    assert list(baseline.state_dict()) == list(candidate.state_dict())
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0.0, atol=0.0)

    _force_all_stages_run(baseline)
    _force_all_stages_run(candidate)
    tokens = make_batch(3, EVAL_SEED + 138505).tokens[:, : 2 * baseline.cfg.chunk_size]
    with torch.no_grad():
        old = baseline(tokens, hard=True, route_mode="hard_sparse", update_memory=True)
        new = candidate(tokens, hard=True, route_mode="hard_sparse", update_memory=True)

    torch.testing.assert_close(old["logits"], new["logits"], rtol=RTOL, atol=ATOL)
    _assert_state_close(old["state"], new["state"])
    for old_stage, new_stage in zip(baseline.stages, candidate.stages):
        assert old_stage.last_candidate_count == new_stage.last_candidate_count
        assert old_stage.last_selected_count == new_stage.last_selected_count
        assert old_stage.last_vectorized_update_calls == new_stage.last_vectorized_update_calls == 1
        assert torch.equal(old_stage.last_selected_indices, new_stage.last_selected_indices)
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


def test_issue381_stable_compaction_preserves_differentiable_memory_gradients():
    baseline, candidate = _models(138506)
    baseline.set_memory_pretraining_mode(True)
    candidate.set_memory_pretraining_mode(True)
    old_stage = baseline.stages[0]
    new_stage = candidate.stages[0]
    g = torch.Generator().manual_seed(138507)
    events0 = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)
    events1 = torch.randn(2, baseline.cfg.chunk_size, baseline.cfg.d_model, generator=g)

    def run(stage):
        stage.zero_grad(set_to_none=True)
        out0, state0, _ = stage.forward_chunk(
            events0.clone().requires_grad_(True), None, hard=False, update_memory=True
        )
        out1, state1, _ = stage.forward_chunk(
            events1.clone().requires_grad_(True), state0, hard=False, update_memory=True
        )
        loss = (
            out0.float().square().mean()
            + out1.float().square().mean()
            + 0.01 * state1.memory.keys.float().square().mean()
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
        return out0, out1, state1, loss.detach(), grads

    old0, old1, old_state, old_loss, old_grads = run(old_stage)
    new0, new1, new_state, new_loss, new_grads = run(new_stage)
    torch.testing.assert_close(old0, new0, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(old1, new1, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(old_loss, new_loss, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(old_state.memory.keys, new_state.memory.keys, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(old_state.memory.values, new_state.memory.values, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        old_state.memory.strengths, new_state.memory.strengths, rtol=RTOL, atol=ATOL
    )
    assert torch.equal(old_state.memory.valid, new_state.memory.valid)
    for name in old_grads:
        assert old_grads[name] is not None, name
        assert new_grads[name] is not None, name
        torch.testing.assert_close(old_grads[name], new_grads[name], rtol=RTOL, atol=ATOL)


def test_issue381_compact_rebuild_has_no_concat_priority_topk_or_gather_tail_and_protocol_stays_frozen():
    source = inspect.getsource(
        StableCompactExecutionEquivalentFactorizedIdentityContextMemory._stable_compact_state
    )
    assert "torch.cat" not in source
    assert "torch.topk" not in source
    assert "_gather_slots" not in source
    assert "priority" not in source
    protocol = stable_compaction_v25_1_protocol()
    assert protocol["stable_compaction"] is True
    assert protocol["write_state_semantics_changed"] is False
    assert protocol["duplicate_semantics_changed"] is False
    assert protocol["incoming_order_changed"] is False
    assert protocol["learned_parameter_count_changed"] is False
    assert protocol["state_dict_schema_changed"] is False
    assert protocol["routing_policy_changed"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["100m_authorized"] is False
