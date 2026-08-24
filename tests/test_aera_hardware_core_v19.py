from __future__ import annotations

import torch

from tam_research.aera import AERAState, FastMemoryState
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v18 import HardwareAwareAERATextLMV18
from tam_research.aera_hardware_core_v19 import (
    HardwareAwareAERATextLMV19,
    TokenwiseFastMemoryStage,
    memory_addressing_protocol,
)


def _small_cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=32,
        d_model=16,
        n_stages=4,
        n_heads=4,
        chunk_size=4,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=4,
        max_reason_steps=4,
        block_size=2,
    )


def test_v19_preserves_v18_checkpoint_parameters_and_memory_off_outputs() -> None:
    cfg = _small_cfg()
    torch.manual_seed(1)
    reference = HardwareAwareAERATextLMV18(cfg).eval()
    torch.manual_seed(1)
    candidate = HardwareAwareAERATextLMV19(cfg).eval()

    assert set(reference.state_dict()) == set(candidate.state_dict())
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in candidate.parameters())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], value, atol=0.0, rtol=0.0)

    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = reference(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
        got = candidate(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
    torch.testing.assert_close(got["logits"], ref["logits"], atol=0.0, rtol=0.0)


def test_tokenwise_memory_read_is_content_addressed_not_broadcast() -> None:
    cfg = _small_cfg()
    torch.manual_seed(2)
    model = HardwareAwareAERATextLMV19(cfg).eval()
    stage = model.stages[0]
    assert isinstance(stage, TokenwiseFastMemoryStage)

    with torch.no_grad():
        stage.memory.q.weight.zero_()
        stage.memory.out.weight.zero_()
        stage.memory.q.weight[: cfg.memory_dim, : cfg.memory_dim] = torch.eye(cfg.memory_dim)
        stage.memory.out.weight[: cfg.memory_dim, :] = torch.eye(cfg.memory_dim)

    h = torch.zeros(1, 2, cfg.d_model)
    h[0, 0, 0] = 1.0
    h[0, 1, 1] = 1.0
    state = AERAState(
        stream=torch.zeros(1, cfg.d_model),
        memory=FastMemoryState(torch.eye(cfg.memory_dim).unsqueeze(0)),
    )
    controls = {
        "state_read": torch.zeros(1, 1),
        "memory_read": torch.ones(1, 1),
    }
    context, recall = stage._tokenwise_context(h, state, controls)

    assert recall.shape == (1, 2, cfg.d_model)
    torch.testing.assert_close(context, recall, atol=0.0, rtol=0.0)
    assert not torch.equal(recall[:, 0], recall[:, 1])
    assert float(recall[0, 0, 0]) > 0.99
    assert float(recall[0, 1, 1]) > 0.99
    assert float(recall[0, 0, 1].abs()) < 1e-7
    assert float(recall[0, 1, 0].abs()) < 1e-7


def test_tokenwise_memory_read_does_not_leak_future_current_chunk_tokens() -> None:
    cfg = _small_cfg()
    torch.manual_seed(3)
    model = HardwareAwareAERATextLMV19(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (2, 8))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 1) % cfg.vocab_size

    with torch.no_grad():
        out_a = model(a, hard=False, route_mode="soft", update_memory=True)
        out_b = model(b, hard=False, route_mode="soft", update_memory=True)
    logits_a = out_a["logits"]
    logits_b = out_b["logits"]
    assert isinstance(logits_a, torch.Tensor) and isinstance(logits_b, torch.Tensor)
    # Only the changed final input position may differ. All prior logits must be
    # invariant, including positions in the same chunk as the changed future token.
    torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1], atol=0.0, rtol=0.0)


def test_v19_pretraining_gradient_reaches_memory_read_and_write_parameters() -> None:
    cfg = _small_cfg()
    torch.manual_seed(4)
    model = HardwareAwareAERATextLMV19(cfg).train()
    model.set_memory_pretraining_mode(True)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(tokens, hard=False, route_mode="soft", update_memory=True)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    loss = logits[:, cfg.chunk_size :].float().square().mean()
    loss.backward()
    model.set_memory_pretraining_mode(False)

    stage = model.stages[0]
    for parameter in (
        stage.memory.q.weight,
        stage.memory.k.weight,
        stage.memory.v.weight,
        stage.memory.out.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_v19_deployment_write_does_not_mutate_base_parameters() -> None:
    cfg = _small_cfg()
    torch.manual_seed(5)
    model = HardwareAwareAERATextLMV19(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    before = [p._version for p in model.parameters()]
    with torch.no_grad():
        out = model(tokens, hard=False, route_mode="soft", update_memory=True)
    after = [p._version for p in model.parameters()]
    assert before == after
    state = out["state"]
    assert state is not None
    assert any(float(s.memory.matrix.abs().sum()) > 0.0 for s in state.stages)


def test_v19_protocol_is_cpu_only_and_changes_only_read_addressing() -> None:
    protocol = memory_addressing_protocol()
    assert protocol["memory_equation_changed"] is False
    assert protocol["memory_write_rule_changed"] is False
    assert protocol["memory_write_timing_changed"] is False
    assert protocol["memory_dimension_changed"] is False
    assert protocol["stored_parameter_count_changed"] is False
    assert protocol["routing_changed_from_v17"] is False
    assert protocol["read_gate_changed"] is False
    assert protocol["read_granularity"] == "token-wise"
    assert protocol["memory_state_during_current_chunk"] == "fixed prior-chunk state"
    assert protocol["current_chunk_future_information_in_memory"] is False
    assert protocol["gpu_authorized"] is False
