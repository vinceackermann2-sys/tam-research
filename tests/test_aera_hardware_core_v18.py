from __future__ import annotations

import torch

from tam_research.aera import FastMemoryState
from tam_research.aera_delta_memory import DeltaFastMemory
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v17 import HardwareAwareAERATextLMV17
from tam_research.aera_hardware_core_v18 import (
    HardwareAwareAERATextLMV18,
    PretrainableDeltaFastMemory,
    memory_pretraining_protocol,
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


def test_differentiable_update_matches_deployment_delta_rule() -> None:
    torch.manual_seed(1)
    reference = DeltaFastMemory(8, 4, lr=0.2, decay=0.999)
    candidate = PretrainableDeltaFastMemory(8, 4, lr=0.2, decay=0.999)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    x = torch.randn(3, 2, 8)
    strength = torch.sigmoid(torch.randn(3, 2, 1))
    state = FastMemoryState(torch.randn(3, 4, 4))

    local = reference.local_update(x, strength, state)
    differentiable = candidate.differentiable_update(x, strength, state)
    torch.testing.assert_close(differentiable.matrix.detach(), local.matrix, atol=1e-7, rtol=1e-6)


def test_differentiable_update_trains_memory_and_write_strength() -> None:
    torch.manual_seed(2)
    memory = PretrainableDeltaFastMemory(8, 4)
    state = memory.empty_state(2, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 2, 8, requires_grad=True)
    raw_strength = torch.randn(2, 2, 1, requires_grad=True)
    strength = torch.sigmoid(raw_strength)
    query = torch.randn(2, 1, 8)

    updated = memory.differentiable_update(x, strength, state)
    recalled = memory.read(query, updated)
    loss = recalled.square().mean()
    loss.backward()

    for parameter in (memory.q.weight, memory.k.weight, memory.v.weight, memory.out.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    assert raw_strength.grad is not None
    assert torch.isfinite(raw_strength.grad).all()
    assert float(raw_strength.grad.abs().sum()) > 0.0


def test_deployment_local_update_is_detached_and_does_not_mutate_parameters() -> None:
    torch.manual_seed(3)
    memory = PretrainableDeltaFastMemory(8, 4)
    before = {name: p.detach().clone() for name, p in memory.named_parameters()}
    state = memory.empty_state(2, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 2, 8, requires_grad=True)
    strength = torch.sigmoid(torch.randn(2, 2, 1, requires_grad=True))

    assert not memory.differentiable_pretraining
    updated = memory.local_update(x, strength, state)
    assert not updated.matrix.requires_grad
    for name, parameter in memory.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name], atol=0.0, rtol=0.0)


def test_v18_preserves_v17_checkpoint_layout_parameters_and_memory_disabled_outputs() -> None:
    cfg = _small_cfg()
    torch.manual_seed(4)
    reference = HardwareAwareAERATextLMV17(cfg).eval()
    torch.manual_seed(4)
    candidate = HardwareAwareAERATextLMV18(cfg).eval()

    assert set(reference.state_dict()) == set(candidate.state_dict())
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in candidate.parameters())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], value, atol=0.0, rtol=0.0)

    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = reference(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
        got = candidate(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
    torch.testing.assert_close(got["logits"], ref["logits"], atol=0.0, rtol=0.0)


def test_memory_write_is_causal_and_pretraining_matches_deployment_semantics() -> None:
    cfg = _small_cfg()
    torch.manual_seed(5)
    model = HardwareAwareAERATextLMV18(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))

    with torch.no_grad():
        off = model(tokens, hard=False, route_mode="soft", update_memory=False)

    model.set_memory_pretraining_mode(True)
    diff = model(tokens, hard=False, route_mode="soft", update_memory=True)
    model.set_memory_pretraining_mode(False)
    with torch.no_grad():
        local = model(tokens, hard=False, route_mode="soft", update_memory=True)

    # The write happens only after finishing chunk 0, so no token in chunk 0 can
    # depend on that write. Chunk 1 is allowed to read it.
    torch.testing.assert_close(diff["logits"][:, : cfg.chunk_size], off["logits"][:, : cfg.chunk_size], atol=0.0, rtol=0.0)
    later_delta = (diff["logits"][:, cfg.chunk_size :] - off["logits"][:, cfg.chunk_size :]).abs().max()
    assert float(later_delta) > 0.0

    # Differentiable pretraining and detached deployment use the same numerical
    # delta equation; only autograd behavior differs.
    torch.testing.assert_close(diff["logits"].detach(), local["logits"], atol=1e-6, rtol=1e-5)
    assert not model.memory_pretraining_mode()


def test_v18_protocol_does_not_claim_gpu_or_architecture_changes() -> None:
    protocol = memory_pretraining_protocol()
    assert protocol["memory_equation_changed"] is False
    assert protocol["stored_parameter_count_changed"] is False
    assert protocol["routing_changed_from_v17"] is False
    assert protocol["deployment_local_update_detached"] is True
    assert protocol["base_pretraining_update_differentiable"] is True
    assert protocol["gpu_authorized"] is False
