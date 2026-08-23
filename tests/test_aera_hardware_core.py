from __future__ import annotations

import torch

from tam_research.aera_hardware_core import (
    HardwareAERAConfig,
    HardwareAwareAERATextLM,
    StackedChunkExpertBank,
    hardware_parameter_accounting,
)


def tiny_cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=31,
        d_model=32,
        n_stages=1,
        n_heads=4,
        chunk_size=8,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=3,
        block_size=3,
    )


def test_hardware_core_shapes_and_block_head():
    torch.manual_seed(1)
    cfg = tiny_cfg()
    model = HardwareAwareAERATextLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 19))
    out = model(tokens, return_block_logits=True)
    assert out["logits"].shape == (2, 19, cfg.vocab_size)
    assert out["hidden"].shape == (2, 19, cfg.d_model)
    assert out["block_logits"].shape == (2, 19, cfg.block_size, cfg.vocab_size)
    assert len(out["state"].stages) == cfg.n_stages


def test_hardware_core_is_causal_within_and_across_chunks():
    torch.manual_seed(2)
    cfg = tiny_cfg()
    model = HardwareAwareAERATextLM(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 20))
    b = a.clone()
    b[:, 5:] = torch.randint(0, cfg.vocab_size, b[:, 5:].shape)
    with torch.no_grad():
        ya = model(a, hard=True)["logits"]
        yb = model(b, hard=True)["logits"]
    # Changing token 5 and everything after it must not affect logits through token 4.
    assert torch.allclose(ya[:, :5], yb[:, :5], atol=1e-5, rtol=1e-5)


def test_carried_state_changes_future_chunk_but_fresh_state_does_not_leak():
    torch.manual_seed(3)
    cfg = tiny_cfg()
    model = HardwareAwareAERATextLM(cfg).eval()
    first = torch.randint(0, cfg.vocab_size, (1, cfg.chunk_size))
    second = torch.randint(0, cfg.vocab_size, (1, cfg.chunk_size))
    with torch.no_grad():
        first_out = model(first, hard=True)
        carried = model(second, state=first_out["state"], hard=True)["logits"]
        fresh = model(second, hard=True)["logits"]
    assert not torch.allclose(carried, fresh)


def test_fast_memory_update_is_state_local_and_base_parameters_unchanged():
    torch.manual_seed(4)
    cfg = tiny_cfg()
    model = HardwareAwareAERATextLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.chunk_size))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        updated = model(tokens, hard=True, update_memory=True)["state"]
        fresh = model.empty_state(tokens)
    assert updated.stages[0].memory.matrix.abs().sum() > 0
    assert fresh.stages[0].memory.matrix.abs().sum() == 0
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])


def test_stacked_expert_bank_routes_gradients_to_expert_and_count_logits():
    torch.manual_seed(5)
    cfg = tiny_cfg()
    bank = StackedChunkExpertBank(cfg)
    x = torch.randn(3, 7, cfg.d_model, requires_grad=True)
    expert_logits = torch.randn(3, cfg.n_experts, requires_grad=True)
    count_logits = torch.randn(3, 2, requires_grad=True)
    y = bank(x, expert_logits, count_logits, hard=False)
    loss = y.square().mean()
    loss.backward()
    assert expert_logits.grad is not None and expert_logits.grad.abs().sum() > 0
    assert count_logits.grad is not None and count_logits.grad.abs().sum() > 0
    assert bank.w1.grad is not None and bank.w1.grad.abs().sum() > 0
    assert bank.w2.grad is not None and bank.w2.grad.abs().sum() > 0


def test_hard_expert_count_is_one_or_two():
    torch.manual_seed(6)
    cfg = tiny_cfg()
    bank = StackedChunkExpertBank(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    routes = torch.randn(2, cfg.n_experts)
    counts = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    bank(x, routes, counts, hard=True)
    stats = bank.stats()
    assert stats is not None
    assert stats["min_active_experts"] == 1
    assert stats["max_active_experts"] == 2
    assert stats["mean_active_experts"] == 1.5


def test_objective_backpropagates_controller_routing_and_depth():
    torch.manual_seed(7)
    cfg = tiny_cfg()
    model = HardwareAwareAERATextLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens, hard=False, return_block_logits=True)
    losses = model.objective(tokens, out)
    losses["total"].backward()
    grad = model.stages[0].controller.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_parameter_accounting_separates_stored_and_active():
    cfg = tiny_cfg()
    model = HardwareAwareAERATextLM(cfg)
    acc = hardware_parameter_accounting(model, mean_active_experts=1.5)
    assert acc["stored_parameters"] > acc["estimated_active_parameters_per_chunk"]
    assert 0 < acc["estimated_active_fraction"] < 1
