from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v10 import HardwareAwareAERATextLMV10


def cfg(vocab_size: int = 101) -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_stages=2,
        n_heads=4,
        chunk_size=8,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=2,
    )


def test_tied_forecast_heads_remove_vocab_scaled_parameter_cost():
    model = HardwareAwareAERATextLMV10(cfg(vocab_size=50_257), stream_forecast_tokens=4)
    accounting = model.predictive_head_accounting()
    assert accounting["tied_forecast_parameters"] == 2 * 32 * 4 * 32
    assert accounting["legacy_equivalent_parameters"] == 2 * 32 * 4 * 50_257
    assert accounting["fraction_of_legacy"] < 0.001
    assert not hasattr(model, "stream_forecast_heads")


def test_tied_stream_forecast_loss_is_finite_and_trains_projector_and_embedding():
    torch.manual_seed(101)
    model = HardwareAwareAERATextLMV10(cfg()).train()
    # Force stages on so the state path is definitely exercised in this unit test.
    with torch.no_grad():
        for router in model.stage_routers:
            router.proj.weight.zero_()
            router.proj.bias.fill_(10.0)
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 24))
    out = model(tokens, route_mode="straight_through", return_block_logits=False)
    terms = model.soft_objective(
        tokens,
        out,
        event_weight=0.0,
        block_weight=0.0,
        stream_forecast_weight=0.2,
    )
    assert torch.isfinite(terms["stream_forecast"])
    assert float(terms["stream_forecast"]) > 0.0
    terms["total"].backward()
    grad = model.stream_forecast_projectors[0].proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
    assert model.token_emb.weight.grad is not None
    assert torch.isfinite(model.token_emb.weight.grad).all()


def test_hard_sparse_task_loss_uses_tied_forecast_without_legacy_head():
    torch.manual_seed(102)
    model = HardwareAwareAERATextLMV10(cfg(n_stages=2) if False else cfg()).train()
    with torch.no_grad():
        for router in model.stage_routers:
            router.proj.weight.zero_()
            router.proj.bias.fill_(10.0)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 24))
    out = model(tokens, hard=True, route_mode="hard_sparse")
    terms = model.hard_sparse_task_loss(
        tokens,
        out,
        event_weight=0.0,
        block_weight=0.0,
        stream_forecast_weight=0.2,
    )
    assert torch.isfinite(terms["total"])
    assert torch.isfinite(terms["stream_forecast"])


def test_forward_runtime_outputs_are_unchanged_shape():
    torch.manual_seed(103)
    model = HardwareAwareAERATextLMV10(cfg()).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 17))
    with torch.no_grad():
        out = model(tokens, hard=True, route_mode="hard_sparse")
    assert out["logits"].shape == (2, 17, model.cfg.vocab_size)
    assert torch.isfinite(out["logits"].float()).all()
