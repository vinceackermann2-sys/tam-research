from __future__ import annotations

import types

import torch

from tam_research.aera import AERAState, FastMemoryState
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
    model = HardwareAwareAERATextLMV10(cfg()).train()
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


def test_straight_through_merge_restores_residual_and_state_dtype_and_gradient():
    torch.manual_seed(103)
    model = HardwareAwareAERATextLMV10(cfg()).train()
    stage = model.stages[0]
    router = model.stage_routers[0]

    def routed(self, first_event, stream, *, mode):
        assert mode == "straight_through"
        # Keep a live sigmoid-like tensor so the blend remains differentiable.
        logits = self.proj(torch.cat((first_event.float(), stream.float()), dim=-1))
        prob = torch.sigmoid(logits)
        hard = (prob >= 0.5).to(prob.dtype)
        gate = hard.detach() - prob.detach() + prob
        return gate, logits

    router.forward = types.MethodType(routed, router)

    def promoted(self, events, state, *, hard, update_memory):
        assert not hard
        y = events.float() + 0.25
        new_state = AERAState(
            stream=state.stream.float() + 0.5,
            memory=FastMemoryState(state.memory.matrix.float() + 0.75),
        )
        controls = {
            "start": {
                "expert_logits": torch.zeros(events.size(0), model.cfg.n_experts),
                "expert_count_logits": torch.zeros(events.size(0), 2),
            },
            "end": {"depth_logits": torch.zeros(events.size(0), model.cfg.max_reason_steps)},
        }
        return y, new_state, controls

    stage.forward_chunk = types.MethodType(promoted, stage)
    x = torch.randn(2, 8, model.cfg.d_model, dtype=torch.bfloat16)
    state = AERAState(
        stream=torch.randn(2, model.cfg.d_model, dtype=torch.bfloat16),
        memory=FastMemoryState(
            torch.randn(2, model.cfg.memory_dim, model.cfg.memory_dim, dtype=torch.bfloat16)
        ),
    )
    y, new_state, _ = model._route_one_stage(
        x,
        stage,
        state,
        router,
        route_mode="straight_through",
        update_memory=False,
    )
    assert y.dtype == torch.bfloat16
    assert new_state.stream.dtype == torch.bfloat16
    assert new_state.memory.matrix.dtype == torch.bfloat16
    y.float().sum().backward()
    assert router.proj.weight.grad is not None
    assert torch.isfinite(router.proj.weight.grad).all()


def test_forward_runtime_outputs_are_unchanged_shape():
    torch.manual_seed(104)
    model = HardwareAwareAERATextLMV10(cfg()).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 17))
    with torch.no_grad():
        out = model(tokens, hard=True, route_mode="hard_sparse")
    assert out["logits"].shape == (2, 17, model.cfg.vocab_size)
    assert torch.isfinite(out["logits"].float()).all()
