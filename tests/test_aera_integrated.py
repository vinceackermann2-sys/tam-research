import torch

from tam_research.aera_full import FullAERAConfig
from tam_research.aera_integrated import (
    BlockDraftHead,
    IntegratedAERATextLM,
    TrainableSparseExpertLayer,
)


def cfg() -> FullAERAConfig:
    return FullAERAConfig(
        vocab_size=23,
        d_model=16,
        n_stages=1,
        n_heads=4,
        local_window=8,
        max_seq_len=12,
        n_experts=4,
        top_k_experts=1,
        expert_mult=2,
        memory_dim=4,
        max_reason_steps=2,
        block_size=3,
    )


def test_top1_sparse_router_receives_task_gradient():
    torch.manual_seed(3001)
    layer = TrainableSparseExpertLayer(cfg())
    x = torch.randn(2, 5, 16)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    out = layer(x, logits)
    loss = out.square().mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
    stats = layer.stats()
    assert stats is not None
    assert stats["assignments"] == 2 * 5
    assert stats["active_fraction_of_experts_per_event"] == 0.25


def test_block_draft_head_has_offset_specific_predictions_and_gradients():
    torch.manual_seed(3002)
    c = cfg()
    model = IntegratedAERATextLM(c)
    hidden = torch.randn(2, 7, c.d_model, requires_grad=True)
    logits = model.block_draft(hidden, model.lm_head)
    assert logits.shape == (2, 7, c.block_size, c.vocab_size)
    assert not torch.allclose(logits[:, :, 0], logits[:, :, 1])
    logits.square().mean().backward()
    assert hidden.grad is not None
    assert float(hidden.grad.abs().sum()) > 0.0
    assert model.block_draft.offset.weight.grad is not None
    assert float(model.block_draft.offset.weight.grad.abs().sum()) > 0.0


def test_integrated_objective_backpropagates_into_router_and_block_head():
    torch.manual_seed(3003)
    c = cfg()
    model = IntegratedAERATextLM(c)
    tokens = torch.randint(0, c.vocab_size, (4, 12))
    out = model(tokens, return_block_logits=True)
    losses = model.objective(tokens, out)
    losses["total"].backward()
    router_grad = model.stages[0].controller.proj.weight.grad
    block_grad = model.block_draft.offset.weight.grad
    assert router_grad is not None and float(router_grad.abs().sum()) > 0.0
    assert block_grad is not None and float(block_grad.abs().sum()) > 0.0
    assert all(torch.isfinite(v) for v in losses.values())
