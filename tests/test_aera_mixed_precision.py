import torch

from tam_research.aera_full import FullAERAConfig
from tam_research.aera_integrated import TrainableSparseExpertLayer


def test_trainable_sparse_expert_layer_preserves_bfloat16_accumulator_dtype():
    torch.manual_seed(6201)
    cfg = FullAERAConfig(
        d_model=32,
        n_experts=4,
        top_k_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
    )
    layer = TrainableSparseExpertLayer(cfg).to(dtype=torch.bfloat16)
    x = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    logits = torch.randn(2, 7, 4, dtype=torch.bfloat16, requires_grad=True)

    out = layer(x, logits)

    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()

    out.float().square().mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad.float()).all()
    assert float(logits.grad.float().abs().sum()) > 0.0
