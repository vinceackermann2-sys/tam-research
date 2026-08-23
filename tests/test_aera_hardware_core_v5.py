from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v5 import (
    DtypeSafeChunkLatentReasoner,
    HardwareAwareAERATextLMV5,
)


def test_hard_latent_depth_survives_bfloat16_autocast_and_preserves_output_dtype():
    torch.manual_seed(51)
    reasoner = DtypeSafeChunkLatentReasoner(d_model=32, max_steps=4)
    summary = torch.randn(8, 32, dtype=torch.float32, requires_grad=True)
    depth_logits = torch.randn(8, 4, dtype=torch.float32, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = reasoner(summary, depth_logits, hard=True)
    assert out.dtype == summary.dtype
    assert out.shape == summary.shape
    out.square().mean().backward()
    assert summary.grad is not None


def test_soft_latent_depth_survives_bfloat16_autocast():
    torch.manual_seed(52)
    reasoner = DtypeSafeChunkLatentReasoner(d_model=32, max_steps=3)
    summary = torch.randn(6, 32, dtype=torch.float32)
    depth_logits = torch.randn(6, 3, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = reasoner(summary, depth_logits, hard=False)
    assert out.dtype == summary.dtype
    assert torch.isfinite(out).all()


def test_full_v5_hard_forward_runs_under_bfloat16_autocast():
    torch.manual_seed(53)
    cfg = HardwareAERAConfig(
        vocab_size=127,
        d_model=64,
        n_stages=1,
        n_heads=4,
        chunk_size=16,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=3,
        block_size=3,
    )
    model = HardwareAwareAERATextLMV5(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 31))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(tokens, hard=True, update_memory=False)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 31, cfg.vocab_size)
    assert torch.isfinite(logits).all()
