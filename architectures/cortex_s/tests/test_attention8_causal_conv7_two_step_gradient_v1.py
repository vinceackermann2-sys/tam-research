"""CPU-only two-step autograd sanity check for the existing Conv7 block.

Not an LM benchmark, engineering run, scientific seed, Modal invocation, or
permission to train at 100M/2B. The synthetic regression target is intentionally
constructed to make the zero-initialized gate receive an unambiguous gradient.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from architectures.cortex_s.attention8_causal_conv7_v1 import (
    CausalConv7ScheduledBlock,
    KERNEL_SIZE,
    VARIANT,
)


def test_zero_gate_opens_before_depthwise_filters_receive_gradient():
    torch.manual_seed(91468)
    block = CausalConv7ScheduledBlock(index=0, variant=VARIANT).cpu().train()
    optimizer = torch.optim.AdamW(
        block.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    x = torch.randn(1, 9, 512)
    with torch.no_grad():
        normalized = block.norm_ff(x)
        local_mixer = block.depthwise_conv(
            F.pad(normalized.transpose(1, 2), (KERNEL_SIZE - 1, 0))
        ).transpose(1, 2)
        target = block(x).detach() + 0.4 * local_mixer.detach()
    original_conv = block.depthwise_conv.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    first_loss = F.mse_loss(block(x), target)
    first_loss.backward()
    assert torch.isfinite(first_loss)
    assert float(block.conv_gain.detach()) == 0.0
    assert block.conv_gain.grad is not None
    assert torch.isfinite(block.conv_gain.grad)
    assert abs(float(block.conv_gain.grad.detach())) > 1e-6
    assert block.depthwise_conv.weight.grad is not None
    assert torch.count_nonzero(block.depthwise_conv.weight.grad).item() == 0
    optimizer.step()
    opened_gain = float(block.conv_gain.detach())
    assert opened_gain > 0.0
    assert torch.isfinite(block.conv_gain.detach())

    optimizer.zero_grad(set_to_none=True)
    second_loss = F.mse_loss(block(x), target)
    second_loss.backward()
    assert torch.isfinite(second_loss)
    assert float(second_loss.detach()) < float(first_loss.detach())
    assert block.depthwise_conv.weight.grad is not None
    assert torch.isfinite(block.depthwise_conv.weight.grad).all()
    assert block.depthwise_conv.weight.grad.abs().sum().item() > 0
    optimizer.step()
    assert not torch.equal(block.depthwise_conv.weight.detach(), original_conv)


def test_two_step_trained_mixer_is_still_strictly_causal():
    torch.manual_seed(91469)
    block = CausalConv7ScheduledBlock(index=0, variant=VARIANT).cpu().train()
    optimizer = torch.optim.AdamW(block.parameters(), lr=3e-4)
    x = torch.randn(1, 11, 512)
    with torch.no_grad():
        norm = block.norm_ff(x)
        mix = block.depthwise_conv(
            F.pad(norm.transpose(1, 2), (KERNEL_SIZE - 1, 0))
        ).transpose(1, 2)
        target = block(x).detach() + mix.detach() * 0.4
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(block(x), target).backward()
        optimizer.step()
    block.eval()
    x_future_changed = x.detach().clone()
    x_future_changed[:, 8:, :] += 5
    with torch.no_grad():
        original = block(x)
        changed = block(x_future_changed)
    torch.testing.assert_close(original[:, :8], changed[:, :8], rtol=0, atol=0)


def test_cpu_smoke_does_not_add_any_gpu_training_entrypoint():
    from pathlib import Path
    source = Path(__file__).read_text(encoding="utf-8")
    for forbidden in ("import modal", "modal run", ".remote(", "gpu=", "cuda",
                      "h100_train_one(", "training_authorized = True"):
        assert forbidden not in source
