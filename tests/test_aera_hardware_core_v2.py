from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v2 import HardwareAwareAERATextLMV2


def cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=31,
        d_model=32,
        n_stages=1,
        n_heads=4,
        chunk_size=7,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=3,
    )


def test_state_read_control_no_longer_attenuates_canonical_stream_residual():
    torch.manual_seed(11)
    model = HardwareAwareAERATextLMV2(cfg()).eval()
    first = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.chunk_size))
    second = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.chunk_size))
    with torch.no_grad():
        state = model(first, hard=False)["state"]
        stage = model.stages[0]
        # state_read is the final controller scalar. Alter only that output's bias.
        state_read_index = (
            model.cfg.n_experts + 2 + model.cfg.max_reason_steps
            + len(stage.controller.CONTROL_NAMES) - 1
        )
        original = stage.controller.proj.bias[state_read_index].clone()
        stage.controller.proj.bias[state_read_index] = -20.0
        low = model(second, state=state, hard=False)["logits"]
        stage.controller.proj.bias[state_read_index] = 20.0
        high = model(second, state=state, hard=False)["logits"]
        stage.controller.proj.bias[state_read_index] = original
    # The state_read telemetry signal no longer gates the core stream path.
    assert torch.allclose(low, high, atol=1e-5, rtol=1e-5)


def test_v2_remains_causal():
    torch.manual_seed(12)
    model = HardwareAwareAERATextLMV2(cfg()).eval()
    a = torch.randint(0, model.cfg.vocab_size, (1, 18))
    b = a.clone()
    b[:, 5:] = torch.randint(0, model.cfg.vocab_size, b[:, 5:].shape)
    with torch.no_grad():
        ya = model(a, hard=True)["logits"]
        yb = model(b, hard=True)["logits"]
    assert torch.allclose(ya[:, :5], yb[:, :5], atol=1e-5, rtol=1e-5)
