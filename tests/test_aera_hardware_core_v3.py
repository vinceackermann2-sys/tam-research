from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v3 import HardwareAwareAERATextLMV3


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


def test_v3_remains_strictly_causal_across_chunks():
    torch.manual_seed(21)
    model = HardwareAwareAERATextLMV3(cfg()).eval()
    a = torch.randint(0, model.cfg.vocab_size, (1, 18))
    b = a.clone()
    b[:, 9:] = torch.randint(0, model.cfg.vocab_size, b[:, 9:].shape)
    with torch.no_grad():
        ya = model(a, hard=True)["logits"]
        yb = model(b, hard=True)["logits"]
    assert torch.allclose(ya[:, :9], yb[:, :9], atol=1e-5, rtol=1e-5)


def test_v3_records_one_stream_per_stage_per_chunk():
    torch.manual_seed(22)
    model = HardwareAwareAERATextLMV3(cfg())
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 18))
    out = model(tokens)
    history = out["stream_history"]
    assert isinstance(history, list)
    assert len(history) == 3  # ceil(18 / 7)
    assert all(len(chunk_streams) == model.cfg.n_stages for chunk_streams in history)
    assert history[0][0].shape == (3, model.cfg.d_model)


def test_predictive_stream_loss_trains_forecast_head_and_recurrent_state():
    torch.manual_seed(23)
    model = HardwareAwareAERATextLMV3(cfg(), stream_forecast_tokens=3)
    tokens = torch.randint(0, model.cfg.vocab_size, (8, 21))
    out = model(tokens, return_block_logits=True)
    losses = model.objective(tokens, out, stream_forecast_weight=1.0)
    forecast = losses["stream_forecast"]
    assert forecast.ndim == 0
    assert float(forecast.detach()) > 0.0

    model.zero_grad(set_to_none=True)
    forecast.backward()
    head_grad = model.stream_forecast_heads[0].weight.grad
    stream_grad = model.stages[0].stream_cell.weight_ih.grad
    assert head_grad is not None and float(head_grad.abs().sum()) > 0.0
    assert stream_grad is not None and float(stream_grad.abs().sum()) > 0.0
