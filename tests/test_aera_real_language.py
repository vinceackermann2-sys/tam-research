from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v10 import HardwareAwareAERATextLMV10
from tam_research.aera_real_language import (
    CALIBRATION_WARMUP_STEPS,
    TOKEN_BUDGET,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    aera_matched_loss,
    aera_route_mode,
    build_aera,
    build_transformer,
    parameter_accounting,
)


def test_small_gate_has_exact_token_accounting_and_mostly_hard_training():
    assert TOKEN_BUDGET == 8_388_608
    assert TOKENS_PER_STEP == 16_384
    assert TOTAL_STEPS == 512
    modes = [aera_route_mode(i) for i in range(TOTAL_STEPS)]
    assert modes[:CALIBRATION_WARMUP_STEPS] == ["straight_through"] * 32
    assert modes.count("hard_sparse") == 420
    assert modes.count("straight_through") == 92


def test_aera_and_transformer_are_stored_parameter_matched():
    aera = build_aera(torch.device("cpu"))
    transformer = build_transformer(torch.device("cpu"))
    counts = parameter_accounting(aera, transformer)
    assert counts["aera_stored_parameters"] == 24_317_092
    assert counts["transformer_parameters"] == 24_940_288
    assert abs(counts["stored_parameter_delta_fraction"]) < 0.03
    predictive = counts["aera_predictive_state"]
    assert predictive["fraction_of_legacy"] < 0.01


def test_matched_primary_loss_uses_external_next_token_targets_and_router_gets_gradient():
    torch.manual_seed(8201)
    cfg = HardwareAERAConfig(
        vocab_size=61,
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
    model = HardwareAwareAERATextLMV10(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    y = (x + 7) % cfg.vocab_size
    total, terms, mode = aera_matched_loss(model, x, y, step=0)
    assert mode == "straight_through"
    assert torch.isfinite(total)
    assert torch.isfinite(terms["next_token"])
    total.backward()
    grad = model.stage_routers[0].proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_hard_sparse_real_language_loss_is_finite():
    torch.manual_seed(8202)
    cfg = HardwareAERAConfig(
        vocab_size=61,
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
    model = HardwareAwareAERATextLMV10(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    total, terms, mode = aera_matched_loss(
        model, x, y, step=CALIBRATION_WARMUP_STEPS + 1
    )
    assert mode == "hard_sparse"
    assert torch.isfinite(total)
    assert torch.isfinite(terms["stream_forecast"])
    total.backward()
    assert model.token_emb.weight.grad is not None
