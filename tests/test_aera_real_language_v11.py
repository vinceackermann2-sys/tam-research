from __future__ import annotations

import torch

from tam_research.aera_real_language_v11 import (
    CHUNK_SIZE,
    DENSE_WARMUP_STEPS,
    ROUTER_CALIBRATION_END,
    SPARSE_CALIBRATION_EVERY,
    TOTAL_STEPS,
    aera_v11_config,
    build_aera,
    build_transformer,
    phase_for_step,
    route_mode_for_step,
    set_stage_router_trainable,
)


def test_v11_uses_two_large_chunks_per_sequence():
    cfg = aera_v11_config()
    assert CHUNK_SIZE == 256
    assert cfg.chunk_size == 256
    assert 512 // cfg.chunk_size == 2


def test_v11_progressive_routing_curriculum_is_frozen():
    assert 0 < DENSE_WARMUP_STEPS < ROUTER_CALIBRATION_END < TOTAL_STEPS
    assert SPARSE_CALIBRATION_EVERY == 4
    assert phase_for_step(0) == "representation_warmup"
    assert phase_for_step(DENSE_WARMUP_STEPS - 1) == "representation_warmup"
    assert phase_for_step(DENSE_WARMUP_STEPS) == "router_calibration"
    assert phase_for_step(ROUTER_CALIBRATION_END - 1) == "router_calibration"
    assert phase_for_step(ROUTER_CALIBRATION_END) == "mostly_hard_sparse"
    assert route_mode_for_step(0) == "straight_through"
    assert route_mode_for_step(DENSE_WARMUP_STEPS) == "straight_through"
    assert route_mode_for_step(ROUTER_CALIBRATION_END) == "straight_through"
    assert route_mode_for_step(ROUTER_CALIBRATION_END + 1) == "hard_sparse"


def test_v11_stage_router_freeze_only_changes_router_grad_flags():
    model = build_aera(torch.device("cpu"))
    others = [p.requires_grad for name, p in model.named_parameters() if not name.startswith("stage_routers.")]
    set_stage_router_trainable(model, False)
    assert all(not p.requires_grad for p in model.stage_routers.parameters())
    assert all(p.requires_grad for name, p in model.named_parameters() if not name.startswith("stage_routers."))
    set_stage_router_trainable(model, True)
    assert all(p.requires_grad for p in model.stage_routers.parameters())
    assert all(others)


def test_v11_stored_parameter_match_remains_within_five_percent():
    aera = build_aera(torch.device("cpu"))
    transformer = build_transformer(torch.device("cpu"))
    a = sum(p.numel() for p in aera.parameters())
    t = sum(p.numel() for p in transformer.parameters())
    assert abs(a - t) / t < 0.05
