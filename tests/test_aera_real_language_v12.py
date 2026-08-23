from __future__ import annotations

import math

import torch

from tam_research.aera_hardware_core_v12 import HardwareAwareAERATextLMV12
from tam_research.aera_real_language import SEQ_LEN, TOKEN_BUDGET, TOKENS_PER_STEP, TOTAL_STEPS
from tam_research.aera_real_language_v12 import (
    CHUNK_SIZE,
    CPU_DIAGNOSTIC_SEED,
    DENSE_WARMUP_STEPS,
    ROUTER_CALIBRATION_END,
    SPARSE_CALIBRATION_EVERY,
    aera_v12_config,
    build_aera,
    build_transformer,
    cpu_preflight,
    initial_cpu_fairness,
    per_chunk_language_loss,
    phase_for_step,
    route_mode_for_step,
    set_optional_stage_router_trainable,
)


def test_v12_preserves_256_chunk_geometry_and_small_dev_budget():
    cfg = aera_v12_config()
    assert CHUNK_SIZE == 256
    assert cfg.chunk_size == 256
    assert SEQ_LEN == 512
    assert SEQ_LEN // CHUNK_SIZE == 2
    assert TOKEN_BUDGET == 8_388_608
    assert TOKENS_PER_STEP == 16_384
    assert TOTAL_STEPS == 512


def test_v12_progressive_curriculum_is_cpu_frozen_before_any_gpu_runner_exists():
    assert 0 < DENSE_WARMUP_STEPS < ROUTER_CALIBRATION_END < TOTAL_STEPS
    assert SPARSE_CALIBRATION_EVERY == 4
    assert phase_for_step(0) == "representation_warmup"
    assert phase_for_step(DENSE_WARMUP_STEPS - 1) == "representation_warmup"
    assert phase_for_step(DENSE_WARMUP_STEPS) == "difficulty_router_calibration"
    assert phase_for_step(ROUTER_CALIBRATION_END - 1) == "difficulty_router_calibration"
    assert phase_for_step(ROUTER_CALIBRATION_END) == "mostly_hard_sparse"
    assert route_mode_for_step(0) == "straight_through"
    assert route_mode_for_step(DENSE_WARMUP_STEPS) == "straight_through"
    assert route_mode_for_step(ROUTER_CALIBRATION_END) == "straight_through"
    assert route_mode_for_step(ROUTER_CALIBRATION_END + 1) == "hard_sparse"


def test_v12_only_optional_stage_routers_can_be_unfrozen():
    model = build_aera(torch.device("cpu"))
    set_optional_stage_router_trainable(model, False)
    assert all(not p.requires_grad for router in model.stage_routers for p in router.parameters())
    set_optional_stage_router_trainable(model, True)
    assert all(not p.requires_grad for p in model.stage_routers[0].parameters())
    assert all(p.requires_grad for router in model.stage_routers[1:] for p in router.parameters())


def test_v12_stored_parameter_match_and_initial_nll_remain_fair():
    report = cpu_preflight()
    counts = report["parameter_accounting"]
    assert abs(counts["stored_parameter_delta_fraction"]) < 0.05
    assert report["gpu_authorized"] is False
    assert report["routing"]["target_mean_total_stage_execution"] == 0.50
    init = report["initialization_fairness"]
    assert abs(init["nll_gap"]) < 0.50
    assert abs(init["aera_initial_nll"] - math.log(50_257)) < 0.75


def test_v12_initial_fairness_is_deterministic_for_cpu_diagnostic_seed():
    first = initial_cpu_fairness(CPU_DIAGNOSTIC_SEED)
    second = initial_cpu_fairness(CPU_DIAGNOSTIC_SEED)
    assert first == second


def test_v12_per_chunk_language_loss_is_detached_and_preserves_two_chunk_shape():
    generator = torch.Generator().manual_seed(1221)
    logits = torch.randn(3, 512, 19, generator=generator, requires_grad=True)
    targets = torch.randint(0, 19, (3, 512), generator=generator)
    losses = per_chunk_language_loss(logits, targets)
    assert losses.shape == (3, 2)
    assert losses.requires_grad is False
    assert torch.isfinite(losses).all()


def test_v12_and_transformer_parameter_counts_are_still_comparable_directly():
    aera = build_aera(torch.device("cpu"))
    transformer = build_transformer(torch.device("cpu"))
    assert isinstance(aera, HardwareAwareAERATextLMV12)
    a = sum(parameter.numel() for parameter in aera.parameters())
    t = sum(parameter.numel() for parameter in transformer.parameters())
    assert abs(a - t) / t < 0.05
