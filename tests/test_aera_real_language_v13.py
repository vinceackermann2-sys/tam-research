from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v12 import HardwareAwareAERATextLMV12
from tam_research.aera_real_language_v13 import (
    DENSE_WARMUP_STEPS,
    ROUTER_CALIBRATION_END,
    SPARSE_CALIBRATION_EVERY,
    aera_matched_loss,
    cpu_preflight,
    router_policy_for_step,
)


def _model() -> HardwareAwareAERATextLMV12:
    cfg = HardwareAERAConfig(
        vocab_size=47,
        d_model=32,
        n_stages=4,
        n_heads=4,
        chunk_size=256,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=2,
    )
    return HardwareAwareAERATextLMV12(cfg)


def _batch(model: HardwareAwareAERATextLMV12, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, model.cfg.vocab_size, (6, 256), generator=g)
    y = torch.randint(0, model.cfg.vocab_size, (6, 256), generator=g)
    return x, y


def test_v13_router_policy_keeps_sparse_calibration_supervised_and_hard_steps_frozen():
    warmup = router_policy_for_step(DENSE_WARMUP_STEPS - 1)
    dense_cal = router_policy_for_step(DENSE_WARMUP_STEPS)
    last_dense_cal = router_policy_for_step(ROUTER_CALIBRATION_END - 1)
    sparse_cal = router_policy_for_step(ROUTER_CALIBRATION_END)
    hard_sparse = router_policy_for_step(ROUTER_CALIBRATION_END + 1)
    next_sparse_cal = router_policy_for_step(
        ROUTER_CALIBRATION_END + SPARSE_CALIBRATION_EVERY
    )

    assert warmup["trainable"] is False and warmup["supervised"] is False
    assert dense_cal["trainable"] is True and dense_cal["supervised"] is True
    assert last_dense_cal["trainable"] is True and last_dense_cal["supervised"] is True
    assert sparse_cal["route_mode"] == "straight_through"
    assert sparse_cal["trainable"] is True and sparse_cal["supervised"] is True
    assert hard_sparse["route_mode"] == "hard_sparse"
    assert hard_sparse["trainable"] is False and hard_sparse["supervised"] is False
    assert next_sparse_cal["route_mode"] == "straight_through"
    assert next_sparse_cal["trainable"] is True and next_sparse_cal["supervised"] is True


def test_v13_sparse_calibration_has_explicit_budget_loss_and_router_gradients():
    torch.manual_seed(1301)
    model = _model().train()
    x, y = _batch(model, seed=1311)

    total, terms, mode, phase = aera_matched_loss(
        model,
        x,
        y,
        step=ROUTER_CALIBRATION_END,
    )
    assert mode == "straight_through"
    assert phase == "mostly_hard_sparse"
    assert "stage_difficulty_bce" in terms
    assert "stage_budget" in terms
    assert "stage_polarization" in terms
    total.backward()

    for parameter in model.stage_routers[0].parameters():
        assert parameter.grad is None
    for router in model.stage_routers[1:]:
        assert all(parameter.requires_grad for parameter in router.parameters())
        grad = router.proj.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0


def test_v13_hard_sparse_task_steps_cannot_pull_optional_router_budget_upward():
    torch.manual_seed(1302)
    model = _model().train()
    x, y = _batch(model, seed=1312)

    total, terms, mode, phase = aera_matched_loss(
        model,
        x,
        y,
        step=ROUTER_CALIBRATION_END + 1,
    )
    assert mode == "hard_sparse"
    assert phase == "mostly_hard_sparse"
    assert "stage_difficulty_bce" not in terms
    total.backward()

    for router in model.stage_routers[1:]:
        assert all(not parameter.requires_grad for parameter in router.parameters())
        for parameter in router.parameters():
            assert parameter.grad is None


def test_v13_cpu_preflight_records_no_gpu_authorization_and_router_policy():
    result = cpu_preflight()
    assert result["gpu_authorized"] is False
    assert result["version"] == "aera-v13-persistent-router-budget"
    policy = result["router_update_policy"]
    assert policy["first_sparse_calibration"]["supervised"] is True
    assert policy["first_hard_sparse"]["trainable"] is False
