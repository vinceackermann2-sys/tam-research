from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v14 import HardwareAwareAERATextLMV14


def _model() -> HardwareAwareAERATextLMV14:
    cfg = HardwareAERAConfig(
        vocab_size=47,
        d_model=32,
        n_stages=4,
        n_heads=4,
        chunk_size=16,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=2,
    )
    return HardwareAwareAERATextLMV14(cfg)


def _batch(model: HardwareAwareAERATextLMV14, seed: int = 1401) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, model.cfg.vocab_size, (6, 16), generator=g)
    y = torch.randint(0, model.cfg.vocab_size, (6, 16), generator=g)
    return x, y


def _optional_router_grad_vector(model: HardwareAwareAERATextLMV14) -> torch.Tensor:
    pieces = []
    for router in model.stage_routers[1:]:
        for parameter in router.parameters():
            if parameter.grad is None:
                pieces.append(torch.zeros_like(parameter).reshape(-1))
            else:
                pieces.append(parameter.grad.detach().clone().reshape(-1))
    return torch.cat(pieces)


def test_v14_primary_task_loss_has_zero_optional_router_gradient_when_isolated():
    torch.manual_seed(1402)
    model = _model().eval()
    model.set_optional_stage_routers_trainable(True)
    model.set_router_task_gradient_isolation(True)
    x, y = _batch(model)

    out = model(x, hard=False, route_mode="straight_through", update_memory=False)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    primary = F.cross_entropy(logits.float().reshape(-1, model.cfg.vocab_size), y.reshape(-1))
    primary.backward()

    grad = _optional_router_grad_vector(model)
    assert torch.count_nonzero(grad) == 0


def test_v14_explicit_routing_supervision_still_trains_optional_routers():
    torch.manual_seed(1403)
    model = _model().eval()
    model.set_optional_stage_routers_trainable(True)
    model.set_router_task_gradient_isolation(True)
    x, _ = _batch(model, seed=1404)

    out = model(x, hard=False, route_mode="straight_through", update_memory=False)
    chunk_losses = torch.linspace(0.1, 1.0, x.size(0)).view(x.size(0), 1)
    terms = model.routing_supervision(out, chunk_losses)
    route_loss = (
        terms["stage_difficulty_bce"]
        + terms["stage_budget"]
        + terms["stage_polarization"]
    )
    route_loss.backward()

    grad = _optional_router_grad_vector(model)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
    for parameter in model.stage_routers[0].parameters():
        assert parameter.grad is None


def test_v14_even_1000x_task_pressure_cannot_change_router_gradient():
    torch.manual_seed(1405)
    model = _model().eval()
    model.set_optional_stage_routers_trainable(True)
    model.set_router_task_gradient_isolation(True)
    x, y = _batch(model, seed=1406)
    chunk_losses = torch.tensor([[0.1], [0.2], [0.3], [0.7], [0.8], [1.0]])

    out = model(x, hard=False, route_mode="straight_through", update_memory=False)
    terms = model.routing_supervision(out, chunk_losses)
    route_loss = (
        0.10 * terms["stage_difficulty_bce"]
        + 0.05 * terms["stage_budget"]
        + 0.01 * terms["stage_polarization"]
    )
    route_loss.backward()
    routing_only = _optional_router_grad_vector(model)

    model.zero_grad(set_to_none=True)
    out = model(x, hard=False, route_mode="straight_through", update_memory=False)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    primary = F.cross_entropy(logits.float().reshape(-1, model.cfg.vocab_size), y.reshape(-1))
    terms = model.routing_supervision(out, chunk_losses)
    combined = (
        1000.0 * primary
        + 0.10 * terms["stage_difficulty_bce"]
        + 0.05 * terms["stage_budget"]
        + 0.01 * terms["stage_polarization"]
    )
    combined.backward()
    with_task_pressure = _optional_router_grad_vector(model)

    assert float(routing_only.abs().sum()) > 0.0
    assert torch.allclose(with_task_pressure, routing_only, atol=1e-6, rtol=1e-5)


def test_v14_isolation_flag_does_not_change_hard_sparse_inference():
    torch.manual_seed(1407)
    model = _model().eval()
    x, _ = _batch(model, seed=1408)
    with torch.no_grad():
        model.set_router_task_gradient_isolation(False)
        ordinary = model(x, hard=True, route_mode="hard_sparse", update_memory=False)
        model.set_router_task_gradient_isolation(True)
        isolated_flag = model(x, hard=True, route_mode="hard_sparse", update_memory=False)
    ordinary_logits = ordinary["logits"]
    isolated_logits = isolated_flag["logits"]
    assert isinstance(ordinary_logits, torch.Tensor)
    assert isinstance(isolated_logits, torch.Tensor)
    assert torch.equal(ordinary_logits, isolated_logits)
