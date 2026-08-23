from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v12 import HardwareAwareAERATextLMV12


def _cfg(*, chunk_size: int = 8) -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=47,
        d_model=32,
        n_stages=4,
        n_heads=4,
        chunk_size=chunk_size,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=2,
    )


def test_v12_foundation_stage_is_frozen_and_hard_runs():
    torch.manual_seed(1201)
    model = HardwareAwareAERATextLMV12(_cfg()).eval()
    foundation = model.stage_routers[0]
    assert all(not parameter.requires_grad for parameter in foundation.parameters())
    assert abs(float(foundation.proj.bias) - 12.0) < 1e-7
    for router in model.stage_routers[1:]:
        assert torch.count_nonzero(router.proj.weight) == 0
        assert torch.count_nonzero(router.proj.bias) == 0

    tokens = torch.randint(0, model.cfg.vocab_size, (3, 8))
    with torch.no_grad():
        out = model(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
    routes = out["stage_routes"]
    assert isinstance(routes, list) and routes
    foundation_route = routes[0][0]
    assert foundation_route["executed_fraction"] == 1.0
    assert torch.all(foundation_route["stage_route_gate"] == 1)


def test_v12_difficulty_targets_allocate_more_optional_stages_to_harder_examples():
    losses = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
    targets = HardwareAwareAERATextLMV12.chunk_difficulty_stage_targets(losses)
    assert targets.shape == (6, 3)
    requested = targets.sum(dim=1)
    assert torch.all(requested[1:] >= requested[:-1])
    assert requested[0] == 0
    assert requested[-1] == 3
    # Reference target: ~50%, ~33%, ~17% optional-stage run rates.
    assert torch.equal(targets.sum(dim=0).to(torch.int64), torch.tensor([3, 2, 1]))


def test_v12_router_supervision_reaches_optional_routers_not_foundation():
    torch.manual_seed(1202)
    model = HardwareAwareAERATextLMV12(_cfg()).train()
    model.set_optional_stage_routers_trainable(True)
    tokens = torch.randint(0, model.cfg.vocab_size, (6, 8))
    out = model(tokens, route_mode="straight_through", update_memory=False)
    chunk_losses = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
    terms = model.soft_objective(
        tokens,
        out,
        chunk_losses=chunk_losses,
        event_weight=0.0,
        compute_weight=0.0,
        balance_weight=0.0,
        block_weight=0.0,
        stream_forecast_weight=0.0,
    )
    terms["total"].backward()

    for parameter in model.stage_routers[0].parameters():
        assert parameter.grad is None
    for router in model.stage_routers[1:]:
        grad = router.proj.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0
    assert abs(float(terms["optional_stage_target_fraction"]) - 1.0 / 3.0) < 1e-6
    assert torch.isfinite(terms["stage_difficulty_bce"])
    assert torch.isfinite(terms["stage_budget"])
    assert torch.isfinite(terms["stage_polarization"])


def test_v12_difficulty_supervision_can_learn_noncollapsed_hard_stage_policy():
    torch.manual_seed(1203)
    model = HardwareAwareAERATextLMV12(_cfg()).train()
    model.set_optional_stage_routers_trainable(True)

    batch = 60
    difficulty = torch.linspace(-2.0, 2.0, batch)
    first_event = torch.zeros(batch, model.cfg.d_model)
    first_event[:, 0] = difficulty
    stream = torch.zeros_like(first_event)
    losses = difficulty[:, None] + 3.0
    targets = model.chunk_difficulty_stage_targets(losses)
    optimizer = torch.optim.Adam(
        [parameter for router in model.stage_routers[1:] for parameter in router.parameters()],
        lr=0.08,
    )

    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        probabilities = []
        for router in model.stage_routers[1:]:
            gate, _ = router(first_event, stream, mode="soft")
            probabilities.append(gate)
        p = torch.cat(probabilities, dim=1)
        loss = F.binary_cross_entropy(p, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        hard = []
        for router in model.stage_routers[1:]:
            gate, _ = router(first_event, stream, mode="hard_sparse")
            hard.append(gate)
        decisions = torch.cat(hard, dim=1)
    accuracy = float((decisions == targets).float().mean())
    assert accuracy >= 0.95
    assert float(decisions[0].sum()) == 0.0
    assert float(decisions[-1].sum()) == 3.0
    assert 0.25 <= float(decisions.mean()) <= 0.42


def test_v12_carries_state_exactly_across_two_256_token_chunks():
    torch.manual_seed(1204)
    model = HardwareAwareAERATextLMV12(_cfg(chunk_size=256)).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, 512))
    with torch.no_grad():
        full = model(tokens, route_mode="straight_through", update_memory=False)
        first = model(tokens[:, :256], route_mode="straight_through", update_memory=False)
        second = model(
            tokens[:, 256:],
            state=first["state"],
            route_mode="straight_through",
            update_memory=False,
        )
    full_logits = full["logits"]
    second_logits = second["logits"]
    assert isinstance(full_logits, torch.Tensor) and isinstance(second_logits, torch.Tensor)
    assert torch.allclose(full_logits[:, 256:], second_logits, atol=2e-5, rtol=2e-5)


def test_v12_has_no_future_token_causality_regression_with_256_chunks():
    torch.manual_seed(1205)
    model = HardwareAwareAERATextLMV12(_cfg(chunk_size=256)).eval()
    a = torch.randint(0, model.cfg.vocab_size, (1, 512))
    b = a.clone()
    b[:, 400:] = (b[:, 400:] + 11) % model.cfg.vocab_size
    with torch.no_grad():
        ya = model(a, route_mode="straight_through", update_memory=False)["logits"]
        yb = model(b, route_mode="straight_through", update_memory=False)["logits"]
    assert isinstance(ya, torch.Tensor) and isinstance(yb, torch.Tensor)
    assert torch.allclose(ya[:, :400], yb[:, :400], atol=2e-5, rtol=2e-5)
