from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v15 import HardwareAwareAERATextLMV15
from tam_research.aera_hardware_core_v16 import HardwareAwareAERATextLMV16


def _cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=32,
        d_model=16,
        n_stages=4,
        n_heads=4,
        chunk_size=8,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=2,
    )


def _chunk_losses() -> torch.Tensor:
    # Six ranks -> exact per-chunk target fractions 3/6, 2/6, 1/6.
    base = torch.arange(6, dtype=torch.float32)
    return torch.stack((base, base.flip(0)), dim=1)


def _constant_output(values: tuple[float, float, float], *, requires_grad: bool = False):
    history = []
    tensors = []
    for _ in range(2):
        chunk = [{"stage_route_probability": torch.full((6, 1), 0.999), "stage_route_gate": torch.ones(6, 1)}]
        for value in values:
            p = torch.full((6, 1), value, requires_grad=requires_grad)
            tensors.append(p)
            chunk.append({"stage_route_probability": p, "stage_route_gate": (p.detach() >= 0.5).float()})
        history.append(chunk)
    return {"stage_routes": history}, tensors


def _exact_hard_fraction_output():
    # Per chunk: stage1 3/6 hard runs, stage2 2/6, stage3 1/6.
    patterns = [
        torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]).unsqueeze(1),
        torch.tensor([0.9, 0.9, 0.1, 0.1, 0.1, 0.1]).unsqueeze(1),
        torch.tensor([0.9, 0.1, 0.1, 0.1, 0.1, 0.1]).unsqueeze(1),
    ]
    history = []
    for _ in range(2):
        chunk = [{"stage_route_probability": torch.full((6, 1), 0.999), "stage_route_gate": torch.ones(6, 1)}]
        for p in patterns:
            chunk.append({"stage_route_probability": p.clone(), "stage_route_gate": (p >= 0.5).float()})
        history.append(chunk)
    return {"stage_routes": history}


def test_v16_rejects_soft_budget_match_with_hard_middle_deep_collapse() -> None:
    v15 = HardwareAwareAERATextLMV15(_cfg())
    v16 = HardwareAwareAERATextLMV16(_cfg())
    output, _ = _constant_output((0.5, 1.0 / 3.0, 1.0 / 6.0))
    losses = _chunk_losses()

    old = v15.routing_supervision(output, losses)
    new = v16.routing_supervision(output, losses)

    assert float(old["stage_budget"]) < 1e-10
    assert torch.allclose(new["optional_stage_hard_run_fractions"], torch.tensor([1.0, 0.0, 0.0]))
    assert float(new["stage_budget"]) > 0.10


def test_v16_budget_zero_when_actual_hard_run_fractions_match_targets() -> None:
    model = HardwareAwareAERATextLMV16(_cfg())
    terms = model.routing_supervision(_exact_hard_fraction_output(), _chunk_losses())
    assert torch.allclose(
        terms["optional_stage_hard_run_fractions"],
        torch.tensor([0.5, 1.0 / 3.0, 1.0 / 6.0]),
        atol=1e-6,
    )
    assert float(terms["stage_budget"]) < 1e-10


def test_v16_hard_budget_keeps_straight_through_probability_gradient() -> None:
    model = HardwareAwareAERATextLMV16(_cfg())
    output, tensors = _constant_output((0.45, 0.30, 0.15), requires_grad=True)
    terms = model.routing_supervision(output, _chunk_losses())
    terms["stage_budget"].backward()
    grads = [p.grad for p in tensors]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0.0


def test_v16_does_not_change_parameter_count_from_v15() -> None:
    v15 = HardwareAwareAERATextLMV15(_cfg())
    v16 = HardwareAwareAERATextLMV16(_cfg())
    assert sum(p.numel() for p in v15.parameters()) == sum(p.numel() for p in v16.parameters())
