from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v16 import HardwareAwareAERATextLMV16
from tam_research.aera_hardware_core_v17 import HardwareAwareAERATextLMV17


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


def _output_from_stage_probabilities(stage_probabilities: list[torch.Tensor]):
    batch = stage_probabilities[0].numel()
    history = []
    for _ in range(2):
        chunk = [
            {
                "stage_route_probability": torch.full((batch, 1), 0.999),
                "stage_route_gate": torch.ones(batch, 1),
            }
        ]
        for values in stage_probabilities:
            p = values.reshape(batch, 1)
            chunk.append(
                {
                    "stage_route_probability": p,
                    "stage_route_gate": (p.detach() >= 0.5).float(),
                }
            )
        history.append(chunk)
    return {"stage_routes": history}


def test_legacy_microbatch8_targets_are_quantized_away_from_nominal_rates() -> None:
    model = HardwareAwareAERATextLMV16(_cfg())
    losses = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    targets = model.chunk_difficulty_stage_targets(losses).reshape(8, 1, 3)
    fractions = targets.mean(dim=(0, 1))
    assert torch.allclose(fractions, torch.tensor([0.5, 0.375, 0.125]))
    nominal = torch.tensor(model.OPTIONAL_STAGE_RUN_RATES)
    assert not torch.allclose(fractions, nominal)


def test_pairwise_teacher_prefers_router_scores_ordered_by_difficulty() -> None:
    losses = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    ordered = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1).repeat(1, 1, 3)
    reversed_scores = ordered.flip(0)
    good = HardwareAwareAERATextLMV17.pairwise_difficulty_ranking_loss(ordered, losses)
    bad = HardwareAwareAERATextLMV17.pairwise_difficulty_ranking_loss(reversed_scores, losses)
    assert float(good) < float(bad)


def test_pairwise_teacher_uses_all_examples_and_backpropagates_to_every_stage() -> None:
    losses = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.2, 0.7, 0.3, 0.6]).unsqueeze(1)
    logits = torch.zeros(8, 1, 3, requires_grad=True)
    loss = HardwareAwareAERATextLMV17.pairwise_difficulty_ranking_loss(logits, losses)
    loss.backward()
    assert logits.grad is not None
    per_stage = logits.grad.abs().sum(dim=(0, 1))
    assert torch.all(per_stage > 0)


def test_v17_budget_uses_exact_nominal_rates_not_quantized_binary_means() -> None:
    model = HardwareAwareAERATextLMV17(_cfg())
    base = torch.linspace(0.2, 0.8, 8)
    output = _output_from_stage_probabilities([base, base.clone(), base.clone()])
    losses = torch.stack((torch.arange(8), torch.arange(8).flip(0)), dim=1).float()
    terms = model.routing_supervision(output, losses)
    assert torch.allclose(
        terms["optional_stage_target_fractions"],
        torch.tensor([0.5, 1.0 / 3.0, 1.0 / 6.0]),
        atol=1e-7,
    )


def test_v17_keeps_hard_budget_gradient_and_difficulty_ranking_gradient() -> None:
    model = HardwareAwareAERATextLMV17(_cfg())
    p1 = torch.linspace(0.35, 0.65, 8, requires_grad=True)
    p2 = torch.linspace(0.30, 0.55, 8, requires_grad=True)
    p3 = torch.linspace(0.20, 0.45, 8, requires_grad=True)
    output = _output_from_stage_probabilities([p1, p2, p3])
    losses = torch.stack((torch.arange(8), torch.arange(8).flip(0)), dim=1).float()
    terms = model.routing_supervision(output, losses)
    combined = terms["stage_difficulty_rank"] + terms["stage_budget"]
    combined.backward()
    for p in (p1, p2, p3):
        assert p.grad is not None
        assert float(p.grad.abs().sum()) > 0.0


def test_v17_does_not_change_parameter_count_from_v16() -> None:
    v16 = HardwareAwareAERATextLMV16(_cfg())
    v17 = HardwareAwareAERATextLMV17(_cfg())
    assert sum(p.numel() for p in v16.parameters()) == sum(p.numel() for p in v17.parameters())
