from __future__ import annotations

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v14 import HardwareAwareAERATextLMV14
from tam_research.aera_hardware_core_v15 import HardwareAwareAERATextLMV15


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


def _route_output(probabilities: tuple[float, float, float], *, batch: int = 6, chunks: int = 2):
    history = []
    for _ in range(chunks):
        chunk = [
            {
                "stage_route_probability": torch.full((batch, 1), 0.999),
                "stage_route_gate": torch.ones(batch, 1),
            }
        ]
        for p in probabilities:
            chunk.append(
                {
                    "stage_route_probability": torch.full((batch, 1), p),
                    "stage_route_gate": torch.full((batch, 1), float(p >= 0.5)),
                }
            )
        history.append(chunk)
    return {"stage_routes": history}


def _chunk_losses() -> torch.Tensor:
    # Six unique ranks per chunk produce exact target fractions 3/6, 2/6, 1/6.
    base = torch.arange(6, dtype=torch.float32)
    return torch.stack((base, base.flip(0)), dim=1)


def test_v15_penalizes_budget_redistribution_that_v14_pooled_budget_misses() -> None:
    v14 = HardwareAwareAERATextLMV14(_cfg())
    v15 = HardwareAwareAERATextLMV15(_cfg())
    output = _route_output((0.999, 0.001, 0.001))
    losses = _chunk_losses()

    old = v14.routing_supervision(output, losses)
    new = v15.routing_supervision(output, losses)

    # 0.999 + 0.001 + 0.001 ~= 1.0, so the old pooled mean nearly matches
    # the intended total optional budget even though stages 2/3 have collapsed.
    assert float(old["stage_budget"]) < 1e-5
    assert float(new["stage_budget"]) > 0.10
    assert torch.allclose(
        new["optional_stage_target_fractions"],
        torch.tensor([0.5, 1.0 / 3.0, 1.0 / 6.0]),
        atol=1e-6,
    )


def test_v15_per_stage_budget_is_zero_at_existing_target_rates() -> None:
    model = HardwareAwareAERATextLMV15(_cfg())
    output = _route_output((0.5, 1.0 / 3.0, 1.0 / 6.0))
    terms = model.routing_supervision(output, _chunk_losses())
    assert float(terms["stage_budget"]) < 1e-10
    assert torch.allclose(terms["optional_stage_budget_errors"], torch.zeros(3), atol=1e-10)


def test_v15_does_not_change_parameter_count() -> None:
    v14 = HardwareAwareAERATextLMV14(_cfg())
    v15 = HardwareAwareAERATextLMV15(_cfg())
    assert sum(p.numel() for p in v14.parameters()) == sum(p.numel() for p in v15.parameters())
