from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from architectures.cortex_s.grouped_moe import (
    PhysicalPaddedGroupedSparseMoE,
    convert_to_production_grouped,
    grouped_training_builder,
    padded_hidden,
)
from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig, TrulySparseMoE, parameter_count


def test_grouped_moe_matches_legacy_cpu_reference_and_gradients() -> None:
    torch.manual_seed(2026090905)
    legacy = TrulySparseMoE(d_model=16, num_experts=4, top_k=2, hidden=10)
    candidate = PhysicalPaddedGroupedSparseMoE(d_model=16, num_experts=4, top_k=2, hidden=10)
    candidate.copy_from_legacy(legacy)

    assert padded_hidden(10) == 16
    assert parameter_count(candidate) == parameter_count(legacy)

    x1 = torch.randn(3, 7, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    target = torch.randn(3, 7, 16)
    legacy_out = legacy(x1)
    candidate_out = candidate(x2)
    assert torch.allclose(candidate_out, legacy_out, rtol=1e-5, atol=1e-6)

    legacy_loss = F.mse_loss(legacy_out, target)
    candidate_loss = F.mse_loss(candidate_out, target)
    legacy_loss.backward()
    candidate_loss.backward()
    assert torch.allclose(x2.grad, x1.grad, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(x2.grad).all()


def test_conversion_preserves_parameters_and_does_not_advance_rng() -> None:
    cfg = CortexSLMConfig(
        vocab_size=257,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        state_size=8,
        num_experts=4,
        top_k=2,
        expert_hidden=10,
        attention_every=3,
    )
    torch.manual_seed(1234)
    model = CortexSLM(cfg)
    before_count = parameter_count(model)
    state_before = torch.random.get_rng_state().clone()
    converted = convert_to_production_grouped(model)
    state_after = torch.random.get_rng_state()

    assert parameter_count(converted) == before_count
    assert torch.equal(state_before, state_after)
    assert all(isinstance(block.moe, PhysicalPaddedGroupedSparseMoE) for block in converted.blocks)


def test_grouped_router_counts_are_kept_on_device_until_report_time() -> None:
    cfg = CortexSLMConfig(
        vocab_size=257,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        state_size=8,
        num_experts=4,
        top_k=2,
        expert_hidden=10,
        attention_every=2,
    )
    torch.manual_seed(55)
    model = convert_to_production_grouped(CortexSLM(cfg)).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        logits = model(tokens)
    assert torch.isfinite(logits).all()
    assert all(isinstance(block.moe.last_counts, torch.Tensor) for block in model.blocks)


def test_grouped_training_builder_is_scoped_and_restores_generic_builder() -> None:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.grouped_moe import build_production_grouped_cortex_100m

    original = training_module.build_cortex_100m
    with grouped_training_builder():
        assert training_module.build_cortex_100m is build_production_grouped_cortex_100m
    assert training_module.build_cortex_100m is original


def test_production_grouped_module_has_no_scientific_claim_or_seed_mutation() -> None:
    from architectures.cortex_s.experiments.scale100m_2b.protocol import protocol_snapshot

    snapshot = protocol_snapshot()
    assert snapshot["production_moe_backend"] == "physical_padded_grouped_bf16"
    assert snapshot["repair4_engineering_evidence"]["scientific_evidence"] is False
    assert snapshot["breakthrough_claim_allowed"] is False
    assert snapshot["continual_learning_claim_allowed"] is False
    json.dumps(snapshot)
