from __future__ import annotations

import copy

import torch

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from tam_research.aera_hardware_core_v17 import HardwareAwareAERATextLMV17
from tam_research.aera_v17_systems_backend import (
    GROUPED_MM_SELECTED_CROSSOVER,
    HybridMeasuredSparseExpertBank,
    LowOverheadHardSparseReasoner,
    install_v17_systems_backend,
    systems_backend_protocol,
)


def _depth_logits(chosen: torch.Tensor, steps: int = 4) -> torch.Tensor:
    logits = torch.full((chosen.numel(), steps), -7.0)
    logits.scatter_(1, (chosen - 1)[:, None], 7.0)
    return logits


def _cfg() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=97,
        d_model=16,
        n_stages=4,
        n_heads=4,
        chunk_size=8,
        n_experts=8,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=4,
        block_size=2,
    )


def test_low_overhead_reasoner_matches_reference_hard_cpu() -> None:
    torch.manual_seed(1)
    ref = DtypeSafeChunkLatentReasoner(12, 4).eval()
    candidate = LowOverheadHardSparseReasoner(12, 4).eval()
    candidate.load_state_dict(ref.state_dict(), strict=True)
    summary = torch.randn(8, 12)
    logits = _depth_logits(torch.tensor([1, 2, 3, 4, 1, 4, 2, 3]))
    with torch.no_grad():
        expected = ref(summary, logits, hard=True)
        actual = candidate(summary, logits, hard=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert candidate.last_steps is not None
    assert candidate.last_steps.device == summary.device
    assert candidate.last_expected is None


def test_low_overhead_reasoner_matches_reference_when_later_steps_empty_cpu() -> None:
    torch.manual_seed(2)
    ref = DtypeSafeChunkLatentReasoner(10, 4).eval()
    candidate = LowOverheadHardSparseReasoner(10, 4).eval()
    candidate.load_state_dict(ref.state_dict(), strict=True)
    summary = torch.randn(6, 10)
    logits = _depth_logits(torch.ones(6, dtype=torch.long))
    with torch.no_grad():
        expected = ref(summary, logits, hard=True)
        actual = candidate(summary, logits, hard=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_low_overhead_reasoner_soft_mode_is_reference_path_cpu() -> None:
    torch.manual_seed(3)
    ref = DtypeSafeChunkLatentReasoner(10, 4).eval()
    candidate = LowOverheadHardSparseReasoner(10, 4).eval()
    candidate.load_state_dict(ref.state_dict(), strict=True)
    summary = torch.randn(5, 10)
    logits = torch.randn(5, 4)
    with torch.no_grad():
        expected = ref(summary, logits, hard=False)
        actual = candidate(summary, logits, hard=False)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_hybrid_expert_kernel_choice_uses_measured_crossover() -> None:
    c = GROUPED_MM_SELECTED_CROSSOVER
    assert c == 64
    assert HybridMeasuredSparseExpertBank.choose_hard_kernel(
        c - 1, grouped_available=True
    ) == "bmm"
    assert HybridMeasuredSparseExpertBank.choose_hard_kernel(
        c, grouped_available=True
    ) == "grouped_mm"
    assert HybridMeasuredSparseExpertBank.choose_hard_kernel(
        c * 2, grouped_available=False
    ) == "bmm"


def test_installer_preserves_state_dict_keys_parameters_and_hard_output_cpu() -> None:
    torch.manual_seed(4)
    reference = HardwareAwareAERATextLMV17(_cfg()).eval()
    candidate = copy.deepcopy(reference).eval()

    keys_before = tuple(candidate.state_dict().keys())
    params_before = sum(p.numel() for p in candidate.parameters())
    install_v17_systems_backend(candidate)
    keys_after = tuple(candidate.state_dict().keys())
    params_after = sum(p.numel() for p in candidate.parameters())

    assert keys_after == keys_before
    assert params_after == params_before
    assert all(isinstance(stage.reasoner, LowOverheadHardSparseReasoner) for stage in candidate.stages)
    assert all(isinstance(stage.experts, HybridMeasuredSparseExpertBank) for stage in candidate.stages)

    tokens = torch.randint(0, _cfg().vocab_size, (3, 16))
    with torch.no_grad():
        ref_out = reference(tokens, hard=True, route_mode="hard_sparse", update_memory=False)
        candidate_out = candidate(tokens, hard=True, route_mode="hard_sparse", update_memory=False)

    torch.testing.assert_close(
        candidate_out["logits"], ref_out["logits"], rtol=1e-5, atol=1e-6
    )


def test_systems_backend_protocol_is_execution_only() -> None:
    p = systems_backend_protocol()
    assert p["architecture_changed"] is False
    assert p["routing_changed"] is False
    assert p["training_objective_changed"] is False
    assert p["checkpoint_weights_changed"] is False
    assert p["hard_selected_depth_changed"] is False
    assert p["hard_expert_selection_changed"] is False
    assert p["grouped_mm_selected_crossover"] == 64
