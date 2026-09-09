from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig, TrulySparseMoE, parameter_count
from architectures.cortex_s.experiments.scale100m_2b.systems_microbench_v1 import (
    CLASSIFICATION,
    CONSUMED_ENGINEERING_SEEDS,
    ENGINEERING_SEED,
    FORBIDDEN_SCIENTIFIC_SEEDS,
    MEASURED_STEPS,
    MIN_PROMISING_SPEEDUP,
    PackedBF16GroupedSparseMoE,
    STOP_GROUPED_BELOW_SPEEDUP,
    TRIGGER_TITLE,
    convert_to_packed_bf16_grouped,
    validate_microbench_protocol,
)


def tiny_config() -> CortexSLMConfig:
    return CortexSLMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        state_size=8,
        num_experts=4,
        top_k=2,
        expert_hidden=24,
        attention_every=2,
    )


def test_microbench_seed_namespace_is_engineering_only_and_fresh_against_known_seeds():
    protocol = validate_microbench_protocol()
    assert protocol["classification"] == CLASSIFICATION == "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
    assert ENGINEERING_SEED not in set(CONSUMED_ENGINEERING_SEEDS)
    assert ENGINEERING_SEED not in set(FORBIDDEN_SCIENTIFIC_SEEDS)
    assert 8_100 in FORBIDDEN_SCIENTIFIC_SEEDS
    assert 910_001 in CONSUMED_ENGINEERING_SEEDS
    assert {48_131, 48_132, 48_133}.issubset(set(FORBIDDEN_SCIENTIFIC_SEEDS))
    assert TRIGGER_TITLE == "[modal-cortex-s-100m-systems-microbench-v1]"
    assert protocol["full_training_authorized"] is False
    assert MEASURED_STEPS == 20
    assert STOP_GROUPED_BELOW_SPEEDUP == 1.10
    assert MIN_PROMISING_SPEEDUP == 1.20


def test_packed_candidate_matches_legacy_forward_and_backward_on_cpu():
    torch.manual_seed(6001)
    legacy = TrulySparseMoE(d_model=24, num_experts=8, top_k=2, hidden=16)
    candidate = PackedBF16GroupedSparseMoE(d_model=24, num_experts=8, top_k=2, hidden=16)
    candidate.copy_from_legacy(legacy)
    assert parameter_count(candidate) == parameter_count(legacy)

    x_legacy = torch.randn(3, 11, 24, requires_grad=True)
    x_candidate = x_legacy.detach().clone().requires_grad_(True)
    y_legacy = legacy(x_legacy)
    y_candidate = candidate(x_candidate)
    torch.testing.assert_close(y_candidate, y_legacy, atol=2e-6, rtol=2e-6)

    loss_legacy = y_legacy.square().mean()
    loss_candidate = y_candidate.square().mean()
    loss_legacy.backward()
    loss_candidate.backward()
    torch.testing.assert_close(x_candidate.grad, x_legacy.grad, atol=3e-6, rtol=3e-6)

    # Compare corresponding expert gradients despite the benchmark-friendly
    # transposed storage layout.
    for index, expert in enumerate(legacy.experts):
        torch.testing.assert_close(
            candidate.expert_w1.grad[index],
            expert[0].weight.grad.T,
            atol=3e-6,
            rtol=3e-6,
        )
        torch.testing.assert_close(
            candidate.expert_w2.grad[index],
            expert[2].weight.grad.T,
            atol=3e-6,
            rtol=3e-6,
        )
    torch.testing.assert_close(candidate.router.weight.grad, legacy.router.weight.grad, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(candidate.router.bias.grad, legacy.router.bias.grad, atol=3e-6, rtol=3e-6)
    assert candidate.theoretical_executed_fraction == 0.25


def test_full_model_conversion_preserves_parameter_count_logits_and_state_on_cpu():
    torch.manual_seed(6002)
    legacy = CortexSLM(tiny_config()).eval()
    candidate = deepcopy(legacy)
    before = parameter_count(candidate)
    candidate = convert_to_packed_bf16_grouped(candidate).eval()
    assert parameter_count(candidate) == before == parameter_count(legacy)
    assert all(isinstance(block.moe, PackedBF16GroupedSparseMoE) for block in candidate.blocks)

    tokens = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        legacy_logits, legacy_state = legacy(tokens, return_state=True)
        candidate_logits, candidate_state = candidate(tokens, return_state=True)
    torch.testing.assert_close(candidate_logits, legacy_logits, atol=3e-6, rtol=3e-6)
    for candidate_layer, legacy_layer in zip(candidate_state, legacy_state):
        torch.testing.assert_close(candidate_layer, legacy_layer, atol=3e-6, rtol=3e-6)


def test_candidate_preserves_exact_topk_router_choices():
    torch.manual_seed(6003)
    legacy = TrulySparseMoE(d_model=32, num_experts=8, top_k=2, hidden=20)
    candidate = PackedBF16GroupedSparseMoE(d_model=32, num_experts=8, top_k=2, hidden=20)
    candidate.copy_from_legacy(legacy)
    x = torch.randn(57, 32)
    legacy_indices = legacy.router(x).topk(2, dim=-1).indices
    candidate_indices = candidate.router(x).topk(2, dim=-1).indices
    assert torch.equal(candidate_indices, legacy_indices)

    with torch.no_grad():
        candidate(x.view(3, 19, 32))
    assert candidate.last_counts is not None
    assert int(candidate.last_counts.sum()) == 57 * 2
    assert int(candidate.last_counts.sum()) < 57 * 8


def test_bf16_casts_are_differentiable_on_cpu_reference_graph():
    # CPU uses the reference grouped path, but explicit dtype casts themselves must
    # preserve autograd to FP32 master tensors because the CUDA candidate relies on it.
    torch.manual_seed(6004)
    weight = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
    x = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    y = x.to(torch.bfloat16) @ weight.to(torch.bfloat16)
    loss = y.float().square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_grouped_cuda_execution_keeps_bf16_output_between_expert_gemms(monkeypatch):
    candidate = PackedBF16GroupedSparseMoE(d_model=8, num_experts=2, top_k=1, hidden=4)
    packed = torch.randn(7, 8, dtype=torch.float32, requires_grad=True)
    matrices = torch.randn(2, 8, 4, dtype=torch.float32, requires_grad=True)
    offsets = torch.tensor([3, 7], dtype=torch.int32)
    seen: dict[str, torch.dtype] = {}

    monkeypatch.setattr(candidate, "grouped_cuda_available", lambda _: True)

    def fake_grouped_mm(left, right, *, offs):
        seen["left"] = left.dtype
        seen["right"] = right.dtype
        assert torch.equal(offs, offsets)
        # Shape-only stand-in for the CUDA kernel; the dtype contract is what this
        # zero-credit test freezes before the H100 benchmark.
        return torch.zeros(left.size(0), right.size(-1), dtype=torch.bfloat16)

    monkeypatch.setattr(F, "grouped_mm", fake_grouped_mm, raising=False)
    out = candidate._grouped_mm(packed, matrices, offsets)
    assert seen == {"left": torch.bfloat16, "right": torch.bfloat16}
    assert out.dtype == torch.bfloat16


def test_candidate_parameter_layout_is_grouped_mm_friendly():
    candidate = PackedBF16GroupedSparseMoE(d_model=24, num_experts=8, top_k=2, hidden=16)
    assert candidate.expert_w1.shape == (8, 24, 16)
    assert candidate.expert_w2.shape == (8, 16, 24)
    assert candidate.expert_w1.is_contiguous()
    assert candidate.expert_w2.is_contiguous()


def test_modal_app_has_only_single_bounded_engineering_h100_entrypoint():
    repo = Path(__file__).resolve().parents[3]
    app = (repo / "modal_cortex_s_100m_systems_microbench_v1.py").read_text(encoding="utf-8")
    assert "H100_TIMEOUT_SECONDS = 15 * 60" in app
    assert "def h100_microbenchmark(" in app
    assert "def full_2b(" not in app
    assert "train_full_2b" not in app
    assert "full-v2" not in app
    assert "ENGINEERING_SEED = 2_026_090_901" in app
    assert "H100_DISPATCH_CONSUMED.json" in app


def test_workflow_has_no_manual_dispatch_and_only_exact_owner_issue_trigger():
    repo = Path(__file__).resolve().parents[3]
    workflow = (repo / ".github/workflows/modal-cortex-s-100m-systems-microbench-v1.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch" not in workflow
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "github.event.issue.title == '[modal-cortex-s-100m-systems-microbench-v1]'" in workflow
    assert "git rev-parse origin/main" in workflow
    assert "modal_cortex_s_100m_systems_microbench_v1.py" in workflow
