from __future__ import annotations

from pathlib import Path
import sys

import torch

# The dedicated CORTEX-S workflow runs from architectures/cortex_s with PYTHONPATH=.
# Add the repository root so this systems-only test can use package imports without
# changing the established workflow environment used by the older tests.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architectures.cortex_s.language_model import TrulySparseMoE, affine_scan, parameter_count
from architectures.cortex_s.systems_optimization import (
    GroupedSparseMoECandidate,
    affine_scan_slice_candidate,
    pack_topk_assignments,
)


def _stack_legacy_grads(legacy: TrulySparseMoE, projection: int) -> torch.Tensor:
    grads = []
    for expert in legacy.experts:
        grad = expert[projection].weight.grad
        assert grad is not None
        grads.append(grad)
    return torch.stack(grads)


def test_routing_pack_preserves_exactly_top_k_assignments() -> None:
    top_indices = torch.tensor(
        [
            [3, 1],
            [0, 3],
            [2, 1],
            [3, 2],
        ],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [
            [0.7, 0.3],
            [0.4, 0.6],
            [0.8, 0.2],
            [0.55, 0.45],
        ],
        dtype=torch.float32,
    )
    plan = pack_topk_assignments(top_indices, weights, num_experts=4)

    assert int(plan.counts.sum()) == top_indices.numel()
    assert int(plan.offsets[-1]) == top_indices.numel()
    assert plan.offsets.dtype == torch.int32
    assert plan.counts.tolist() == [1, 2, 2, 3]

    # Reconstruct the multiset of (token, expert, weight) assignments and compare
    # it with the ungrouped router output.  Grouping is physical ordering only.
    grouped_experts = torch.repeat_interleave(torch.arange(4), plan.counts)
    observed = sorted(
        (int(token), int(expert), float(weight))
        for token, expert, weight in zip(
            plan.token_indices,
            grouped_experts,
            plan.assignment_weights,
        )
    )
    expected = sorted(
        (token, int(top_indices[token, slot]), float(weights[token, slot]))
        for token in range(top_indices.size(0))
        for slot in range(top_indices.size(1))
    )
    assert observed == expected


def test_grouped_moe_candidate_matches_legacy_forward_and_backward_on_cpu() -> None:
    torch.manual_seed(77101)
    legacy = TrulySparseMoE(d_model=16, num_experts=4, top_k=2, hidden=12)
    candidate = GroupedSparseMoECandidate(
        d_model=16,
        num_experts=4,
        top_k=2,
        hidden=12,
    ).copy_from_legacy(legacy)

    assert parameter_count(candidate) == parameter_count(legacy)

    x_legacy = torch.randn(3, 7, 16, requires_grad=True)
    x_candidate = x_legacy.detach().clone().requires_grad_(True)
    upstream = torch.randn(3, 7, 16)

    out_legacy = legacy(x_legacy)
    out_candidate = candidate(x_candidate)
    torch.testing.assert_close(out_candidate, out_legacy, atol=2e-6, rtol=2e-6)

    (out_legacy * upstream).sum().backward()
    (out_candidate * upstream).sum().backward()

    torch.testing.assert_close(x_candidate.grad, x_legacy.grad, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        candidate.router.weight.grad,
        legacy.router.weight.grad,
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        candidate.router.bias.grad,
        legacy.router.bias.grad,
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        candidate.expert_w1.grad,
        _stack_legacy_grads(legacy, 0),
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        candidate.expert_w2.grad,
        _stack_legacy_grads(legacy, 2),
        atol=3e-6,
        rtol=3e-6,
    )

    assert candidate.last_counts is not None
    assert int(candidate.last_counts.sum()) == 3 * 7 * 2
    assert candidate.theoretical_executed_fraction == legacy.theoretical_executed_fraction


def test_grouped_candidate_reduces_logical_expert_gemms_without_dense_execution() -> None:
    candidate = GroupedSparseMoECandidate(
        d_model=32,
        num_experts=8,
        top_k=2,
        hidden=24,
    )
    plan = candidate.logical_kernel_plan()
    assert plan["legacy_expert_gemms_per_layer"] == 16
    assert plan["candidate_grouped_gemms_per_layer"] == 2
    assert plan["selected_assignments_per_token"] == 2
    assert plan["possible_assignments_per_token"] == 8
    assert candidate.theoretical_executed_fraction == 0.25


def test_grouped_mm_cuda_path_is_fail_closed_on_cpu() -> None:
    candidate = GroupedSparseMoECandidate(
        d_model=16,
        num_experts=4,
        top_k=2,
        hidden=8,
    )
    assert not candidate._grouped_mm_available(torch.randn(2, 16))


def test_scan_slice_candidate_matches_existing_scan_values_and_gradients() -> None:
    torch.manual_seed(77102)
    raw_a = torch.randn(2, 17, 9)
    a_ref = torch.sigmoid(raw_a).detach().requires_grad_(True)
    b_ref = torch.randn(2, 17, 9, requires_grad=True)
    initial_ref = torch.randn(2, 9, requires_grad=True)

    a_candidate = a_ref.detach().clone().requires_grad_(True)
    b_candidate = b_ref.detach().clone().requires_grad_(True)
    initial_candidate = initial_ref.detach().clone().requires_grad_(True)

    out_ref = affine_scan(a_ref, b_ref, initial_ref)
    out_candidate = affine_scan_slice_candidate(a_candidate, b_candidate, initial_candidate)
    torch.testing.assert_close(out_candidate, out_ref, atol=2e-6, rtol=2e-6)

    upstream = torch.randn_like(out_ref)
    out_ref.backward(upstream)
    out_candidate.backward(upstream)
    torch.testing.assert_close(a_candidate.grad, a_ref.grad, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(b_candidate.grad, b_ref.grad, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        initial_candidate.grad,
        initial_ref.grad,
        atol=3e-6,
        rtol=3e-6,
    )
