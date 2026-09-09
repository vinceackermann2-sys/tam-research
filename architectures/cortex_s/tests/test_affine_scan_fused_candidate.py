from __future__ import annotations

import torch

from architectures.cortex_s.affine_scan_fused_candidate import (
    affine_scan_analytic_backward_reference,
    affine_scan_sequential_reference,
    kernel_contract,
)
from architectures.cortex_s.language_model import affine_scan


def _production_grads(a0: torch.Tensor, b0: torch.Tensor, initial0: torch.Tensor, grad: torch.Tensor):
    a = a0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)
    initial = initial0.detach().clone().requires_grad_(True)
    states = affine_scan(a, b, initial)
    (states * grad).sum().backward()
    return states.detach(), a.grad.detach(), b.grad.detach(), initial.grad.detach()


def test_sequential_reference_matches_parallel_scan_forward_and_gradients() -> None:
    torch.manual_seed(20260909)
    for length in (1, 2, 7, 16, 31):
        a0 = torch.sigmoid(torch.randn(3, length, 5, dtype=torch.float64))
        b0 = torch.randn(3, length, 5, dtype=torch.float64) * 0.2
        initial0 = torch.randn(3, 5, dtype=torch.float64)
        grad = torch.randn(3, length, 5, dtype=torch.float64)

        prod_states, prod_ga, prod_gb, prod_gi = _production_grads(a0, b0, initial0, grad)

        a = a0.detach().clone().requires_grad_(True)
        b = b0.detach().clone().requires_grad_(True)
        initial = initial0.detach().clone().requires_grad_(True)
        ref_states = affine_scan_sequential_reference(a, b, initial)
        (ref_states * grad).sum().backward()

        torch.testing.assert_close(ref_states.detach(), prod_states, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(a.grad, prod_ga, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(b.grad, prod_gb, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(initial.grad, prod_gi, rtol=1e-10, atol=1e-10)


def test_explicit_reverse_recurrence_matches_autograd() -> None:
    torch.manual_seed(20260910)
    a0 = torch.sigmoid(torch.randn(2, 19, 7, dtype=torch.float64))
    b0 = torch.randn(2, 19, 7, dtype=torch.float64) * 0.1
    initial0 = torch.randn(2, 7, dtype=torch.float64)
    grad = torch.randn(2, 19, 7, dtype=torch.float64)

    states, prod_ga, prod_gb, prod_gi = _production_grads(a0, b0, initial0, grad)
    ga, gb, gi = affine_scan_analytic_backward_reference(a0, states, grad, initial0)

    torch.testing.assert_close(ga, prod_ga, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(gb, prod_gb, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(gi, prod_gi, rtol=1e-10, atol=1e-10)


def test_chunk_carry_matches_one_shot_recurrence() -> None:
    torch.manual_seed(20260911)
    a = torch.sigmoid(torch.randn(4, 23, 6, dtype=torch.float64))
    b = torch.randn(4, 23, 6, dtype=torch.float64) * 0.1
    initial = torch.randn(4, 6, dtype=torch.float64)

    whole = affine_scan_sequential_reference(a, b, initial)
    first = affine_scan_sequential_reference(a[:, :9], b[:, :9], initial)
    second = affine_scan_sequential_reference(a[:, 9:], b[:, 9:], first[:, -1])
    stitched = torch.cat((first, second), dim=1)
    torch.testing.assert_close(stitched, whole, rtol=1e-12, atol=1e-12)


def test_candidate_contract_is_non_authorizing() -> None:
    contract = kernel_contract()
    assert contract["parameter_count_delta"] == 0
    assert contract["scientific_semantics_changed"] is False
    assert contract["production_wired"] is False
    assert contract["gpu_benchmark_authorized"] is False
