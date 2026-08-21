import torch

from tam_research.models import diagonal_affine_scan
from tam_research.scan_experiments import chunked_diagonal_affine_scan


def sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    state = torch.zeros_like(a[:, 0])
    states = []
    for t in range(a.size(1)):
        state = a[:, t] * state + b[:, t]
        states.append(state)
    return torch.stack(states, dim=1)


def test_chunked_scan_matches_reference_across_lengths():
    torch.manual_seed(41)
    for time in (7, 8, 17, 31, 64):
        a = torch.sigmoid(torch.randn(2, time, 5, dtype=torch.float64))
        b = torch.randn(2, time, 5, dtype=torch.float64)
        reference = sequential_scan(a, b)
        hillis = diagonal_affine_scan(a, b)
        torch.testing.assert_close(hillis, reference, rtol=1e-10, atol=1e-11)
        for chunk_size in (4, 8, 16):
            got = chunked_diagonal_affine_scan(a, b, chunk_size=chunk_size)
            torch.testing.assert_close(got, reference, rtol=1e-10, atol=1e-11)


def test_chunked_scan_gradients_match_reference():
    torch.manual_seed(42)
    base_a = torch.sigmoid(torch.randn(2, 23, 4, dtype=torch.float64))
    base_b = torch.randn(2, 23, 4, dtype=torch.float64)
    weight = torch.randn(2, 23, 4, dtype=torch.float64)

    a_ref = base_a.detach().clone().requires_grad_(True)
    b_ref = base_b.detach().clone().requires_grad_(True)
    loss_ref = (diagonal_affine_scan(a_ref, b_ref) * weight).sum()
    grad_a_ref, grad_b_ref = torch.autograd.grad(loss_ref, (a_ref, b_ref))

    for chunk_size in (4, 8, 16):
        a = base_a.detach().clone().requires_grad_(True)
        b = base_b.detach().clone().requires_grad_(True)
        loss = (
            chunked_diagonal_affine_scan(a, b, chunk_size=chunk_size) * weight
        ).sum()
        grad_a, grad_b = torch.autograd.grad(loss, (a, b))
        torch.testing.assert_close(grad_a, grad_a_ref, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(grad_b, grad_b_ref, rtol=1e-9, atol=1e-10)
