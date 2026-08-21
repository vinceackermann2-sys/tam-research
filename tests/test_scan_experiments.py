import torch

from tam_research.models import TAMV3Mixer, diagonal_affine_scan, parameter_count
from tam_research.scan_experiments import (
    ProjectionFusedTAMV3Mixer,
    chunked_diagonal_affine_scan,
)


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


def _copy_tamv3_into_fused(
    source: TAMV3Mixer,
    target: ProjectionFusedTAMV3Mixer,
) -> None:
    inner = source.attn.inner
    state = source.world.state_size
    with torch.no_grad():
        target.in_proj.weight[: 3 * inner].copy_(source.attn.qkv.weight)
        target.in_proj.weight[3 * inner : 3 * inner + state].copy_(
            source.world.candidate.weight
        )
        target.in_proj.weight[3 * inner + state :].copy_(source.world.keep.weight)
        target.keep_bias.copy_(source.world.keep.bias)
        target.out_proj.weight[:, :inner].copy_(source.attn.out.weight)
        target.out_proj.weight[:, inner:].copy_(source.world.out.weight)
        target.gate_logit.copy_(source.gate_logit)


def test_projection_fusion_is_parameter_exact_and_function_equivalent():
    torch.manual_seed(43)
    source = TAMV3Mixer(
        d_model=64,
        n_heads=4,
        attention_inner=52,
        state_size=16,
    ).double()
    fused = ProjectionFusedTAMV3Mixer(
        d_model=64,
        n_heads=4,
        attention_inner=52,
        state_size=16,
        chunk_size=None,
    ).double()
    _copy_tamv3_into_fused(source, fused)

    assert parameter_count(source) == parameter_count(fused)
    x = torch.randn(2, 13, 64, dtype=torch.float64)
    expected = source(x)
    actual = fused(x)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_projection_fusion_stays_equivalent_with_chunked_scan():
    torch.manual_seed(44)
    source = TAMV3Mixer(
        d_model=64,
        n_heads=4,
        attention_inner=52,
        state_size=16,
    ).double()
    fused = ProjectionFusedTAMV3Mixer(
        d_model=64,
        n_heads=4,
        attention_inner=52,
        state_size=16,
        chunk_size=8,
    ).double()
    _copy_tamv3_into_fused(source, fused)

    x = torch.randn(2, 19, 64, dtype=torch.float64)
    expected = source(x)
    actual = fused(x)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
