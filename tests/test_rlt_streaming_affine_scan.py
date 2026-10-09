from __future__ import annotations

import pytest
import torch

from experiments.rlt.model_gated_scan import (
    associative_affine_scan, sequential_affine_reference,
)
from experiments.rlt.streaming_affine_scan import chunked_affine_scan


def sample(
    length: int,
    *,
    dtype: torch.dtype = torch.float64,
    per_batch_initial: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(502300 + length)
    gates = (0.5 + 0.48 * torch.rand(2, length, 5, generator=gen)).to(dtype)
    writes = (0.05 * torch.randn(2, length, 5, generator=gen)).to(dtype)
    initial = (0.1 * torch.randn(
        (2, 5) if per_batch_initial else (5,), generator=gen
    )).to(dtype)
    return gates, writes, initial


@pytest.mark.parametrize("length", [1, 3, 17, 65, 129])
@pytest.mark.parametrize("chunk_size", [1, 2, 8, 16, 64, 256])
def test_chunked_equals_full_and_sequential(
    length: int, chunk_size: int,
) -> None:
    gates, writes, init = sample(length)
    states, final = chunked_affine_scan(
        gates, writes, init, chunk_size=chunk_size,
    )
    full = associative_affine_scan(gates, writes, init)
    sequential = sequential_affine_reference(gates, writes, init)
    assert states.shape == (2, length, 5)
    assert final.shape == (2, 5)
    torch.testing.assert_close(states, full, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(states, sequential, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(final, full[:, -1], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("shape", ["shared", "per_batch"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_state_layout_and_dtype(shape: str, dtype: torch.dtype) -> None:
    gates, writes, init = sample(
        47, dtype=dtype, per_batch_initial=(shape == "per_batch"),
    )
    states, final = chunked_affine_scan(gates, writes, init, chunk_size=9)
    expected = associative_affine_scan(gates, writes, init)
    assert states.dtype == dtype and final.dtype == dtype
    torch.testing.assert_close(states, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("length,chunk", [(3, 1), (21, 6), (65, 16), (129, 31)])
def test_full_bptt_matches_monolithic_gradient(length: int, chunk: int) -> None:
    gates0, writes0, init0 = sample(length)
    gates_a = gates0.clone().requires_grad_()
    writes_a = writes0.clone().requires_grad_()
    init_a = init0.clone().requires_grad_()
    chunked, final = chunked_affine_scan(
        gates_a, writes_a, init_a, chunk_size=chunk,
    )
    objective = final.square().sum() + chunked[:, length // 2].square().sum()
    objective.backward()
    assert gates_a.grad is not None and writes_a.grad is not None
    assert init_a.grad is not None
    assert bool(torch.isfinite(gates_a.grad).all())
    assert bool(torch.isfinite(writes_a.grad).all())
    assert bool(torch.isfinite(init_a.grad).all())
    assert bool((writes_a.grad[:, :1].abs().sum() > 0))

    gates_b = gates0.clone().requires_grad_()
    writes_b = writes0.clone().requires_grad_()
    init_b = init0.clone().requires_grad_()
    full = associative_affine_scan(gates_b, writes_b, init_b)
    (full[:, -1].square().sum() + full[:, length // 2].square().sum()).backward()
    torch.testing.assert_close(gates_a.grad, gates_b.grad, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(writes_a.grad, writes_b.grad, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(init_a.grad, init_b.grad, atol=2e-11, rtol=2e-10)


def test_future_gates_and_writes_do_not_change_prefix_states() -> None:
    gates, writes, init = sample(81)
    baseline, _ = chunked_affine_scan(gates, writes, init, chunk_size=13)
    changed_g = gates.clone()
    changed_w = writes.clone()
    changed_g[:, 29:] = 0.3
    changed_w[:, 29:] = 13.0
    changed, _ = chunked_affine_scan(
        changed_g, changed_w, init, chunk_size=9,
    )
    torch.testing.assert_close(
        baseline[:, :29], changed[:, :29], atol=2e-12, rtol=2e-12,
    )


def test_streaming_resume_across_independent_calls() -> None:
    g, w, init = sample(113)
    part1, state1 = chunked_affine_scan(g[:, :37], w[:, :37], init, chunk_size=8)
    part2, state2 = chunked_affine_scan(g[:, 37:80], w[:, 37:80], state1, chunk_size=12)
    part3, state3 = chunked_affine_scan(g[:, 80:], w[:, 80:], state2, chunk_size=13)
    combined = torch.cat([part1, part2, part3], dim=1)
    reference = associative_affine_scan(g, w, init)
    torch.testing.assert_close(combined, reference, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(state3, reference[:, -1], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("size", [0, -1, 2.5, True, None])
def test_reject_invalid_chunk_size(size) -> None:
    g, w, init = sample(8)
    with pytest.raises(ValueError, match="chunk_size"):
        chunked_affine_scan(g, w, init, chunk_size=size)


def test_reject_malformed_tensors() -> None:
    g, w, init = sample(7)
    with pytest.raises(ValueError, match="shape|share"):
        chunked_affine_scan(g[:, 0], w[:, 0], init, chunk_size=2)
    with pytest.raises(ValueError, match="shape|share"):
        chunked_affine_scan(g, w[:, :3], init, chunk_size=2)
    with pytest.raises(ValueError, match="state"):
        chunked_affine_scan(g, w, torch.zeros(2, 6), chunk_size=2)
    with pytest.raises(ValueError, match="token"):
        chunked_affine_scan(g[:, :0], w[:, :0], init, chunk_size=2)
    with pytest.raises(ValueError, match="dtype"):
        chunked_affine_scan(g, w.float(), init, chunk_size=2)
    with pytest.raises(TypeError, match="floating"):
        chunked_affine_scan(g.int(), w.int(), init, chunk_size=2)


def test_inputs_not_modified() -> None:
    g, w, init = sample(65)
    old = tuple(t.clone() for t in (g, w, init))
    chunked_affine_scan(g, w, init, chunk_size=12)
    for a, b in zip((g, w, init), old):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
