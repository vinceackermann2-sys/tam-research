from __future__ import annotations

import pytest
import torch

from experiments.rlt.model_gated_scan import (
    associative_affine_scan,
    sequential_affine_reference,
)
from experiments.rlt.streaming_scan import (
    chunked_affine_scan,
    chunked_affine_scan_inference,
)


def case(batch: int, steps: int, width: int, *, batched_init: bool = True):
    torch.manual_seed(9000 + 100 * batch + steps + width)
    dtype = torch.float64
    gates = 0.2 + 0.7 * torch.rand(batch, steps, width, dtype=dtype)
    writes = 0.1 * torch.randn(batch, steps, width, dtype=dtype)
    init = 0.2 * torch.randn(
        (batch, width) if batched_init else (width,), dtype=dtype,
    )
    return gates, writes, init


@pytest.mark.parametrize("length,chunk", [
    (1, 1), (3, 1), (9, 2), (13, 3), (17, 7),
    (32, 8), (33, 16), (64, 64), (64, 9),
])
def test_all_states_and_final_carry_match_full_scan(
    length: int, chunk: int,
) -> None:
    gates, writes, initial = case(3, length, 8)
    expected = associative_affine_scan(gates, writes, initial)
    actual, last = chunked_affine_scan(
        gates, writes, initial, chunk_size=chunk,
    )
    assert actual is not None
    assert actual.shape == expected.shape
    assert last.shape == (3, 8)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(last, expected[:, -1], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("chunk", [1, 2, 4, 7, 99])
def test_batch_independent_initial_state(chunk: int) -> None:
    gates, writes, initial = case(2, 11, 5, batched_init=False)
    output, final = chunked_affine_scan(
        gates, writes, initial, chunk_size=chunk,
    )
    expected = sequential_affine_reference(gates, writes, initial)
    assert output is not None
    torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(final, expected[:, -1], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("chunk", [1, 3, 7, 16])
def test_no_grad_inference_matches_full_final_without_history(chunk: int) -> None:
    gates, writes, initial = case(3, 29, 4)
    inference = chunked_affine_scan_inference(
        gates, writes, initial, chunk_size=chunk,
    )
    assert inference.shape == (3, 4)
    assert inference.grad_fn is None
    expected = associative_affine_scan(gates, writes, initial)[:, -1]
    torch.testing.assert_close(inference, expected, atol=1e-10, rtol=1e-10)
    states, last = chunked_affine_scan(
        gates, writes, initial, chunk_size=chunk,
        collect_states=False,
    )
    assert states is None
    torch.testing.assert_close(last, inference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("chunk", [1, 2, 5, 16])
def test_full_backprop_through_chunk_boundaries(chunk: int) -> None:
    a, b, state = case(2, 17, 6)
    gates = a.clone().detach().requires_grad_(True)
    writes = b.clone().detach().requires_grad_(True)
    init = state.clone().detach().requires_grad_(True)
    weights = torch.randn(2, 17, 6, dtype=torch.float64)
    reference = associative_affine_scan(gates, writes, init)
    loss = (reference * weights).sum() + reference[:, -1].square().sum()
    gold = torch.autograd.grad(loss, (gates, writes, init))

    gates = a.clone().detach().requires_grad_(True)
    writes = b.clone().detach().requires_grad_(True)
    init = state.clone().detach().requires_grad_(True)
    outputs, last = chunked_affine_scan(
        gates, writes, init, chunk_size=chunk,
    )
    assert outputs is not None
    objective = (outputs * weights).sum() + last.square().sum()
    grads = torch.autograd.grad(objective, (gates, writes, init))
    for actual, expected in zip(grads, gold):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)
    assert bool((grads[1][:, :chunk].abs() > 1e-12).any())


@pytest.mark.parametrize("prefix", [1, 4, 9])
def test_future_updates_do_not_affect_previous_states(prefix: int) -> None:
    a, b, initial = case(2, 13, 4)
    modified_b = b.clone()
    modified_b[:, prefix:] += 0.5
    original, _ = chunked_affine_scan(a, b, initial, chunk_size=3)
    altered, _ = chunked_affine_scan(a, modified_b, initial, chunk_size=3)
    assert original is not None and altered is not None
    torch.testing.assert_close(
        original[:, :prefix], altered[:, :prefix], atol=0, rtol=0,
    )
    assert not torch.allclose(original[:, prefix:], altered[:, prefix:])


def test_chunk_state_is_functionally_required() -> None:
    gates, writes, initial = case(2, 13, 4)
    correct, last = chunked_affine_scan(
        gates, writes, initial, chunk_size=4,
    )
    assert correct is not None
    wrong_chunks = [
        associative_affine_scan(
            gates[:, k:k+4], writes[:, k:k+4], initial,
        )
        for k in range(0, 13, 4)
    ]
    wrong = torch.cat(wrong_chunks, dim=1)
    assert not torch.allclose(correct[:, 4:], wrong[:, 4:])
    torch.testing.assert_close(
        last, correct[:, -1], atol=0, rtol=0,
    )


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_reject_invalid_chunk_size(invalid) -> None:
    g, w, s = case(1, 4, 2)
    with pytest.raises(ValueError, match="chunk_size"):
        chunked_affine_scan(g, w, s, chunk_size=invalid)


def test_reject_wrong_shapes() -> None:
    g, w, s = case(2, 4, 3)
    with pytest.raises(ValueError, match="gates and writes"):
        chunked_affine_scan(g, w[:, :-1], s, chunk_size=2)
    with pytest.raises(ValueError, match="initial state"):
        chunked_affine_scan(g, w, torch.zeros(7), chunk_size=2)
