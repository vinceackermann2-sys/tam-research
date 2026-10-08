from __future__ import annotations

import torch

from tam_research.cpw_v5_afm.model import (
    exclusive_fast_weight_read,
    exclusive_fast_weight_read_reference,
)


def test_eight_key_value_bindings_recalled_at_long_delay() -> None:
    """Constructed features: verify mathematical capacity, not learned recall."""
    torch.manual_seed(1296)
    length = 320
    rank = 8
    value_dim = 8
    q = torch.zeros(1, length, rank)
    k = torch.zeros_like(q)
    v = torch.zeros(1, length, value_dim)
    gate = torch.zeros(1, length, 1)

    values = torch.randperm(rank)
    for key in range(rank):
        position = 10 + key * 3
        k[0, position, key] = 1
        v[0, position, values[key]] = 1
        gate[0, position, 0] = 1

    # All eight bindings coexist through a long filler interval.
    q[0, -1, 6] = 1
    read = exclusive_fast_weight_read(q, k, v, gate)
    expected = values[6]
    assert read[0, -1].argmax().item() == expected
    torch.testing.assert_close(
        read[0, -1, expected], torch.tensor(1.0), rtol=0, atol=1e-6
    )


def test_irrelevant_and_future_writes_cannot_contaminate_query() -> None:
    torch.manual_seed(1297)
    b, t, rk, rv = 2, 48, 6, 7
    q = torch.rand(b, t, rk)
    k = torch.rand(b, t, rk)
    v = torch.rand(b, t, rv)
    gate = torch.sigmoid(torch.randn(b, t, 1))
    gate[:, 0] = 0

    earlier = exclusive_fast_weight_read(q, k, v, gate)
    k2 = k.clone()
    v2 = v.clone()
    gate2 = gate.clone()
    k2[:, 35:] = 100 * torch.randn_like(k2[:, 35:])
    v2[:, 35:] = 100 * torch.randn_like(v2[:, 35:])
    gate2[:, 35:] = 1
    changed = exclusive_fast_weight_read(q, k2, v2, gate2)

    torch.testing.assert_close(earlier[:, :36], changed[:, :36], rtol=0, atol=1e-4)


def test_vectorized_matches_slow_reference_at_320_positions() -> None:
    torch.manual_seed(1298)
    q = torch.rand(2, 320, 8) + 0.01
    k = torch.rand(2, 320, 8) + 0.01
    v = torch.randn(2, 320, 8)
    gate = torch.rand(2, 320, 1)
    gate[:, 0] = 0
    fast = exclusive_fast_weight_read(q, k, v, gate)
    slow = exclusive_fast_weight_read_reference(q, k, v, gate)
    torch.testing.assert_close(fast, slow, rtol=1e-4, atol=1e-4)
