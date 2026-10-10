from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.cpu_bounded_streaming_aw import (
    BATCH_SIZE, CHUNK_SIZE, DISTANCES, EVAL_PER_CLASS, PARAMETERS,
    SEQ_LEN, assert_no_target_information_in_bounded_suffix,
    balanced_holdout, binary_logits, models,
)
from experiments.rlt.cpu_lastwrite_probe_ar import (
    QUERY, SET0, last_write_labels, make_batch,
)
from experiments.rlt.model import parameter_count


def test_equal_active_parameters_and_initialization() -> None:
    carry, reset = models()
    assert parameter_count(carry) == parameter_count(reset) == PARAMETERS
    assert carry.carry_between_chunks is True
    assert reset.carry_between_chunks is False
    assert carry.chunk_size == reset.chunk_size == CHUNK_SIZE
    assert set(carry.state_dict()) == set(reset.state_dict())
    for name, value in carry.state_dict().items():
        torch.testing.assert_close(value, reset.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("gap", DISTANCES)
def test_independent_balanced_eval_oracle(gap: int) -> None:
    x, y = balanced_holdout(gap, seed=20291081 + gap)
    assert x.shape == (2 * EVAL_PER_CLASS, SEQ_LEN)
    assert int(y.sum()) == EVAL_PER_CLASS
    assert bool((x[:, -1] == QUERY).all())
    assert torch.equal(x[:, SEQ_LEN - 1 - gap], y + SET0)
    assert torch.equal(last_write_labels(x), y)


def test_off_window_labels_cannot_leak_into_final_chunk() -> None:
    assert_no_target_information_in_bounded_suffix()
    carry, reset = models()
    x, y = balanced_holdout(64, seed=20291081 + 64)
    other = x.clone()
    other[:, SEQ_LEN - 1 - 64] = (1 - y) + SET0
    assert torch.equal(x[:, -CHUNK_SIZE:], other[:, -CHUNK_SIZE:])
    assert torch.equal(last_write_labels(other), 1 - y)
    with torch.no_grad():
        a = binary_logits(reset, x[:4])
        b = binary_logits(reset, other[:4])
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("gap", [1, 16, 127])
def test_forward_loss_and_gradients_for_both_models(gap: int) -> None:
    x, y, _ = make_batch(BATCH_SIZE, (gap,), 20261081 + gap, seq_len=SEQ_LEN)
    for model in models():
        logits = binary_logits(model, x)
        assert logits.shape == (BATCH_SIZE, 2)
        loss = F.cross_entropy(logits, y)
        assert math.isfinite(float(loss))
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        assert all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
