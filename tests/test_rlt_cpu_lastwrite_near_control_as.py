from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.cpu_lastwrite_near_control_as import (
    BATCH_SIZE, CANDIDATES, DISTANCES, EVAL_PER_CLASS, EXPECTED_PARAMETERS,
    QUERY, SEQ_LEN, SET0, TRAIN_SECONDS_PER_CANDIDATE, balanced_holdout,
)
from experiments.rlt.cpu_lastwrite_probe_ar import (
    last_write_labels, make_batch, make_models, query_binary_logits,
)
from experiments.rlt.model import parameter_count


def test_balanced_independent_holdout_and_next_token_label() -> None:
    x, y = balanced_holdout()
    assert x.shape == (2 * EVAL_PER_CLASS, SEQ_LEN)
    assert int(y.sum()) == EVAL_PER_CLASS
    assert bool((x[:, -1] == QUERY).all())
    assert torch.equal(x[:, -2], y + SET0)
    assert torch.equal(last_write_labels(x), y)
    assert TRAIN_SECONDS_PER_CANDIDATE == 25.0


def test_train_seed_is_separate_from_heldout() -> None:
    x, y, d = make_batch(BATCH_SIZE, DISTANCES, 20261062, seq_len=SEQ_LEN)
    val, _, = balanced_holdout()
    assert not torch.equal(x, val[:BATCH_SIZE])
    assert bool((d == 1).all())
    assert torch.equal(last_write_labels(x), y)


@pytest.mark.parametrize("name", CANDIDATES)
def test_exact_counts_and_finite_loss_gradients(name: str) -> None:
    torch.manual_seed(20301062)
    model = make_models()[name]
    assert parameter_count(model) == EXPECTED_PARAMETERS
    x, y, _ = make_batch(2, DISTANCES, 20261622, seq_len=SEQ_LEN)
    logits = query_binary_logits(model, x)
    assert logits.shape == (2, 2)
    loss = F.cross_entropy(logits, y)
    assert math.isfinite(float(loss))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
