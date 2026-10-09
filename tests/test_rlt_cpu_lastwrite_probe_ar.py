from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.cpu_lastwrite_probe_ar import (
    BATCH_SIZE, CANDIDATES, DISTANCES, EXPECTED_PARAMETERS, QUERY,
    SEQ_LEN, SET0, SET1, VOCAB,
    last_write_labels, make_batch, make_models, query_binary_logits,
)
from experiments.rlt.model import parameter_count


@pytest.mark.parametrize("distance", DISTANCES)
def test_oracle_matches_nonleaking_generator(distance: int) -> None:
    x, targets, distances = make_batch(16, (distance,), 20261009 + distance)
    assert x.shape == (16, SEQ_LEN)
    assert targets.shape == (16,)
    assert bool((distances == distance).all())
    assert bool((x[:, -1] == QUERY).all())
    assert bool(((x[:, :-1] >= SET0) & (x[:, :-1] < VOCAB)).all())
    assert torch.equal(last_write_labels(x), targets)
    last_location = SEQ_LEN - 1 - distance
    assert torch.equal(x[:, last_location], targets + SET0)
    # No later SET instruction may overwrite the forced final SET.
    if last_location + 1 < SEQ_LEN - 1:
        assert not bool(
            ((x[:, last_location + 1 : -1] == SET0) |
             (x[:, last_location + 1 : -1] == SET1)).any()
        )


def test_determinism_and_balanced_target_distribution() -> None:
    a = make_batch(512, DISTANCES, 20261611)
    b = make_batch(512, DISTANCES, 20261611)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(last_write_labels(a[0]), a[1])
    assert 0.4 <= float(a[1].float().mean()) <= 0.6
    assert set(a[2].tolist()) == set(DISTANCES)


def test_counterfactual_last_write_flips_target() -> None:
    x, y, distance = make_batch(32, DISTANCES, 20261612)
    changed = x.clone()
    for row in range(len(y)):
        last = SEQ_LEN - 1 - int(distance[row])
        changed[row, last] = SET1 if int(y[row]) == 0 else SET0
    assert torch.equal(last_write_labels(changed), 1 - y)
    assert bool((changed[:, -1] == QUERY).all())


def test_no_answer_in_final_input_and_parameter_equality() -> None:
    x, _, _ = make_batch(BATCH_SIZE, DISTANCES, 20261613)
    assert bool((x[:, -1] == QUERY).all())
    models = make_models()
    assert list(models) == list(CANDIDATES)
    assert {k: parameter_count(m) for k, m in models.items()} == {
        k: EXPECTED_PARAMETERS for k in CANDIDATES
    }
    for name, model in models.items():
        with torch.no_grad():
            logits = query_binary_logits(model, x[:, :])
        assert logits.shape == (BATCH_SIZE, 2), name
        assert bool(torch.isfinite(logits).all()), name


@pytest.mark.parametrize("name", CANDIDATES)
def test_finite_query_loss_and_gradient(name: str) -> None:
    torch.manual_seed(20261614)
    model = make_models()[name]
    x, y, _ = make_batch(2, (1, 16, 63), 20261614)
    loss = F.cross_entropy(query_binary_logits(model, x), y)
    assert math.isfinite(float(loss))
    loss.backward()
    assert any(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )


def test_reject_missing_write_and_query() -> None:
    x, _, _ = make_batch(1, (63,), 20261615)
    bad = x.clone()
    bad[:, 0] = 4
    with pytest.raises(ValueError, match="missing state write"):
        last_write_labels(bad)
    bad = x.clone()
    bad[:, -1] = 4
    with pytest.raises(ValueError, match="missing final QUERY"):
        last_write_labels(bad)
