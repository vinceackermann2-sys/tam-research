from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.cpu_lastwrite_curriculum_at import (
    DISTANCES, EASY_SEQ_LEN, EASY_SECONDS, EVAL_PER_CLASS,
    EXPECTED_PARAMETERS, LONG_SECONDS, SEQ_LEN,
    balanced_batch_for_distance, timed_phase,
)
from experiments.rlt.cpu_lastwrite_probe_ar import (
    CANDIDATES, QUERY, SET0, last_write_labels,
    make_batch, make_models, query_binary_logits,
)
from experiments.rlt.model import parameter_count


@pytest.mark.parametrize("distance", DISTANCES)
def test_balanced_heldout_state_and_gap(distance: int) -> None:
    x, y = balanced_batch_for_distance(distance, SEQ_LEN, 20293100 + distance)
    assert x.shape == (2 * EVAL_PER_CLASS, SEQ_LEN)
    assert bool((x[:, -1] == QUERY).all())
    assert int(y.sum()) == EVAL_PER_CLASS
    assert torch.equal(last_write_labels(x), y)
    assert torch.equal(x[:, SEQ_LEN - 1 - distance], y + SET0)


def test_heldout_determinism_and_disjoint_train() -> None:
    x, y = balanced_batch_for_distance(1, EASY_SEQ_LEN, 20293101)
    x2, y2 = balanced_batch_for_distance(1, EASY_SEQ_LEN, 20293101)
    assert torch.equal(x, x2) and torch.equal(y, y2)
    training, _, _ = make_batch(8, (1,), 20261063, seq_len=EASY_SEQ_LEN)
    assert not torch.equal(x[:8], training)
    assert EASY_SECONDS == 20.0 and LONG_SECONDS == 90.0


def test_cpu_model_parameter_counts() -> None:
    models = make_models()
    assert list(models)==list(CANDIDATES)
    assert all(parameter_count(m)==EXPECTED_PARAMETERS for m in models.values())


def test_short_cpu_phase_trains_and_consumes_budget() -> None:
    torch.manual_seed(20301063)
    model = make_models()["transformer_control"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    trained = timed_phase(
        model, optimizer, seconds=0.02, seed=20261063,
        seq_len=EASY_SEQ_LEN, distances=(1,),
    )
    assert trained["steps"]>=1
    assert trained["train_seconds"]>=0.02
    assert trained["tokens_seen"]==trained["steps"]*8*EASY_SEQ_LEN
    assert trained["tokens_per_second"]>0


@pytest.mark.parametrize("name", CANDIDATES)
def test_finite_query_logits_for_long_distance(name: str) -> None:
    torch.manual_seed(20301063)
    model = make_models()[name]
    x, y, _ = make_batch(2, (63,), 20261063, seq_len=SEQ_LEN)
    logits=query_binary_logits(model, x)
    assert logits.shape==(2,2)
    loss=F.cross_entropy(logits,y)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
