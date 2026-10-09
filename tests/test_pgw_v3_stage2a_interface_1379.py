"""PGW-v3 Stage-2A CPU-only structural contract; no optimizer/training."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from tam_research.pgw_v3_stage0.model import ROUTE_MODES
from tam_research.pgw_v3_stage1b.oracle import (
    NOT_FOUND, QUERY, READ, make_example,
)
from tam_research.pgw_v3_stage2a.model import (
    PGWV3Stage2A, PGWV3Stage2AConfig, batch_verified_examples,
    instantiated_parameter_count,
)


def _example(*, index=0, delay=1, missing=False, last_position=4):
    return make_example(
        split="test", index=index, last_position=last_position,
        overwrite_count=1, interference=True, delay_chunks=delay, missing=missing,
    )


def _model(mode="hybrid"):
    torch.manual_seed(1379)  # structural test fixture, NOT scientific seed
    return PGWV3Stage2A(PGWV3Stage2AConfig(route_mode=mode)).eval()


def test_all_modes_have_exact_frozen_instantiated_parameter_count():
    for mode in sorted(ROUTE_MODES):
        assert instantiated_parameter_count(_model(mode)) == 20354


def test_stage1b_batch_verified_shape_and_external_labels():
    present, missing = _example(missing=False), _example(index=1, missing=True)
    tokens, anchors, answers = batch_verified_examples([present, missing])
    assert tokens.shape == (2, 80)
    assert anchors.tolist() == [74, 74]
    assert answers.tolist() == [present.expected_value - 32, NOT_FOUND - 32]
    assert tokens[0, 72].item() == READ
    assert tokens[0, 74].item() == QUERY
    assert all(not (tokens[i, -8:] == present.expected_value).any().item() for i in range(2))
    assert _model()(tokens, anchors).shape == (2, 33)


@pytest.mark.parametrize("mode", sorted(ROUTE_MODES))
def test_forward_all_routing_modes_batch_and_class_space(mode):
    batch = [_example(index=4), _example(index=5, missing=True)]
    t, a, y = batch_verified_examples(batch)
    with torch.no_grad():
        result = _model(mode)(t, a)
    assert result.shape == (2, 33)
    assert torch.isfinite(result).all()
    assert y.min() >= 0 and y.max() <= 32


def test_future_read_chunk_filler_does_not_change_query_logits():
    t, a, _ = batch_verified_examples([_example()])
    changed = t.clone()
    changed[0, -1] = 200 if changed[0, -1].item() != 200 else 201
    model = _model("hybrid")
    with torch.no_grad():
        assert torch.equal(model(t, a), model(changed, a))


def test_without_workspace_earlier_complete_chunk_cannot_affect_final_query():
    t, a, _ = batch_verified_examples([_example()])
    changed = t.clone()
    changed[0, 6] = 201 if changed[0, 6].item() != 201 else 202
    model = _model("no_workspace")
    with torch.no_grad():
        assert torch.equal(model(t, a), model(changed, a))


def test_with_workspace_recent_completed_chunk_affects_query():
    t, a, _ = batch_verified_examples([_example()])
    changed = t.clone()
    # Delay chunk immediately preceding READ is WRITE. Recency selects pos 6,7.
    edit = t.shape[1] - 16 + 6
    changed[0, edit] = 204 if changed[0, edit].item() != 204 else 205
    model = _model("recency")
    with torch.no_grad():
        first, second = model(t, a), model(changed, a)
    assert (first - second).abs().max().item() > 1e-10


def test_single_cpu_backward_reaches_workspace_but_not_detached_predictor():
    tokens, anchors, targets = batch_verified_examples([_example(index=9)])
    model = _model("hybrid")
    logits = model(tokens, anchors)
    loss = F.cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()  # No optimizer.step(), no training, no scientific seed
    for weight in (model.workspace.read_out.weight, model.workspace.event_value.weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum().item() > 0
    assert model.workspace.predict_down.weight.grad is None
    assert model.workspace.predict_up.weight.grad is None
    assert model.workspace.utility.weight.grad is not None
    assert model.workspace.utility.weight.grad.abs().sum().item() > 0


def test_no_workspace_still_has_no_workspace_gradient():
    tokens, anchors, targets = batch_verified_examples([_example(index=10)])
    model = _model("no_workspace")
    F.cross_entropy(model(tokens, anchors), targets).backward()
    assert all(p.grad is None for p in model.workspace.parameters())
    assert model.answer_head.weight.grad is not None


def test_batch_rejects_mixed_delays_tampering_and_incorrect_external_answer():
    a, b = _example(), _example(delay=2)
    with pytest.raises(ValueError, match="mixed lengths"):
        batch_verified_examples([a, b])
    with pytest.raises(ValueError, match="fingerprint"):
        batch_verified_examples([replace(a, sha256="0" * 64)])
    with pytest.raises(ValueError, match="target"):
        batch_verified_examples([replace(a, expected_value=NOT_FOUND)])
    with pytest.raises(ValueError, match="anchor"):
        batch_verified_examples([replace(a, query_anchor=a.query_anchor - 1)])
    with pytest.raises(ValueError, match="empty"):
        batch_verified_examples([])


def test_adapter_rejects_wrong_shape_type_invalid_ids_and_anchors():
    model = _model()
    t, a, _ = batch_verified_examples([_example()])
    with pytest.raises(ValueError, match="LongTensor"):
        model(t.float(), a)
    with pytest.raises(ValueError, match="LongTensor"):
        model(t.unsqueeze(0), a)
    with pytest.raises(ValueError, match="vocabulary"):
        model(t.clone().index_fill(1, torch.tensor([5]), 256), a)
    with pytest.raises(ValueError, match="anchors"):
        model(t, a.to(torch.int32))
    with pytest.raises(ValueError, match="anchors"):
        model(t, torch.tensor([3]))
    invalid_query = t.clone()
    invalid_query[0, -6] = 210
    with pytest.raises(ValueError, match="QUERY"):
        model(invalid_query, a)
    invalid_read = t.clone()
    invalid_read[0, -8] = 211
    with pytest.raises(ValueError, match="READ"):
        model(invalid_read, a)


def test_geometry_changes_explicitly_refused():
    for bad in (dict(d_model=64), dict(vocab_size=512), dict(chunk_size=16),
                dict(heads=8), dict(ff_width=128), dict(route_mode="unknown")):
        with pytest.raises(ValueError):
            PGWV3Stage2AConfig(**bad)
