"""Stage-2B0 #1384: CPU-only predictor auxiliary derivative/causality tests.

No optimizer, no training loop, no scientific seeds, no GPU usage.
"""
import pytest
import torch
from torch.nn import functional as F

from tam_research.pgw_v3_stage1b.oracle import make_example
from tam_research.pgw_v3_stage2a.model import (
    PGWV3Stage2A,
    PGWV3Stage2AConfig,
    batch_verified_examples,
    instantiated_parameter_count,
)
from tam_research.pgw_v3_stage2b0.auxiliary import (
    causal_event_hidden,
    predictor_auxiliary_loss,
    predictor_step_errors,
)


def _fixture(*, index=0, delay=1, missing=False):
    return make_example(
        split="validation",
        index=index,
        last_position=5,
        overwrite_count=2,
        interference=True,
        delay_chunks=delay,
        missing=missing,
    )


def _model(mode="hybrid"):
    torch.manual_seed(1384)  # local fixture initialization, NOT a scientific seed
    return PGWV3Stage2A(PGWV3Stage2AConfig(route_mode=mode)).eval()


def test_untrained_auxiliary_shape_exact_parameter_budget_and_positive_loss():
    model = _model()
    tokens, anchors, _ = batch_verified_examples([_fixture(), _fixture(index=3, missing=True)])
    hidden = causal_event_hidden(model, tokens, anchors)
    assert tuple(hidden.shape) == (2, 8, 8, 32)
    per_step = predictor_step_errors(model, hidden)
    assert tuple(per_step.shape) == (2, 8, 7)
    assert torch.isfinite(per_step).all()
    assert (per_step >= 0).all()
    total = predictor_auxiliary_loss(model, tokens, anchors)
    assert torch.isfinite(total)
    assert total.item() > 0
    assert torch.allclose(total, per_step.mean(), rtol=0, atol=0)
    assert instantiated_parameter_count(model) == 20354


def test_pure_auxiliary_backward_reaches_two_predictor_matrices_not_answer_head():
    model = _model()
    tokens, anchors, _ = batch_verified_examples([_fixture(index=9)])
    loss = predictor_auxiliary_loss(model, tokens, anchors)
    loss.backward()  # derivative probe only, NO optimizer.step()
    for weight in (
        model.workspace.predict_down.weight,
        model.workspace.predict_up.weight,
    ):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in model.answer_head.parameters())
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_detached_targets_last_position_has_zero_gradient_when_hidden_is_leaf():
    model = _model()
    torch.manual_seed(1385)  # isolated CPU test only
    hidden = torch.randn(1, 8, 8, 32, requires_grad=True)
    predictor_step_errors(model, hidden).mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    # Position7 appears only as target, never as previous input (p=1..7).
    assert torch.count_nonzero(hidden.grad[:, :, 7, :]).item() == 0
    assert hidden.grad[:, :, :7, :].abs().sum().item() > 0


def test_future_event_position_cannot_change_earlier_causal_hidden_or_losses():
    model = _model()
    tokens, anchors, _ = batch_verified_examples([_fixture(index=11)])
    changed = tokens.clone()
    # Edit last filler in event chunk 3; remains legal and must not affect p<=6.
    index = 3 * 8 + 7
    changed[0, index] = 200 if tokens[0, index] != 200 else 201
    with torch.no_grad():
        before = causal_event_hidden(model, tokens, anchors)
        after = causal_event_hidden(model, changed, anchors)
        errors_before = predictor_step_errors(model, before)
        errors_after = predictor_step_errors(model, after)
    assert torch.equal(before[:, 3, :7], after[:, 3, :7])
    assert torch.equal(errors_before[:, 3, :6], errors_after[:, 3, :6])
    assert torch.equal(before[:, :3], after[:, :3])


def test_all_delays_and_read_suffix_are_excluded_from_auxiliary_objective():
    model = _model()
    tokens, anchors, _ = batch_verified_examples([_fixture(index=13, delay=4)])
    changed = tokens.clone()
    # Legal edits in unrelated delay WRITE chunks and READ suffix after QUERY.
    changed[0, 8 * 8 + 5] = 213 if tokens[0, 8 * 8 + 5] != 213 else 214
    changed[0, -1] = 219 if tokens[0, -1] != 219 else 220
    with torch.no_grad():
        a = predictor_auxiliary_loss(model, tokens, anchors)
        b = predictor_auxiliary_loss(model, changed, anchors)
    assert torch.equal(a, b)


def test_matched_auxiliary_identical_for_hybrid_and_no_workspace_control():
    hybrid = _model("hybrid")
    control = _model("no_workspace")
    control.load_state_dict(hybrid.state_dict(), strict=True)
    tokens, anchors, _ = batch_verified_examples([_fixture(index=15)])
    assert instantiated_parameter_count(hybrid) == instantiated_parameter_count(control)
    with torch.no_grad():
        assert torch.equal(
            predictor_auxiliary_loss(hybrid, tokens, anchors),
            predictor_auxiliary_loss(control, tokens, anchors),
        )
    predictor_auxiliary_loss(control, tokens, anchors).backward()
    assert control.workspace.predict_down.weight.grad is not None
    assert control.workspace.predict_down.weight.grad.abs().sum().item() > 0


def test_answer_only_ce_still_cannot_train_detached_predictor():
    model = _model()
    tokens, anchors, targets = batch_verified_examples([_fixture(index=18)])
    F.cross_entropy(model(tokens, anchors), targets).backward()
    assert model.workspace.predict_down.weight.grad is None
    assert model.workspace.predict_up.weight.grad is None


@pytest.mark.parametrize("bad", (
    "wrong_dtype", "wrong_rank", "empty_batch", "short", "invalid_token",
    "wrong_anchor", "anchor_dtype", "no_write_event", "missing_read", "missing_query",
))
def test_auxiliary_fail_closed_for_malformed_inputs(bad):
    model = _model()
    tokens, anchors, _ = batch_verified_examples([_fixture(index=23)])
    if bad == "wrong_dtype":
        tokens = tokens.float()
    elif bad == "wrong_rank":
        tokens = tokens.unsqueeze(0)
    elif bad == "empty_batch":
        tokens, anchors = tokens[:0], anchors[:0]
    elif bad == "short":
        tokens = tokens[:, :8]
    elif bad == "invalid_token":
        tokens = tokens.clone()
        tokens[0, 20] = 256
    elif bad == "wrong_anchor":
        anchors = anchors - 1
    elif bad == "anchor_dtype":
        anchors = anchors.to(torch.int32)
    elif bad == "no_write_event":
        tokens = tokens.clone()
        tokens[0, 8] = 3
    elif bad == "missing_read":
        tokens = tokens.clone()
        tokens[0, -8] = 3
    elif bad == "missing_query":
        tokens = tokens.clone()
        tokens[0, -6] = 3
    with pytest.raises(ValueError):
        predictor_auxiliary_loss(model, tokens, anchors)


@pytest.mark.parametrize("shape", ((8, 8, 32), (1, 9, 8, 32), (1, 8, 8, 31), (0, 8, 8, 32)))
def test_bad_hidden_geometry_rejected(shape):
    with pytest.raises(ValueError):
        predictor_step_errors(_model(), torch.randn(*shape))
