"""Stage-2B2 #1400: CPU causal references, NO training/optimizer/science seed."""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from tam_research.pgw_v3_stage1b.oracle import make_example
from tam_research.pgw_v3_stage2a.model import (
    PGWV3Stage2A, batch_verified_examples, instantiated_parameter_count,
)
from tam_research.pgw_v3_stage2b2.reference import (
    CausalReference, CausalReferenceConfig, count_instantiated, count_modules,
)


def _example(*, index=0, delay=1, missing=False):
    return make_example(
        split="validation", index=index, last_position=5, overwrite_count=2,
        interference=True, delay_chunks=delay, missing=missing,
    )


def _batch(*, delay=1):
    return batch_verified_examples(
        [_example(index=2, delay=delay, missing=False),
         _example(index=7, delay=delay, missing=True)]
    )


def _model(scope="full"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1400)  # only structural test initializer
        return CausalReference(CausalReferenceConfig(attention_scope=scope)).cpu().eval()


def test_exact_near_instantiated_parity_and_fully_accounted_live_components():
    assert instantiated_parameter_count(PGWV3Stage2A()) == 20354
    expected = {
        "shared_embeddings": 8448,
        "causal_attention": 4224,
        "encoder_ffn_norms": 6010,
        "answer": 1153,
        "predictor_auxiliary": 512,
    }
    for scope in ("full", "chunk"):
        m = _model(scope)
        assert count_instantiated(m) == 20347
        assert count_modules(m) == expected
        assert abs(20347 - 20354) == 7
        assert abs(20347 - 20354) / 20354 <= 0.001


@pytest.mark.parametrize("scope", ("full", "chunk"))
@pytest.mark.parametrize("delay", (1, 2, 4))
def test_stage1b_validated_present_missing_forward_aux_shape(scope, delay):
    tokens, anchors, targets = _batch(delay=delay)
    model = _model(scope)
    logits, aux = model.forward_with_aux(tokens, anchors)
    assert logits.shape == (2, 33)
    assert targets.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux)
    assert aux.item() > 0
    with torch.no_grad():
        torch.testing.assert_close(model(tokens, anchors), logits.detach(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("scope", ("full", "chunk"))
def test_future_query_suffix_is_invisible_to_both_logits_and_aux(scope):
    tokens, anchors, _ = _batch(delay=4)
    changed = tokens.clone()
    changed[:, -1] = 234
    changed[:, -2] = 233
    model = _model(scope)
    with torch.no_grad():
        logits0, aux0 = model.forward_with_aux(tokens, anchors)
        logits1, aux1 = model.forward_with_aux(changed, anchors)
    assert torch.equal(logits0, logits1)
    assert torch.equal(aux0, aux1)


def test_paired_scopes_cross_chunk_causality_and_intervention():
    tokens, anchors, _ = _batch(delay=1)
    edited = tokens.clone()
    # Event chunk0's value token; remains a valid WRITE-value token.
    old = int(edited[0, 2])
    edited[0, 2] = 32 if old != 32 else 33
    full = _model("full")
    chunk = _model("chunk")
    chunk.load_state_dict(full.state_dict(), strict=True)
    with torch.no_grad():
        full_orig, full_changed = full(tokens, anchors), full(edited, anchors)
        chunk_orig, chunk_changed = chunk(tokens, anchors), chunk(edited, anchors)
    assert torch.equal(chunk_orig, chunk_changed)
    assert not torch.equal(full_orig, full_changed)
    assert not torch.equal(full_orig, chunk_orig)


def test_positional_input_gradient_reaches_older_event_only_with_full_scope():
    tokens, anchors, _ = _batch(delay=1)
    full = _model("full")
    chunk = _model("chunk")
    chunk.load_state_dict(full.state_dict(), strict=True)
    for model, should_reach in ((full, True), (chunk, False)):
        # Leaf embeddings isolate *positions*, unlike shared token tables.
        inputs = model.input_embeddings(tokens).detach().requires_grad_(True)
        hidden = model.encode_embeddings(inputs)
        score = model.answer_from_hidden(hidden, anchors).square().sum()
        score.backward()  # derivative, NOT optimization
        assert inputs.grad is not None
        old_prefix_grad = inputs.grad[:, :64, :]
        if should_reach:
            assert old_prefix_grad.abs().sum().item() > 0
        else:
            assert torch.count_nonzero(old_prefix_grad).item() == 0
        assert inputs.grad[:, -8:-5, :].abs().sum().item() > 0
        # Tokens after QUERY cannot affect the answer.
        assert torch.count_nonzero(inputs.grad[:, -5:, :]).item() == 0


@pytest.mark.parametrize("scope", ("full", "chunk"))
def test_aux_predictor_gradients_exist_but_answer_only_does_not_train_predictor(scope):
    tokens, anchors, targets = _batch(delay=2)
    model = _model(scope)
    frozen = {name: p.detach().clone() for name, p in model.state_dict().items()}
    F.cross_entropy(model(tokens, anchors), targets).backward()
    assert model.predict_down.weight.grad is None
    assert model.predict_up.weight.grad is None
    assert model.answer_head.weight.grad is not None
    model.zero_grad(set_to_none=True)
    _, aux = model.forward_with_aux(tokens, anchors)
    aux.backward()
    for weight in (model.predict_down.weight, model.predict_up.weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in model.answer_head.parameters())
    assert all(torch.equal(value.detach(), frozen[name])
               for name, value in model.state_dict().items())


@pytest.mark.parametrize("scope", ("full", "chunk"))
def test_auxiliary_excludes_delay_even_for_full_attention(scope):
    tokens, anchors, _ = _batch(delay=4)
    changed = tokens.clone()
    changed[:, 8*8 + 4] = 230  # first delay chunk filler, valid symbol
    model = _model(scope)
    with torch.no_grad():
        _, original_aux = model.forward_with_aux(tokens, anchors)
        _, changed_aux = model.forward_with_aux(changed, anchors)
    assert torch.equal(original_aux, changed_aux)


@pytest.mark.parametrize("kwargs", (
    {"attention_scope": "none"}, {"vocab_size": 257}, {"d_model": 64},
    {"chunk_size": 16}, {"heads": 2}, {"ff_width": 89},
    {"predictor_rank": 16},
))
def test_config_changes_fail_closed(kwargs):
    with pytest.raises(ValueError):
        CausalReferenceConfig(**kwargs)


@pytest.mark.parametrize("bad", (
    "float", "wrong_rank", "out_of_range", "anchor_dtype",
    "anchor_position", "missing_read", "missing_query", "missing_write",
    "too_short",
))
def test_reference_rejects_malformed_inputs(bad):
    tokens, anchors, _ = _batch()
    t = tokens.clone()
    a = anchors.clone()
    if bad == "float":
        t = t.float()
    elif bad == "wrong_rank":
        t = t.unsqueeze(0)
    elif bad == "out_of_range":
        t[0, 3] = 256
    elif bad == "anchor_dtype":
        a = a.int()
    elif bad == "anchor_position":
        a[0] -= 1
    elif bad == "missing_read":
        t[0, -8] = 3
    elif bad == "missing_query":
        t[0, -6] = 3
    elif bad == "missing_write":
        t[0, 0] = 3
    elif bad == "too_short":
        t = t[:, :8]
    with pytest.raises(ValueError):
        _model()(t, a)


def test_local_initialization_does_not_mutate_global_torch_random_state():
    original = torch.get_rng_state().clone()
    a = _model("full")
    b = _model("chunk")
    assert torch.equal(torch.get_rng_state(), original)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
