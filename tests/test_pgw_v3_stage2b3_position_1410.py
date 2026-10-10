"""Stage-2B3 #1411: global-position causal reference + static counts, no training."""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from tam_research.pgw_v3_stage1b.oracle import make_example
from tam_research.pgw_v3_stage2a.model import (
    PGWV3Stage2A, batch_verified_examples, instantiated_parameter_count,
)
from tam_research.pgw_v3_stage2b2.reference import (
    CausalReference, CausalReferenceConfig, count_instantiated,
)
from tam_research.pgw_v3_stage2b3.position import (
    GlobalPositionCausalReference, absolute_sinusoidal_positions,
)
from tam_research.pgw_v3_stage2b3.accounting import attention_operation_counts


def _model(global_positions=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1411)  # local unit test initialization, not science seed
        if global_positions:
            return GlobalPositionCausalReference().cpu().eval()
        return CausalReference(CausalReferenceConfig(attention_scope="full")).cpu().eval()


def _batch(delay=1):
    examples = [
        make_example(
            split="validation", index=i, last_position=5,
            overwrite_count=2, interference=True, delay_chunks=delay,
            missing=missing,
        )
        for i, missing in ((3, False), (4, True))
    ]
    return batch_verified_examples(examples)


@pytest.mark.parametrize("length", (80, 88, 104))
def test_sinusoid_is_deterministic_global_and_parameter_free(length):
    pe = absolute_sinusoidal_positions(length, width=32)
    assert pe.shape == (length, 32)
    assert pe.dtype == torch.float32
    assert pe.device.type == "cpu"
    assert torch.isfinite(pe).all()
    assert torch.equal(pe, absolute_sinusoidal_positions(length, width=32))
    assert torch.equal(pe[0, 0::2], torch.zeros(16))
    assert torch.equal(pe[0, 1::2], torch.ones(16))
    assert not torch.equal(pe[2], pe[10])
    assert torch.testing.assert_close(pe[1, 0], torch.sin(torch.tensor(1.0))) is None
    assert torch.testing.assert_close(pe[1, 1], torch.cos(torch.tensor(1.0))) is None


def test_global_position_separates_same_token_at_same_local_offset():
    tokens, _, _ = _batch()
    same_token = tokens.clone()
    same_token[0, 2] = same_token[0, 10] = 39  # both are valid event value IDs
    old = _model(False)
    glob = _model(True)
    glob.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        local_emb = old.input_embeddings(same_token)
        global_emb = glob.input_embeddings(same_token)
    assert torch.equal(local_emb[0, 2], local_emb[0, 10])
    assert not torch.equal(global_emb[0, 2], global_emb[0, 10])
    pe = absolute_sinusoidal_positions(same_token.shape[1])
    torch.testing.assert_close(global_emb - local_emb, pe[None].expand_as(global_emb), rtol=1e-6, atol=1e-6)

    swapped = same_token.clone()
    swapped[0, 2], swapped[0, 10] = 40, 41
    reversed_events = swapped.clone()
    reversed_events[0, 2], reversed_events[0, 10] = 41, 40
    with torch.no_grad():
        a = glob.input_embeddings(swapped)
        b = glob.input_embeddings(reversed_events)
    assert not torch.equal(a[0, [2, 10]], b[0, [2, 10]])
    # Intervening READ labels are NOT used to generate these representations.


def test_state_dict_and_counts_identical_to_frozen_original_reference():
    old = _model(False)
    new = _model(True)
    assert set(old.state_dict()) == set(new.state_dict())
    assert all(torch.equal(x, new.state_dict()[name]) for name, x in old.state_dict().items())
    assert count_instantiated(new) == count_instantiated(old) == 20347
    assert instantiated_parameter_count(PGWV3Stage2A()) == 20354
    assert len(list(new.named_buffers())) == len(list(old.named_buffers())) == 0
    with pytest.raises(ValueError, match="requires full attention"):
        GlobalPositionCausalReference(CausalReferenceConfig(attention_scope="chunk"))


@pytest.mark.parametrize("delay", (1, 2, 4))
def test_global_reference_validated_inputs_future_safe_aux_and_counts(delay):
    tokens, anchors, targets = _batch(delay)
    new = _model(True)
    old = _model(False)
    old.load_state_dict(new.state_dict(), strict=True)
    future = tokens.clone()
    future[:, -1] = 231
    future[:, -2] = 232
    with torch.no_grad():
        logits, aux = new.forward_with_aux(tokens, anchors)
        logits2, aux2 = new.forward_with_aux(future, anchors)
        old_logits = old(tokens, anchors)
    assert logits.shape == (2, 33)
    assert targets.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux) and aux.item() > 0
    torch.testing.assert_close(logits, logits2, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(aux, aux2, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(logits, old_logits, rtol=1e-6, atol=1e-6)


def test_past_only_attention_and_causal_cross_chunk_gradient():
    tokens, anchors, _ = _batch()
    new = _model(True)
    inputs = new.input_embeddings(tokens).detach().requires_grad_(True)
    h = new.encode_embeddings(inputs)
    new.answer_from_hidden(h, anchors).square().sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad[:, :64, :].abs().sum().item() > 0
    assert inputs.grad[:, -8:-5, :].abs().sum().item() > 0
    assert torch.count_nonzero(inputs.grad[:, -5:, :]).item() == 0

    # Full causal block cannot make prefix output depend on future input.
    changed = inputs.detach().clone()
    changed[:, 30:, :] += 0.25
    with torch.no_grad():
        earlier = new.encode_embeddings(inputs.detach())[:, :25]
        altered = new.encode_embeddings(changed)[:, :25]
    torch.testing.assert_close(earlier, altered, rtol=1e-6, atol=1e-6)


def test_aux_only_predictor_and_answer_only_head_gradient_isolation():
    tokens, anchors, targets = _batch(delay=2)
    model = _model(True)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    F.cross_entropy(model(tokens, anchors), targets).backward()
    assert model.predict_down.weight.grad is None
    assert model.predict_up.weight.grad is None
    assert model.answer_head.weight.grad is not None
    model.zero_grad(set_to_none=True)
    _, aux = model.forward_with_aux(tokens, anchors)
    aux.backward()
    for w in (model.predict_down.weight, model.predict_up.weight):
        assert w.grad is not None and torch.isfinite(w.grad).all()
        assert w.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in model.answer_head.parameters())
    assert all(torch.equal(x, original[k]) for k, x in model.state_dict().items())


@pytest.mark.parametrize("bad", ("float", "anchor", "future_marker", "write_marker", "out_of_range"))
def test_global_reference_inherits_frozen_fail_closed_guards(bad):
    t, a, _ = _batch()
    t, a = t.clone(), a.clone()
    if bad == "float":
        t = t.float()
    elif bad == "anchor":
        a -= 1
    elif bad == "future_marker":
        t[0, -8] = 8
    elif bad == "write_marker":
        t[0, 0] = 8
    elif bad == "out_of_range":
        t[0, 2] = 300
    with pytest.raises(ValueError):
        _model(True)(t, a)


@pytest.mark.parametrize("length", (80, 88, 104))
def test_operation_accounting_counts_fairly_labeled(length):
    b = 2
    counts = attention_operation_counts(batch=b, tokens=length)
    chunks = length // 8
    assert counts.chunks == chunks and counts.head_width == 8
    assert counts.full_dense_pair_ops == 2*b*4*length**2
    assert counts.full_causal_pair_ops == b*4*length*(length+1)
    assert counts.chunk_dense_pair_ops == 2*b*4*chunks*8**2
    assert counts.chunk_causal_pair_ops == b*4*chunks*8*9
    assert counts.full_dense_qk_av_macs == counts.full_dense_pair_ops * 8
    assert counts.chunk_dense_qk_av_macs == counts.chunk_dense_pair_ops * 8
    assert counts.pgw_selected_write_calls == b*chunks*2
    assert counts.pgw_read_slot_key_comparisons == b*length*4
    assert counts.pgw_write_slot_key_comparisons == b*chunks*2*4
    assert counts.full_dense_pair_ops > counts.chunk_dense_pair_ops
    assert counts.full_causal_pair_ops > counts.chunk_causal_pair_ops


@pytest.mark.parametrize("kwargs", (
    {"batch": 0, "tokens": 80}, {"batch": 1, "tokens": 81},
    {"batch": 1, "tokens": 8}, {"batch": 1.5, "tokens": 80},
    {"batch": True, "tokens": 80}, {"batch": 1, "tokens": 80, "heads": 8},
    {"batch": 1, "tokens": 80, "width": 64},
    {"batch": 1, "tokens": 80, "workspace_slots": 3},
    {"batch": 1, "tokens": 80, "selected_events": 1},
))
def test_accounting_rejects_nonfrozen_geometry(kwargs):
    with pytest.raises(ValueError):
        attention_operation_counts(**kwargs)


@pytest.mark.parametrize("length,width", ((0,32), (80,31), (80,0), (True,32)))
def test_sinusoid_rejects_bad_geometry(length, width):
    with pytest.raises(ValueError):
        absolute_sinusoidal_positions(length, width=width)


def test_no_global_random_state_consumption():
    state = torch.get_rng_state().clone()
    a = absolute_sinusoidal_positions(104)
    b = attention_operation_counts(batch=2, tokens=104)
    assert a.shape == (104, 32)
    assert b.chunks == 13
    assert torch.equal(state, torch.get_rng_state())
