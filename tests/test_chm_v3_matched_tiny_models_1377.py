from __future__ import annotations

import inspect
import math
from dataclasses import fields, replace

import pytest
import torch

import tam_research.chm_v3_matched_tiny_models_1377 as models
import tam_research.chm_v3_counterfactual_multiset_suite_1373 as suite
from tam_research.chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from tam_research.chm_v3_counterfactual_model_view_1365 import (
    SealedModelView, oracle_read, seal_model_view,
)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread_for_tiny_smoke():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("family", ("direct", "overwrite", "two_hop", "no_match"))
def test_sealed_tokenizer_is_deterministic_and_does_not_require_gold_label(family):
    ep = suite.generate_episode("test", family, 2, 5)
    view = seal_model_view(ep)
    assert list(vars(view)) == ["query", "memory_text", "answer_options"]
    a, b = models.encode_sealed_view(view), models.encode_sealed_view(view)
    assert a == b
    assert 8 < a.memory_length < len(a.token_ids) <= models.MAX_TOKENS
    assert len(a.code_tokens) == (9 if family == "overwrite" else 8)
    assert len(set(a.line_anchors)) == len(a.line_anchors)
    assert all(pos < a.memory_length for pos, _ in a.code_tokens)
    assert all(0 <= token < models.VOCAB_SIZE for token in a.token_ids)
    assert set(code for _, code in a.code_tokens).issubset(set(ANSWER_IDS) | {19001})
    assert not hasattr(view, "variant") and not hasattr(view, "gold_answer")
    with pytest.raises(TypeError, match="SealedModelView"):
        models.encode_sealed_view(ep)


@pytest.mark.parametrize("family", ("direct", "overwrite", "two_hop", "no_match"))
def test_gold_token_targets_bound_only_in_training_loss(family):
    ep = suite.generate_episode("train", family, 0, 2)
    a, b, c = models.train_only_targets(ep)
    if family == "no_match":
        assert (a, b, c) == (8, None, None)
    else:
        assert a == ANSWER_IDS.index(ep.gold_answer)
        assert b is not None and c is not None
        visible = models.encode_sealed_view(seal_model_view(ep))
        assert b < visible.memory_length
        assert c < len(visible.code_tokens)
        assert visible.code_tokens[c][1] == ep.gold_answer
        _, relation, answer_pos = oracle_read(ep)
        if family == "two_hop":
            assert relation is not None
            assert dict(visible.line_anchors)[relation] == b
        else:
            assert dict(visible.line_anchors)[answer_pos] == b
    source = inspect.getsource(models.TinyTwoHopPointer.forward)
    for prohibited in ("ep.gold_answer", "oracle_read(", "ep.variant", "ep.facts",
                       "ep.split", "train_only_targets("):
        assert prohibited not in source


def test_backbone_same_initialization_and_parameters_within_one_percent():
    a, b, c = models.matched_initial_models()
    na, nb, nc = (models.trainable_parameter_count(x) for x in (a, b, c))
    assert na > 0 and nb > 0 and nc == nb
    assert abs(nb - na) / nb <= 0.01
    for p1, p2 in zip(a.base.parameters(), b.base.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(b.parameters(), c.parameters()):
        assert torch.equal(p1, p2)
    assert len(list(b.base.decoder.layers)) == 1
    assert b.base.decoder.layers[0].self_attn.num_heads == 4


def test_all_three_arms_forward_finite_on_exact_same_sealed_input():
    a, b, c = models.matched_initial_models()
    for family in ("direct", "overwrite", "two_hop", "no_match"):
        view = seal_model_view(suite.generate_episode("development", family, 1, 4))
        for model in (a, b, c):
            out = model(view)
            assert out.answer_logits.shape == (9,)
            assert torch.isfinite(out.answer_logits).all().item()
            predicted = models.predict_from_sealed_view(model, view)
            assert predicted["predicted_answer_id"] in (*ANSWER_IDS, None)
            assert 0 < predicted["confidence"] <= 1
            assert predicted["abstained"] == (predicted["predicted_answer_id"] is None)
            if isinstance(model, models.TinyDecoderOnly):
                assert out.first_scores is None and out.second_scores is None
                assert predicted["first_selected_visible_token_index"] is None
                assert predicted["second_selected_visible_token_index"] is None
            else:
                assert out.first_scores is not None
                assert out.second_scores is not None
                assert torch.isfinite(out.first_scores).all().item()
                assert torch.isfinite(out.second_scores).all().item()
                assert 0 <= predicted["first_selected_visible_token_index"] < (
                    models.encode_sealed_view(view).memory_length
                )
                assert predicted["second_selected_visible_token_index"] in (
                    i for i, _ in models.encode_sealed_view(view).code_tokens
                )


def test_loss_gradients_reach_pointer_and_base_on_positive_training_case():
    a, b, c = models.matched_initial_models()
    ep = suite.generate_episode("train", "two_hop", 0, 3)
    for model, supervised in ((a, False), (b, True), (c, False)):
        loss, metrics = models.learning_objective(
            model, ep, pointer_supervision=supervised,
        )
        assert torch.isfinite(loss).item()
        assert metrics["answer_nll"] > 0
        assert (metrics["address_ce"] > 0) == supervised
        loss.backward()
        assert model.base.embedding.weight.grad is not None
        assert torch.isfinite(model.base.embedding.weight.grad).all().item()
        if isinstance(model, models.TinyTwoHopPointer):
            assert model.query_projection.weight.grad is not None
            assert torch.isfinite(model.query_projection.weight.grad).all().item()
            assert model.key_projection.weight.grad is not None
            assert torch.isfinite(model.key_projection.weight.grad).all().item()


def test_training_schedule_only_uses_training_split_and_is_bounded():
    for step in range(48):
        ep = models.training_episode(step)
        assert ep.split == "train"
        assert ep.family in ("direct", "overwrite", "two_hop", "no_match")
        assert ep.gold_answer is None or ep.gold_answer in ANSWER_IDS
    for invalid in (-1, 48, 2013161):
        with pytest.raises(ValueError, match="envelope"):
            models.training_episode(invalid)
    with pytest.raises(ValueError, match="cap"):
        models.train_cpu_models(steps=0)
    with pytest.raises(ValueError, match="cap"):
        models.train_cpu_models(steps=49)


def test_single_training_step_has_all_three_arms_and_no_evaluation_peek():
    a, b, c, meta = models.train_cpu_models(steps=1)
    assert meta["classification"] == "CHM_V3_1377_TINY_CPU_ENGINEERING_ONLY"
    assert meta["steps_per_arm"] == 1
    assert meta["training_example_count_per_arm"] == 1
    assert meta["training_token_count_per_arm"] > 0
    assert meta["no_gpu"] is True
    assert meta["new_scientific_attempt"] is False
    assert meta["historical_scientific_status"] == models.ORIGINAL_SCIENTIFIC_STATUS
    assert meta["parameters"]["pointer_ce"] == meta["parameters"]["pointer_no_ce"]
    assert isinstance(a, models.TinyDecoderOnly)
    assert isinstance(b, models.TinyTwoHopPointer)
    assert isinstance(c, models.TinyTwoHopPointer)
    with pytest.raises(TypeError):
        b(suite.generate_episode("test", "direct", 0, 1))


def test_cpu_only_source_no_modal_checkpoint_or_history_mutation():
    src = inspect.getsource(models)
    assert models.GPU_AUTHORIZED is False
    assert models.SCIENTIFIC_SEED_CONSUMED == 2013161
    assert models.ORIGINAL_SCIENTIFIC_STATUS == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    for banned in ("modal.App(", "modal.run(", "torch.cuda", ".cuda(", "checkpoint.load(",
                   "torch.load(", "generate_aligned_probe_suite("):
        assert banned not in src
    assert "generate_episode(\"train\"" in src
