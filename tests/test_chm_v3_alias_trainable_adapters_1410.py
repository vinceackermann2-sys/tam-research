from __future__ import annotations

"""#1410 Stage A strict TRAIN-only tensor math and gradient integrity."""

from dataclasses import replace
import inspect

import pytest
import torch

import tam_research.chm_v3_alias_trainable_adapters_1410 as adapters
from tam_research.chm_v3_balanced_train_schedule_1391 import (
    balanced_training_episode, TRAIN_STEPS,
)
from tam_research.chm_v3_counterfactual_model_view_1365 import seal_model_view
from tam_research.chm_v3_matched_tiny_models_1377 import (
    TinyDecoderOnly, TinyTwoHopPointer, encode_sealed_view,
    train_only_targets, trainable_parameter_count,
)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread_only():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def test_adapters_exact_parameter_and_initial_tensor_parity():
    h_d,a_d,h_p,a_p = adapters.matched_alias_hash_initial_models()
    assert isinstance(h_d,TinyDecoderOnly)
    assert isinstance(a_d,adapters.AliasedTinyDecoderOnly)
    assert isinstance(h_p,TinyTwoHopPointer)
    assert isinstance(a_p,adapters.AliasedTinyTwoHopPointer)
    assert trainable_parameter_count(h_d)==trainable_parameter_count(a_d)==271049
    assert trainable_parameter_count(h_p)==trainable_parameter_count(a_p)==273097
    for old,new in ((h_d,a_d),(h_p,a_p)):
        assert type(old.base) is type(new.base)
        assert list(old.state_dict())==list(new.state_dict())
        assert all(torch.equal(p,q) for p,q in zip(old.parameters(),new.parameters()))
        assert old.base.decoder.layers[0].self_attn.num_heads==4


@pytest.mark.parametrize("step", (0,12,58,130,255))
def test_frozen_math_completely_matches_original_when_input_hash_is_same(step):
    """Catch any reimplementation drift in both forward paths."""
    view=seal_model_view(balanced_training_episode(step))
    hash_input=encode_sealed_view(view)
    h_d,a_d,h_p,a_p=adapters.matched_alias_hash_initial_models()
    for frozen,reimplemented in (
        (h_d,adapters._decoder_math(h_d,hash_input)),
        (h_p,adapters._pointer_math(h_p,hash_input)),
    ):
        original=frozen(view)
        torch.testing.assert_close(reimplemented.answer_logits,original.answer_logits,rtol=0,atol=0)
        if original.first_scores is None:
            assert reimplemented.first_scores is None
            assert reimplemented.second_scores is None
        else:
            torch.testing.assert_close(reimplemented.first_scores,original.first_scores,rtol=0,atol=0)
            torch.testing.assert_close(reimplemented.second_scores,original.second_scores,rtol=0,atol=0)
        assert reimplemented.first_candidate_token_positions == original.first_candidate_token_positions
        assert reimplemented.second_candidate_token_positions == original.second_candidate_token_positions
    aliased=adapters.alias_encoded_view(view)
    assert aliased.line_anchors==hash_input.line_anchors
    assert aliased.code_tokens==hash_input.code_tokens
    assert aliased.memory_length==hash_input.memory_length
    assert len(aliased.token_ids)==len(hash_input.token_ids)


@pytest.mark.parametrize("step", (0,12,58,130,255))
def test_alias_forward_uses_sealed_visible_input_and_train_labels_outside(step):
    ep=balanced_training_episode(step)
    view=seal_model_view(ep)
    for model in adapters.matched_alias_hash_initial_models():
        out=model(view)
        assert out.answer_logits.shape==(9,)
        assert bool(torch.isfinite(out.answer_logits).all().item())
        if isinstance(model,TinyTwoHopPointer):
            assert out.first_scores is not None and out.second_scores is not None
            assert bool(torch.isfinite(out.first_scores).all().item())
            assert bool(torch.isfinite(out.second_scores).all().item())
            gold,first,second=train_only_targets(ep)
            if first is not None:
                assert 0 <= first < adapters.alias_encoded_view(view).memory_length
                assert 0 <= second < len(adapters.alias_encoded_view(view).code_tokens)
            else:
                assert (first,second)==(None,None)
        else:
            assert out.first_scores is None and out.second_scores is None
        with pytest.raises(TypeError):
            model(ep)


def test_exact_balanced_train_geometry_256_and_no_evaluation():
    result=adapters.verify_train_only_encoder_parity()
    assert result["training_split_only"] is True
    assert result["train_examples"]==TRAIN_STEPS==256
    assert result["identical_input_token_lengths"] is True
    assert result["identical_length_total_per_arm"]==19264
    assert result["max_input_length_per_arm"]==83
    assert result["optimizer_updates"]==0
    assert result["scored_heldout_cases"]==0
    assert result["gpu_used"] is False
    assert result["historical_stage_c_status"]=="CHM_V3_100M_DAEC_STAGE_C_STOP"


def test_one_train_step_each_arm_gradient_and_no_generalization_claim():
    r=adapters.one_train_step_four_arm_smoke()
    assert r["smoke_updates_per_arm"]==1
    assert r["same_train_episode_per_arm"] is True
    assert set(r["arms"])=={
        "hash_decoder","alias_decoder","hash_pointer_ce","alias_pointer_ce"
    }
    assert all(c["steps"]==1 and c["finite_train_only_loss"]
               and c["embedding_gradients_present"] for c in r["arms"].values())
    assert r["arms"]["hash_decoder"]["trainable_parameters"]==r["arms"]["alias_decoder"]["trainable_parameters"]
    assert r["arms"]["hash_pointer_ce"]["trainable_parameters"]==r["arms"]["alias_pointer_ce"]["trainable_parameters"]
    assert all(c["train_tokens"]==r["arms"]["hash_decoder"]["train_tokens"]
               for c in r["arms"].values())
    assert all(not c["pointer_gradients_present"] for k,c in r["arms"].items() if k.endswith("decoder"))
    assert all(c["pointer_gradients_present"] for k,c in r["arms"].items() if k.endswith("pointer_ce"))
    assert r["scored_heldout_cases"]==0
    assert r["new_scientific_attempt"] is False
    assert r["gpu_used"] is False


def test_frozen_source_fully_unchanged_and_adapter_does_not_use_oracle_to_forward():
    assert adapters.GPU_AUTHORIZED is False
    assert adapters.SCORED_HELDOUT_AUTHORIZED is False
    assert adapters.FULL_TRAINING_AUTHORIZED is False
    assert adapters.NEW_SCIENTIFIC_ATTEMPT is False
    assert adapters.HISTORICAL_SCIENTIFIC_SEED_CONSUMED==2013161
    assert adapters.ENCODER_PREPROCESSOR_CONFUND_DISCLOSED is True
    for cls in (adapters.AliasedTinyDecoderOnly,adapters.AliasedTinyTwoHopPointer):
        src=inspect.getsource(cls.forward)
        assert "alias_encoded_view(view)" in src
        for forbidden in ("oracle_read(", "train_only_targets(", "ep.", "gold_answer", "variant", "Fact(", "split"):
            assert forbidden not in src
    mod=inspect.getsource(adapters)
    for forbidden in (
        "modal.App(", "torch.cuda", ".cuda(", "torch.load(",
        'generate_episode("development"', 'generate_episode("test"',
        'paired_group("test"', 'paired_group("development"',
        "predict_from_sealed_view(", "score_split(", "heldout_cases(",
    ):
        assert forbidden not in mod
