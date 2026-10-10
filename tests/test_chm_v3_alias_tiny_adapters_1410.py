from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

import tam_research.chm_v3_alias_tiny_adapters_1410 as adapter
from tam_research.chm_v3_matched_tiny_models_1377 import (
    TinyDecoderOnly, TinyTwoHopPointer, encode_sealed_view,
    learning_objective, trainable_parameter_count, predict_from_sealed_view,
)
from tam_research.chm_v3_balanced_train_schedule_1391 import balanced_training_episode
from tam_research.chm_v3_counterfactual_model_view_1365 import seal_model_view


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize(
    "legacy,aliased",
    [
        (TinyDecoderOnly.forward, adapter.TinyAliasDecoderOnly.forward),
        (TinyTwoHopPointer.forward, adapter.TinyAliasTwoHopPointer.forward),
    ],
)
def test_forward_source_ast_identical_except_one_explicit_visible_encoder_call(
    legacy, aliased,
):
    original = textwrap.dedent(inspect.getsource(legacy))
    source = textwrap.dedent(inspect.getsource(aliased))
    assert source.count("_encode_alias_view(view)") == 1
    normalized = source.replace("_encode_alias_view(view)", "encode_sealed_view(view)")
    # AST parity enforces EXACT forward math, even for soft-hop / copy mixture.
    assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(
        ast.parse(original), include_attributes=False
    )


def test_adapter_initial_parameters_and_parameter_counts_are_exactly_matched():
    models = adapter.matched_original_and_alias_models()
    assert set(models) == {
        "whole_decoder", "alias_decoder", "whole_pointer_ce", "alias_pointer_ce"
    }
    count = {name: trainable_parameter_count(model) for name,model in models.items()}
    assert count == {
        "whole_decoder": 271049,
        "alias_decoder": 271049,
        "whole_pointer_ce": 273097,
        "alias_pointer_ce": 273097,
    }
    for original, adapted in (("whole_decoder","alias_decoder"),
                              ("whole_pointer_ce","alias_pointer_ce")):
        assert all(torch.equal(models[original].state_dict()[k],
                               models[adapted].state_dict()[k])
                   for k in models[original].state_dict())


@pytest.mark.parametrize("step", (0,1,2,3,12,31,80))
def test_alias_changes_only_identifier_token_ids_and_never_source_indices(step):
    ep = balanced_training_episode(step)
    assert ep.split == "train"
    view = seal_sealed = seal_model_view(ep)
    x, y = encode_sealed_view(view), adapter._encode_alias_view(view)
    assert len(x.token_ids) == len(y.token_ids)
    assert x.memory_length == y.memory_length
    assert x.line_anchors == y.line_anchors
    assert x.code_tokens == y.code_tokens
    assert len(x.token_ids) < 384
    assert x.token_ids[-1] == y.token_ids[-1]


@pytest.mark.parametrize("step", (0,1,2,3))
def test_forward_outputs_and_train_only_losses_are_finite(step):
    ep = balanced_training_episode(step)
    models = adapter.matched_original_and_alias_models()
    for name, model in models.items():
        view = seal_model_view(ep)
        result = model(view)
        assert result.answer_logits.shape == (9,)
        assert bool(torch.isfinite(result.answer_logits).all().item())
        result_info = predict_from_sealed_view(model, view)
        assert result_info["predicted_answer_id"] is None or result_info[
            "predicted_answer_id"
        ] in (18001,18002,18003,18004,18005,18006,18007,18008)
        loss, details = learning_objective(
            model, ep, pointer_supervision=name.endswith("pointer_ce"),
        )
        assert bool(torch.isfinite(loss).item())
        assert details["answer_nll"] > 0
        if name.endswith("pointer_ce") and ep.gold_answer is not None:
            assert details["address_ce"] > 0
        loss.backward()
        assert model.base.embedding.weight.grad is not None
        assert bool(torch.isfinite(model.base.embedding.weight.grad).all().item())
        if "pointer" in name:
            assert model.query_projection.weight.grad is not None


def test_model_forward_refuses_full_episode_and_all_hidden_oracle_metadata():
    models = adapter.matched_original_and_alias_models()
    ep = balanced_training_episode(9)
    for model in models.values():
        with pytest.raises(TypeError):
            model(ep)
    source = inspect.getsource(adapter.TinyAliasTwoHopPointer.forward)
    for forbidden in (
        "oracle_read(", ".gold_answer", ".variant", ".facts", ".split",
        "train_only_targets(", "answer_line", "gold_position",
    ):
        assert forbidden not in source


def test_exactly_one_cpu_train_only_optimizer_step_is_permitted():
    info = adapter.cpu_train_only_smoke(steps=1)
    assert info["classification"] == "CHM_V3_1410_ALIASED_MODEL_CPU_TRAIN_ONLY_SMOKE"
    assert info["samples_per_arm"] == 1
    assert info["scored_heldout_constructed"] is False
    assert info["model_training_on_dev_test"] is False
    assert info["old_tokenizer_training_sample_tokens"] == (
        info["aliased_tokenizer_training_sample_tokens"]
    )
    assert info["trainable_parameters"]["whole_decoder"] == (
        info["trainable_parameters"]["alias_decoder"]
    )
    assert info["trainable_parameters"]["whole_pointer_ce"] == (
        info["trainable_parameters"]["alias_pointer_ce"]
    )
    assert info["gpu_used"] is False
    assert info["new_scientific_attempt"] is False
    assert info["old_100m_scientific_classification"] == (
        "CHM_V3_100M_DAEC_STAGE_C_STOP"
    )
    for invalid in (0,2,256,True,1.0):
        with pytest.raises(ValueError,match="only exactly one"):
            adapter.cpu_train_only_smoke(steps=invalid)


def test_source_no_modal_gpu_new_splits_or_old_science_run():
    assert adapter.GPU_AUTHORIZED is False
    assert adapter.SCORED_HELDOUT_AUTHORIZED is False
    assert adapter.CONSUMED_100M_SCIENCE_SEED == 2013161
    src = inspect.getsource(adapter)
    for forbidden in (
        "modal.App(", "modal.run(", "torch.cuda", ".cuda(", "torch.load(",
        'generate_episode("test"', 'generate_episode("development"',
        "checkpoint.load(",
    ):
        assert forbidden not in src
    assert "balanced_training_episode(step)" in src
