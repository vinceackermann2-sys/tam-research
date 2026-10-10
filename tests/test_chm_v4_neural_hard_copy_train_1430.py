from __future__ import annotations

import ast
from dataclasses import replace
import inspect

import pytest
import torch

import tam_research.chm_v4_neural_hard_copy_train_1430 as neural
from tam_research.chm_v4_independent_binding_benchmark_1421 import (
    CODES, FAMILIES, SealedInput, train_counterfactual_group, seal_input,
    oracle_read,
)
from tam_research.chm_v3_matched_tiny_models_1377 import trainable_parameter_count


@pytest.fixture(scope="module",autouse=True)
def one_cpu_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("entity_index",(0,1,7,31,63))
def test_v4_encoder_uses_only_visible_memory_anchor_and_code_tokens(family,entity_index):
    episode=train_counterfactual_group(family,entity_index)[0]
    assert episode.split=="train"
    view=seal_input(episode)
    encoded=neural.encode_visible_v4(view)
    assert len(encoded.model_tokens.token_ids)<=384
    assert len(encoded.code_candidate_positions)==8
    assert set(encoded.code_candidate_ids)==set(CODES)
    assert len(encoded.code_candidate_source_physical)==8
    assert len(encoded.first_source_physical)==len(episode.records)
    assert encoded.first_source_physical==tuple(r.position for r in episode.records)
    assert all(pos in encoded.first_source_physical for pos in
               encoded.code_candidate_source_physical)
    assert any(s.startswith("V4-") for s in neural.V4_TOKEN_PATTERN.findall(view.query))
    if family=="overwrite":
        assert len(encoded.model_tokens.code_tokens)==9
    else:
        assert len(encoded.model_tokens.code_tokens)==8


def test_hard_output_copies_from_own_code_choice_never_hidden_gold():
    episode=train_counterfactual_group("two_hop",0)[0]
    encoded=neural.encode_visible_v4(seal_input(episode))
    choice=3
    line_logits=torch.zeros(len(encoded.first_source_physical))
    line_logits[2]=10.0
    code_logits=torch.zeros(8)
    code_logits[choice]=10.0
    present=neural.PointerV4Output(
        presence_logits=torch.tensor([0.0,1.0]),
        first_line_logits=line_logits,
        second_code_logits=code_logits,
        first_source_physical=encoded.first_source_physical,
        code_source_physical=encoded.code_candidate_source_physical,
        code_candidate_ids=encoded.code_candidate_ids,
    )
    predicted=neural.hard_copy_prediction(present)
    assert predicted.predicted_answer_id==encoded.code_candidate_ids[choice]
    assert predicted.payload_source_physical==encoded.code_candidate_source_physical[choice]
    assert predicted.first_source_physical==encoded.first_source_physical[2]
    assert not predicted.did_abstain
    absent=replace(present,presence_logits=torch.tensor([1.0,0.0]))
    abstained=neural.hard_copy_prediction(absent)
    assert abstained.did_abstain and abstained.predicted_answer_id is None
    # No oracle/source-target logic inside output decoding.
    src=inspect.getsource(neural.hard_copy_prediction)
    assert "oracle_read(" not in src and ".gold_answer" not in src


def test_train_only_positive_loss_addresses_and_nonmatch_gate():
    model=neural.TinyV4HardCopyPointer()
    for family in FAMILIES:
        episode=train_counterfactual_group(family,3)[0]
        view=seal_input(episode)
        output=model(view)
        assert output.presence_logits.shape==(2,)
        assert output.first_line_logits.shape==(len(episode.records),)
        assert output.second_code_logits.shape==(8,)
        assert bool(torch.isfinite(output.presence_logits).all().item())
        loss,components=neural.train_only_supervised_loss(output,episode)
        assert bool(torch.isfinite(loss).item())
        assert components["presence_ce"]>0
        if family=="no_match":
            assert components["first_address_ce"]==0
            assert components["payload_address_ce"]==0
        else:
            assert components["first_address_ce"]>0
            assert components["payload_address_ce"]>0
        copy=neural.hard_copy_prediction(output)
        assert copy.predicted_answer_id in (*CODES,None)
        assert copy.first_source_physical in (r.position for r in episode.records)
        assert copy.payload_source_physical in (r.position for r in episode.records)


def test_model_rejects_evaluator_only_episode_hidden_labels_oracle():
    model=neural.TinyV4HardCopyPointer()
    episode=train_counterfactual_group("direct",0)[0]
    with pytest.raises(TypeError,match="SealedInput"):
        model(episode)
    src=inspect.getsource(model.forward)
    for forbidden in (
        "oracle_read(", ".gold_answer", ".records", ".family", ".variant",
        ".split", "train_only_supervised_loss", "source_target",
    ):
        assert forbidden not in src


def test_encoder_rejects_invalid_source_addresses_and_code_bags():
    episode=train_counterfactual_group("direct",0)[0]
    view=seal_input(episode)
    with pytest.raises(TypeError,match="SealedInput"):
        neural.encode_visible_v4(episode)
    with pytest.raises(ValueError,match="options"):
        neural.encode_visible_v4(replace(view,answer_options=("CODE-18001",)))
    with pytest.raises(ValueError,match="duplicate"):
        neural.encode_visible_v4(replace(
            view,memory_text=view.memory_text+"\n"+view.memory_text.splitlines()[0]
        ))
    with pytest.raises(ValueError,match="CODE candidates"):
        neural.encode_visible_v4(replace(
            view,memory_text=view.memory_text.replace("CODE-18002","CODE-18001",1)
        ))
    with pytest.raises(ValueError,match="source"):
        neural.encode_visible_v4(replace(
            view,memory_text=view.memory_text.replace("[","X",1)
        ))
    with pytest.raises(ValueError,match="unknown"):
        neural.encode_visible_v4(replace(
            view,memory_text=view.memory_text.replace("CODE-18001","CODE-18099",1)
        ))


def test_exactly_one_train_only_cpu_optimizer_step_with_nonzero_gradients():
    result=neural.cpu_one_train_step_smoke(steps=1)
    assert result["classification"]==(
        "CHM_V4_1430_TRAIN_ONLY_NEURAL_HARD_COPY_GRADIENT_SMOKE"
    )
    assert result["train_only_steps"]==1
    assert result["training_examples_seen"]==1
    assert 0<result["visible_input_token_count"]<=384
    assert result["model_parameters"]==trainable_parameter_count(
        neural.TinyV4HardCopyPointer()
    )
    assert set(result["abs_grad_sum"])=={
        "first_query","first_key","second_query","gate_head","base_embeddings"
    }
    assert all(value>0 for value in result["abs_grad_sum"].values())
    assert result["preupdate_output_is_hard_copy_of_visible_code_or_abstention"] is True
    assert result["soft_first_hop_then_hard_final_copy"] is True
    assert result["genuinely_hard_conditioned_second_hop"] is False
    assert result["scored_dev_test_cases_constructed"] is False
    assert result["gpu_used"] is False
    assert result["new_scientific_attempt"] is False
    assert result["historical_scientific_status"]=="CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert result["historical_scientific_seed_consumed"]==2013161
    for invalid in (0,2,256,True,1.0):
        with pytest.raises(ValueError,match="exactly one"):
            neural.cpu_one_train_step_smoke(steps=invalid)


def test_ast_no_devtest_generator_gpu_modal_or_hidden_oracle_in_forward():
    tree=ast.parse(inspect.getsource(neural))
    funcs={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    disallowed={
        "modal.App","modal.run","torch.cuda","torch.load","torch.save",
        "generate_episode","v4.generate_episode","optimizer.load_state_dict",
        "model.eval",
    }
    assert disallowed.isdisjoint(funcs)
    assert "train_counterfactual_group" in funcs
    assert neural.GPU_AUTHORIZED is False
    assert neural.HELDOUT_SCORING_AUTHORIZED is False
    assert neural.NEW_SCIENTIFIC_ATTEMPT is False
    assert "h = self.base(encoded.model_tokens)" in inspect.getsource(
        neural.TinyV4HardCopyPointer.forward
    )
    src=inspect.getsource(neural.TinyV4HardCopyPointer.forward)
    assert "line-end" in src or "starts[k+1]" in src
    assert "oracle_read" not in src
