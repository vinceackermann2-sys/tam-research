from __future__ import annotations

"""#1431: static TRAIN-only source/copy/gate intervention factorization."""

import ast
from collections import Counter
from dataclasses import replace
import inspect

import pytest

import tam_research.chm_v4_independent_binding_benchmark_1421 as v4
import tam_research.chm_v4_factorized_interventions_1431 as fac
from tam_research.chm_v4_visible_hard_copy_train_1427 import (
    _parse_completed_memory, read_visible_text,
)


@pytest.mark.parametrize("family",v4.FAMILIES)
@pytest.mark.parametrize("index",(0,1,7,23,42,63))
def test_factorized_reader_agrees_with_original_only_on_train(family,index):
    episodes=v4.train_counterfactual_group(family,index)
    assert len(episodes)==8
    seen=[]
    for ep in episodes:
        assert ep.split=="train"
        visible=v4.seal_input(ep)
        source=read_visible_text(visible)
        hard=fac.copy_from_proposed_source(visible,source.first_physical_source)
        gated=fac.gate_copied_value(hard.copied_code)
        gold, relation, payload=v4.oracle_read(ep)
        first=relation if relation is not None else payload
        assert gated.candidate_code==gold==ep.gold_answer
        assert hard.first_physical_source==first
        assert hard.payload_physical_source==payload
        assert gated.abstained==(family=="no_match")
        assert hard.route==("absent" if family=="no_match" else
                            "relation" if family=="two_hop" else "direct")
        if payload is not None:
            assert fac.copy_value_at_explicit_payload(visible,payload)==gold
        seen.append(gated.candidate_code)
    assert len(set(seen))==(1 if family=="no_match" else 8)


@pytest.mark.parametrize("family",v4.POSITIVES)
def test_proposed_wrong_visible_decoy_address_and_value_both_flip_answer(family):
    for index in (0,1,8,21,39,63):
        ep=v4.train_counterfactual_group(family,index)[4]
        view=v4.seal_input(ep)
        correct=read_visible_text(view)
        correct_code=correct.answer_id
        candidates=[r for r in _parse_completed_memory(view)
                    if r.code in v4.CODES and r.code!=correct_code]
        assert len(candidates)==7
        for row in candidates:
            forced=fac.copy_from_proposed_source(view,row.position)
            assert forced.copied_code==row.code
            assert forced.copied_code!=correct_code
            assert fac.copy_value_at_explicit_payload(view,row.position)==row.code
            assert fac.gate_copied_value(forced.copied_code).candidate_code!=correct_code
        assert fac.gate_copied_value(
            correct_code,policy="force_abstain"
        )==fac.GatedAnswer(None,True)


def test_two_hop_old_relation_and_new_relation_are_separate_first_sources():
    for index in (0,1,13,39,63):
        ep=v4.train_counterfactual_group("two_hop",index)[1]
        view=v4.seal_input(ep)
        relations=sorted(
            (r for r in _parse_completed_memory(view)
             if r.subject==ep.entity and r.linked_document is not None),
            key=lambda r:r.position,
        )
        assert len(relations)==2
        outdated=fac.copy_from_proposed_source(view,relations[0].position)
        current=fac.copy_from_proposed_source(view,relations[1].position)
        assert outdated.route==current.route=="relation"
        assert outdated.first_physical_source!=current.first_physical_source
        assert outdated.payload_physical_source!=current.payload_physical_source
        assert outdated.copied_code!=current.copied_code==ep.gold_answer


def test_overwrite_outdated_noncandidate_and_no_match_forced_answer():
    for index in (0,2,12,33,63):
        ep=v4.train_counterfactual_group("overwrite",index)[5]
        view=v4.seal_input(ep)
        old=[r for r in _parse_completed_memory(view)
             if r.subject==ep.entity and r.code==v4.STALE_CODE]
        assert len(old)==1
        with pytest.raises(ValueError,match="no valid candidate"):
            fac.copy_from_proposed_source(view,old[0].position)
        absent=v4.train_counterfactual_group("no_match",index)[0]
        gone=v4.seal_input(absent)
        expected=fac.copy_from_proposed_source(gone,None)
        assert expected==fac.PhysicalCopy(None,None,None,"absent")
        assert fac.gate_copied_value(expected.copied_code)==fac.GatedAnswer(None,True)
        wrong=fac.gate_copied_value(None,policy="force_default")
        assert wrong.abstained is False and wrong.candidate_code==v4.CODES[0]


def test_fail_closed_on_invalid_external_source_payload_and_gate():
    ep=v4.train_counterfactual_group("direct",0)[0]
    view=v4.seal_input(ep)
    with pytest.raises(TypeError):
        fac.copy_from_proposed_source(ep,12)
    with pytest.raises(TypeError):
        fac.copy_from_proposed_source(view,True)
    with pytest.raises(ValueError,match="missing"):
        fac.copy_from_proposed_source(view,127+200)
    with pytest.raises(ValueError,match="payload"):
        fac.copy_value_at_explicit_payload(view,999)
    with pytest.raises(TypeError):
        fac.copy_value_at_explicit_payload(view,None)
    with pytest.raises(ValueError):
        fac.gate_copied_value(19001)
    with pytest.raises(ValueError):
        fac.gate_copied_value(None,policy="invalid")
    with pytest.raises(ValueError,match="candidate"):
        fac.copy_from_proposed_source(
            replace(view,answer_options=("CODE-18001",)),None
        )


def test_entire_train_only_suite_and_counterfactual_intervention_counts():
    result=fac.train_only_factorized_audit()
    assert result==fac.train_only_factorized_audit()
    assert result["classification"]==fac.CLASSIFICATION
    assert result["training_cases_only"]==2048
    assert result["train_independent_entities"]==64
    for x in (
        "natural_answer_correct_by_family",
        "natural_first_address_correct_by_family",
        "natural_payload_address_correct_by_family",
    ):
        assert result[x]==dict.fromkeys(v4.FAMILIES,512)
    for x in (
        "wrong_first_source_changes_answer_by_positive_family",
        "wrong_payload_source_changes_answer_by_positive_family",
        "forced_abstain_loses_answer_by_positive_family",
    ):
        assert result[x]==dict.fromkeys(v4.POSITIVES,512)
    assert result["outdated_two_hop_relation_changes_answer"]==512
    assert result["stale_overwrite_first_source_rejected"]==512
    assert result["no_match_natural_abstentions"]==512
    assert result["no_match_forced_default_false_positives"]==512
    assert result["development_test_examples_created"]==0
    assert result["trained_model_forward_calls"]==0
    assert result["optimizer_steps"]==0
    assert result["gpu_used"] is False
    assert result["historical_status"]=="CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert result["historical_seed_consumed"]==2013161
    assert "no neural learning" in result["interpretation"]


def test_train_audit_runtime_guard_rejects_all_heldout_generation(monkeypatch):
    original=v4.generate_episode
    splits=Counter()
    def guard(split,family,index,variant):
        splits[split]+=1
        if split!="train":
            raise RuntimeError("attempted heldout generation")
        return original(split,family,index,variant)
    monkeypatch.setattr(v4,"generate_episode",guard)
    fac.train_only_factorized_audit()
    assert splits==Counter({"train":2048})


def test_static_forward_apis_never_receive_gold_or_model_metadata():
    assert fac.GPU_AUTHORIZED is False
    assert fac.SCORED_HELDOUT_AUTHORIZED is False
    assert fac.SCIENTIFIC_EXECUTION_AUTHORIZED is False
    for fn in (
        fac.copy_from_proposed_source,fac.copy_value_at_explicit_payload,
        fac.gate_copied_value,
    ):
        body=inspect.getsource(fn)
        for banned in ("oracle_read(", "ep.gold_answer", "ep.records", "train_counterfactual_group("):
            assert banned not in body
    tree=ast.parse(inspect.getsource(fac))
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    for prohibited in ("torch.cuda", "modal.App", "model.forward", "loss.backward",
                       "optimizer.step", "generate_episode", "torch.save"):
        assert prohibited not in calls
    assert "read_visible_text(view)" in inspect.getsource(fac.train_only_factorized_audit)
    assert "oracle_read(ep)" in inspect.getsource(fac.train_only_factorized_audit)
    assert inspect.getsource(fac.train_only_factorized_audit).index(
        "read_visible_text(view)"
    ) < inspect.getsource(fac.train_only_factorized_audit).index("oracle_read(ep)")
