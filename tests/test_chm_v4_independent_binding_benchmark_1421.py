from __future__ import annotations

"""#1421: zero-GPU TRAIN-only anti-shortcut and counterfactual geometry tests."""

from collections import Counter
from dataclasses import fields, replace
import inspect
import json
import re

import pytest

import tam_research.chm_v4_independent_binding_benchmark_1421 as v4


def test_frozen_manifest_is_only_names_no_heldout_cases():
    m1 = v4.prospective_split_manifest()
    m2 = v4.prospective_split_manifest()
    assert m1 == m2
    assert m1["namespace"] == v4.STUDY_NAMESPACE
    assert m1["rng_namespace"] == v4.RNG_NAMESPACE
    assert m1["heldout_examples_generated"] is False
    assert m1["heldout_predictions_made"] == 0
    assert m1["scored_test_authorized"] is False
    assert {x:y["independent_entities"] for x,y in m1["splits"].items()} == {
        "train":64, "development":16, "test":32,
    }
    assert all(len(x["manifest_sha256"]) == 64 for x in m1["splits"].values())
    assert v4.entity_name("train",0) == "V4-train-E0000"
    assert v4.entity_name("train",63) == "V4-train-E0063"
    assert v4.entity_name("development",0) != v4.entity_name("test",0)
    assert v4.entity_name("test",0) != "V3-test-E003"
    for invalid in (-1,64,True,1.1):
        with pytest.raises(ValueError):
            v4.entity_name("train",invalid)
    for forbidden_split in ("unknown", "", None):
        with pytest.raises(ValueError):
            v4.entity_name(forbidden_split,0)


@pytest.mark.parametrize("family",v4.FAMILIES)
@pytest.mark.parametrize("entity",(0,1,2,17,42,63))
def test_eight_counterfactual_train_rotations_are_oracle_sound(family,entity):
    group=v4.train_counterfactual_group(family,entity)
    assert len(group)==8
    assert {ep.split for ep in group}=={"train"}
    assert {ep.family for ep in group}=={family}
    assert all(ep.entity==f"V4-train-E{entity:04d}" for ep in group)
    assert all(ep.memory_length==128 for ep in group)
    if family in v4.POSITIVES:
        assert {ep.gold_answer for ep in group}==set(v4.CODES)
    else:
        assert {ep.gold_answer for ep in group}=={None}
    views=[v4.seal_input(ep) for ep in group]
    assert all(view.query==views[0].query for view in views)
    assert len({view.answer_options for view in views})==1
    assert all(set(view.answer_options)==set(f"CODE-{c}" for c in v4.CODES) for view in views)
    assert all(not re.search(r"CODE-\d+", view.query) for view in views)
    if family in v4.POSITIVES:
        assert len({view.memory_text for view in views})==8
    else:
        assert len({view.memory_text for view in views})==1
    for ep in group:
        answer,relation,value=v4.oracle_read(ep)
        assert answer==ep.gold_answer
        if family == "no_match":
            assert relation is None and value is None
        elif family == "two_hop":
            assert relation is not None and value is not None
        else:
            assert relation is None and value is not None
        assert Counter(r.code for r in ep.records if r.code in v4.CODES)==Counter(v4.CODES)


@pytest.mark.parametrize("family",v4.FAMILIES)
def test_adversarial_record_layout_and_query_bag_redaction_shortcuts(family):
    for entity in (0,3,9,23,37,63):
        group=v4.train_counterfactual_group(family,entity)
        views=[v4.seal_input(ep) for ep in group]
        layout=[
            tuple((r.position,r.role,r.subject,r.document) for r in ep.records)
            for ep in group
        ]
        assert len(set(layout))==1
        redacted=[
            re.sub(r"CODE-\d+", "CODE-REDACTED", view.memory_text)
            for view in views
        ]
        assert len(set(redacted))==1
        bags=[
            tuple(sorted(re.findall(r"CODE-\d+",view.memory_text)))
            for view in views
        ]
        assert len(set(bags))==1
        assert len(set(view.query for view in views))==1
        if family in v4.POSITIVES:
            # Any blind query/layout/bag-only predictor emits the same answer
            # on eight rotations; the gold is each candidate exactly once.
            assert len(set(ep.gold_answer for ep in group))==8
            assert sum(ep.gold_answer==v4.CODES[0] for ep in group)==1
        else:
            assert all(ep.gold_answer is None for ep in group)


def test_two_hop_query_relation_rebinding_has_latest_authoritative_record():
    for entity in (0,1,7,19,36,63):
        ep=v4.train_counterfactual_group("two_hop",entity)[0]
        rels=[r for r in ep.records if r.role=="relation" and r.subject==ep.entity]
        assert len(rels)==2
        newest=max(rels,key=lambda x:x.position)
        oldest=min(rels,key=lambda x:x.position)
        assert newest.document != oldest.document
        assert any(r.subject==newest.document and r.role=="record" for r in ep.records)
        assert any(r.subject==oldest.document and r.role=="record" for r in ep.records)
        answer,link,record_pos=v4.oracle_read(ep)
        assert link==newest.position
        assert record_pos in {r.position for r in ep.records
                              if r.subject==newest.document and r.role=="record"}
        assert answer==ep.gold_answer
        assert newest.document in v4.seal_input(ep).memory_text


def test_overwrite_chooses_newest_subject_value_not_old_stale_code():
    for entity in (0,2,5,18,63):
        ep=v4.train_counterfactual_group("overwrite",entity)[0]
        values=[r for r in ep.records if r.subject==ep.entity and r.role=="value"]
        assert len(values)==2
        older,newer=sorted(values,key=lambda r:r.position)
        assert older.code==v4.STALE_CODE and newer.code==ep.gold_answer
        result=v4.oracle_read(ep)
        assert result==(newer.code,None,newer.position)
        assert "CODE-19001" in v4.seal_input(ep).memory_text


def test_fail_closed_corrupted_train_episode_different_code_or_position():
    group=v4.train_counterfactual_group("direct",0)
    ep=group[0]
    with pytest.raises(ValueError):
        v4.generate_episode("train","direct",0,True)
    with pytest.raises(ValueError):
        v4.generate_episode("train","direct",0,8)
    records=list(ep.records)
    records[0]=replace(records[0],position=9999)
    with pytest.raises(ValueError):
        v4.validate_episode(replace(ep,records=tuple(records)))
    records=list(ep.records)
    records[0]=replace(records[0],code=25001)
    with pytest.raises(ValueError):
        v4.validate_episode(replace(ep,records=tuple(records)))
    with pytest.raises(ValueError):
        v4.validate_train_group(group[:7])
    with pytest.raises(ValueError):
        v4.validate_train_group(tuple(replace(x,split="test") for x in group))


def test_all_training_entities_all_families_report_and_heldout_zero():
    report=v4.train_only_audit()
    assert report==v4.train_only_audit()
    assert report["train_groups"]==256
    assert report["train_episodes"]==2048
    assert report["independent_training_entities"]==64
    assert report["train_group_bag_invariants"]==256
    assert report["train_group_layout_invariants"]==256
    assert report["heldout_episodes_constructed"]==0
    assert report["model_training_steps"]==0
    assert report["trained_predictions"]==0
    assert report["gpu_used"] is False
    assert report["new_scientific_attempt"] is False
    assert report["historical_stop"]=="CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert report["historical_seed_consumed"]==2013161
    assert report["blind_positive_ceiling_per_eight_variants"]=="1/8"
    assert set(report["template_counts"])=={0,1}
    assert sum(report["template_counts"].values())==256


def test_sealed_forward_view_excludes_gold_roles_source_or_episode_metadata():
    assert [x.name for x in fields(v4.SealedInput)] == [
        "query","memory_text","answer_options"
    ]
    ep=v4.train_counterfactual_group("direct",0)[0]
    view=v4.seal_input(ep)
    assert not hasattr(view,"gold_answer")
    assert not hasattr(view,"variant")
    assert not hasattr(view,"records")
    assert not hasattr(view,"split")
    assert v4.GPU_AUTHORIZED is False
    assert v4.NEW_SCIENTIFIC_ATTEMPT is False
    assert v4.SCORED_HELDOUT_AUTHORIZED is False
    src=inspect.getsource(v4.train_only_audit)
    assert 'train_counterfactual_group(family,index)' in src
    for banned in ('generate_episode("test"', 'generate_episode("development"',
                   "torch.cuda", "modal.App(", "torch.load("):
        assert banned not in src
