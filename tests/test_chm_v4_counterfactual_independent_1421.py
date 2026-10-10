from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import inspect
import json
import re

import pytest

import tam_research.chm_v4_counterfactual_independent_1421 as v4
from tam_research.chm_v3_counterfactual_model_view_1365 import SealedModelView


@pytest.mark.parametrize("family", v4.FAMILIES)
@pytest.mark.parametrize("entity_index", (0, 1, 2, 7, 19, 63))
def test_train_groups_have_identical_input_except_two_swapped_values(
    family, entity_index,
):
    group = v4.v4_train_group(family, entity_index)
    assert len(group) == 8
    assert {x.variant for x in group} == set(range(8))
    assert {x.entity for x in group} == {f"V4-train-E{entity_index:04d}"}
    assert len({x.query for x in group}) == 1
    result = v4.validate_v4_group(group)
    assert result["query_layout_codebag_invariant"]
    assert result["independent_entity_clusters"] == 1
    assert result["oracle_position_is_variant_invariant"]
    assert {v4.code_bag(v4.seal_v4_view(x)) for x in group} == {
        tuple(sorted(v4.ANSWER_IDS))
    }
    before = group[0]
    for ep in group:
        assert ep.render_style == before.render_style
        view = v4.seal_v4_view(ep)
        assert type(view) is SealedModelView
        assert "V4-train" in view.query
        assert len({f.position for f in ep.facts}) == len(ep.facts)
        assert all(line.startswith("[") for line in view.memory_text.splitlines())
        assert len(re.findall(r"CODE-(?:1800[1-8])\b", view.memory_text)) == 8
        assert v4.redacted_view(view) == v4.redacted_view(v4.seal_v4_view(before))
        changed = [p for p,(x,y) in enumerate(zip(before.facts, ep.facts))
                   if x.code != y.code]
        assert len(changed) == (2 if ep.variant and family in v4.POSITIVES else 0)
    if family in v4.POSITIVES:
        assert {x.gold_answer for x in group} == set(v4.ANSWER_IDS)
    else:
        assert {x.gold_answer for x in group} == {None}


def test_train_only_full_2048_cases_balance_entities_answers_addresses():
    audit = v4.train_only_shortcut_audit()
    assert audit["classification"] == (
        "CHM_V4_1421_INDEPENDENT_ENTITY_TRAIN_ONLY_STATIC_AUDIT"
    )
    assert audit["training_cases"] == 64 * 4 * 8 == 2048
    assert audit["training_8_way_counterfactual_groups"] == 64 * 4
    assert audit["training_gold_frequency_per_candidate_per_positive_family"] == 64
    assert audit["training_entities_per_family"] == 64
    assert audit["render_styles_used"] == [0,1,2]
    assert all(value >= 16 for value in
               audit["distinct_target_addresses_per_positive_family"].values())
    assert audit["scored_test_cases_constructed"] is False
    assert audit["model_forward_calls"] == 0
    assert audit["training_optimizer_steps"] == 0
    assert audit["gpu_used"] is False and audit["new_scientific_attempt"] is False
    assert audit["original_100m_scientific_classification"] == (
        "CHM_V3_100M_DAEC_STAGE_C_STOP"
    )
    assert audit["original_scientific_seed_consumed"] == 2013161


def test_training_audit_cannot_access_heldout_split_during_execution(monkeypatch):
    original = v4.generate_v4_episode
    accessed: list[str] = []

    def train_only_guard(split, family, entity_index, variant):
        accessed.append(split)
        if split != "train":
            raise AssertionError("CI accessed a heldout group")
        return original(split, family, entity_index, variant)

    monkeypatch.setattr(v4, "generate_v4_episode", train_only_guard)
    report = v4.train_only_shortcut_audit()
    assert report["training_cases"] == len(accessed) == 2048
    assert set(accessed) == {"train"}


def test_latest_overwrite_and_latest_relation_binding_train_only():
    for entity in (0,2,7,19,63):
        overwrite = v4.v4_train_group("overwrite", entity)[3]
        old, new = sorted((f for f in overwrite.facts
                           if f.subject == overwrite.entity and f.kind == "value"),
                          key=lambda f:f.position)
        assert old.code == v4.STALE_NONCANDIDATE
        assert new.code == overwrite.gold_answer
        code, first, second = v4.oracle_reference(overwrite)
        assert code == new.code and first == second == new.position
        linked = v4.v4_train_group("two_hop", entity)[4]
        relations = sorted((f for f in linked.facts
                            if f.kind == "relation" and f.subject == linked.entity),
                           key=lambda f:f.position)
        assert len(relations) == 2
        assert relations[0].document != relations[1].document
        current_doc = relations[-1].document
        selected = [f for f in linked.facts
                    if f.kind == "record" and f.subject == current_doc]
        old_records = [f for f in linked.facts
                       if f.kind == "record"
                       and f.subject == relations[0].document]
        assert len(selected) == len(old_records) == 1
        assert selected[0].code == linked.gold_answer
        assert old_records[0].code != selected[0].code
        answer, first, second = v4.oracle_reference(linked)
        assert answer == linked.gold_answer
        assert first == relations[-1].position
        assert second == selected[0].position
        assert first != second


def test_negative_has_all_candidates_but_no_query_entity_in_memory():
    for index in (0,1,7,37,63):
        episodes = v4.v4_train_group("no_match", index)
        assert len({v4.seal_v4_view(ep).memory_text for ep in episodes}) == 1
        for ep in episodes:
            assert ep.gold_answer is None
            assert ep.entity not in v4.seal_v4_view(ep).memory_text
            assert v4.oracle_reference(ep) == (None,None,None)
            assert v4.code_bag(v4.seal_v4_view(ep)) == tuple(sorted(v4.ANSWER_IDS))


def test_tampering_with_input_labels_source_records_or_oracle_is_rejected():
    base = v4.v4_train_group("overwrite", 0)[2]
    first = base.facts[0]
    with pytest.raises(ValueError, match="query"):
        v4.validate_episode(replace(base,query=base.query+" CODE-18001"))
    with pytest.raises(ValueError,match="oracle|binding|candidate"):
        v4.validate_episode(replace(base,gold_answer=18001 if base.gold_answer!=18001 else 18002))
    with pytest.raises(ValueError,match="duplicate"):
        v4.validate_episode(replace(base,facts=base.facts+(
            replace(first,position=first.position),)))
    with pytest.raises(ValueError,match="candidate"):
        v4.validate_episode(replace(base,facts=tuple(
            replace(f,code=18001) if f.code==18002 else f for f in base.facts
        )))
    with pytest.raises(ValueError,match="stale|candidate"):
        v4.validate_episode(replace(base,facts=tuple(
            replace(f,code=18008) if f.code==v4.STALE_NONCANDIDATE else f
            for f in base.facts
        )))
    with pytest.raises(ValueError,match="group|eight"):
        v4.validate_v4_group(v4.v4_train_group("direct",0)[:-1])


@pytest.mark.parametrize("split,valid", (("train",63),("development",15),("test",31)))
def test_declarative_split_bounds_without_constructing_heldout(split,valid):
    assert v4.SPLIT_ENTITIES[split] == valid+1
    assert v4._spec(split,"direct",valid)[0] == f"V4-{split}-E{valid:04d}"
    with pytest.raises(ValueError,match="outside"):
        v4._spec(split,"direct",valid+1)
    with pytest.raises(ValueError,match="outside"):
        v4._spec(split,"direct",-1)
    with pytest.raises(ValueError,match="outside"):
        v4._spec(split,"direct",True)


def test_declarative_manifest_digest_without_building_test_cases(monkeypatch):
    def no_generation(*args,**kwargs):
        raise AssertionError("manifest must not instantiate episodes")
    monkeypatch.setattr(v4,"generate_v4_episode",no_generation)
    manifest = v4.declarative_heldout_manifest()
    assert manifest == v4.declarative_heldout_manifest()
    assert manifest["splits"] == {"train":64,"development":16,"test":32}
    assert manifest["test_independent_entities_per_family"] == 32
    assert manifest["development_independent_entities_per_family"] == 16
    assert manifest["expected_test_cases"] == 32 * 4 * 8 == 1024
    assert manifest["expected_development_cases"] == 16 * 4 * 8 == 512
    assert manifest["scored_heldout_constructed"] is False
    assert manifest["payload_sha256_computed"] is False
    assert len(manifest["declarative_manifest_sha256"]) == 64
    copy = dict(manifest)
    fingerprint = copy.pop("declarative_manifest_sha256")
    payload = json.dumps(copy,sort_keys=True,separators=(",",":"))
    assert fingerprint == hashlib.sha256(payload.encode()).hexdigest()


def test_source_is_data_only_and_evaluation_entrypoints_not_triggered_in_ci():
    source = inspect.getsource(v4)
    for forbidden in (
        "modal.App(", "modal.run(", "torch.cuda", "torch.load(",
        "optimizer.step(", "model.forward(", "loss.backward(", "load_checkpoint(",
    ):
        assert forbidden not in source
    assert "v4_train_group(family, idx)" in source
    assert 'generate_v4_episode("train"' in source
    assert "old scientific seed" not in v4.V4_NAMESPACE.lower()
    assert v4.GPU_AUTHORIZED is False
    assert v4.SCORED_HELDOUT_AUTHORIZED is False
