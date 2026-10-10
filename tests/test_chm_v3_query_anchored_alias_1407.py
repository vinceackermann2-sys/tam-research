from __future__ import annotations

from collections import Counter
from dataclasses import replace
import inspect

import pytest

import tam_research.chm_v3_query_anchored_alias_1407 as alias
from tam_research.chm_v3_balanced_train_schedule_1391 import balanced_training_episode
from tam_research.chm_v3_counterfactual_multiset_suite_1373 import FAMILIES, POSITIVES, paired_group
from tam_research.chm_v3_counterfactual_model_view_1365 import seal_model_view
from tam_research.chm_v3_matched_tiny_models_1377 import encode_sealed_view
from tam_research.chm_v3_identifier_composition_audit_1399 import CODE_SURFACE


def test_local_alias_encoder_preserves_exact_memory_codes_and_token_budget_train_only():
    for step in range(256):
        ep = balanced_training_episode(step)
        assert ep.split == "train"
        view = seal_model_view(ep)
        old = encode_sealed_view(view)
        new = alias.alias_sealed_input(view)
        assert len(old.token_ids) == len(new.token_ids)
        assert old.memory_length == new.memory_length
        assert old.line_anchors == new.line_anchors
        assert old.code_tokens == new.code_tokens
        assert new.surfaces[new.memory_length] == "QUESTION"
        assert new.surfaces[-1] == "ANSWER"
        assert all(new.surfaces[i] == f"CODE-{code}" for i,code in new.code_tokens)
        ids = dict(new.alias_dictionary_evaluator_only)
        assert ids[ep.entity] == "ID_ALIAS_Q"
        assert new.surfaces.count("ID_ALIAS_Q") >= 1
        assert len(set(ids.values())) == len(ids)
        assert all(i < new.memory_length for i,code in new.code_tokens)
        assert all(0 <= idx < 8192 for idx in new.token_ids)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("entity", (0, 1, 3, 7))
def test_train_counterfactual_groups_have_two_code_swaps_and_identical_aliases(family, entity):
    result = alias.validate_train_pair_group(family, entity)
    assert result["variants"] == 8
    assert result["positive_labels_balanced"]
    assert result["same_query_and_layout"]
    assert result["identical_alias_mapping_across_counterfactuals"]
    assert result["original_token_budget_parity"]
    assert result["max_swapped_code_token_positions"] == (2 if family in POSITIVES else 0)
    groups = paired_group("train", family, entity)
    views = [alias.alias_sealed_input(seal_model_view(x)) for x in groups]
    assert len({v.alias_dictionary_evaluator_only for v in views}) == 1
    assert len({v.memory_length for v in views}) == 1
    bags = [Counter(code for _,code in v.code_tokens) for v in views]
    assert all(bag == bags[0] for bag in bags)


def test_local_alias_preserves_visible_two_hop_relation_record_name_equality():
    ep = paired_group("train", "two_hop", 0)[0]
    view = seal_model_view(ep)
    result = alias.alias_sealed_input(view)
    dictionary = dict(result.alias_dictionary_evaluator_only)
    relation = next(x for x in ep.facts if x.kind == "relation")
    assert relation.document is not None
    doc = relation.document
    assert doc in dictionary
    token = dictionary[doc]
    assert result.surfaces.count(token) >= 2
    assert dictionary[ep.entity] == alias.QUERY_ANCHOR
    assert token != alias.QUERY_ANCHOR
    # A decoy document is neither a gold-selected source nor a model oracle.
    assert all(k != doc or v == token for k,v in dictionary.items())


def test_no_match_query_anchor_appears_only_in_query_not_completed_memory():
    for entity in (0,2,7):
        ep = paired_group("train", "no_match", entity)[0]
        view = seal_model_view(ep)
        tokenized = alias.alias_sealed_input(view)
        assert tokenized.surfaces[:tokenized.memory_length].count(alias.QUERY_ANCHOR) == 0
        assert tokenized.surfaces[tokenized.memory_length:].count(alias.QUERY_ANCHOR) == 1


def test_text_only_aliasing_rejects_ambiguous_query_and_bad_input():
    base = seal_model_view(balanced_training_episode(0))
    with pytest.raises(TypeError,match="sealed"):
        alias.alias_sealed_input(object())
    with pytest.raises(ValueError,match="one and only one"):
        alias.alias_sealed_input(replace(base,query="What is the current access code?"))
    with pytest.raises(ValueError,match="one and only one"):
        alias.alias_sealed_input(replace(base,query=base.query+" V3-hypothetical-E900"))
    with pytest.raises(ValueError,match="candidate"):
        alias.alias_sealed_input(replace(base,answer_options=("CODE-99999",)))
    with pytest.raises(ValueError,match="duplicate"):
        first=base.memory_text.splitlines()[0]
        alias.alias_sealed_input(replace(base,memory_text=base.memory_text+"\n"+first))


def test_static_audit_trains_zero_models_and_never_reads_consumed_test_panels():
    stats = alias.training_only_static_report()
    assert stats["classification"] == "CHM_V3_1407_QUERY_ANCHORED_ALIAS_STATIC_TRAIN_ONLY"
    assert stats["training_episodes"] == 256
    assert stats["counterfactual_groups"] == 32
    assert stats["alias_mapping_invariant_groups"] == 32
    assert stats["original_tokens_sum"] == stats["alias_tokens_sum"] == 19264
    assert stats["original_max_tokens"] == stats["alias_max_tokens"] == 83
    assert 1 <= stats["alias_cardinality_min"] <= stats["alias_cardinality_max"] <= 32
    assert stats["model_forward_calls"] == 0
    assert stats["optimizer_updates"] == 0
    assert stats["scored_heldout_inputs_constructed"] is False
    assert stats["gpu_used"] is False
    assert stats["new_scientific_attempt"] is False
    assert stats["old_100m_scientific_status"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert stats["old_scientific_seed_consumed"] == 2013161
    for forbidden in ("development", "test"):
        assert forbidden not in inspect.getsource(alias.training_only_static_report)
    for forbidden in ("modal.App(", "torch.cuda", "optimizer.step(", "model.forward("):
        assert forbidden not in inspect.getsource(alias)


def test_same_query_and_memory_strings_across_train_examples_have_same_alias_id():
    for entity in (0,1,2,7):
        for family in FAMILIES:
            ep = paired_group("train",family,entity)[0]
            result = alias.alias_sealed_input(seal_model_view(ep))
            surfaces = result.surfaces
            assert surfaces[-2] == alias.QUERY_ANCHOR or alias.QUERY_ANCHOR in surfaces[result.memory_length:]
            assert result.alias_dictionary_evaluator_only
            # No gold answer appears in query after aliasing.
            assert all(not CODE_SURFACE.fullmatch(x) for x in surfaces[result.memory_length:])
