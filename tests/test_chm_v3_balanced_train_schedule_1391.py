from __future__ import annotations

from collections import Counter, defaultdict
import inspect

import pytest
import torch

import tam_research.chm_v3_balanced_train_schedule_1391 as balanced
from tam_research.chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS, SPLITS
from tam_research.chm_v3_counterfactual_multiset_suite_1373 import FAMILIES, POSITIVES
from tam_research.chm_v3_counterfactual_model_view_1365 import seal_model_view
from tam_research.chm_v3_matched_tiny_models_1377 import TinyDecoderOnly, TinyTwoHopPointer


def test_all_256_training_episodes_balanced_by_entity_family_and_answer():
    schedule = balanced.balanced_train_schedule()
    assert len(schedule) == len(set(schedule)) == 256
    assert schedule == balanced.balanced_train_schedule()
    groups = defaultdict(list)
    for index, (family, entity, variant) in enumerate(schedule):
        ep = balanced.balanced_training_episode(index)
        assert ep.split == "train"
        assert (ep.family, ep.variant) == (family, variant)
        assert ep.entity == f"V3-train-E{entity:03d}"
        assert 0 <= entity < SPLITS["train"] == 8
        assert len(ep.facts) >= 8
        assert not hasattr(seal_model_view(ep), "gold_answer")
        groups[(family, entity)].append(ep)
    assert len(groups) == 4 * 8
    for (family, entity), episodes in groups.items():
        assert len(episodes) == 8
        assert {ep.variant for ep in episodes} == set(range(8))
        assert len({ep.query for ep in episodes}) == 1
        assert len({tuple((r.position,r.subject,r.kind,r.document)
                          for r in ep.facts) for ep in episodes}) == 1
        if family in POSITIVES:
            assert {ep.gold_answer for ep in episodes} == set(ANSWER_IDS)
            assert len({ep.gold_answer for ep in episodes}) == 8
        else:
            assert {ep.gold_answer for ep in episodes} == {None}
    assert groups[("direct", 0)][0].gold_answer is not None


def test_model_no_longer_gets_one_constant_answer_per_positive_family():
    by_family = defaultdict(list)
    for step in range(balanced.TRAIN_STEPS):
        ep = balanced.balanced_training_episode(step)
        by_family[ep.family].append(ep.gold_answer)
    assert Counter({family:len(v) for family,v in by_family.items()}) == {
        "direct":64,"overwrite":64,"two_hop":64,"no_match":64
    }
    for family in POSITIVES:
        assert set(by_family[family]) == set(ANSWER_IDS)
        assert all(by_family[family].count(code) == 8 for code in ANSWER_IDS)
    assert set(by_family["no_match"]) == {None}


def test_locked_train_only_audit_does_not_peek_into_previous_heldout_entity_zero():
    result = balanced.audit_training_balance()
    assert result["steps_per_arm"] == 256
    assert result["positive_groups"] == 24
    assert result["all_positive_groups_cover_all_eight_answers"] is True
    assert result["development_entity_reserved"] == 1
    assert result["test_entity_reserved"] == 2
    assert result["old_entity_index_zero_not_reused_in_scored_followup"] is True
    assert result["gpu_allocated"] is False
    assert result["new_scientific_attempt"] is False
    assert balanced.ORIGINAL_SCIENTIFIC_SEED_CONSUMED == 2013161
    for field in ("development", "test"):
        assert field not in inspect.getsource(balanced.balanced_training_episode)


def test_train_schedule_rejects_out_of_range_or_float_indices():
    for index in (-1,256,999,1.0,True):
        with pytest.raises(ValueError, match="256-step"):
            balanced.balanced_training_episode(index)
    for invalid in (0,257,1.0,True):
        with pytest.raises(ValueError, match="256 cap"):
            balanced.train_balanced_cpu_models(steps=invalid)


def test_one_cpu_training_step_has_same_initial_backbone_and_no_heldout_scoring():
    initial_threads = torch.get_num_threads()
    try:
        a,b,c,metadata = balanced.train_balanced_cpu_models(steps=1)
        assert isinstance(a,TinyDecoderOnly)
        assert isinstance(b,TinyTwoHopPointer)
        assert isinstance(c,TinyTwoHopPointer)
        assert metadata["steps_per_arm"] == 1
        assert metadata["tokens_per_arm"] > 0
        assert metadata["cpu_only"] is True
        assert metadata["new_scientific_attempt"] is False
        assert metadata["historical_scientific_status"] == (
            "CHM_V3_100M_DAEC_STAGE_C_STOP"
        )
        assert metadata["historical_scientific_seed_consumed"] is True
        counts = metadata["trainable_parameters"]
        assert counts["pointer_ce"] == counts["pointer_no_ce"]
        assert abs(counts["pointer_ce"] - counts["transformer"])/max(counts.values()) < .01
    finally:
        torch.set_num_threads(initial_threads)


def test_training_module_never_runs_modal_or_scores_heldout():
    code = inspect.getsource(balanced)
    assert balanced.GPU_AUTHORIZED is False
    assert balanced.SCIENTIFIC_RUN_AUTHORIZED is False
    assert balanced.NEVER_REUSE_OLD_REPORT_TEST_ENTITY_INDEX == 0
    assert balanced.BALANCED_SCHEDULE_SEED == 13772001
    for banned in (
        "torch.cuda", ".cuda(", "modal.App(", "modal.run(", "torch.load(",
        "checkpoint.load(", "eval_cases(", "score_split(",
    ):
        assert banned not in code
    assert 'generate_episode(\n        "train"' in code
    assert "pointer_supervision=(arm == 1)" in code
