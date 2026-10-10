from __future__ import annotations

import inspect
from collections import Counter

import pytest
import torch

from tam_research.chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from tam_research.chm_v3_counterfactual_model_view_1365 import seal_model_view
from tam_research.chm_v3_counterfactual_multiset_suite_1373 import FAMILIES
from tam_research.chm_v3_matched_tiny_models_1377 import (
    TinyDecoderOnly, TinyTwoHopPointer, encode_sealed_view,
)
from tam_research.chm_v3_matched_tiny_cpu_report_1385 import (
    ARCHIVED_SCIENTIFIC_SEED_CONSUMED, CPU_ONLY, HELDOUT_SPLITS, MEMORY_LENGTH,
    TRAIN_STEPS_PER_ARM, _evaluate_case, frozen_heldout_groups,
    redacted_code_control_view, run_cpu_one_shot_report,
)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_precommitted_panels_are_exactly_32_per_split():
    groups = frozen_heldout_groups()
    assert len(groups) == 8
    for split in HELDOUT_SPLITS:
        selected = [(family, eps) for sp, family, eps in groups if sp == split]
        assert [family for family, _ in selected] == list(FAMILIES)
        assert sum(len(eps) for _, eps in selected) == 32
        for family, episodes in selected:
            assert len(episodes) == 8
            assert tuple(ep.variant for ep in episodes) == tuple(range(8))
            assert all(ep.memory_length == MEMORY_LENGTH == 128 for ep in episodes)
            if family != "no_match":
                assert {ep.gold_answer for ep in episodes} == set(ANSWER_IDS)
            else:
                assert all(ep.gold_answer is None for ep in episodes)
    assert 48 == TRAIN_STEPS_PER_ARM
    assert ARCHIVED_SCIENTIFIC_SEED_CONSUMED == 2013161


def test_redacted_control_is_identical_across_all_eight_positive_rotations():
    for _, family, episodes in frozen_heldout_groups():
        if family == "no_match":
            continue
        views = [redacted_code_control_view(seal_model_view(ep)) for ep in episodes]
        assert len(set(views)) == 1
        assert all("CODE-19001" in view.memory_text for view in views)
        assert all("CODE-1800" not in view.memory_text for view in views)
        assert all(len(encode_sealed_view(view).code_tokens) >= 8 for view in views)


def test_evaluator_records_gold_outside_model_and_predicts_without_gold():
    torch.manual_seed(1385)
    models = {
        "transformer": TinyDecoderOnly(),
        "pointer_ce": TinyTwoHopPointer(),
    }
    for family in FAMILIES:
        episode = next(ep for split, fam, eps in frozen_heldout_groups()
                       if split == "development" and fam == family for ep in eps[:1])
        for arm, model in models.items():
            model.eval()
            row = _evaluate_case(model, episode, arm)
            assert row["family"] == family
            assert row["arm"] == arm
            assert row["variant"] == 0
            assert row["encoded_tokens"] > 8
            assert row["predicted_answer_id"] in (*ANSWER_IDS, None)
            assert row["correct"] == (row["gold_answer_id"] == row["predicted_answer_id"])
            if arm == "transformer":
                assert row["predicted_first_token_index"] is None
                assert row["predicted_second_code_token_index"] is None
            else:
                assert row["predicted_first_token_index"] is not None
                assert row["predicted_second_code_token_index"] is not None
            if family == "no_match":
                assert row["gold_answer_id"] is None
                assert row["no_match_false_positive"] == (not row["abstained"])
            else:
                assert row["gold_answer_id"] in ANSWER_IDS
                assert row["gold_first_token_index"] is not None
                assert row["gold_second_code_token_index"] is not None


def test_report_source_is_grad_enabled_only_on_training_and_no_scientific_gpu():
    import tam_research.chm_v3_matched_tiny_cpu_report_1385 as mod
    source = inspect.getsource(mod)
    assert "@torch.no_grad()\ndef run_cpu_one_shot_report" not in source
    assert CPU_ONLY is True
    assert mod.PAID_TRAINING_AUTHORIZED is False
    assert "train_cpu_models(" in source
    assert "steps=TRAIN_STEPS_PER_ARM" in source
    assert "modal.App(" not in source
    assert "torch.cuda" not in source
    assert "CHM_V3_100M_DAEC_STAGE_C_STOP" in source
    assert "scientific_quality_pass_authorized" in source
    assert mod.MAX_TOTAL_WALL_SECONDS == 2400


def test_cpu_one_shot_report_is_expensive_and_not_executed_by_repo_ci():
    # Repo-wide CI only validates source/functional smoke; one scored report is
    # run by the frozen issue-trigger workflow after exact-head/main CI.
    assert callable(run_cpu_one_shot_report)
    assert len(frozen_heldout_groups()) == 8
