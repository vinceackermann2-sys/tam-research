from __future__ import annotations

"""#1395 PR CI: source/shape tests only. No heldout score/training run."""

import math
import pytest
import torch

from tam_research.chm_v3_balanced_heldout_1395 import (
    REPORT_MARKER, CLASSIFICATION, EVAL_ENTITIES, HISTORICAL_STOP,
    ARMS, SOURCE_BLOBS, heldout_cases, score_one, aggregate, evaluate_models,
)
from tam_research.chm_v3_balanced_train_schedule_1391 import (
    TRAIN_STEPS, BALANCED_SCHEDULE_SEED, balanced_training_episode,
)
from tam_research.chm_v3_matched_tiny_models_1377 import matched_initial_models


def test_source_locked_shape_and_split_reservations():
    assert REPORT_MARKER == "CHM_V3_1395_BALANCED_CPU_HELDOUT_RESULT="
    assert CLASSIFICATION.endswith("EXPLORATORY")
    assert HISTORICAL_STOP == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert EVAL_ENTITIES == {"development": 1, "test": 2}
    assert ARMS == ("transformer", "pointer_ce", "pointer_no_ce")
    assert TRAIN_STEPS == 256 and BALANCED_SCHEDULE_SEED == 13772001
    assert SOURCE_BLOBS["balanced_schedule"] == "6d40dafc7a487cfc8d5a39d5067ff00190e60181"
    assert SOURCE_BLOBS["model"] == "ffe14b0701e18493a2bf9b45a1238b5acd5e3eab"


def test_no_original_or_train_entities_in_heldout_panel():
    with pytest.raises(ValueError):
        heldout_cases("train")
    with pytest.raises(ValueError):
        heldout_cases("invalid")
    assert all(ep.split == "train" for ep in (
        balanced_training_episode(0), balanced_training_episode(255)
    ))


def test_train_only_synthetic_prediction_and_evaluator_isolation():
    # PR tests never score the reserved development or test entities.
    torch.set_num_threads(1)
    a, b, c = matched_initial_models()
    ep = balanced_training_episode(0)
    with torch.no_grad():
        for arm, model in zip(ARMS, (a, b, c)):
            row = score_one(model, ep, arm)
            assert row["split"] == "train"
            assert row["variant"] in range(8)
            assert row["predicted_answer_id"] is None or isinstance(row["predicted_answer_id"], int)
            assert math.isfinite(row["confidence"])
            assert 0 <= row["confidence"] <= 1
            if arm == "transformer":
                assert row["first_selected_visible_token_index"] is None
                assert row["second_selected_visible_token_index"] is None


def test_fail_closed_incomplete_aggregation():
    with pytest.raises(ValueError, match="32 predictions"):
        aggregate([])
    with pytest.raises(RuntimeError, match="full 256-step"):
        evaluate_models(matched_initial_models(), {"steps_per_arm": 1}, 0.0)
