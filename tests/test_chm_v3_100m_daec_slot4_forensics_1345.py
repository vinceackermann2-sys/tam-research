from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from tam_research.chm_v1_long_memory_eval_v3 import _answer_slot as frozen_v3_slot
from tam_research.chm_v3_100m_daec_slot4_forensics_1345 import (
    answer_slot, audit_slot4_success_geometry,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def original_archives() -> tuple[dict, dict]:
    correctness = json.loads(
        (FIXTURES / "chm_v3_1345_archived_correctness_rows.json").read_text(encoding="utf-8")
    )
    oracle = json.loads(
        (FIXTURES / "chm_v3_1340_archived_hard_copy_traces.json").read_text(encoding="utf-8")
    )
    return correctness, oracle


def test_original_v3_answer_schedule_matches_independent_zero_gpu_formula() -> None:
    for fam in range(4):
        for index in range(128):
            case_id = fam * 100000 + index
            assert answer_slot(case_id, fam) == frozen_v3_slot(case_id, (1, 3, 5, 7)[fam])


def test_source_bound_original_model_correctness_sets_match_constant_slot4() -> None:
    correctness, oracle = original_archives()
    orig = json.dumps((correctness, oracle), sort_keys=True)
    result = audit_slot4_success_geometry(correctness, oracle)
    assert json.dumps((correctness, oracle), sort_keys=True) == orig
    assert result["source_scored_probe_rows"] == 512
    assert result["shared_model_correctness_case_sets"] is True
    assert result["all_model_correctness_sets_match_synthetic_always_slot4"] is True
    assert result["frozen_answer_candidate_slot4"] == {"word": "north", "gpt2_token_id": 5093}
    assert result["logged_wrong_choice_predicted_ids_available"] is False
    assert result["original_scientific_seed_consumed"] is True
    assert result["original_scientific_classification"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert result["original_scientific_result_reclassified"] is False
    assert result["replayed_trained_checkpoint"] is False
    assert result["new_gpu_or_training_attempt"] is False

    offset_for_family = {
        "rare_fact": 7, "overwrite": 5, "two_hop": 3, "local_negative": 1
    }
    for family, start in offset_for_family.items():
        stats = result["family_stats"][family]
        assert stats["cases"] == 128
        assert stats["synthetic_always_slot4_correct"] == 16
        assert stats["three_models_success_sets_equal"] is True
        expected = [("rare_fact", 0), ("overwrite", 1),
                    ("two_hop", 2), ("local_negative", 3)]
        family_index = dict(expected)[family]
        expected_cases = [family_index * 100000 + x for x in range(start, 128, 8)]
        assert stats["synthetic_always_slot4_correct_case_ids"] == expected_cases
        for model in ("local", "raw", "daec"):
            item = stats["models"][model]
            assert item["correct"] == 16
            assert item["correct_by_answer_slot"] == [0, 0, 0, 0, 16, 0, 0, 0]
            assert item["correct_case_ids"] == expected_cases
            assert item["success_set_identical_to_synthetic_always_slot4"] is True
            assert 0 < item["mean_eight_way_candidate_nll"] < 20


def test_no_inference_of_unlogged_predictions_even_given_identical_success_sets() -> None:
    correctness, oracle = original_archives()
    assert correctness["original_probe_rows_with_explicit_predicted_candidate_id"] == 0
    assert not any("predicted" in k for k in correctness["original_probe_row_fields"])
    report = audit_slot4_success_geometry(correctness, oracle)
    assert report["logged_wrong_choice_predicted_ids_available"] is False
    assert "unproven" in report["interpretation_ceiling"]


@pytest.mark.parametrize("field,replacement", [
    ("source_sha", "0" * 40),
    ("scientific_seed", 2013162),
    ("scientific_seed_consumed", False),
    ("classification", "CHM_V3_100M_DAEC_POSITIVE_DEVELOPMENT_SCREEN"),
    ("passed", True),
    ("archived_job_id", 113255052798),
    ("probe_seed", 977302),
    ("original_integrity_parity_flag_present", True),
    ("original_probe_rows_with_explicit_predicted_candidate_id", 1),
])
def test_correctness_archive_provenance_cannot_be_rewritten(field: str, replacement: object) -> None:
    correctness, oracle = original_archives()
    correctness[field] = replacement
    with pytest.raises(ValueError):
        audit_slot4_success_geometry(correctness, oracle)


def test_cross_archive_join_detects_bad_correctness_and_row_identity() -> None:
    correctness, oracle = original_archives()
    correctness["rows"][0]["case_id"] = 999999
    with pytest.raises(ValueError, match="ID/order"):
        audit_slot4_success_geometry(correctness, oracle)

    correctness, oracle = original_archives()
    correctness["rows"][0]["daec_correct"] = True
    with pytest.raises(ValueError, match="correctness mismatch"):
        audit_slot4_success_geometry(correctness, oracle)

    correctness, oracle = original_archives()
    correctness["rows"][0]["local_candidate_nll"] = float("nan")
    with pytest.raises(ValueError, match="candidate NLL"):
        audit_slot4_success_geometry(correctness, oracle)

    correctness, oracle = original_archives()
    correctness["rows"][0]["raw_correct"] = 1
    with pytest.raises(ValueError, match="boolean"):
        audit_slot4_success_geometry(correctness, oracle)

    correctness, oracle = original_archives()
    oracle["scientific_seed_consumed"] = False
    with pytest.raises(ValueError, match="scientific_seed_consumed"):
        audit_slot4_success_geometry(correctness, oracle)


def test_alternative_success_set_does_not_get_misclassified_as_constant_slot4() -> None:
    correctness, oracle = original_archives()
    # Synthetic counterexample: preserve 16/128 accuracy, but swap a success
    # at case 7 (slot 4) with a wrong choice at case 0 (slot 1).
    for model in ("local", "raw", "daec"):
        correctness["rows"][7][model + "_correct"] = False
        correctness["rows"][0][model + "_correct"] = True
    oracle["rows"][7]["correct"] = False
    oracle["rows"][0]["correct"] = True
    report = audit_slot4_success_geometry(correctness, oracle)
    assert report["family_stats"]["rare_fact"]["models"]["daec"]["correct"] == 16
    assert report["shared_model_correctness_case_sets"] is True
    assert report["all_model_correctness_sets_match_synthetic_always_slot4"] is False


def test_static_cpu_only_scope() -> None:
    import tam_research.chm_v3_100m_daec_slot4_forensics_1345 as helper
    src = inspect.getsource(helper)
    for forbidden in ("modal.App(", "modal.run(", "torch.cuda",
                      "optimizer.step(", "torch.load(", "checkpoint.load("):
        assert forbidden not in src
