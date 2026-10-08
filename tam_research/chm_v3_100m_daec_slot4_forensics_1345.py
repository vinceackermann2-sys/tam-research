from __future__ import annotations

"""#1345: CPU-only correctness-set audit of the immutable Stage-C scientific STOP.

Pure analysis of preserved per-case candidate NLL/correctness booleans. It never
runs a model, infers missing predicted IDs, changes the historical classifier,
or launches training/Modal. Equality to a constant-choice *correctness set* is
not evidence that an unlogged prediction was constant on incorrect cases.
"""

from collections.abc import Mapping
import math
from typing import Any

FAMILIES = ("rare_fact", "overwrite", "two_hop", "local_negative")
OFFSETS = (1, 3, 5, 7)
MODELS = ("local", "raw", "daec")
NUM_CASES = 128
FROZEN_CANDIDATES = (
    ("amber", 36505), ("delta", 25979), ("granite", 41013),
    ("mosaic", 47076), ("north", 5093), ("pearl", 43836),
    ("quartz", 47969), ("river", 7850),
)
EXPECTED_SOURCE = "eb4f0e36ee6c4095e550f26cb7a52c20c5116aa9"
STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
GENERATOR = "chm-v1-100m-heldout-aligned-v4"
ORIGINAL_PROBE_FIELDS = (
    "case_id", "daec_candidate_nll", "daec_copied_token_id",
    "daec_copy_matches_answer", "daec_correct", "daec_retrieval_traces",
    "daec_stale_choice", "daec_trace", "evidence_distance", "family",
    "generator_version", "local_candidate_nll", "local_correct",
    "local_stale_choice", "raw_candidate_nll", "raw_correct",
    "raw_retrieval_traces", "raw_stale_choice",
)


def answer_slot(case_id: int, family_index: int) -> int:
    """Frozen v3 balanced 8-way schedule, independently checked in CPU tests."""
    if type(case_id) is not int or not (0 <= family_index < len(FAMILIES)):
        raise ValueError("invalid answer slot inputs")
    return ((case_id % 100_000) * 5 + OFFSETS[family_index]) % 8


def _expect_exact(data: Mapping[str, Any], field: str, value: Any) -> None:
    actual = data.get(field)
    if type(actual) is not type(value) or actual != value:
        raise ValueError(f"frozen {field} mismatch")


def _validate_science_identity(
    original: Mapping[str, Any],
    *, correctness: bool,
) -> None:
    if correctness:
        _expect_exact(original, "evidence_kind",
                      "VERBATIM_CORRECTNESS_AND_EIGHT_WAY_NLL_FIELDS_FROM_ORIGINAL_STAGE_C_RESULT")
        _expect_exact(original, "original_integrity_parity_flag_present", False)
        _expect_exact(original, "original_probe_rows_with_explicit_predicted_candidate_id", 0)
        if tuple(original.get("original_probe_row_fields", ())) != ORIGINAL_PROBE_FIELDS:
            raise ValueError("original frozen scored-row schema drift")
        _expect_exact(original, "generator_version", GENERATOR)
        _expect_exact(original, "probe_seed", 977301)
    else:
        _expect_exact(original, "source_kind", "READ_ONLY_EXTRACT_OF_ARCHIVED_RESULT_FROM_COMPLETED_RUN")
        _expect_exact(original, "generator_version", GENERATOR)
        _expect_exact(original, "probe_seed", 977301)
        _expect_exact(original, "terminal_issue_comment", 6_058_454_735)
        counts = original.get("counts")
        if not isinstance(counts, Mapping) or counts.get("probes") != 512:
            raise ValueError("frozen 1340 probe-count mismatch")

    renames = (
        ("archived_run_id", "archived_actions_run_id", 37_760_374_218),
        ("archived_job_id", "archived_actions_job_id", 113_255_052_797),
        ("terminal_comment_id", "terminal_issue_comment", 6_058_454_735),
        ("source_sha", "source_sha", EXPECTED_SOURCE),
        ("scientific_seed", "scientific_seed", 2_013_161),
        ("scientific_seed_consumed", "scientific_seed_consumed", True),
        ("classification", "scientific_classification", STOP),
        ("passed", "scientific_passed", False),
        ("status", "original_status", "COMPLETE"),
    )
    for left, right, value in renames:
        _expect_exact(original, left if correctness else right, value)


def audit_slot4_success_geometry(
    archived_correctness: Mapping[str, Any],
    archived_oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate source-bound 512-row joins and describe the success sets."""
    _validate_science_identity(archived_correctness, correctness=True)
    _validate_science_identity(archived_oracle, correctness=False)

    rows = archived_correctness.get("rows")
    oracle_rows = archived_oracle.get("rows")
    if not isinstance(rows, list) or len(rows) != len(FAMILIES) * NUM_CASES:
        raise ValueError("512-case correctness envelope required")
    if not isinstance(oracle_rows, list) or len(oracle_rows) != len(rows):
        raise ValueError("512-case oracle envelope required")

    success_ids: dict[str, dict[str, set[int]]] = {}
    family_stats: dict[str, Any] = {}
    all_model_sets_equal = True
    all_model_sets_equal_to_constant = True
    for family_index, family in enumerate(FAMILIES):
        success_ids[family] = {}
        model_metrics: dict[str, Any] = {}
        expected_constant = {
            family_index * 100_000 + k
            for k in range(NUM_CASES)
            if answer_slot(family_index * 100_000 + k, family_index) == 4
        }
        for model in MODELS:
            success_ids[family][model] = set()

        for offset in range(NUM_CASES):
            n = family_index * NUM_CASES + offset
            case_id = family_index * 100_000 + offset
            row, ora = rows[n], oracle_rows[n]
            if not isinstance(row, Mapping) or not isinstance(ora, Mapping):
                raise ValueError("invalid original row shape")
            if (row.get("case_id") != case_id or row.get("family") != family
                    or row.get("generator_version") != GENERATOR
                    or ora.get("case_id") != case_id or ora.get("family") != family):
                raise ValueError("original aligned-v4 row ID/order mismatch")

            for model in MODELS:
                correct = row.get(f"{model}_correct")
                nll = row.get(f"{model}_candidate_nll")
                if type(correct) is not bool:
                    raise ValueError("original correctness must be boolean")
                if type(nll) not in (int, float) or not math.isfinite(nll) or nll < 0:
                    raise ValueError("original candidate NLL must be finite nonnegative")
                if correct:
                    success_ids[family][model].add(case_id)

            if type(ora.get("correct")) is not bool or ora.get("correct") != row["daec_correct"]:
                raise ValueError("independent #1340 DAEC correctness mismatch")

        for model in MODELS:
            counts_by_slot = [
                sum(answer_slot(case_id, family_index) == slot
                    for case_id in success_ids[family][model])
                for slot in range(8)
            ]
            family_rows = rows[family_index * NUM_CASES:(family_index + 1) * NUM_CASES]
            mean_nll = math.fsum(float(r[f"{model}_candidate_nll"]) for r in family_rows) / NUM_CASES
            match = success_ids[family][model] == expected_constant
            all_model_sets_equal_to_constant &= match
            model_metrics[model] = {
                "correct": len(success_ids[family][model]),
                "correct_by_answer_slot": counts_by_slot,
                "correct_case_ids": sorted(success_ids[family][model]),
                "mean_eight_way_candidate_nll": mean_nll,
                "success_set_identical_to_synthetic_always_slot4": match,
            }
        pair_sets = [success_ids[family][m] for m in MODELS]
        family_all_equal = all(pair_sets[0] == s for s in pair_sets[1:])
        all_model_sets_equal &= family_all_equal
        family_stats[family] = {
            "cases": NUM_CASES,
            "each_candidate_answers_count": 16,
            "synthetic_always_slot4_correct": len(expected_constant),
            "synthetic_always_slot4_correct_case_ids": sorted(expected_constant),
            "three_models_success_sets_equal": family_all_equal,
            "models": model_metrics,
        }

    return {
        "classification": "CHM_V3_1345_SLOT4_CORRECTNESS_SET_FORENSIC_CPU_ONLY",
        "original_scientific_classification": STOP,
        "original_scientific_seed_consumed": True,
        "original_scientific_result_reclassified": False,
        "replayed_trained_checkpoint": False,
        "new_gpu_or_training_attempt": False,
        "source_scored_probe_rows": len(rows),
        "frozen_answer_candidate_slot4": {
            "word": FROZEN_CANDIDATES[4][0],
            "gpt2_token_id": FROZEN_CANDIDATES[4][1],
        },
        "shared_model_correctness_case_sets": all_model_sets_equal,
        "all_model_correctness_sets_match_synthetic_always_slot4": all_model_sets_equal_to_constant,
        "logged_wrong_choice_predicted_ids_available": False,
        "family_stats": family_stats,
        "interpretation_ceiling": (
            "Correctness-set identity with a constant slot-4 predictor is observed, "
            "but wrong-case predicted candidate IDs were not logged; therefore "
            "constant-prediction collapse is unproven, as is any causal mechanism. "
            "The original scientific STOP and missing parity evidence remain."
        ),
    }


__all__ = ["answer_slot", "audit_slot4_success_geometry"]
