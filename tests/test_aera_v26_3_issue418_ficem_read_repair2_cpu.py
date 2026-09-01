from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import subprocess

import torch

from tam_research.aera_v26_3_ficem_read_probe import (
    BATCH_SIZES,
    BF16_ATOL,
    BF16_RTOL,
    CALLS_PER_ROUND,
    CAPACITY,
    DESIGN_SEED,
    DTYPE_NAMES,
    D_MODEL,
    FP32_ATOL,
    FP32_RTOL,
    MAX_FULL_EVENT_RATIO,
    MAX_GEOMEAN_LATENCY_RATIO,
    MAX_ROW_LATENCY_RATIO,
    MEMORY_DIM,
    TIME,
    TIMED_ROUNDS,
    VALIDITY_KINDS,
    WARMUP_CALLS,
    _tie_aware_top4_equivalence,
    correctness_row,
    cpu_contract_preflight,
    make_case,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
BACKEND_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
LAUNCHER_PATH = ROOT / "modal_aera_v26_3_ficem_read_repair2_app.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair2.yml"
SOURCE_BACKEND_GIT_BLOB = "e8ab9778fe3f3b853e6b18327cbea2c73250624c"
SOURCE_MAKE_CASE_SHA256 = "ca32ff5e47202bbd2aee2225212ef5a6dbf4819d5276974e36f8092b65c642d8"


def _selection(
    logits: list[float],
    reference: list[int],
    candidate: list[int],
    *,
    valid: list[bool] | None = None,
):
    tensor = torch.tensor(logits, dtype=torch.float32).view(1, 1, -1)
    validity = torch.tensor(
        valid if valid is not None else [True] * len(logits), dtype=torch.bool
    ).view(1, -1)
    return _tie_aware_top4_equivalence(
        tensor,
        validity,
        torch.tensor(reference, dtype=torch.long).view(1, 1, 4),
        torch.tensor(candidate, dtype=torch.long).view(1, 1, 4),
    )


def test_issue418_keeps_original_issue411_geometry_timing_and_thresholds():
    assert DESIGN_SEED == 408_411
    assert (D_MODEL, MEMORY_DIM, CAPACITY, TIME) == (200, 50, 48, 256)
    assert BATCH_SIZES == (8, 64)
    assert DTYPE_NAMES == ("float32", "bfloat16")
    assert VALIDITY_KINDS == ("mixed", "full")
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 100)
    assert (FP32_ATOL, FP32_RTOL, BF16_ATOL, BF16_RTOL) == (
        1e-5,
        1e-5,
        1e-2,
        1e-2,
    )
    assert MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert MAX_ROW_LATENCY_RATIO == 1.05
    assert MAX_FULL_EVENT_RATIO == 0.75
    check = cpu_contract_preflight()
    assert check["gpu_authorized_by_cpu_preflight"] is False
    assert check["synthetic_only"] is True
    assert check["scientific_seed_consumed"] is False


def test_issue418_original_make_case_generation_is_byte_stable():
    source = inspect.getsource(make_case)
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_MAKE_CASE_SHA256
    probe = PROBE_PATH.read_text()
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in probe
    assert "case = make_case(" in probe
    assert "rejection" not in probe.lower()
    assert "resampl" not in probe.lower()
    assert "nudge" not in probe.lower()


def test_issue418_backend_is_byte_for_byte_frozen_from_source_main():
    actual = subprocess.check_output(
        ["git", "hash-object", str(BACKEND_PATH)], text=True
    ).strip()
    assert actual == SOURCE_BACKEND_GIT_BLOB


def test_distinct_boundary_still_requires_exact_selected_set():
    good = _selection([5, 4, 3, 2, 1, 0], [0, 1, 2, 3], [3, 2, 1, 0])
    assert good["tie_query_count"] == 0
    assert good["distinct_selected_set_exact"] is True
    assert good["selection_semantically_equivalent"] is True

    bad = _selection([5, 4, 3, 2, 1, 0], [0, 1, 2, 3], [0, 1, 2, 4])
    assert bad["tie_query_count"] == 0
    assert bad["distinct_selected_set_exact"] is False
    assert bad["selection_semantically_equivalent"] is False


def test_tied_boundary_accepts_alternate_mathematically_valid_top4_set():
    result = _selection([5, 4, 3, 2, 2, 1], [0, 1, 2, 3], [0, 1, 2, 4])
    assert result["tie_query_count"] == 1
    assert result["raw_selected_set_equal_all_queries"] is False
    assert result["tied_selection_semantically_valid"] is True
    assert result["selection_semantically_equivalent"] is True


def test_tied_boundary_rejects_duplicate_below_cutoff_invalid_and_missing_strict_slots():
    duplicate = _selection([5, 4, 3, 2, 2, 1], [0, 1, 2, 3], [0, 1, 2, 2])
    below = _selection([5, 4, 3, 2, 2, 1], [0, 1, 2, 3], [0, 1, 2, 5])
    missing_strict = _selection([5, 4, 3, 2, 2, 1], [0, 1, 2, 3], [0, 1, 3, 4])
    invalid = _selection(
        [5, 4, 3, 2, 2, 1],
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        valid=[True, True, True, True, False, True],
    )
    out_of_range = _selection([5, 4, 3, 2, 2, 1], [0, 1, 2, 3], [0, 1, 2, 99])
    assert duplicate["selection_semantically_equivalent"] is False
    assert below["selection_semantically_equivalent"] is False
    assert missing_strict["selection_semantically_equivalent"] is False
    assert invalid["selection_semantically_equivalent"] is False
    assert out_of_range["selection_semantically_equivalent"] is False


def test_issue418_full_tensor_numerical_gates_remain_unmasked():
    source = inspect.getsource(correctness_row)
    assert "reference_recalled, candidate_recalled, atol=atol, rtol=rtol" in source
    assert "reference_result.recalled," in source
    assert "candidate_result.recalled," in source
    assert "selection[\"selection_semantically_equivalent\"]" in source
    assert "distinct_mask" not in source
    assert "tied_mask" not in source
    assert "issue411 synthetic row has a tied fourth/fifth read boundary" not in source


def test_issue418_probe_has_no_fixture_selection_or_candidate_based_admission():
    source = PROBE_PATH.read_text()
    run_source = source.split("def run_ficem_read_probe()", 1)[1]
    assert run_source.count("generator = torch.Generator().manual_seed(DESIGN_SEED)") == 1
    assert "while " not in run_source
    assert "for attempt" not in run_source
    assert "candidate_ordinal" not in run_source
    assert "eligible_case" not in run_source
    assert "make_case(" in run_source


def test_issue418_launcher_is_unique_l4_duplicate_safe_and_synthetic_only():
    source = LAUNCHER_PATH.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue418-ficem-read-repair2/result.json"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in source
    assert "result_path.exists()" in source
    assert "refusing duplicate issue418 FICEM read repair2 run" in source
    assert "AERA_V26_ISSUE418_FICEM_READ_REPAIR2_RESULT_JSON=" in source
    assert "run_ficem_read_probe" in source
    assert "candidate_backend_changed_by_repair2\": False" in source
    assert "torch.load(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "seed8471" not in source.lower()


def test_issue418_workflow_is_one_attempt_exact_bound_and_no_retry_path():
    source = WORKFLOW_PATH.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-3-ficem-read-l4-repair2]" in source
    assert 'if [ "${GITHUB_RUN_ATTEMPT}" != "1" ]; then' in source
    assert "33497787407" in source
    assert "tied fourth/fifth read boundary" in source
    assert "Expected exactly one issue418 FICEM read repair2 trigger" in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 513d8d186f69aa69a4be4d53c76e718b36133310 HEAD" in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_repair2_app.py") == 1
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source


def test_issue418_workflow_permissions_are_minimal_for_source_guard():
    source = WORKFLOW_PATH.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "contents: write" not in permissions
    assert "actions: write" not in permissions
    assert "pull-requests: write" not in permissions
