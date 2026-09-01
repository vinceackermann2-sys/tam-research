from __future__ import annotations

from pathlib import Path

from tam_research.aera_v26_state_movement_probe import (
    BATCH_SIZES,
    CALLS_PER_ROUND,
    CAPACITY,
    D_MODEL,
    DESIGN_SEED,
    MAX_BATCH64_GEOMEAN_LATENCY_RATIO,
    MAX_BATCH64_ROW_LATENCY_RATIO,
    MAX_BATCH8_ROW_LATENCY_RATIO,
    MAX_KERNEL_RATIO,
    MEMORY_DIM,
    PACKED_FLOAT_WIDTH,
    SELECTED_FRACTIONS,
    TIMED_ROUNDS,
    WARMUP_CALLS,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / ".github" / "workflows" / "aera-v26-state-movement-l4.yml"
REPAIR = ROOT / ".github" / "workflows" / "aera-v26-state-movement-l4-repair1.yml"
HARNESS = ROOT / "tam_research" / "aera_v26_state_movement_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_state_movement_app.py"


def _text(path: Path) -> str:
    return path.read_text()


def test_issue403_repair1_is_permission_guard_only_and_original_stays_consumed():
    original = _text(ORIGINAL)
    repair = _text(REPAIR)

    assert "[aera-v26-state-movement-l4]" in original
    assert "[aera-v26-state-movement-l4-repair1]" not in original
    assert "pull-requests: read" not in original
    assert "actions: read" not in original

    assert "[aera-v26-state-movement-l4-repair1]" in repair
    assert "actions: read" in repair
    assert "contents: read" in repair
    assert "issues: write" in repair
    assert "pull-requests: read" in repair
    assert "actions: write" not in repair
    assert "pull-requests: write" not in repair
    assert "Issue #403 permits Actions attempt 1 only" in repair
    assert "cancel-in-progress: false" in repair


def test_issue403_repair1_freezes_source_failure_and_proves_no_source_l4():
    repair = _text(REPAIR)
    assert "33489236409" in repair
    assert "99796514541" in repair
    assert "ee46b17b55711e481d8b29fac98b9a66dba1ff67" in repair
    assert "Verify issue400 one-attempt synthetic boundary" in repair
    assert 'test "${source_guard_conclusion}" = "failure"' in repair
    assert 'test "${modal_auth_conclusion}" = "skipped"' in repair
    assert 'test "${l4_conclusion}" = "skipped"' in repair
    assert "Expected exactly one issue400 repair1 trigger" in repair
    assert "source_result_comment_count" in repair
    assert "repair_result_comment_count" in repair


def test_issue403_repair1_reuses_exact_issue400_launcher_marker_and_duplicate_path():
    repair = _text(REPAIR)
    launcher = _text(LAUNCHER)

    assert repair.count("modal run modal_aera_v26_state_movement_app.py") == 1
    assert "AERA_V26_ISSUE400_STATE_MOVEMENT_RESULT_JSON=" in repair
    assert 'RESULT_PATH = "/vol/aera-v26/issue400-state-movement/result.json"' in launcher
    assert 'APP_NAME = "tam-research-aera-v26-issue400-state-movement"' in launcher
    assert "MAX_GPU_SECONDS = 300" in launcher
    assert 'gpu="L4"' in launcher
    assert "refusing duplicate issue400 state-movement run because result exists" in launcher


def test_issue403_repair1_does_not_change_frozen_geometry_or_threshold_contract():
    assert DESIGN_SEED == 398_400
    assert (D_MODEL, MEMORY_DIM, CAPACITY, PACKED_FLOAT_WIDTH) == (200, 50, 48, 5_048)
    assert BATCH_SIZES == (8, 64)
    assert SELECTED_FRACTIONS == (0.25, 0.50, 0.75, 1.00)
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 200)
    assert MAX_KERNEL_RATIO == 0.80
    assert MAX_BATCH64_GEOMEAN_LATENCY_RATIO == 0.90
    assert MAX_BATCH64_ROW_LATENCY_RATIO == 1.05
    assert MAX_BATCH8_ROW_LATENCY_RATIO == 1.10

    harness = _text(HARNESS)
    assert "torch.cuda.Event(enable_timing=True)" in harness
    assert "torch.profiler.ProfilerActivity.CUDA" in harness
    assert "pack_ephemeral_epi_state(case.base)" in harness
    assert "pack_ephemeral_epi_state(case.update)" in harness


def test_issue403_repair1_has_no_rerun_redispatch_or_alternate_measurement_path():
    repair = _text(REPAIR).lower()
    forbidden = (
        "gh run rerun",
        "workflow_dispatch",
        "/rerun-failed-jobs",
        "/rerun",
        "rerun_workflow",
        "modal deploy",
    )
    for token in forbidden:
        assert token not in repair

    assert "run sole issue400 repair1 synthetic l4 probe" in repair
    assert "no rerun, automatic retry, or timeout increase is authorized" in repair
    assert "harness, launcher, synthetic geometry, timing, profiling, thresholds" in repair
