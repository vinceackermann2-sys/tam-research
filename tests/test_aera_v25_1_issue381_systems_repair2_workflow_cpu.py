from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPAIR2 = ROOT / ".github" / "workflows" / "aera-v25-1-systems-l4-repair2.yml"
REPAIR1 = ROOT / ".github" / "workflows" / "aera-v25-1-systems-l4-repair1.yml"
ORIGINAL = ROOT / ".github" / "workflows" / "aera-v25-1-systems-l4.yml"
LAUNCHER = ROOT / "modal_aera_v25_1_systems_app.py"


def _text(path: Path) -> str:
    return path.read_text()


def test_issue381_repair2_has_minimum_actions_read_and_distinct_single_attempt_trigger():
    src = _text(REPAIR2)
    assert "actions: read" in src
    assert "contents: read" in src
    assert "issues: write" in src
    assert "[aera-v25-1-systems-l4-repair2]" in src
    assert "[research] AERA-v25.1 issue381 systems repair2: grant Actions read for frozen source-run guard" in src
    assert "Issue #395 permits Actions attempt 1 only" in src
    assert "GITHUB_RUN_ATTEMPT" in src
    assert "cancel-in-progress: false" in src


def test_issue381_repair2_freezes_consumed_repair1_source_identity_and_pre_gpu_boundary():
    src = _text(REPAIR2)
    assert "33485403310" in src
    assert "99784191657" in src
    assert "b9733602a8ea1d858e41b044fe98a31d58806dee" in src
    assert "[aera-v25-1-systems-l4-repair1] issue381 sole workflow-only continuation" in src
    assert 'test "$(jq -r \'.run_id\' <<<"${source_job}")" = "33485403310"' in src
    assert 'select(.name == "Verify source pre-GPU failure and refuse duplicate repair1")' in src
    assert 'select(.name == "Authenticate Modal")' in src
    assert 'select(.name == "Run sole repair1 L4 systems comparison")' in src
    assert 'test "${source_guard_conclusion}" = "failure"' in src
    assert 'test "${modal_auth_conclusion}" = "skipped"' in src
    assert 'test "${l4_conclusion}" = "skipped"' in src


def test_issue381_repair2_requires_unique_historical_and_current_triggers_and_zero_results():
    src = _text(REPAIR2)
    assert "original_trigger_count" in src
    assert "repair1_trigger_count" in src
    assert "repair2_trigger_count" in src
    assert 'test "${original_trigger_count}" = "1"' in src
    assert 'test "${repair1_trigger_count}" = "1"' in src
    assert "Expected exactly one issue381 systems repair2 trigger" in src
    assert "original_result_count" in src
    assert "repair1_result_count" in src
    assert "repair2_result_count" in src
    assert 'test "${original_result_count}" = "0"' in src
    assert 'test "${repair1_result_count}" = "0"' in src
    assert 'test "${repair2_result_count}" = "0"' in src


def test_issue381_repair2_reporting_is_rest_best_effort_and_not_a_gpu_prerequisite():
    src = _text(REPAIR2)
    assert "gh issue comment" not in src
    assert 'gh api "repos/${GITHUB_REPOSITORY}/issues/${REPAIR_TRIGGER}/comments"' in src
    assert "Record frozen repair2 guard (best effort)" in src
    assert "Record authoritative repair2 result (best effort)" in src
    assert "Record repair2 workflow failure (best effort)" in src
    assert src.count("continue-on-error: true") == 3
    assert "warning: repair2 guard comment could not be posted; reporting is non-authoritative" in src
    assert "warning: authoritative result exists but GitHub reporting comment failed" in src
    assert "warning: repair2 failure reporting comment could not be posted" in src


def test_issue381_repair2_keeps_authoritative_modal_result_marker_and_no_retry_logic():
    src = _text(REPAIR2)
    assert "modal run modal_aera_v25_1_systems_app.py" in src
    assert "AERA_V25_1_ISSUE381_SYSTEMS_RESULT_JSON=" in src
    assert "if [ \"${rc}\" = \"0\" ]; then exit 1; fi" in src
    assert "No automatic retry is authorized" in src
    forbidden = (
        "gh run rerun",
        "workflow_dispatch",
        "actions/runs/${GITHUB_RUN_ID}/rerun",
        "/rerun-failed-jobs",
        "rerun_workflow",
        "modal deploy",
    )
    lowered = src.lower()
    for token in forbidden:
        assert token.lower() not in lowered


def test_issue381_repair2_preserves_frozen_systems_launcher_contract():
    launcher = _text(LAUNCHER)
    src = _text(REPAIR2)
    assert "MAX_GPU_SECONDS = 600" in launcher
    assert 'gpu="L4"' in launcher
    assert "run_guarded_systems_comparison" in launcher
    assert "SOURCE_RUN_DIR = \"/vol/aera-real-language/v25-dev-seed8471\"" in launcher
    assert "RESULT_PATH = \"/vol/aera-real-language/v25-1-issue381-systems/result.json\"" in launcher
    assert "refusing duplicate issue381 systems run because result exists" in launcher
    assert "Syntax-check unchanged systems launcher" in src


def test_issue381_original_and_repair1_workflows_remain_historical_and_distinct():
    original = _text(ORIGINAL)
    repair1 = _text(REPAIR1)
    repair2 = _text(REPAIR2)
    assert "[aera-v25-1-systems-l4]" in original
    assert "[aera-v25-1-systems-l4-repair1]" in repair1
    assert "[aera-v25-1-systems-l4-repair2]" in repair2
    assert "[aera-v25-1-systems-l4-repair2]" not in original
    assert "[aera-v25-1-systems-l4-repair2]" not in repair1
    assert "Run sole L4 systems comparison" in original
    assert "Run sole repair1 L4 systems comparison" in repair1
    assert "Run sole repair2 L4 systems comparison" in repair2
