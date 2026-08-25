from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / ".github" / "workflows" / "aera-v25-1-systems-l4-repair1.yml"
ORIGINAL = ROOT / ".github" / "workflows" / "aera-v25-1-systems-l4.yml"
LAUNCHER = ROOT / "modal_aera_v25_1_systems_app.py"


def _text(path: Path) -> str:
    return path.read_text()


def test_issue381_repair1_has_distinct_single_attempt_trigger_and_source_identity():
    src = _text(REPAIR)
    assert "[aera-v25-1-systems-l4-repair1]" in src
    assert "[research] AERA-v25.1 issue381 systems repair1: pre-GPU issue-comment permission failure" in src
    assert "Issue #392 permits Actions attempt 1 only" in src
    assert "GITHUB_RUN_ATTEMPT" in src
    assert "32885050371" in src
    assert "97923438291" in src
    assert "a34511f2c535644edf1bcb4170ba51a17a3ec6a3" in src
    assert "test \"$(jq -r '.conclusion' <<<\"${source_run}\")\" = \"failure\"" in src
    assert "test \"$(jq -r '.run_attempt' <<<\"${source_run}\")\" = \"1\"" in src
    assert 'test "${guard_conclusion}" = "failure"' in src
    assert 'test "${modal_auth_conclusion}" = "skipped"' in src
    assert 'test "${l4_conclusion}" = "skipped"' in src
    assert "repair_trigger_count" in src
    assert "Expected exactly one issue381 systems repair1 trigger" in src
    assert "cancel-in-progress: false" in src


def test_issue381_repair1_reporting_is_rest_best_effort_not_a_gpu_prerequisite():
    src = _text(REPAIR)
    assert "gh issue comment" not in src
    assert 'gh api "repos/${GITHUB_REPOSITORY}/issues/${REPAIR_TRIGGER}/comments"' in src
    assert "Record frozen repair1 guard (best effort)" in src
    assert "Record authoritative repair1 result (best effort)" in src
    assert "Record repair1 workflow failure (best effort)" in src
    assert src.count("continue-on-error: true") == 3
    assert "warning: repair1 guard comment could not be posted; reporting is non-authoritative" in src
    assert "warning: authoritative result exists but GitHub reporting comment failed" in src
    assert "warning: repair1 failure reporting comment could not be posted" in src


def test_issue381_repair1_keeps_authoritative_modal_result_marker_and_no_retry_logic():
    src = _text(REPAIR)
    assert "modal run modal_aera_v25_1_systems_app.py" in src
    assert "AERA_V25_1_ISSUE381_SYSTEMS_RESULT_JSON=" in src
    assert "if [ \"${rc}\" = \"0\" ]; then exit 1; fi" in src
    assert "No automatic retry is authorized" in src
    # Safety prose may say "no rerun"; reject executable rerun/re-dispatch mechanisms instead.
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


def test_issue381_repair1_preserves_frozen_systems_launcher_contract():
    launcher = _text(LAUNCHER)
    src = _text(REPAIR)
    assert "MAX_GPU_SECONDS = 600" in launcher
    assert 'gpu="L4"' in launcher
    assert "run_guarded_systems_comparison" in launcher
    assert "SOURCE_RUN_DIR = \"/vol/aera-real-language/v25-dev-seed8471\"" in launcher
    assert "RESULT_PATH = \"/vol/aera-real-language/v25-1-issue381-systems/result.json\"" in launcher
    assert "refusing duplicate issue381 systems run because result exists" in launcher
    assert "Syntax-check unchanged systems launcher" in src


def test_issue381_original_trigger_remains_distinct_and_unmodified_in_semantics():
    original = _text(ORIGINAL)
    repair = _text(REPAIR)
    assert "[aera-v25-1-systems-l4]" in original
    assert "[aera-v25-1-systems-l4-repair1]" not in original
    assert "[aera-v25-1-systems-l4-repair1]" in repair
    assert "Run sole L4 systems comparison" in original
    assert "Run sole repair1 L4 systems comparison" in repair
    # The failed source command remains historical evidence; repair1 does not rewrite it.
    assert "gh issue comment" in original


# Fresh-head CI validation after a runner-start failure with zero executed steps.
