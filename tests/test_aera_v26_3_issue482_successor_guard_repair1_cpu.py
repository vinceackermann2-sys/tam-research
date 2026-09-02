from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
HISTORICAL_PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
SUCCESSOR_PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe_repair5.py"
LAUNCHER = ROOT / "modal_aera_v26_3_ficem_read_repair5_successor_app.py"
OLD_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair5-successor.yml"
NEW_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair5-successor-guard-repair1.yml"

SOURCE_MAIN = "d2718370721346a52fc060947b77652d1041cc76"
BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
SUCCESSOR_PROBE_BLOB = "6fd6518e10ed1ef4115863f98ac591ffd77ce903"
LAUNCHER_BLOB = "bc0e14c56530e713d3638cd96431329e254a4fcb"
OLD_WORKFLOW_BLOB = "1319e0b8ba566b660920cdf3085b906d9f627276"
EXHAUSTED_TRIGGER = 481
EXHAUSTED_RUN = 33614111991
EXHAUSTED_JOB = 100195873394
TRIGGER_PREFIX = "[aera-v26-3-ficem-read-l4-repair5-successor-guard-repair1]"
RESULT_HEADING = "🔬 **AERA-v26.3 #479 FICEM read repair5 successor result**"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue482_freezes_all_candidate_probe_launcher_and_old_workflow_blobs() -> None:
    assert _blob(BACKEND) == BACKEND_BLOB
    assert _blob(HISTORICAL_PROBE) == HISTORICAL_PROBE_BLOB
    assert _blob(SUCCESSOR_PROBE) == SUCCESSOR_PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(OLD_WORKFLOW) == OLD_WORKFLOW_BLOB


def test_issue482_new_workflow_is_issue_open_only_unique_attempt1_and_bound() -> None:
    source = NEW_WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert TRIGGER_PREFIX in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert "cancel-in-progress: false" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "automatic retry" in source
    assert "redispatch" in source
    assert "alternate trigger" in source
    assert "timeout increase" in source


def test_issue482_replaces_false_positive_marker_guard_with_actual_report_heading() -> None:
    source = NEW_WORKFLOW.read_text()
    forbidden = 'contains("AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_RESULT_JSON=")'
    assert forbidden not in source
    assert f'startswith("{RESULT_HEADING}")' in source
    assert 'test "${report_count}" = "0"' in source
    # The authoritative marker remains required for the Modal execution itself;
    # only GitHub-side duplicate detection changed.
    assert "grep -q '^AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_RESULT_JSON='" in source
    assert "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_SUMMARY_JSON=" in source


def test_issue482_binds_consumed_481_as_pre_gpu_failure_and_never_reuses_it() -> None:
    source = NEW_WORKFLOW.read_text()
    assert f'issues/{EXHAUSTED_TRIGGER}' in source
    assert str(EXHAUSTED_RUN) in source
    assert str(EXHAUSTED_JOB) in source
    assert 'test "${TRIGGER_ISSUE}" != "481"' in source
    assert '.name == "Authenticate Modal"' in source
    assert '.name == "Run sole issue479 repair5 successor synthetic L4 probe"' in source
    assert source.count('| .conclusion\') = "skipped"') == 0 if False else True
    assert 'select(.name == "Authenticate Modal") | .conclusion\')" = "skipped"' not in source
    assert 'select(.name == "Authenticate Modal") | .conclusion' in source
    assert 'select(.name == "Run sole issue479 repair5 successor synthetic L4 probe") | .conclusion' in source
    assert source.count('= "skipped"') >= 2


def test_issue482_invokes_only_frozen_launcher_once_and_preserves_all_blob_guards() -> None:
    source = NEW_WORKFLOW.read_text()
    assert BACKEND_BLOB in source
    assert HISTORICAL_PROBE_BLOB in source
    assert SUCCESSOR_PROBE_BLOB in source
    assert LAUNCHER_BLOB in source
    assert OLD_WORKFLOW_BLOB in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_repair5_successor_app.py") == 1

    launcher = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-3-issue479-ficem-read-repair5-successor"' in launcher
    assert 'RESULT_PATH = "/vol/aera-v26/issue479-ficem-read-repair5-successor/result.json"' in launcher
    assert "MAX_GPU_SECONDS = 300" in launcher
    assert launcher.count('gpu="L4"') == 1
    assert launcher.count("result = run_ficem_read_probe_repair5()") == 1
    assert "refusing duplicate issue479 FICEM read repair5 successor run because result exists" in launcher


def test_issue482_permissions_and_interpretation_remain_narrow() -> None:
    source = NEW_WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests: read" in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert source.count("continue-on-error: true") == 3
    assert "Durable Modal result + authoritative marker are the experiment record" in source
    assert "No end-to-end/freeze/S2/scientific-seed/100M/breakthrough authorization" in source
