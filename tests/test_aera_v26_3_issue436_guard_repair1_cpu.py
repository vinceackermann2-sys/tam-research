from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
LAUNCHER = ROOT / "modal_aera_v26_3_ficem_read_repair3_app.py"
FAILED_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair3.yml"
REPAIR_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair3-guard-repair1.yml"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue436_preserves_every_frozen_experiment_file():
    assert _blob(PROBE) == "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
    assert _blob(BACKEND) == "b6b37f0379b280eea4e5c2b16f349951dadc4df9"
    assert _blob(LAUNCHER) == "0768818305985809dc8ba232f8d5b4a115c7ea24"
    assert _blob(FAILED_WORKFLOW) == "19649b4f9abe42ccee1451071eb115f0461d1a70"


def test_issue436_is_a_narrow_permission_repair():
    source = REPAIR_WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests: read" in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert "packages:" not in permissions


def test_issue436_requires_failed_attempt1_and_proves_old_l4_was_skipped():
    source = REPAIR_WORKFLOW.read_text()
    assert "33507022732" in source
    assert "99853390510" not in source  # job identity is selected by exact job name below.
    assert "Run sole issue433 repair3 synthetic L4 probe" in source
    assert "source_run_conclusion" not in source  # compact exact gh-api assertions only.
    assert "'.conclusion')\" = \"failure\"" in source
    assert "'.run_attempt')\" = \"1\"" in source
    assert "| .conclusion')\" = \"skipped\"" in source
    assert "8b2f850e35a21cbfc6a9266a9bbba8c3821c971f" in source


def test_issue436_has_one_new_trigger_and_no_retry_or_old_trigger_reuse():
    source = REPAIR_WORKFLOW.read_text()
    prefix = "[aera-v26-3-ficem-read-l4-repair3-guard-repair1]"
    assert prefix in source
    assert "[aera-v26-3-ficem-read-l4-repair3]'" not in source
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_repair3_app.py") == 1
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "retry" in source.lower()
    assert "redispatch" in source.lower()
    assert "alternate trigger" in source.lower()
    assert "timeout increase" in source.lower()


def test_issue436_keeps_result_path_and_authoritative_marker_in_frozen_launcher():
    launcher = LAUNCHER.read_text()
    workflow = REPAIR_WORKFLOW.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue433-ficem-read-repair3/result.json"' in launcher
    assert "MAX_GPU_SECONDS = 300" in launcher
    assert launcher.count('gpu="L4"') == 1
    marker = "AERA_V26_ISSUE433_FICEM_READ_REPAIR3_RESULT_JSON="
    assert marker in launcher
    assert marker in workflow
    assert "bfloat16_batch8_mixed" in launcher


def test_issue436_repair_guard_keeps_exact_source_hashes_and_binding():
    source = REPAIR_WORKFLOW.read_text()
    assert "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b" in source
    assert "b6b37f0379b280eea4e5c2b16f349951dadc4df9" in source
    assert "0768818305985809dc8ba232f8d5b4a115c7ea24" in source
    assert "19649b4f9abe42ccee1451071eb115f0461d1a70" in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 8b2f850e35a21cbfc6a9266a9bbba8c3821c971f HEAD" in source
