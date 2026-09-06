from __future__ import annotations

import hashlib
from pathlib import Path

AUDIT = Path("modal_aera_v26_9_issue678_issue665_result_path_absence_audit.py")
WORKFLOW = Path(".github/workflows/aera-v26-9-issue678-issue665-result-path-absence-audit.yml")
TEST_FILE = Path("tests/test_aera_v26_9_issue678_issue665_result_path_absence_audit_cpu.py")
ISSUE665 = Path("modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py")
ISSUE675_WORKFLOW = Path(".github/workflows/aera-v26-9-issue675-issue665-workflow-registration-repair2.yml")

EXPECTED_AUDIT_BLOB = "5d3f9df21bbee2d2b7dc1491bb8d1ce4dde5ec76"
EXPECTED_WORKFLOW_BLOB = "d6e70a36cd9d8311e94443ceaa6ddd85bc965b86"
EXPECTED_ISSUE665_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"
EXPECTED_ISSUE675_WORKFLOW_BLOB = "942f71f626c85d450fff81fd9f5fc13eefd3d0aa"
PREFIX = "[aera-v26-9-issue678-issue665-result-path-absence-audit]"
AUTH = "## #678 sole read-only result-path audit authorization"
MANIFEST = "## #678 pre-implementation read-only result-path absence audit freeze"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_exact_issue678_scope_and_frozen_predecessors() -> None:
    issue678_files = sorted(
        str(path)
        for path in Path(".").rglob("*issue678*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert issue678_files == sorted((str(AUDIT), str(WORKFLOW), str(TEST_FILE)))
    assert _blob(ISSUE665) == EXPECTED_ISSUE665_BLOB
    assert _blob(ISSUE675_WORKFLOW) == EXPECTED_ISSUE675_WORKFLOW_BLOB


def test_audit_and_workflow_blobs_are_exact() -> None:
    assert _blob(AUDIT) == EXPECTED_AUDIT_BLOB
    assert _blob(WORKFLOW) == EXPECTED_WORKFLOW_BLOB
    assert EXPECTED_WORKFLOW_BLOB not in WORKFLOW.read_text()


def test_audit_is_preflight_only_and_read_only() -> None:
    text = AUDIT.read_text()
    assert text.count("issue665.preflight.remote()") == 1
    assert "run_diagnostic.remote" not in text
    assert "gpu=" not in text
    assert "volume.commit" not in text
    assert "import torch" not in text
    assert "load_models" not in text
    assert "optimizer" not in text.lower()
    assert "backward" not in text.lower()
    assert "result_absent" in text
    assert "AERA_V26_9_ISSUE678_ISSUE665_RESULT_PATH_ABSENCE_AUDIT_JSON=" in text


def test_workflow_registration_and_manifest_contract() -> None:
    text = WORKFLOW.read_text()
    assert "on:\n  issues:\n    types: [opened]" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert f"startsWith(github.event.issue.title, '{PREFIX}')" in text
    assert text.count(f'startswith("{PREFIX}")') >= 2
    assert MANIFEST in text
    for field in (
        "SOURCE_MAIN=",
        "SOURCE_TREE=",
        "BRANCH=",
        "TRIGGER_PREFIX=",
        "RESULT_PATH=",
        "FROZEN_ISSUE665_LAUNCHER_BLOB=",
        "FROZEN_ISSUE675_WORKFLOW_BLOB=",
        "AUDIT_LAUNCHER_BLOB=",
        "WORKFLOW_BLOB=",
        "CPU_TEST_BLOB=",
        "FROZEN_TREE=",
        "FROZEN_COMMIT=",
    ):
        assert field in text
    assert AUTH in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text


def test_workflow_yaml_block_shape_and_safe_reporting() -> None:
    text = WORKFLOW.read_text()
    allowed = ("name:", "on:", "permissions:", "concurrency:", "jobs:")
    unexpected = [
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if line and not line[0].isspace() and not line.startswith(allowed)
    ]
    assert unexpected == []
    assert "          python - <<'PY' >/tmp/issue678-summary.md\n" in text
    assert "\nPY\n" not in text
    assert "          PY\n" in text


def test_workflow_executes_only_readonly_audit_launcher_once() -> None:
    text = WORKFLOW.read_text()
    command = "modal run modal_aera_v26_9_issue678_issue665_result_path_absence_audit.py::audit_main"
    assert text.count(command) == 1
    assert text.index("Verify issue678 canonical read-only audit boundary") < text.index(command)
    assert text.index("Re-verify issue678 trigger immediately before Modal") < text.index(command)
    assert text.index("Authenticate Modal") < text.index(command)
    assert "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py::" not in text
    assert "run_diagnostic.remote" not in text


def test_workflow_freezes_canary_and_negative_authority() -> None:
    text = WORKFLOW.read_text()
    assert "issues/677" in text
    assert "34054750867" in text
    assert "5561583050" in text
    assert "result_absent" in text
    lowered = text.lower()
    assert "workflow_dispatch" not in lowered
    assert "modal deploy" not in lowered
    for phrase in (
        "systems_pass_earned == true",
        "optimization_authorized == true",
        "architecture_freeze_authorized == true",
        "s2_authorized == true",
        "fresh_scientific_seed_authorized == true",
        "independent_replication_credit == true",
        "100m_authorized == true",
        "breakthrough_proven == true",
    ):
        assert phrase not in lowered
