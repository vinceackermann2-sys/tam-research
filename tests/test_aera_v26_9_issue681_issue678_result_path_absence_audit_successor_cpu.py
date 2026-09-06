from __future__ import annotations

import hashlib
from pathlib import Path

WORKFLOW = Path(".github/workflows/aera-v26-9-issue681-issue678-result-path-absence-audit-successor.yml")
TEST_FILE = Path("tests/test_aera_v26_9_issue681_issue678_result_path_absence_audit_successor_cpu.py")
AUDIT_LAUNCHER = Path("modal_aera_v26_9_issue678_issue665_result_path_absence_audit.py")
ISSUE678_CPU_TEST = Path("tests/test_aera_v26_9_issue678_issue665_result_path_absence_audit_cpu.py")
ISSUE665_LAUNCHER = Path("modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py")
SCIENTIFIC_ADAPTER = Path("tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py")
RUNTIME = Path("tam_research/aera_hardware_core_v26.py")
BACKEND = Path("tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py")

EXPECTED_WORKFLOW_BLOB = "2671d2856aeb04c0bb76ccc466ed1e8e178695fb"
EXPECTED_AUDIT_LAUNCHER_BLOB = "5d3f9df21bbee2d2b7dc1491bb8d1ce4dde5ec76"
EXPECTED_ISSUE678_CPU_TEST_BLOB = "7c008d1afa505e41eed4f04aa5d13592bc812aa7"
EXPECTED_ISSUE665_LAUNCHER_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"
EXPECTED_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
EXPECTED_RUNTIME_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
EXPECTED_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"

MANIFEST_HEADING = "## #681 pre-implementation read-only result-path successor freeze"
TRIGGER_PREFIX = "[aera-v26-9-issue681-issue678-result-path-absence-audit-successor]"
AUTH_HEADING = "## #681 sole read-only result-path audit authorization"


def _blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_exact_issue681_additive_scope_and_frozen_existing_bytes() -> None:
    issue681_files = sorted(
        str(path)
        for path in Path(".").rglob("*issue681*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert issue681_files == sorted((str(WORKFLOW), str(TEST_FILE)))
    assert _blob_sha(AUDIT_LAUNCHER) == EXPECTED_AUDIT_LAUNCHER_BLOB
    assert _blob_sha(ISSUE678_CPU_TEST) == EXPECTED_ISSUE678_CPU_TEST_BLOB
    assert _blob_sha(ISSUE665_LAUNCHER) == EXPECTED_ISSUE665_LAUNCHER_BLOB
    assert _blob_sha(SCIENTIFIC_ADAPTER) == EXPECTED_ADAPTER_BLOB
    assert _blob_sha(RUNTIME) == EXPECTED_RUNTIME_BLOB
    assert _blob_sha(BACKEND) == EXPECTED_BACKEND_BLOB


def test_successor_workflow_blob_and_manifest_schema() -> None:
    assert _blob_sha(WORKFLOW) == EXPECTED_WORKFLOW_BLOB
    text = WORKFLOW.read_text()
    assert EXPECTED_WORKFLOW_BLOB not in text
    assert MANIFEST_HEADING in text
    fields = (
        "SOURCE_MAIN=",
        "SOURCE_TREE=",
        "BRANCH=",
        "TRIGGER_PREFIX=",
        "RESULT_PATH=",
        "FROZEN_ISSUE678_AUDIT_LAUNCHER_BLOB=",
        "FROZEN_ISSUE678_CPU_TEST_BLOB=",
        "FROZEN_ISSUE665_LAUNCHER_BLOB=",
        "WORKFLOW_BLOB=",
        "CPU_TEST_BLOB=",
        "FROZEN_TREE=",
        "FROZEN_COMMIT=",
    )
    for field in fields:
        assert field in text


def test_registration_safe_yaml_top_level_and_heredoc() -> None:
    text = WORKFLOW.read_text()
    allowed = ("name:", "on:", "permissions:", "concurrency:", "jobs:")
    unexpected = [
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if line and not line[0].isspace() and not line.startswith(allowed)
    ]
    assert unexpected == []
    assert "\nPY\n" not in text
    assert "          PY\n" in text
    assert "          python - <<'PY' >/tmp/issue681-summary.md\n" in text


def test_fresh_trigger_authorization_and_consumed_boundaries() -> None:
    text = WORKFLOW.read_text()
    assert "on:\n  issues:\n    types: [opened]" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert f"startsWith(github.event.issue.title, '{TRIGGER_PREFIX}')" in text
    assert text.count(f'startswith("{TRIGGER_PREFIX}")') >= 2
    assert AUTH_HEADING in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert "5561787353" in text
    assert "pulls/680" in text
    assert "actions/runs/34056221019" in text
    assert "actions/jobs/101548494965" in text
    assert "7d9235415b2d059af699d8f427f6e9874e961607" in text


def test_only_readonly_audit_launcher_can_execute() -> None:
    text = WORKFLOW.read_text()
    command = "modal run modal_aera_v26_9_issue678_issue665_result_path_absence_audit.py::audit_main"
    modal_run_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("modal run ")]
    assert modal_run_lines == [command + " 2>&1 | tee /tmp/issue681.log"]
    assert text.index("Verify issue681 canonical read-only successor boundary") < text.index(command)
    assert text.index("Re-verify issue681 trigger immediately before Modal") < text.index(command)
    assert text.index("Authenticate Modal") < text.index(command)
    assert "modal run modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py" not in text


def test_audit_launcher_remains_zero_gpu_preflight_only() -> None:
    text = AUDIT_LAUNCHER.read_text()
    assert text.count("issue665.preflight.remote()") == 1
    assert "run_diagnostic.remote" not in text
    assert "gpu=" not in text
    assert "volume.commit" not in text
    lowered = text.lower()
    for forbidden in ("optimizer", ".backward(", "load_checkpoint", "model ="):
        assert forbidden not in lowered


def test_evidence_contract_requires_absence_and_no_higher_stage_credit() -> None:
    text = WORKFLOW.read_text()
    assert 'payload.get("research_issue") != 665' in text
    assert 'payload.get("result_absent") is not True' in text
    assert "914615db5267565563dcc9e82bfc31f444a656a68bd560f50447a8fd03588431" in text
    assert "FAIL_FROZEN_E2E_SYSTEMS_GATE" in text
    required_false = (
        "gpu_used",
        "model_constructed",
        "new_measurement_performed",
        "systems_pass_earned",
        "optimization_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    )
    for key in required_false:
        assert f'"{key}"' in text


def test_no_alternate_dispatch_deploy_or_true_authority() -> None:
    lowered = WORKFLOW.read_text().lower()
    assert "workflow_dispatch" not in lowered
    assert "modal deploy" not in lowered
    for phrase in (
        "systems_pass_earned=true",
        "optimization_authorized=true",
        "architecture_freeze_authorized=true",
        "s2_authorized=true",
        "fresh_scientific_seed_authorized=true",
        "independent_replication_credit=true",
        "100m_authorized=true",
        "breakthrough_proven=true",
    ):
        assert phrase not in lowered
