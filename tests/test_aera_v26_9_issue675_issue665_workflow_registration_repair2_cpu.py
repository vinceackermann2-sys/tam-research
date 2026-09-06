from __future__ import annotations

import hashlib
from pathlib import Path

WORKFLOW = Path(".github/workflows/aera-v26-9-issue675-issue665-workflow-registration-repair2.yml")
TEST_FILE = Path("tests/test_aera_v26_9_issue675_issue665_workflow_registration_repair2_cpu.py")
FROZEN_ISSUE665_LAUNCHER = Path("modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py")
FROZEN_BAD_ISSUE665_WORKFLOW = Path(".github/workflows/aera-v26-9-issue665-frozen-throughput-component-attribution.yml")
SCIENTIFIC_ADAPTER = Path("tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py")
RUNTIME = Path("tam_research/aera_hardware_core_v26.py")
BACKEND = Path("tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py")

EXPECTED_WORKFLOW_BLOB = "942f71f626c85d450fff81fd9f5fc13eefd3d0aa"
EXPECTED_ISSUE665_LAUNCHER_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"
EXPECTED_BAD_ISSUE665_WORKFLOW_BLOB = "d2642917ed1e9009edfe1b255131d0e55723ea6a"
EXPECTED_SCIENTIFIC_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
EXPECTED_RUNTIME_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
EXPECTED_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"

MANIFEST_HEADING = "## #675 pre-implementation workflow-registration repair2 freeze"
MANIFEST_FIELDS = (
    "SOURCE_MAIN=",
    "SOURCE_TREE=",
    "BRANCH=",
    "TRIGGER_PREFIX=",
    "REGISTRATION_CANARY_TITLE=",
    "FROZEN_ISSUE665_LAUNCHER_BLOB=",
    "FROZEN_BAD_ISSUE665_WORKFLOW_BLOB=",
    "WORKFLOW_BLOB=",
    "CPU_TEST_BLOB=",
    "FROZEN_TREE=",
    "FROZEN_COMMIT=",
)
FORBIDDEN_MANIFEST_ALIASES = ("TREE_SHA=", "COMMIT_SHA=")

TRIGGER_PREFIX = "[aera-v26-9-issue675-issue665-workflow-registration-repair2]"
CANARY_TITLE = "[infra-issue675-workflow-registration-canary] post-merge registration probe"
AUTH_HEADING = "## #675 sole diagnostic continuation authorization"
CANARY_PASS_HEADING = "## #675 workflow-registration canary PASS"


def _blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _workflow_text() -> str:
    return WORKFLOW.read_text()


def test_exact_successor_scope_and_frozen_existing_bytes() -> None:
    issue675_files = sorted(
        str(path)
        for path in Path(".").rglob("*issue675*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert issue675_files == sorted((str(WORKFLOW), str(TEST_FILE)))

    assert _blob_sha(FROZEN_ISSUE665_LAUNCHER) == EXPECTED_ISSUE665_LAUNCHER_BLOB
    assert _blob_sha(FROZEN_BAD_ISSUE665_WORKFLOW) == EXPECTED_BAD_ISSUE665_WORKFLOW_BLOB
    assert _blob_sha(SCIENTIFIC_ADAPTER) == EXPECTED_SCIENTIFIC_ADAPTER_BLOB
    assert _blob_sha(RUNTIME) == EXPECTED_RUNTIME_BLOB
    assert _blob_sha(BACKEND) == EXPECTED_BACKEND_BLOB


def test_successor_workflow_blob_is_frozen_without_self_hash_cycle() -> None:
    assert _blob_sha(WORKFLOW) == EXPECTED_WORKFLOW_BLOB
    text = _workflow_text()
    assert EXPECTED_WORKFLOW_BLOB not in text
    assert "WORKFLOW_BLOB=" in text
    assert "CPU_TEST_BLOB=" in text


def test_manifest_schema_is_exact_and_alias_free() -> None:
    text = _workflow_text()
    assert MANIFEST_HEADING in text
    for field in MANIFEST_FIELDS:
        assert field in text
    for alias in FORBIDDEN_MANIFEST_ALIASES:
        assert alias not in text

    assert "SOURCE_MAIN=2fb477ce1323f59460678a3cbb0e260e70559ef2" in text
    assert "SOURCE_TREE=3ed87b6378338d49d2a46c90ab6580714bd5c046" in text
    assert "BRANCH=research/aera-v26-9-issue675-issue665-workflow-registration-repair2" in text
    assert f"TRIGGER_PREFIX={TRIGGER_PREFIX}" in text
    assert f"REGISTRATION_CANARY_TITLE={CANARY_TITLE}" in text
    assert f"FROZEN_ISSUE665_LAUNCHER_BLOB={EXPECTED_ISSUE665_LAUNCHER_BLOB}" in text
    assert f"FROZEN_BAD_ISSUE665_WORKFLOW_BLOB={EXPECTED_BAD_ISSUE665_WORKFLOW_BLOB}" in text


def test_workflow_has_only_registration_safe_top_level_yaml_lines() -> None:
    text = _workflow_text()
    allowed_top_level = ("name:", "on:", "permissions:", "concurrency:", "jobs:")
    unexpected = [
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if line and not line[0].isspace() and not line.startswith(allowed_top_level)
    ]
    assert unexpected == []

    assert "\n`${compact}`\n" not in text
    assert "\nThe immutable #653/#650 decision" not in text
    assert "\nPY\n" not in text
    assert "          PY\n" in text
    assert "          python - <<'PY' >/tmp/issue675-summary.md\n" in text


def test_issue_opened_trigger_and_registration_canary_contract() -> None:
    text = _workflow_text()
    assert "on:\n  issues:\n    types: [opened]" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert f"startsWith(github.event.issue.title, '{TRIGGER_PREFIX}')" in text
    assert text.count(f'startswith("{TRIGGER_PREFIX}")') >= 2
    assert f'.title == "{CANARY_TITLE}"' in text
    assert CANARY_PASS_HEADING in text
    assert AUTH_HEADING in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text


def test_consumed_predecessors_and_cpu_green_blocked_head_are_frozen() -> None:
    text = _workflow_text()
    assert "issues/671" in text
    assert "issues/672" in text
    assert "pulls/674" in text
    assert "b46388347ea290f0a6c4915177445d4f3145871a" in text
    assert "actions/runs/34053254935" in text
    assert "actions/jobs/101540510967" in text
    assert "5561439872" in text


def test_frozen_issue665_launcher_is_invoked_exactly_once_after_guards() -> None:
    text = _workflow_text()
    command = "modal run modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
    assert text.count(command) == 1
    assert text.index("Verify issue675 canonical diagnostic-continuation boundary") < text.index(command)
    assert text.index("Re-verify issue675 trigger immediately before Modal") < text.index(command)
    assert text.index("Authenticate Modal") < text.index(command)
    assert "AERA_V26_9_ISSUE665_COMPONENT_ATTRIBUTION_RESULT_JSON=" in text


def test_fresh_result_path_and_single_l4_lineage_remain_in_frozen_launcher() -> None:
    launcher = FROZEN_ISSUE665_LAUNCHER.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue665-frozen-throughput-component-attribution/result.json"' in launcher
    assert "if result_path.exists():" in launcher
    assert launcher.count('gpu="L4"') == 1
    assert "DIAGNOSTIC_WARMUP_CALLS = 2" in launcher
    assert "DIAGNOSTIC_MEASURED_CALLS = 12" in launcher
    assert "SYSTEM_BATCH_SIZES = (8, 64)" in launcher
    assert "TOKEN_SEED_BASE = 138471" in launcher
    assert "volume.commit()" in launcher


def test_no_alternate_dispatch_or_higher_stage_authority() -> None:
    text = _workflow_text()
    lowered = text.lower()
    assert "workflow_dispatch" not in lowered
    assert "modal deploy" not in lowered
    assert "threshold modification" not in lowered
    assert "route_mode=" not in lowered

    forbidden_true = (
        "systems_pass_earned=true",
        "optimization_authorized=true",
        "architecture_freeze_authorized=true",
        "s2_authorized=true",
        "fresh_scientific_seed_authorized=true",
        "independent_replication_credit=true",
        "100m_authorized=true",
        "breakthrough_proven=true",
    )
    for phrase in forbidden_true:
        assert phrase not in lowered
