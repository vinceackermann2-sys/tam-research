from __future__ import annotations

import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github/workflows/aera-v26-9-issue673-issue665-workflow-registration-repair1.yml"
CPU_TEST = REPO / "tests/test_aera_v26_9_issue673_issue665_workflow_registration_repair1_cpu.py"
FROZEN_LAUNCHER = REPO / "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
FROZEN_BAD_WORKFLOW = REPO / ".github/workflows/aera-v26-9-issue665-frozen-throughput-component-attribution.yml"
SCIENTIFIC_ADAPTER = REPO / "tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py"
RUNTIME_INTERFACE = REPO / "tam_research/aera_hardware_core_v26.py"
V26_9_BACKEND = REPO / "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py"

FROZEN_LAUNCHER_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"
FROZEN_BAD_WORKFLOW_BLOB = "d2642917ed1e9009edfe1b255131d0e55723ea6a"
SCIENTIFIC_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
RUNTIME_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"

PREFIX = "[aera-v26-9-issue673-issue665-workflow-registration-repair1]"
CANARY_TITLE = "[infra-issue673-workflow-registration-canary] post-merge registration probe"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _unexpected_top_level_lines(text: str) -> list[str]:
    allowed = ("name:", "on:", "permissions:", "concurrency:", "jobs:")
    return [
        line
        for line in text.splitlines()
        if line
        and not line.startswith((" ", "#"))
        and not line.startswith(allowed)
    ]


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def test_issue673_scope_is_exactly_two_fresh_files() -> None:
    candidates = set()
    candidates.update(
        str(path.relative_to(REPO))
        for path in REPO.glob(".github/workflows/*issue673*")
        if path.is_file()
    )
    candidates.update(
        str(path.relative_to(REPO))
        for path in REPO.glob("tests/*issue673*.py")
        if path.is_file()
    )
    candidates.update(
        str(path.relative_to(REPO))
        for path in REPO.glob("modal_*issue673*.py")
        if path.is_file()
    )
    candidates.update(
        str(path.relative_to(REPO))
        for path in REPO.glob("tam_research/*issue673*.py")
        if path.is_file()
    )
    assert candidates == {
        ".github/workflows/aera-v26-9-issue673-issue665-workflow-registration-repair1.yml",
        "tests/test_aera_v26_9_issue673_issue665_workflow_registration_repair1_cpu.py",
    }


def test_frozen_issue665_and_scientific_bytes_are_unchanged() -> None:
    assert _git_blob_sha(FROZEN_LAUNCHER) == FROZEN_LAUNCHER_BLOB
    assert _git_blob_sha(FROZEN_BAD_WORKFLOW) == FROZEN_BAD_WORKFLOW_BLOB
    assert _git_blob_sha(SCIENTIFIC_ADAPTER) == SCIENTIFIC_ADAPTER_BLOB
    assert _git_blob_sha(RUNTIME_INTERFACE) == RUNTIME_INTERFACE_BLOB
    assert _git_blob_sha(V26_9_BACKEND) == V26_9_BACKEND_BLOB


def test_frozen_issue665_registration_defect_is_detected() -> None:
    bad = FROZEN_BAD_WORKFLOW.read_text()
    unexpected = _unexpected_top_level_lines(bad)
    assert unexpected, "frozen malformed #665 workflow unexpectedly has no column-zero defect"
    assert any("${compact}" in line for line in unexpected)
    assert any(
        line.startswith("The immutable #653/#650 decision remains FAIL")
        for line in unexpected
    )
    assert '\n\\`${compact}\\`\n' in bad
    assert "\nThe immutable #653/#650 decision remains FAIL" in bad


def test_successor_has_no_unexpected_column_zero_yaml_lines() -> None:
    text = WORKFLOW.read_text()
    assert _unexpected_top_level_lines(text) == []


def test_successor_uses_registration_safe_multiline_reporting() -> None:
    lines = WORKFLOW.read_text().splitlines()
    heredoc_start = next(
        index
        for index, line in enumerate(lines)
        if "python - <<'PY' >/tmp/issue673-summary.md" in line
    )
    run_line = max(
        index
        for index in range(heredoc_start)
        if lines[index].lstrip().startswith("run: |")
    )
    run_indent = _leading_spaces(lines[run_line])
    terminator = next(
        index
        for index in range(heredoc_start + 1, len(lines))
        if lines[index].strip() == "PY"
    )
    for line in lines[heredoc_start : terminator + 1]:
        if line.strip():
            assert _leading_spaces(line) > run_indent
    text = "\n".join(lines)
    assert 'body="$(cat /tmp/issue673-summary.md)"' in text
    assert '\n\\`${compact}\\`\n' not in text
    assert "\nThe immutable #653/#650 decision remains FAIL" not in text


def test_successor_event_and_trigger_contract() -> None:
    text = WORKFLOW.read_text()
    assert "\non:\n  issues:\n    types: [opened]\n" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert f"startsWith(github.event.issue.title, '{PREFIX}')" in text
    assert text.count(PREFIX) >= 3
    assert 'test "${#matching_triggers[@]}" = "1"' in text
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert "## #673 sole diagnostic continuation authorization" in text
    assert "## #673 workflow-registration canary PASS" in text
    assert CANARY_TITLE in text


def test_successor_requires_consumed_predecessor_evidence() -> None:
    text = WORKFLOW.read_text()
    assert "issues/671" in text
    assert "issues/672" in text
    assert "[aera-v26-9-issue665-frozen-throughput-component-attribution] sole #665 diagnostic trigger" in text
    assert "[infra-issue665-workflow-registration-canary] post-#671 registration probe" in text
    assert "5561237545" in text
    assert "#665 terminal classification of consumed registration canary #672 + repair-successor boundary" in text


def test_successor_reuses_frozen_launcher_exactly_once() -> None:
    text = WORKFLOW.read_text()
    command = "modal run modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
    assert text.count(command) == 1
    assert "modal_aera_v26_9_issue673" not in text
    assert FROZEN_LAUNCHER_BLOB in text
    assert FROZEN_BAD_WORKFLOW_BLOB in text
    assert SCIENTIFIC_ADAPTER_BLOB in text
    assert RUNTIME_INTERFACE_BLOB in text
    assert V26_9_BACKEND_BLOB in text

    launcher = FROZEN_LAUNCHER.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue665-frozen-throughput-component-attribution/result.json"' in launcher
    assert 'volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)' in launcher
    assert launcher.count('gpu="L4"') == 1
    assert "DIAGNOSTIC_WARMUP_CALLS = 2" in launcher
    assert "DIAGNOSTIC_MEASURED_CALLS = 12" in launcher


def test_successor_preserves_pre_l4_result_absence_guard() -> None:
    launcher = FROZEN_LAUNCHER.read_text()
    preflight_index = launcher.index("def preflight()")
    l4_index = launcher.index('gpu="L4"')
    preflight_region = launcher[preflight_index:l4_index]
    assert "if result_path.exists()" in preflight_region
    assert "issue665 diagnostic result already exists" in preflight_region
    assert "volume.reload()" in preflight_region

    workflow = WORKFLOW.read_text()
    assert "Authenticate Modal" in workflow
    assert "Run frozen issue665 diagnostic once" in workflow


def test_successor_has_no_forbidden_dispatch_or_reexecution_path() -> None:
    text = WORKFLOW.read_text().lower()
    forbidden = (
        "workflow_dispatch",
        "modal deploy",
        "gh run rerun",
        "/rerun",
        "rerun_workflow",
        "retry-max",
    )
    for token in forbidden:
        assert token not in text


def test_successor_result_reporting_cannot_grant_higher_authority() -> None:
    text = WORKFLOW.read_text()
    required_false = (
        "source_decision_changed",
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
    assert 'payload.get(key) is not False' in text
    assert 'payload.get("source_decision") != "FAIL_FROZEN_E2E_SYSTEMS_GATE"' in text
    assert "cannot earn systems PASS" in text


def test_successor_binds_exact_source_lineage() -> None:
    text = WORKFLOW.read_text()
    assert "2fb477ce1323f59460678a3cbb0e260e70559ef2" in text
    assert "3ed87b6378338d49d2a46c90ab6580714bd5c046" in text
    assert 'git merge-base --is-ancestor 2fb477ce1323f59460678a3cbb0e260e70559ef2 "${bound_main}"' in text
    assert 'test "$(git rev-parse 2fb477ce1323f59460678a3cbb0e260e70559ef2^{tree})" = "3ed87b6378338d49d2a46c90ab6580714bd5c046"' in text


def test_successor_file_itself_is_not_a_scientific_or_launcher_change() -> None:
    assert WORKFLOW.exists()
    assert CPU_TEST.exists()
    assert not (REPO / "modal_aera_v26_9_issue673_issue665_workflow_registration_repair1.py").exists()
    assert not (REPO / "tam_research/aera_v26_9_issue673_issue665_workflow_registration_repair1.py").exists()
