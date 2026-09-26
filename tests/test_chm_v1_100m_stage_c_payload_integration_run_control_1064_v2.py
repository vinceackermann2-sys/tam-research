from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v2.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-v2.yml"
AUDIT = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v2.yml"
PROBE = ROOT / "modal_runtime_admission_probe_1067_v1.py"
SELECTOR = ROOT / "tam_research" / "modal_dual_account_v2.py"
CLI = ROOT / "scripts" / "modal_select_account_v2.py"


def _heredocs(source: str) -> list[str]:
    lines = source.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if "python - <<'PY'" not in lines[i]:
            i += 1
            continue
        i += 1
        block = []
        while i < len(lines) and lines[i].strip() != "PY":
            block.append(lines[i])
            i += 1
        assert i < len(lines), "unterminated Python heredoc"
        blocks.append(textwrap.dedent("\n".join(block)))
        i += 1
    return blocks


def test_v2_runner_preserves_diagnostic_semantics_and_fresh_namespace() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'PHASE = "chm-v1-100m-stage-c-payload-integration-1064-v2"' in source
    assert 'TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-payload-integration-1064-v2]"' in source
    assert 'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-integration/issue-1064/v2"' in source
    assert 'DECOMPOSITION_BLOB = "8ab98658d4c8ced33623c2d28ec899b35339997e"' in source
    assert 'DUAL_ACCOUNT_BLOB = "78e66d827fe6d1b0c7d7fdedaf0b47096a844db9"' in source
    assert 'DUAL_ACCOUNT_CLI_BLOB = "cb38f336f076f9d7e294d104f106bdb4df8a4fda"' in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "torch.save(" not in source
    assert "retries=0" in source
    assert "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v8" not in source


def test_v2_scientific_workflow_requires_runtime_admission_revalidation() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-v2]" in source
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_FINAL_LAUNCHER_AUTHORITY_V2" in source
    assert "scripts/modal_select_account_v2.py" in source
    assert "modal_runtime_admission_probe_1067_v1.py" in source
    assert "runtime_admission_probe_blob=" in source
    assert 'issue-1064/v2' in source
    assert "--force-account" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "automatic_retry_authorized=false" in source
    assert "stage_d_authorized=false" in source


def test_v2_audit_is_fresh_read_only_successor_to_failed_v1_audit() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v2]" in source
    assert '"supersedes_failed_audit_issue":1066' in source
    assert '"supersedes_failed_audit_run":36254024967' in source
    assert "scripts/modal_select_account_v2.py" in source
    assert "modal_runtime_admission_probe_1067_v1.py" in source
    assert "runtime_admission_verified=true" in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run " not in source
    assert "--phase state" not in source
    assert "issue-1064/v2" in source
    assert "result_namespace_unused=true" in source
    assert "gpu_allocated=false" in source
    assert "diagnostic_attempt_consumed=false" in source
    assert "trigger_authorized=false" in source


def test_runtime_probe_is_minimal_zero_gpu_read_only() -> None:
    source = PROBE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "gpu=" not in source
    assert "cpu=0.125" in source
    assert "memory=128" in source
    assert "timeout=60" in source
    assert "retries=0" in source
    assert "create_if_missing=False" in source
    assert "volume.commit" not in source
    assert "write_text(" not in source
    assert '"runtime_admission_ok": True' in source
    assert '"writes_performed": False' in source


def test_v2_selector_requires_runtime_admission_for_health_and_cli_uses_it() -> None:
    selector = SELECTOR.read_text(encoding="utf-8")
    cli = CLI.read_text(encoding="utf-8")
    ast.parse(selector)
    ast.parse(cli)
    assert "runtime_admission_ok" in selector
    assert "and self.runtime_admission_ok" in selector
    assert '["modal", "run", runtime_probe_path]' in selector
    assert "modal_runtime_admission_probe_1067_v1.py" in selector
    assert "modal_dual_account_v2.py" in cli
    assert "--runtime-probe-path" in cli


def test_all_v2_workflow_embedded_python_is_valid() -> None:
    for path in (WORKFLOW, AUDIT):
        source = path.read_text(encoding="utf-8")
        assert "<<:" not in source
        blocks = _heredocs(source)
        assert blocks
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} embedded Python block {index} invalid: {exc}"
                ) from exc


def test_historical_v1_surfaces_still_exist_and_are_not_replaced() -> None:
    assert (ROOT / "tam_research" / "modal_dual_account.py").is_file()
    assert (ROOT / "scripts" / "modal_select_account.py").is_file()
    assert (ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v1.py").is_file()
    assert (
        ROOT
        / ".github"
        / "workflows"
        / "modal-chm-v1-100m-stage-c-payload-integration-1064-v1.yml"
    ).is_file()
