from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v3.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-v3.yml"
AUDIT = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v3.yml"
PROBE = ROOT / "modal_runtime_admission_probe_1067_v1.py"
SELECTOR = ROOT / "tam_research" / "modal_dual_account_v3.py"
CLI = ROOT / "scripts" / "modal_select_account_v3.py"


def _heredocs(source: str) -> list[str]:
    lines = source.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if "python - <<'PY'" not in lines[i]:
            i += 1
            continue
        i += 1
        block: list[str] = []
        while i < len(lines) and lines[i].strip() != "PY":
            block.append(lines[i])
            i += 1
        assert i < len(lines), "unterminated Python heredoc"
        blocks.append(textwrap.dedent("\n".join(block)))
        i += 1
    return blocks


def test_v3_runner_preserves_frozen_diagnostic_and_fresh_namespace() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'PHASE = "chm-v1-100m-stage-c-payload-integration-1064-v3"' in source
    assert 'TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-payload-integration-1064-v3]"' in source
    assert 'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-integration/issue-1064/v3"' in source
    assert 'DECOMPOSITION_BLOB = "8ab98658d4c8ced33623c2d28ec899b35339997e"' in source
    assert 'DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"' in source
    assert 'DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"' in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "torch.save(" not in source
    assert "retries=0" in source
    assert "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v8" not in source


def test_v3_selector_and_cli_are_alias_aware_without_secret_output() -> None:
    selector = SELECTOR.read_text(encoding="utf-8")
    cli = CLI.read_text(encoding="utf-8")
    ast.parse(selector)
    ast.parse(cli)

    for token in (
        "MODAL_TOKEN_ID_SECONDARY",
        "MODAL_TOKEN_SECRET_SECONDARY",
        "MODAL_TOKEN_ID_2",
        "MODAL_TOKEN_SECRET_2",
        "MODAL_TOKEN_ID_B",
        "MODAL_TOKEN_SECRET_B",
        "resolve_secondary_credentials",
        "credential_aliases",
        "runtime_admission_ok",
    ):
        assert token in selector

    assert "modal_dual_account_v3.py" in cli
    assert "secondary_credential_aliases" in cli
    assert "selection_evidence_json" in cli
    assert "token_secret" not in cli
    assert "--runtime-probe-path" in cli


def test_v3_diagnostic_workflow_uses_aliases_and_exact_v3_authority() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-v3]" in source
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_FINAL_LAUNCHER_AUTHORITY_V3" in source
    assert "scripts/modal_select_account_v3.py" in source
    assert "tam_research/modal_dual_account_v3.py" in source
    assert "modal_chm_v1_100m_stage_c_payload_integration_1064_v3.py" in source
    assert "modal_runtime_admission_probe_1067_v1.py" in source
    for secret in (
        "MODAL_TOKEN_ID_SECONDARY",
        "MODAL_TOKEN_SECRET_SECONDARY",
        "MODAL_TOKEN_ID_2",
        "MODAL_TOKEN_SECRET_2",
        "MODAL_TOKEN_ID_B",
        "MODAL_TOKEN_SECRET_B",
    ):
        assert secret in source
    for alias in ("secondary", "account2", "legacy_b"):
        assert f"*,{alias},*)" in source
    assert "SECONDARY_CREDENTIAL_ALIASES" in source
    assert "--force-account" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "automatic_retry_authorized=false" in source
    assert "stage_d_authorized=false" in source
    assert "modal_chm_v1_100m_stage_c_payload_integration_1064_v2.py" not in source
    assert "FINAL_LAUNCHER_AUTHORITY_V2" not in source
    assert "issue-1064/v2" not in source


def test_v3_audit_is_cpu_only_fresh_successor_and_alias_aware() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v3]" in source
    assert '"supersedes_failed_audit_issue":1072' in source
    assert '"supersedes_failed_audit_run":36256923462' in source
    assert '"supersedes_duplicate_audit_issue":1073' in source
    assert '"supersedes_duplicate_audit_run":36257000761' in source
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_AUTHORITY_AUDIT_V3_PASS" in source
    assert "scripts/modal_select_account_v3.py" in source
    assert "modal_chm_v1_100m_stage_c_payload_integration_1064_v3.py" in source
    assert "modal_runtime_admission_probe_1067_v1.py" in source
    for secret in (
        "MODAL_TOKEN_ID_SECONDARY",
        "MODAL_TOKEN_SECRET_SECONDARY",
        "MODAL_TOKEN_ID_2",
        "MODAL_TOKEN_SECRET_2",
        "MODAL_TOKEN_ID_B",
        "MODAL_TOKEN_SECRET_B",
    ):
        assert secret in source
    for alias in ("secondary", "account2", "legacy_b"):
        assert f"*,{alias},*)" in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run " not in source
    assert "--phase state" not in source
    assert "result_namespace_unused=true" in source
    assert "gpu_allocated=false" in source
    assert "diagnostic_attempt_consumed=false" in source
    assert "trigger_authorized=false" in source
    assert "modal_chm_v1_100m_stage_c_payload_integration_1064_v2.py" not in source
    assert "FINAL_LAUNCHER_AUTHORITY_V2" not in source
    assert "issue-1064/v2" not in source


def test_runtime_probe_remains_zero_gpu_read_only_and_immutable() -> None:
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


def test_all_v3_workflow_embedded_python_is_valid() -> None:
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


def test_historical_v1_v2_surfaces_remain_immutable_and_present() -> None:
    for path in (
        ROOT / "tam_research" / "modal_dual_account.py",
        ROOT / "tam_research" / "modal_dual_account_v2.py",
        ROOT / "scripts" / "modal_select_account.py",
        ROOT / "scripts" / "modal_select_account_v2.py",
        ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v1.py",
        ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v2.py",
    ):
        assert path.is_file()
