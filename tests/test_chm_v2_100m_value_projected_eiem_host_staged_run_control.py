from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_value_projected_eiem_host_staged_1137_v1.py"
CORE = ROOT / "tam_research" / "chm_v2_100m_value_projected_eiem_host_staged_rerun.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v2-100m-value-projected-eiem-host-staged-1137-v1.yml"
)
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v2-100m-value-projected-eiem-host-staged-1137-authority-audit-v1.yml"
)

TRIGGER = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1137-seed-2011371-v1]"
AUDIT_TRIGGER = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1137-authority-audit-v1]"
RESULT_ROOT = "/vol/chm-v2/100m-value-projected-eiem-host-staged/issue-1137/seed-2011371-v1"


def _python_heredocs(source: str) -> list[str]:
    lines = source.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if "python - <<'PY'" not in lines[index]:
            index += 1
            continue
        index += 1
        block: list[str] = []
        while index < len(lines) and lines[index].strip() != "PY":
            block.append(lines[index])
            index += 1
        assert index < len(lines), "unterminated Python heredoc"
        blocks.append(textwrap.dedent("\n".join(block)))
        index += 1
    return blocks


def test_frozen_runner_and_core_are_present_and_syntax_valid() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    ast.parse(runner)
    ast.parse(core)
    assert "SCIENTIFIC_SEED = 2_011_371" in runner
    assert RESULT_ROOT in runner
    assert TRIGGER in runner
    assert AUDIT_TRIGGER in runner
    assert "_device_tokens" not in runner
    assert "host_staged_gather" in runner


def test_scientific_workflow_uses_only_fresh_host_staged_identity() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert TRIGGER in source
    assert RESULT_ROOT in source
    assert '"control_issue":1139' in source
    assert '"prereg_issue":1137' in source
    assert '"scientific_seed":2011371' in source
    assert "scientific_seed=2011371" in source
    assert "seed_2011371_consumed=false" in source
    assert "CHM_V2_100M_HOST_STAGED_RERUN_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "github.run_attempt" in source or "GITHUB_RUN_ATTEMPT" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "modal billing rates --json" in source
    assert "worst=hourly*4.0" in source.replace(" ", "")
    assert "worst<=6.00" in source.replace(" ", "")
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "automatic_retry_authorized=false" in source
    assert "checkpoint_resume_authorized=false" in source
    assert "account_switch_after_reservation_authorized=false" in source
    assert "stage_d_authorized=false" in source
    assert "scale_up_authorized=false" in source
    assert "multi_seed_replication_authorized=false" in source


def test_scientific_workflow_binds_exact_fresh_paths_and_blobs() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "tam_research/chm_v2_100m_value_projected_eiem_host_staged_rerun.py:$CORE_SHA" in source
    assert "modal_chm_v2_100m_value_projected_eiem_host_staged_1137_v1.py:$RUNNER_SHA" in source
    assert (
        ".github/workflows/modal-chm-v2-100m-value-projected-eiem-host-staged-1137-v1.yml:$WORKFLOW_SHA"
        in source
    )
    assert (
        ".github/workflows/modal-chm-v2-100m-value-projected-eiem-host-staged-1137-authority-audit-v1.yml:$AUDIT_WORKFLOW_SHA"
        in source
    )
    assert "3d8da59bc09552954d6c15270266b7cc4fd0738e" in source
    assert "f8be5c97eb7da4538565e3e08bd6591907a92505" in source


def test_audit_workflow_is_fresh_cpu_only_preauthority_inspection() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert AUDIT_TRIGGER in source
    assert TRIGGER in source
    assert '"control_issue":1139' in source
    assert '"prereg_issue":1137' in source
    assert '"scientific_seed":2011371' in source
    assert RESULT_ROOT in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "--phase state" not in source
    assert "CHM_V2_100M_HOST_STAGED_RERUN_AUTHORITY_AUDIT_V1_PASS" in source
    assert 'control=gh(f"repos/{repo}/issues/1139")' in source
    assert "len(authority_comments)==0" in source
    assert 'state["scientific_seed"]==2011371' in source
    assert 'state["seed_2011371_consumed"] is False' in source
    assert 'state["gpu_allocated"] is False' in source
    assert 'state["writes_performed"] is False' in source
    assert 'state["trigger_authorized"] is False' in source


def test_retired_1115_scientific_identity_is_absent_from_new_workflows() -> None:
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    for source in (scientific, audit):
        forbidden = (
            "2011121",
            "issue-1115",
            "1115-seed",
            "modal_chm_v2_100m_value_projected_eiem_1115_v1.py",
            "tam_research/chm_v2_100m_value_projected_eiem_run_control.py",
            "modal-chm-v2-100m-value-projected-eiem-1115-authority-audit-v3.yml",
        )
        for token in forbidden:
            assert token not in source


def test_workflows_do_not_reintroduce_known_yaml_registration_failures() -> None:
    for path in (WORKFLOW, AUDIT_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        assert "&bindings" not in source
        assert "*bindings" not in source
        assert "run: |\\n" not in source
        # GitHub comment commands containing # must live under a block scalar,
        # never on a plain YAML run: scalar where # becomes a YAML comment.
        for line in source.splitlines():
            stripped = line.strip()
            assert not (
                stripped.startswith("run: gh issue comment")
                and "#" in stripped
            )


def test_all_embedded_workflow_python_is_syntax_valid() -> None:
    for path in (WORKFLOW, AUDIT_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        blocks = _python_heredocs(source)
        assert blocks, f"no embedded Python in {path.name}"
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} Python heredoc {index} invalid: {exc}"
                ) from exc


def test_audit_and_scientific_workflows_bind_each_other_exactly() -> None:
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    audit_path = (
        ".github/workflows/"
        "modal-chm-v2-100m-value-projected-eiem-host-staged-1137-authority-audit-v1.yml"
    )
    scientific_path = (
        ".github/workflows/"
        "modal-chm-v2-100m-value-projected-eiem-host-staged-1137-v1.yml"
    )
    assert audit_path in scientific
    assert scientific_path in audit
    assert "AUDIT_WORKFLOW_SHA" in scientific
    assert "AUDIT_WORKFLOW_SHA" in audit
    assert "SCIENTIFIC_WORKFLOW_SHA" in audit
