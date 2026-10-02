from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-v1.yml"
)
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v2.yml"
)


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
        assert index < len(lines), "unterminated embedded Python heredoc"
        blocks.append(textwrap.dedent("\n".join(block)))
        index += 1
    return blocks


def test_scientific_workflow_is_fresh_1143_seed_and_1187_authority_only() -> None:
    source = SCIENTIFIC_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch" not in source
    assert "issues:" in source
    assert "types: [opened]" in source
    assert "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1]" in source
    assert '"phase":"chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1"' in source
    assert '"control_issue":1187' in source
    assert '"repair_issue":1143' in source
    assert '"prereg_issue":1137' in source
    assert '"scientific_seed":2011431' in source
    assert (
        '"/vol/chm-v2/100m-value-projected-eiem-host-staged/'
        'issue-1143/seed-2011431-v1"'
    ) in source

    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V2_100M_HOST_STAGED_1143_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "issues/1187/comments" in source
    assert "gh issue comment 1187" in source
    assert "gh issue close 1187" in source

    assert "modal_chm_v2_100m_value_projected_eiem_host_staged_1143_v1.py" in source
    assert "tam_research/chm_v2_100m_value_projected_eiem_host_staged_1143.py" in source
    assert "c6c2ec21126f6c8492475a43e39ef87f1279d0b9" in source
    assert "5aafacab5804f364701f5ef6d9ee9200db493458" in source
    assert "modal billing rates --json" in source
    assert "worst=hourly*4.0" in source
    assert "worst<=6.00" in source

    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "automatic_retry_authorized=false" in source
    assert "checkpoint_resume_authorized=false" in source
    assert "stage_d_authorized=false" in source
    assert "multi_seed_replication_authorized=false" in source

    for stale in (
        "2011371",
        "issue-1137/seed-2011371",
        "modal_chm_v2_100m_value_projected_eiem_host_staged_1137_v1.py",
        "tam_research/chm_v2_100m_value_projected_eiem_host_staged_rerun.py",
        "3d8da59bc09552954d6c15270266b7cc4fd0738e",
        "f8be5c97eb7da4538565e3e08bd6591907a92505",
        "issues/1139/comments",
        "gh issue comment 1139",
        "gh issue close 1139",
    ):
        assert stale not in source


def test_authority_audit_is_cpu_preallocation_only_and_bound_to_1187() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch" not in source
    assert "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v2]" in source
    assert "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1]" in source
    assert '"phase":"chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v2"' in source
    assert '"control_issue":1187' in source
    assert '"repair_issue":1143' in source
    assert '"prereg_issue":1137' in source
    assert '"scientific_seed":2011431' in source

    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "repos/{repo}/issues/1187" in source
    assert "repos/{repo}/issues/1187/comments?per_page=100" in source
    assert "CHM_V2_100M_HOST_STAGED_1143_AUTHORITY_AUDIT_V2_PASS" in source

    assert "modal_chm_v2_100m_value_projected_eiem_host_staged_1143_v1.py" in source
    assert "tam_research/chm_v2_100m_value_projected_eiem_host_staged_1143.py" in source
    assert "c6c2ec21126f6c8492475a43e39ef87f1279d0b9" in source
    assert "5aafacab5804f364701f5ef6d9ee9200db493458" in source

    assert "modal billing rates --json" in source
    assert "worst=hourly*4.0" in source
    assert "worst<=6.00" in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "--phase state" not in source
    assert "result_namespace_unused" in source
    assert "gpu_allocated" in source
    assert 'state["seed_2011431_consumed"] is False' in source
    assert "trigger_authorized=false" in source

    assert '"telemetry_repair_issue":1198' in source
    assert '"supersedes_failed_audit_issue":1195' in source
    assert '"supersedes_failed_audit_run":37031157458' in source
    assert "repos/{}/actions/runs/37031157458" in source
    assert "actions: read" in source

    for stale in (
        "2011371",
        "issue-1137/seed-2011371",
        "modal_chm_v2_100m_value_projected_eiem_host_staged_1137_v1.py",
        "tam_research/chm_v2_100m_value_projected_eiem_host_staged_rerun.py",
        "3d8da59bc09552954d6c15270266b7cc4fd0738e",
        "f8be5c97eb7da4538565e3e08bd6591907a92505",
        "repos/{repo}/issues/1139",
    ):
        assert stale not in source


def test_all_embedded_workflow_python_compiles() -> None:
    for path in (SCIENTIFIC_WORKFLOW, AUDIT_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        blocks = _python_heredocs(source)
        assert blocks, f"expected embedded Python in {path.name}"
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} embedded Python block {index} is invalid: {exc}"
                ) from exc


def test_workflows_pin_exact_repaired_science_and_governance_blobs() -> None:
    scientific = SCIENTIFIC_WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    combined = scientific + "\n" + audit

    frozen = (
        "d0c2186231cead2a7c851d776d21e6ab6f73ec43",
        "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f",
        "863bd038e60da5511503adb0c8e1046a680ed3bd",
        "26668b37a6062c641275e177b622948b36d0f227",
        "bc4ed60885aaf991a2d6b9fe8f634ff973f3f722",
        "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6",
        "440292942066d0b3d3a71d40ca8495092674ce6c",
        "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc",
        "c6c2ec21126f6c8492475a43e39ef87f1279d0b9",
        "5aafacab5804f364701f5ef6d9ee9200db493458",
    )
    for blob in frozen:
        assert blob in combined
