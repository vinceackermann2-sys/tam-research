from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_value_projected_eiem_host_staged_1143_v1.py"
CORE = ROOT / "tam_research" / "chm_v2_100m_value_projected_eiem_host_staged_1143.py"
SCIENTIFIC_WORKFLOW = (
    ROOT / ".github" / "workflows" / "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-v1.yml"
)
AUDIT_V1 = (
    ROOT / ".github" / "workflows" / "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v1.yml"
)
AUDIT_V2 = (
    ROOT / ".github" / "workflows" / "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v2.yml"
)

OLD_RUNNER_BLOB = "1df51fea0ea8e58f9c057ef853fda1e9dacaffd4"
NEW_RUNNER_BLOB = "ed4eb4f50067eecbac9304c6ec53f2a403e2cee3"
FAILED_AUDIT_RUN = "37031157458"
AUDIT_V1_TITLE = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v1]"
AUDIT_V2_TITLE = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v2]"
SCIENTIFIC_TRIGGER = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1]"


def _python_heredocs(source: str) -> list[str]:
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


def test_runner_repair_is_fresh_seed_provenance_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'SCIENTIFIC_SEED = 2_011_431' in source
    assert 'RESULT_ROOT = "/vol/chm-v2/100m-value-projected-eiem-host-staged/issue-1143/seed-2011431-v1"' in source
    assert f'AUDIT_TITLE = "{AUDIT_V2_TITLE}"' in source
    assert '"seed_2011431_consumed": False' in source
    assert '"seed_2011371_consumed": False' not in source
    assert "torch.optim" in source  # scientific runner itself is intentionally unchanged
    assert "def run_scientific(" in source


def test_frozen_core_remains_on_original_audit_identity() -> None:
    source = CORE.read_text(encoding="utf-8")
    assert f'AUDIT_TITLE = "{AUDIT_V1_TITLE}"' in source
    assert "SCIENTIFIC_SEED = 2_011_431" in source
    assert "CONTROL_ISSUE = 1143" in source


def test_scientific_workflow_wires_corrected_runner_and_audit_v2_only() -> None:
    source = SCIENTIFIC_WORKFLOW.read_text(encoding="utf-8")
    assert NEW_RUNNER_BLOB in source
    assert OLD_RUNNER_BLOB not in source
    assert "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v2.yml" in source
    assert "modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v1.yml" not in source
    assert SCIENTIFIC_TRIGGER in source
    assert "scientific_seed=2011431" in source
    assert "seed_2011431_consumed=false" in source
    assert "github.run_attempt" in source
    assert "workflow_dispatch" not in source


def test_audit_v2_is_cpu_read_only_and_supersedes_failed_v1_once() -> None:
    source = AUDIT_V2.read_text(encoding="utf-8")
    assert AUDIT_V2_TITLE in source
    assert "actions: read" in source
    assert FAILED_AUDIT_RUN in source
    assert 'failed["run_attempt"]==1' in source
    assert 'failed["conclusion"]=="failure"' in source
    assert 'failed["path"]==".github/workflows/modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v1.yml"' in source
    assert "--phase inspect-source" in source
    assert "--phase reserve" not in source
    assert "--phase scientific" not in source
    assert "--phase run" not in source
    assert "seed_2011431_consumed" in source
    assert "seed_2011371_consumed" not in source
    assert "result_namespace_unused" in source
    assert "gpu_allocated" in source
    assert "writes_performed" in source
    assert "trigger_authorized" in source
    assert "supersedes_failed_audit_issue=1195" in source
    assert "supersedes_failed_audit_run=37031157458" in source


def test_retired_audit_v1_is_preserved_as_failure_provenance() -> None:
    source = AUDIT_V1.read_text(encoding="utf-8")
    assert AUDIT_V1_TITLE in source
    assert "seed_2011431_consumed" in source
    assert FAILED_AUDIT_RUN not in source


def test_repaired_workflows_embedded_python_is_syntax_valid() -> None:
    for path in (SCIENTIFIC_WORKFLOW, AUDIT_V2):
        blocks = _python_heredocs(path.read_text(encoding="utf-8"))
        assert blocks
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(f"{path.name} Python heredoc {index} invalid: {exc}") from exc
