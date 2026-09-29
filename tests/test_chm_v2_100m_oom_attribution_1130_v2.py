from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_oom_attribution_1127_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v2-100m-oom-attribution-1127-v2.yml"


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
        assert i < len(lines)
        blocks.append(textwrap.dedent("\n".join(block)))
        i += 1
    return blocks


def test_v2_preserves_exact_read_only_runner_boundary() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    ast.parse(runner)
    assert "import torch" not in runner
    assert "torch.load" not in runner
    assert "gpu=" not in runner
    assert "volume.commit" not in runner
    assert 'writes_performed": False' in runner
    assert 'checkpoint_loaded": False' in runner
    assert 'gpu_allocated": False' in runner


def test_v2_uses_fresh_identity_and_retires_v1_attempt() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "[modal-chm-v2-100m-oom-attribution-1127-v2]" in source
    assert '"phase": "chm-v2-100m-oom-attribution-1127-v2"' in source
    assert '"repair_issue": 1130' in source
    assert '"retired_inspector_issue": 1129' in source
    assert '"retired_inspector_run": 36555144448' in source
    assert "repos/" in source and "/actions/runs/36555144448" in source
    assert 'assert v1["run_attempt"] == 1' in source
    assert 'assert v1["conclusion"] == "failure"' in source
    assert "workflow_dispatch" not in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source


def test_v2_prefers_account2_and_never_activates_primary_credentials() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    account2 = source.index('if [ -n "$MODAL_TOKEN_ID_2" ]')
    secondary = source.index('elif [ -n "$MODAL_TOKEN_ID_SECONDARY" ]')
    legacy = source.index('elif [ -n "$MODAL_TOKEN_ID_B" ]')
    assert account2 < secondary < legacy

    assert "MODAL_TOKEN_ID_2" in source
    assert "MODAL_TOKEN_SECRET_2" in source
    assert "MODAL_TOKEN_ID_SECONDARY" in source
    assert "MODAL_TOKEN_SECRET_SECONDARY" in source
    assert "MODAL_TOKEN_ID_B" in source
    assert "MODAL_TOKEN_SECRET_B" in source

    # Generic primary credentials must not be sourced from GitHub secrets in v2.
    assert "secrets.MODAL_TOKEN_ID }}" not in source
    assert "secrets.MODAL_TOKEN_SECRET }}" not in source
    assert "credential_alias=account2" in source
    assert "credential_alias=secondary" in source
    assert "credential_alias=legacy_b" in source
    assert "modal token info" in source


def test_v2_remains_cpu_only_read_only() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "--gpu" not in source
    assert "gpu:" not in source
    assert "torch" not in source.lower()
    assert "modal run modal_chm_v2_100m_oom_attribution_1127_v1.py" in source
    assert "volume.commit" not in source
    assert "torch.load" not in source
    assert "scientific_interpretation=false" in source
    assert "retry_authorized=false" in source
    assert "stage_d_authorized=false" in source


def test_v2_binds_runner_and_own_workflow_across_live_main_drift() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "git -C source merge-base --is-ancestor" in source
    assert "origin/main" in source
    assert 'test "$RUNNER_SHA" = "cebb30730f28ff823925245bfd31f81804fd6abe"' in source
    assert "modal_chm_v2_100m_oom_attribution_1127_v1.py" in source
    assert "modal-chm-v2-100m-oom-attribution-1127-v2.yml" in source


def test_v2_embedded_python_is_syntax_valid() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    blocks = _heredocs(source)
    assert blocks
    for index, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            raise AssertionError(f"embedded Python block {index} invalid: {exc}") from exc
