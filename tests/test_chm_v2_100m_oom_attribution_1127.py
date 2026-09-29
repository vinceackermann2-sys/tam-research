from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_oom_attribution_1127_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v2-100m-oom-attribution-1127-v1.yml"


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


def test_runner_is_cpu_only_read_only_and_never_loads_checkpoint() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)

    assert 'ISSUE = 1127' in source
    assert 'SOURCE_CONTROL_ISSUE = 1115' in source
    assert 'SOURCE_TRIGGER_ISSUE = 1126' in source
    assert 'SOURCE_SEED = 2_011_121' in source
    assert (
        'SOURCE_ROOT = "/vol/chm-v2/100m-value-projected-eiem/issue-1115/seed-2011121-v2"'
        in source
    )
    assert 'FROZEN_STEPS = (512, 1024, 1536, 2048)' in source
    assert 'KINDS = ("local", "raw_eiem", "vp_eiem")' in source

    assert "gpu=" not in source
    assert "torch" not in source.lower()
    assert "volume.commit" not in source
    assert "write_text(" not in source
    assert "write_bytes(" not in source
    assert "unlink(" not in source
    assert "rename(" not in source
    assert "replace(" not in source
    assert "mkdir(" not in source
    assert "open(" not in source

    assert "volume.reload()" in source
    assert ".stat().st_size" in source
    assert 'checkpoint_loaded": False' in source
    assert 'writes_performed": False' in source
    assert 'gpu_allocated": False' in source
    assert 'scientific_interpretation": False' in source


def test_runner_never_reads_pt_payload_contents() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "torch.load" not in source
    assert "_sha256_file" not in source
    assert "read_bytes" not in source
    # .pt path existence/size is allowed, but no content read API may be used.
    assert 'checkpoint_path.is_file()' in source
    assert 'checkpoint_path.stat().st_size' in source


def test_workflow_is_issue_triggered_unique_and_cpu_only() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source
    assert "types: [opened]" in source
    assert "[modal-chm-v2-100m-oom-attribution-1127-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source

    assert "modal run modal_chm_v2_100m_oom_attribution_1127_v1.py" in source
    assert "pip install 'modal>=1.5.4,<1.6'" in source
    assert "torch" not in source.lower()
    assert "gpu" not in source.lower() or "gpu_allocated" in source
    assert "--gpu" not in source
    assert "workflow_sha" in source
    assert "runner_sha" in source
    assert "origin/main" in source
    assert "source_seed_2011121_consumed=true" in source
    assert "scientific_interpretation=false" in source
    assert "stage_d_authorized=false" in source


def test_workflow_embedded_python_is_syntax_valid() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    blocks = _heredocs(source)
    assert blocks
    for index, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            raise AssertionError(f"embedded Python block {index} invalid: {exc}") from exc
