from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
FROZEN_IMPLEMENTATION_BLOB = "fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"
ROOT_IMPL = ROOT / "modal_chm_v1_100m_stage_c_990_v1_impl.py"
PACKAGED_IMPL = ROOT / "tam_research" / "modal_chm_v1_100m_stage_c_990_v1_impl.py"
SHIM = ROOT / "modal_chm_v1_100m_stage_c_990_v1.py"
AUDIT_V3 = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-990-authority-audit-v3.yml"


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _embedded_python_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if "python - <<'PY'" not in lines[index]:
            index += 1
            continue
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip() != "PY":
            body.append(lines[index])
            index += 1
        assert index < len(lines), "unterminated embedded Python heredoc"
        blocks.append(textwrap.dedent("\n".join(body)))
        index += 1
    return blocks


def test_packaged_implementation_is_exact_frozen_git_blob() -> None:
    root_bytes = ROOT_IMPL.read_bytes()
    packaged_bytes = PACKAGED_IMPL.read_bytes()
    assert root_bytes == packaged_bytes
    assert _git_blob_sha(root_bytes) == FROZEN_IMPLEMENTATION_BLOB
    assert _git_blob_sha(packaged_bytes) == FROZEN_IMPLEMENTATION_BLOB


def test_launcher_has_hash_checked_packaged_fallback() -> None:
    source = SHIM.read_text(encoding="utf-8")
    assert f'IMPLEMENTATION_BLOB_SHA = "{FROZEN_IMPLEMENTATION_BLOB}"' in source
    assert 'PACKAGED_IMPLEMENTATION_MODULE = "tam_research"' in source
    assert "def _resolve_implementation_path()" in source
    assert "if sibling.is_file():" in source
    assert "importlib.import_module(PACKAGED_IMPLEMENTATION_MODULE)" in source
    assert "if packaged.is_file():" in source
    assert "_actual_impl_sha = _git_blob_sha(_impl_bytes)" in source
    assert "if _actual_impl_sha != IMPLEMENTATION_BLOB_SHA:" in source
    assert "exec(compile(_impl_bytes" in source


def test_v3_audit_is_fresh_read_only_and_binds_both_copies() -> None:
    text = AUDIT_V3.read_text(encoding="utf-8")
    assert "CHM-v1 100M Stage-C pre-authority audit v3" in text
    assert "[modal-chm-v1-100m-stage-c-990-authority-audit-v3]" in text
    assert '"repair_issue": 996' in text
    assert '"supersedes_failed_audit_issue": 995' in text
    assert '"supersedes_failed_audit_run": 35113534368' in text
    assert "35113534368" in text
    assert 'v2["conclusion"] == "cancelled"' in text
    assert "PACKAGED_IMPLEMENTATION_SHA" in text
    assert "tam_research/modal_chm_v1_100m_stage_c_990_v1_impl.py" in text
    assert "--phase inspect" in text
    assert "--phase preflight" not in text
    assert "--phase reserve" not in text
    assert "--phase run" not in text
    assert "result_namespace_unused=true" in text
    assert "scientific_seed_consumed=false" in text
    assert "gpu_allocated=false" in text
    assert "trigger_authorized=false" in text


def test_v3_embedded_python_is_syntax_valid() -> None:
    blocks = _embedded_python_blocks(AUDIT_V3.read_text(encoding="utf-8"))
    assert len(blocks) >= 4
    for index, block in enumerate(blocks):
        compile(block, f"audit-v3-embedded-{index}.py", "exec")
