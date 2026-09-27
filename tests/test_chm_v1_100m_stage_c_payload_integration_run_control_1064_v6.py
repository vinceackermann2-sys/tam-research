from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER_V4 = ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v4.py"
RUNNER_V5 = ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v6.py"
WORKFLOW_V5 = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-v5.yml"
AUDIT_V5 = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v5.yml"
MIRROR_V5 = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5.yml"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            assert node.end_lineno is not None
            return "\n".join(lines[start:node.end_lineno])
    raise AssertionError(f"{name!r} not found in {path}")


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


def test_v6_runner_uses_verified_v5_mirror_and_fresh_diagnostic_namespace() -> None:
    source = RUNNER_V6.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'PHASE = "chm-v1-100m-stage-c-payload-integration-1064-v6"' in source
    assert 'TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-payload-integration-1064-v6]"' in source
    assert 'SOURCE_RESULT_ROOT = "/vol/chm-v1/source-mirror/issue-1064/v5"' in source
    assert 'SOURCE_MIRROR_MANIFEST_PATH = f"{SOURCE_RESULT_ROOT}/MIRROR_MANIFEST.json"' in source
    assert 'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-integration/issue-1064/v6"' in source
    assert "SOURCE_CHECKPOINT_METADATA_PATH" not in source
    assert "SOURCE_POSTMORTEM_RESULT_PATH" not in source
    assert "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v8" not in source
    assert "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831" in source
    assert "CHECKPOINT_BYTES = 407_424_818" in source


def test_v6_runner_verifies_v5_mirror_manifest_and_actual_checkpoint_hash() -> None:
    inspect = _function_source(RUNNER_V6, "_inspect_source")
    assert '"MIRROR_COMPLETE"' in inspect
    assert '"CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_SOURCE_MIRROR_V5_PASS"' in inspect
    assert '"repair_issue": 1085' in inspect
    assert '"selected_modal_account": "secondary"' in inspect
    assert '"selected_credential_alias": "account2"' in inspect
    assert '"source_and_mirror_byte_identical": True' in inspect
    assert "_sha256_file(checkpoint)" in inspect
    assert 'if actual_sha != CHECKPOINT_SHA256' in inspect
    assert '"mirror_manifest_verified": True' in inspect
    assert '"first_postmortem_payload_integration_weak_verified_by_authority_audit": True' in inspect


def test_v6_scoring_function_is_identical_to_v5() -> None:
    assert _function_source(RUNNER_V6, "run_diagnostic") == _function_source(RUNNER_V5, "run_diagnostic")


def test_verified_v5_mirror_workflow_remains_storage_only_and_immutable() -> None:
    source = MIRROR_V5.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5]" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "modal run " not in source
    assert "modal shell " not in source
    assert "modal deploy " not in source
    assert "modal volume get" in source
    assert "modal volume put" in source
    assert "--force" not in source
    assert "selected_credential_alias" in source and '"account2"' in source
    assert "MODAL_TOKEN_ID_2" in source and "MODAL_TOKEN_SECRET_2" in source
    assert "407424818" in source
    assert "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831" in source
    assert "SOURCE_MIRROR_V5_PASS" in source
    assert "diagnostic result namespace creation: NONE" in source
    assert "final #1064 diagnostic trigger authority remains withheld" in source

    upload_result = source.index(
        "modal volume put tam-research-data mirror-src/RESULT.json "
        "/chm-v1/source-mirror/issue-1064/v5/RESULT.json"
    )
    upload_checkpoint = source.index(
        "modal volume put tam-research-data mirror-src/step-2048.pt "
        "/chm-v1/source-mirror/issue-1064/v5/checkpoints/eiem/step-2048.pt"
    )
    redownload = source.index("Re-download secondary mirror and verify exact parity")
    build_manifest = source.index("Build immutable mirror manifest after parity succeeds")
    upload_manifest = source.index(
        "modal volume put tam-research-data MIRROR_MANIFEST.json "
        "/chm-v1/source-mirror/issue-1064/v5/MIRROR_MANIFEST.json"
    )
    assert upload_result < upload_checkpoint < redownload < build_manifest < upload_manifest


def test_v6_launch_workflow_binds_verified_v5_mirror_and_secondary_only() -> None:
    source = WORKFLOW_V6.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-v6]" in source
    assert '"phase":"chm-v1-100m-stage-c-payload-integration-1064-v6"' in source
    assert '"/vol/chm-v1/100m-stage-c-payload-integration/issue-1064/v6"' in source
    assert "modal_chm_v1_100m_stage_c_payload_integration_1064_v6.py" in source
    assert "modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5.yml" in source
    assert "mirror_workflow_sha" in source
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_FINAL_LAUNCHER_AUTHORITY_V6" in source
    assert 'assert account=="secondary"' in source
    assert "source_mirror_verified=true" in source
    assert "first_postmortem_payload_integration_weak_verified=true" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v8" not in source


def test_v6_audit_retires_incomplete_v5_audit_and_binds_mirror_provenance() -> None:
    source = AUDIT_V6.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v6]" in source
    assert '"supersedes_incomplete_audit_issue":1092' in source
    assert '"supersedes_incomplete_audit_run":36331605001' in source
    assert "modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5" in source
    assert "repos/{repo}/issues/1085" in source
    assert "SOURCE_MIRROR_V5_PASS" in source
    assert "repos/{repo}/issues/1008/comments?per_page=100" in source
    assert 'tags=["PAYLOAD_INTEGRATION_WEAK_SIGNAL"]' in source
    assert "source_mirror_verified=true" in source
    assert "source_mirror_repair_issue=1085" in source
    assert 'assert account=="secondary"' in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run " not in source
    assert "--phase state" not in source
    assert "gpu_allocated=false" in source
    assert "diagnostic_attempt_consumed=false" in source
    assert "trigger_authorized=false" in source


def test_v6_workflow_embedded_python_is_syntax_valid() -> None:
    for path in (MIRROR_V5, WORKFLOW_V6, AUDIT_V6):
        source = path.read_text(encoding="utf-8")
        blocks = _heredocs(source)
        assert blocks, path
        for idx, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(f"{path.name} embedded Python block {idx}: {exc}") from exc


def test_v6_files_preserve_no_training_no_new_seed_no_stage_d() -> None:
    combined = "\n".join(
        p.read_text(encoding="utf-8") for p in (RUNNER_V6, MIRROR_V5, WORKFLOW_V6, AUDIT_V6)
    )
    for forbidden in ("torch.optim", ".backward(", "torch.save(", "optimizer.step"):
        assert forbidden not in combined
    assert "new_scientific_seed_authorized=false" in combined
    assert "stage_d_authorized=false" in combined


def _duplicate_direct_env_keys(source: str) -> list[tuple[int, str, int, int]]:
    lines = source.splitlines()
    duplicates: list[tuple[int, str, int, int]] = []
    for index, line in enumerate(lines):
        if line.strip() != "env:":
            continue
        base = len(line) - len(line.lstrip())
        seen: dict[str, int] = {}
        for child_index in range(index + 1, len(lines)):
            child = lines[child_index]
            if not child.strip():
                continue
            indent = len(child) - len(child.lstrip())
            if indent <= base:
                break
            if indent != base + 2 or ":" not in child.strip():
                continue
            key = child.strip().split(":", 1)[0]
            if key in seen:
                duplicates.append((index + 1, key, seen[key], child_index + 1))
            else:
                seen[key] = child_index + 1
    return duplicates


def test_v6_workflow_registration_regressions_are_blocked() -> None:
    for path in (MIRROR_V5, WORKFLOW_V6, AUDIT_V6):
        source = path.read_text(encoding="utf-8")
        assert _duplicate_direct_env_keys(source) == [], path
        assert not any(
            line.startswith("- ") for line in source.splitlines()
        ), f"{path.name} contains a zero-indented sequence item outside a run block"

    mirror = MIRROR_V5.read_text(encoding="utf-8")
    assert "cat > mirror-comment.md <<'EOF'" in mirror
    assert 'gh issue comment 1085 --repo "$GITHUB_REPOSITORY" -F mirror-comment.md' in mirror
    assert '\n- source Stage-C RESULT + EIEM step-2048 checkpoint' not in mirror


def test_v6_audit_durable_record_exports_every_referenced_uppercase_variable() -> None:
    source = AUDIT_V6.read_text(encoding="utf-8")
    marker = "- name: Record immutable pre-authority audit"
    assert marker in source
    step = source.split(marker, 1)[1]
    # Stop at the next YAML step if one exists.
    if "\n      - name:" in step:
        step = step.split("\n      - name:", 1)[0]

    assert "MIRROR_WORKFLOW_SHA: ${{ steps.cfg.outputs.mirror_workflow_sha }}" in step
    assert "mirror_workflow_blob=$MIRROR_WORKFLOW_SHA" in step
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_AUTHORITY_AUDIT_V6_PASS" in step

    env_part, run_part = step.split("\n        run: |", 1)
    env_keys = set()
    for line in env_part.splitlines():
        stripped = line.strip()
        if ":" in stripped and stripped.split(":", 1)[0].replace("_", "").isalnum():
            key = stripped.split(":", 1)[0]
            if key.isupper():
                env_keys.add(key)

    import re
    referenced = set(re.findall(r"\$([A-Z][A-Z0-9_]*)", run_part))
    # GITHUB_REPOSITORY is injected by Actions globally; every other uppercase
    # shell variable in this immutable-record step must be explicitly bound.
    assert referenced - {"GITHUB_REPOSITORY"} <= env_keys
    assert "MIRROR_WORKFLOW_SHA" in env_keys


def test_v6_fresh_identities_retire_v5_diagnostic_execution_but_reuse_verified_v5_source_mirror_only() -> None:
    runner = RUNNER_V6.read_text(encoding="utf-8")
    workflow = WORKFLOW_V6.read_text(encoding="utf-8")
    audit = AUDIT_V6.read_text(encoding="utf-8")
    combined = "\n".join((runner, workflow, audit))
    assert "/vol/chm-v1/100m-stage-c-payload-integration/issue-1064/v5" not in combined
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-v5]" not in combined
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v5]" not in combined
    assert "/vol/chm-v1/source-mirror/issue-1064/v5" in runner
    assert "modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5.yml" in workflow
    assert "modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5.yml" in audit
