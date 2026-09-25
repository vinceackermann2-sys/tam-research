# Fresh-head CI successor after hosted-runner TLS checkout failure 35915811863; no protocol change.
from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_payload_gate_1017_v6.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-gate-1017-v6.yml"
AUDIT = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v6.yml"


def _func(source: str, name: str) -> str:
    tree=ast.parse(source); lines=source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name==name:
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            return "\n".join(lines[start:node.end_lineno])
    raise AssertionError(name)


def _blocks(source: str) -> list[str]:
    lines=source.splitlines(); out=[]; i=0
    while i<len(lines):
        if "python - <<'PY'" not in lines[i]:
            i+=1; continue
        i+=1; b=[]
        while i<len(lines) and lines[i].strip()!="PY":
            b.append(lines[i]); i+=1
        assert i<len(lines)
        out.append(textwrap.dedent("\n".join(b))); i+=1
    return out


def test_runner_frozen_checkpoint_only_contract() -> None:
    s=RUNNER.read_text()
    ast.parse(s)
    for needle in (
        'CONTROL_ISSUE = 1017',
        'PREREG_ISSUE = 1014',
        'SOURCE_POSTMORTEM_ISSUE = 1008',
        'SOURCE_SCIENTIFIC_SEED = 977_001',
        'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v6"',
        'PAYLOAD_GATE_BLOB = "32a5ea1baca45705e75fef4c99974668b59d73a6"',
        'CHECKPOINT_SHA256 = "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831"',
        'CHECKPOINT_BYTES = 407_424_818',
        'GPU_CLASS = "L4"',
        'MAX_SECONDS = 7_200',
        'MAX_BILLED_COMPUTE_USD = 3.00',
    ):
        assert needle in s
    for forbidden in ("torch.optim", ".backward(", "zero_grad(", "torch.save(", "optimizer.step", "resume_from"):
        assert forbidden not in s
    assert "torch.load(" in s
    assert "payload_gate_probe_row" in s
    assert "gate_state_summary" in s


def test_runner_verifies_upstream_stop_and_postmortem_tag() -> None:
    s=RUNNER.read_text()
    inspect=_func(s,"_inspect_source")
    assert "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH" in inspect
    assert "CHM_V1_100M_STAGE_C_POSTMORTEM_DIAGNOSTIC_ONLY" in inspect
    assert '["PAYLOAD_INTEGRATION_WEAK_SIGNAL"]' in inspect
    assert "source_checkpoint_unchanged" in inspect


def test_inspect_source_is_read_only() -> None:
    s=RUNNER.read_text()
    inspect=_func(s,"inspect_source")
    assert "_atomic_write(" not in inspect
    assert "volume.commit()" not in inspect
    assert ".mkdir(" not in inspect
    assert "gpu=" not in inspect


def test_gpu_attempt_is_consumed_before_checkpoint_hash_load_or_scoring() -> None:
    s=RUNNER.read_text()
    run=_func(s,"run_diagnostic")
    consumed=run.index("_atomic_write(consumed_path, consumed)")
    hashed=run.index("_sha256_file(checkpoint)")
    loaded=run.index("torch.load(checkpoint")
    scored=run.index("payload_gate_probe_row(model, probe)")
    assert consumed < hashed < loaded < scored
    assert "retries=0" in run
    assert "gpu=GPU_CLASS" in run
    assert '"source_scientific_seed_reused": False' in run
    assert '"new_scientific_seed_created": False' in run


def test_scientific_workflow_is_issue_only_one_shot_and_authority_bound() -> None:
    s=WORKFLOW.read_text()
    assert "workflow_dispatch" not in s
    assert "&bindings" not in s and "*bindings" not in s
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v6]" in s
    assert 'test "$RUN_ATTEMPT" = "1"' in s
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_FINAL_LAUNCHER_AUTHORITY_V6" in s
    assert "checkpoint_sha256=846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831" in s
    assert "checkpoint_bytes=407424818" in s
    assert "modal billing rates --json" in s
    assert "--phase preflight" in s
    assert "--phase reserve" in s
    assert "--phase run" in s
    assert "--phase state" in s
    assert "steps.reserve.outcome == 'success'" in s
    assert "continue-on-error: true" in s
    assert "new_scientific_seed_authorized=false" in s
    assert "stage_d_authorized=false" in s


def test_authority_audit_is_cpu_inspect_only() -> None:
    s=AUDIT.read_text()
    assert "workflow_dispatch" not in s
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v6]" in s
    assert 'test "$RUN_ATTEMPT" = "1"' in s
    assert "--phase inspect-source" in s
    assert "--phase preflight" not in s
    assert "--phase reserve" not in s
    assert "--phase run" not in s
    assert "--phase state" not in s
    assert "PAYLOAD_GATE_AUTHORITY_AUDIT_V6_PASS" in s
    assert "result_namespace_unused=true" in s
    assert "trigger_authorized=false" in s


def test_embedded_python_syntax() -> None:
    for path in (WORKFLOW,AUDIT):
        blocks=_blocks(path.read_text())
        assert blocks
        for block in blocks:
            ast.parse(block)


def test_no_new_scientific_seed_in_workflows() -> None:
    s=WORKFLOW.read_text()+AUDIT.read_text()
    assert '"seed":' not in s


def test_v6_comment_recording_never_uses_plain_scalar_with_hash() -> None:
    for path in (WORKFLOW, AUDIT):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            assert not (
                stripped.startswith("run: gh issue comment")
                and "#" in stripped
            ), f"unsafe YAML plain scalar comment command: {line!r}"


def test_v6_has_fresh_namespace_and_never_reuses_v1_trigger() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    assert "/issue-1017/v6" in runner
    assert "/issue-1017/v1" not in runner
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v6]" in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v1]" not in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v6]" in audit


def test_v6_workflows_have_real_block_scalars_and_retire_v1_v2_identities() -> None:
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    for source in (scientific, audit):
        assert "run: |\\n" not in source
        assert "\\n          gh issue comment" not in source
    assert "/issue-1017/v6" in runner
    assert "/issue-1017/v1" not in runner
    assert "/issue-1017/v2" not in runner
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v6]" in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v1]" not in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v2]" not in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v6]" in audit
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v2]" not in audit


def test_v6_runner_binds_dual_account_identity_into_durable_evidence() -> None:
    s = RUNNER.read_text(encoding="utf-8")
    evidence = _func(s, "_evidence")
    assert "selected_modal_account" in evidence
    assert "selected_modal_workspace" in evidence
    assert "account_selection_evidence_sha256" in evidence
    assert '"modal_account_switch_after_reservation_authorized": False' in s
    assert 'DUAL_ACCOUNT_BLOB = "b10c4a19d3c4114d084a721b1a84235dfef45259"' in s
    assert 'DUAL_ACCOUNT_CLI_BLOB = "91053740508c6b7b78374fc7aa4fb76f689b23e9"' in s


def test_v6_gpu_entry_requires_same_reservation_account_before_consumption() -> None:
    s = RUNNER.read_text(encoding="utf-8")
    run = _func(s, "run_diagnostic")
    evidence = run.index("evidence = _evidence(")
    reservation = run.index('_read_json(reserve, "dispatch reservation")')
    drift = run.index("durable reservation binding drift")
    consumed = run.index("_atomic_write(consumed_path, consumed)")
    hashed = run.index("_sha256_file(checkpoint)")
    assert evidence < reservation < drift < consumed < hashed


def test_v6_scientific_workflow_revalidates_only_authority_bound_account() -> None:
    s = WORKFLOW.read_text(encoding="utf-8")
    assert "MODAL_TOKEN_ID_SECONDARY" in s
    assert "MODAL_TOKEN_SECRET_SECONDARY" in s
    assert "MODAL_LOW_CREDIT_ACCOUNT" in s
    assert "MODAL_FORCE_ACCOUNT" not in s
    assert 'modal_select_account.py --required-volume tam-research-data --force-account "$BOUND_MODAL_ACCOUNT"' in s
    assert 'test "${{ steps.account.outputs.selected_account }}" = "$BOUND_MODAL_ACCOUNT"' in s
    assert 'test "${{ steps.account.outputs.selected_workspace_name }}" = "$BOUND_MODAL_WORKSPACE"' in s
    assert "--dual-account-sha" in s
    assert "--dual-account-cli-sha" in s
    assert "--selected-modal-account" in s
    assert "--selected-modal-workspace" in s
    assert "--account-selection-evidence-sha256" in s
    assert "selected_modal_account=" in s
    assert "selected_modal_workspace=" in s
    assert "account_selection_evidence_sha256=" in s


def test_v6_audit_selects_account_once_and_is_still_inspect_only() -> None:
    s = AUDIT.read_text(encoding="utf-8")
    assert "MODAL_TOKEN_ID_SECONDARY" in s
    assert "MODAL_TOKEN_SECRET_SECONDARY" in s
    assert "MODAL_FORCE_ACCOUNT" in s
    assert "MODAL_LOW_CREDIT_ACCOUNT" in s
    assert "modal_select_account.py --required-volume tam-research-data" in s
    assert "Freeze selected account evidence" in s
    assert "account_selection_evidence_sha256" in s
    assert "selected_modal_account=$SELECTED_MODAL_ACCOUNT" in s
    assert "selected_modal_workspace=$SELECTED_MODAL_WORKSPACE" in s
    assert "account_selection_evidence_json=$ACCOUNT_SELECTION_EVIDENCE_JSON" in s
    assert "account_switch_after_reservation_authorized=false" in s
    assert "--phase inspect-source" in s
    assert "--phase preflight" not in s
    assert "--phase reserve" not in s
    assert "--phase run" not in s
    assert "--phase state" not in s


def test_v6_workflows_bind_exact_dual_account_blobs() -> None:
    for path in (WORKFLOW, AUDIT):
        s = path.read_text(encoding="utf-8")
        assert "b10c4a19d3c4114d084a721b1a84235dfef45259" in s
        assert "91053740508c6b7b78374fc7aa4fb76f689b23e9" in s
        assert "tam_research/modal_dual_account.py" in s
        assert "scripts/modal_select_account.py" in s


def test_v6_workflows_never_reference_retired_v3_runner() -> None:
    for path in (WORKFLOW, AUDIT):
        source = path.read_text()
        assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v3.py" not in source
        assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v6.py" in source


def test_v6_selector_invocations_make_repo_root_importable() -> None:
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    qualified = 'PYTHONPATH="$PWD" python scripts/modal_select_account.py'
    assert qualified in scientific
    assert qualified in audit
    for source in (scientific, audit):
        for line in source.splitlines():
            if "python scripts/modal_select_account.py" in line:
                assert 'PYTHONPATH="$PWD"' in line, line


def test_v6_never_reuses_retired_v4_execution_identity() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v4.py" not in scientific
    assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v4.py" not in audit
    assert "/issue-1017/v4" not in runner
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v4]" not in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v4]" not in audit
    assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v6.py" in scientific
    assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v6.py" in audit


def test_v6_preserves_aligned_v4_probe_semantics() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert "generate_aligned_probe_suite" in runner
    assert "LONG_RANGE_FAMILIES" in runner
    assert "len(probes) != 384" in runner


def test_v6_never_reuses_retired_v5_execution_identity() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    assert "/issue-1017/v5" not in runner
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v5]" not in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v5]" not in audit
    assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v5.py" not in scientific
    assert "modal_chm_v1_100m_stage_c_payload_gate_1017_v5.py" not in audit
    assert "/issue-1017/v6" in runner
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v6]" in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v6]" in audit
