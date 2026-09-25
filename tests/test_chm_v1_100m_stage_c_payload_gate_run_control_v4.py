from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_payload_gate_1017_v4.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-gate-1017-v4.yml"
AUDIT = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v4.yml"


def _func(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            return "\n".join(lines[start:node.end_lineno])
    raise AssertionError(name)


def _blocks(source: str) -> list[str]:
    lines = source.splitlines()
    out: list[str] = []
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
        assert i < len(lines), "unterminated heredoc"
        out.append(textwrap.dedent("\n".join(block)))
        i += 1
    return out


def test_v4_runner_is_mechanical_v3_successor_with_account_binding_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    for needle in (
        'PHASE = "chm-v1-100m-stage-c-payload-gate-1017-v4"',
        'TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-payload-gate-1017-v4]"',
        'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v4"',
        'PAYLOAD_GATE_BLOB = "32a5ea1baca45705e75fef4c99974668b59d73a6"',
        'CHECKPOINT_SHA256 = "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831"',
        'CHECKPOINT_BYTES = 407_424_818',
        'GPU_CLASS = "L4"',
        'MAX_SECONDS = 7_200',
        'MAX_BILLED_COMPUTE_USD = 3.00',
        '"selected_modal_account"',
        '"selected_modal_workspace"',
    ):
        assert needle in source
    for forbidden in (
        "/issue-1017/v1",
        "/issue-1017/v2",
        "/issue-1017/v3",
        "torch.optim",
        ".backward(",
        "zero_grad(",
        "torch.save(",
        "optimizer.step",
        "resume_from",
    ):
        assert forbidden not in source
    assert "payload_gate_probe_row" in source
    assert "gate_state_summary" in source


def test_runner_account_binding_is_required_for_all_durable_phases() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    helper = _func(source, "_validate_account_binding")
    assert '{"primary", "secondary"}' in helper
    assert "selected Modal workspace must be non-empty" in helper

    for name in ("verify_zero_gpu", "reserve_dispatch", "run_diagnostic", "inspect_state"):
        fn = _func(source, name)
        assert "selected_account" in fn
        assert "selected_workspace_name" in fn

    evidence = _func(source, "_evidence")
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "_validate_account_binding" in evidence


def test_runner_preserves_one_shot_consumption_before_checkpoint_load() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _func(source, "run_diagnostic")
    consumed = run.index("_atomic_write(consumed_path, consumed)")
    hashed = run.index("_sha256_file(checkpoint)")
    loaded = run.index("torch.load(checkpoint")
    scored = run.index("payload_gate_probe_row(model, probe)")
    assert consumed < hashed < loaded < scored
    assert "retries=0" in run
    assert "gpu=GPU_CLASS" in run
    assert '"source_scientific_seed_reused": False' in run
    assert '"new_scientific_seed_created": False' in run


def test_v4_diagnostic_workflow_binds_selector_and_account_before_reservation() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v4]" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_FINAL_LAUNCHER_AUTHORITY_V4" in source
    assert "MODAL_TOKEN_ID_SECONDARY" in source
    assert "MODAL_TOKEN_SECRET_SECONDARY" in source
    assert "MODAL_LOW_CREDIT_ACCOUNT" in source
    assert "scripts/modal_select_account.py" in source
    assert '--force-account "$BOUND_ACCOUNT"' in source
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "selector_module_blob=" in source
    assert "selector_cli_blob=" in source
    assert "b10c4a19d3c4114d084a721b1a84235dfef45259" in source
    assert "7d9bd073bedcb8bfafcde23e6f03772dddb6d50e" in source
    assert '--selected-account "$SELECTED_MODAL_ACCOUNT"' in source
    assert '--selected-workspace-name "$SELECTED_MODAL_WORKSPACE"' in source

    selection = source.index("Revalidate frozen Modal account before preallocation")
    preflight = source.index("Zero-GPU preflight")
    reserve = source.index("Reserve one-shot dispatch")
    launch = source.index("Launch payload/gate diagnostic exactly once")
    assert selection < preflight < reserve < launch
    # The selector must not be called again after the durable reservation boundary.
    assert "modal_select_account.py" not in source[reserve:]


def test_v4_diagnostic_workflow_never_fails_over_after_reservation() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    reserve = source.index("Reserve one-shot dispatch")
    tail = source[reserve:]
    assert "MODAL_LOW_CREDIT_ACCOUNT" not in tail
    assert "MODAL_FORCE_ACCOUNT" not in tail
    assert "MODAL_TOKEN_ID_SECONDARY" not in tail
    assert "scripts/modal_select_account.py" not in tail
    assert "SELECTED_MODAL_ACCOUNT" in tail
    assert "SELECTED_MODAL_WORKSPACE" in tail
    assert "steps.reserve.outcome == 'success'" in tail
    assert "continue-on-error: true" in tail


def test_v4_audit_selects_account_and_is_read_only() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v4]" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "MODAL_TOKEN_ID_SECONDARY" in source
    assert "MODAL_LOW_CREDIT_ACCOUNT" in source
    assert "MODAL_FORCE_ACCOUNT" in source
    assert "scripts/modal_select_account.py" in source
    assert "--required-volume tam-research-data" in source
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "modal_selection_reason=" in source
    assert "modal_selection_evidence=" in source
    assert "--phase inspect-source" in source
    for forbidden in ("--phase preflight", "--phase reserve", "--phase run", "--phase state"):
        assert forbidden not in source
    assert "PAYLOAD_GATE_AUTHORITY_AUDIT_V4_PASS" in source
    assert "result_namespace_unused=true" in source
    assert "trigger_authorized=false" in source


def test_v4_workflows_bind_current_selector_blobs() -> None:
    for path in (WORKFLOW, AUDIT):
        source = path.read_text(encoding="utf-8")
        assert "b10c4a19d3c4114d084a721b1a84235dfef45259" in source
        assert "7d9bd073bedcb8bfafcde23e6f03772dddb6d50e" in source
        assert "tam_research/modal_dual_account.py" in source
        assert "scripts/modal_select_account.py" in source


def test_v4_workflow_embedded_python_is_syntax_valid() -> None:
    for path in (WORKFLOW, AUDIT):
        blocks = _blocks(path.read_text(encoding="utf-8"))
        assert blocks
        for block in blocks:
            ast.parse(block)


def test_v4_has_fresh_identities_and_no_new_scientific_seed() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    assert "/issue-1017/v4" in runner
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-v4]" in scientific
    assert "[modal-chm-v1-100m-stage-c-payload-gate-1017-authority-audit-v4]" in audit
    for old in ("1017-v1", "1017-v2", "1017-v3"):
        assert old not in runner
        assert old not in scientific
        assert old not in audit
    assert '"seed":' not in scientific
    assert '"seed":' not in audit


def test_v4_runner_diagnostic_semantics_are_unchanged() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    # Same frozen source modules/checkpoint and same scoring calls as v3.
    assert "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH" in source
    assert '["PAYLOAD_INTEGRATION_WEAK_SIGNAL"]' in source
    assert "payload_gate_probe_row(model, probe)" in source
    assert "summarize_payload_gate_diagnostic(rows, gate_state)" in source
    assert "classify_payload_gate_signals(summary)" in source
    assert "len(probes) != 384" in source
    assert "torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16)" in source
