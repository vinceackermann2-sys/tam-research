from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_payload_integration_1064_v1.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v1-100m-stage-c-payload-integration-1064-v1.yml"
)
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v1.yml"
)
RETIRED_V8_ROOT = "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v8"


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            assert node.end_lineno is not None
            return "\n".join(lines[start : node.end_lineno])
    raise AssertionError(name)


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


def test_runner_is_checkpoint_only_and_binds_frozen_decomposition() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)

    assert 'PHASE = "chm-v1-100m-stage-c-payload-integration-1064-v1"' in source
    assert "CONTROL_ISSUE = 1064" in source
    assert "PREREG_ISSUE = 1037" in source
    assert "SOURCE_SCIENTIFIC_ISSUE = 990" in source
    assert "SOURCE_POSTMORTEM_ISSUE = 1008" in source
    assert "SOURCE_SCIENTIFIC_SEED = 977_001" in source
    assert (
        'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-integration/issue-1064/v1"'
        in source
    )
    assert (
        'DECOMPOSITION_BLOB = "8ab98658d4c8ced33623c2d28ec899b35339997e"'
        in source
    )
    assert (
        'CHECKPOINT_SHA256 = '
        '"846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831"'
        in source
    )
    assert "CHECKPOINT_BYTES = 407_424_818" in source
    assert 'GPU_CLASS = "L4"' in source
    assert "MAX_SECONDS = 7_200" in source
    assert "MAX_BILLED_COMPUTE_USD = 3.00" in source

    for forbidden in (
        "torch.optim",
        ".backward(",
        "zero_grad(",
        "torch.save(",
        "optimizer.step",
        "resume_from",
        RETIRED_V8_ROOT,
    ):
        assert forbidden not in source


def test_source_inspector_binds_authoritative_stop_and_first_postmortem_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    inspect = _function_source(source, "_inspect_source")
    assert "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH" in inspect
    assert "CHM_V1_100M_STAGE_C_POSTMORTEM_DIAGNOSTIC_ONLY" in inspect
    assert '["PAYLOAD_INTEGRATION_WEAK_SIGNAL"]' in inspect
    assert "SOURCE_POSTMORTEM_RESULT_PATH" in inspect
    assert "diagnostic_result_namespace_unused" in inspect
    assert "1017" not in inspect
    assert RETIRED_V8_ROOT not in inspect


def test_gpu_attempt_is_consumed_before_checkpoint_load_and_scoring() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_diagnostic")
    consumed_write = run.index("_atomic_write(consumed_path, consumed)")
    first_hash = run.index("_sha256_file(checkpoint)")
    checkpoint_load = run.index("torch.load(checkpoint")
    scoring = run.index("payload_integration_probe_row(model, probe)")
    assert consumed_write < first_hash < checkpoint_load < scoring
    assert '"diagnostic_attempt_consumed": True' in run
    assert "retries=0" in run
    assert "gpu=GPU_CLASS" in run
    assert "timeout=MAX_SECONDS" in run


def test_result_schema_cannot_repeat_v8_benefits_key_bug() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_diagnostic")
    assert 'summary = summarize_payload_integration(rows, gate_stats)' in run
    assert "decision = classify_payload_integration(summary)" in run
    assert '"aggregate": summary["aggregate"]' in run
    assert 'decision["benefits"]' not in run
    assert 'decision["nll_benefits"]' not in run
    run_tree = ast.parse(textwrap.dedent(run))
    decision_keys = {
        node.slice.value
        for node in ast.walk(run_tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "decision"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert {
        "stage_c_result_changed",
        "first_postmortem_result_changed",
        "new_training_authorized",
        "new_scientific_seed_authorized",
        "stage_d_authorized",
    } <= decision_keys


def test_runner_writes_only_own_one_shot_state_and_no_checkpoint() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for name in (
        "ZERO_GPU_GATE.json",
        "DISPATCH_RESERVED.json",
        "ATTEMPT_CONSUMED.json",
        "RAW_ROWS.json",
        "SUMMARY.json",
        "RESULT.json",
        "ATTEMPT_FAILURE.json",
    ):
        assert name in source
    assert "torch.save(" not in source
    assert "_atomic_write(Path(SOURCE_" not in source
    assert '"source_scientific_seed_reused": False' in source
    assert '"new_scientific_seed_created": False' in source


def test_scientific_workflow_requires_exact_authority_account_and_one_shot_sequence() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert "[modal-chm-v1-100m-stage-c-payload-integration-1064-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "account_selection_evidence_sha256" in source
    assert "--force-account" in source
    assert "Verify revalidated Modal account matches frozen authority" in source
    assert "modal billing rates --json" in source
    assert "worst=hourly*2.0" in source
    assert "worst<=3.00" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "automatic_retry_authorized=false" in source
    assert "new_scientific_seed_authorized=false" in source
    assert "stage_d_authorized=false" in source
    assert RETIRED_V8_ROOT in source  # only as a fail-closed grep target
    assert "decision[\"benefits\"]" not in source


def test_audit_workflow_is_cpu_inspect_only_and_excludes_v8_partial_namespace() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert (
        "[modal-chm-v1-100m-stage-c-payload-integration-1064-authority-audit-v1]"
        in source
    )
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "--phase state" not in source
    assert "modal billing rates --json" in source
    assert "result_namespace_unused=true" in source
    assert "gpu_allocated=false" in source
    assert "diagnostic_attempt_consumed=false" in source
    assert "trigger_authorized=false" in source
    assert "retired_v8_partial_artifacts_excluded=true" in source
    assert RETIRED_V8_ROOT in source  # only as an explicit rejection target
    assert "modal_select_account.py" in source
    assert "account_selection_evidence_sha256" in source


def test_workflows_embed_valid_python_and_avoid_yaml_merge_keys() -> None:
    for path in (WORKFLOW, AUDIT_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        assert "<<:" not in source
        blocks = _python_heredocs(source)
        assert blocks
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} embedded Python block {index} invalid: {exc}"
                ) from exc


def test_no_new_scientific_seed_or_v8_result_dependency() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")

    assert '"seed":' not in scientific
    assert '"seed":' not in audit
    assert "source_seed_historical_only" in runner

    # The retired V8 path may appear only in shell grep guards, never as a
    # Python runner source/read path or an authority evidence source.
    assert RETIRED_V8_ROOT not in runner
    assert "SOURCE_PAYLOAD_GATE" not in runner
    assert "issue-1017/v8/RAW_ROWS.json" not in runner
    assert "issue-1017/v8/SUMMARY.json" not in runner
