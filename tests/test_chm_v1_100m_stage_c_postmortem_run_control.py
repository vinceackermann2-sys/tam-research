from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_postmortem_1008_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-postmortem-1008-v1.yml"
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v1-100m-stage-c-postmortem-1008-authority-audit-v1.yml"
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
        assert index < len(lines), "unterminated Python heredoc"
        blocks.append(textwrap.dedent("\n".join(block)))
        index += 1
    return blocks


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            end = node.end_lineno
            assert end is not None
            return "\n".join(lines[start:end])
    raise AssertionError(f"function {name!r} not found")


def test_runner_is_syntax_valid_and_frozen_to_checkpoint_only_diagnostic() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)

    assert 'PHASE = "chm-v1-100m-stage-c-postmortem-1008-v1"' in source
    assert 'CONTROL_ISSUE = 1008' in source
    assert 'PREREG_ISSUE = 1002' in source
    assert 'SOURCE_SCIENTIFIC_ISSUE = 990' in source
    assert 'SOURCE_SCIENTIFIC_SEED = 977_001' in source
    assert 'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-postmortem/issue-1008/v1"' in source
    assert 'SOURCE_CHECKPOINT_PATH = f"{SOURCE_RESULT_ROOT}/checkpoints/eiem/step-2048.pt"' in source
    assert 'GPU_CLASS = "L4"' in source
    assert 'MAX_SECONDS = 7_200' in source
    assert 'MAX_BILLED_COMPUTE_USD = 3.00' in source
    assert 'SOFT_TEMPERATURE = 0.10' in source

    forbidden = (
        "torch.optim",
        ".backward(",
        "zero_grad(",
        "torch.save(",
        "optimizer.step",
        "train_model",
        "resume_from",
    )
    for token in forbidden:
        assert token not in source

    assert "torch.load(" in source
    assert "model.load_state_dict(" in source
    assert "model.eval()" in source
    assert "torch.inference_mode()" in source
    assert "torch.autocast" in source


def test_runner_binds_immutable_analysis_and_source_scientific_blobs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"' in source
    assert 'EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"' in source
    assert 'POSTMORTEM_BLOB = "1612dfb17a613508b0604990c94bed6fecd3776d"' in source
    assert (
        'SCIENTIFIC_IMPLEMENTATION_BLOB = '
        '"fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"'
    ) in source
    assert '"source_seed_historical_only": True' in source
    assert '"new_scientific_seed_created": False' in source
    assert '"source_scientific_seed_reused": False' in source


def test_pre_authority_inspector_is_read_only_and_does_not_create_namespace() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    inspect = _function_source(source, "inspect_source")
    assert "_inspect_source_state()" in inspect
    assert "_atomic_write(" not in inspect
    assert "volume.commit()" not in inspect
    assert ".mkdir(" not in inspect
    assert "gpu=" not in inspect
    assert '"trigger_authorized_by_runner": False' in inspect


def test_gpu_function_consumes_attempt_before_checkpoint_hash_load_or_scoring() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_diagnostic")
    consumed_write = run.index("_atomic_write(consumed_path, consumed)")
    first_hash = run.index("_sha256_file(checkpoint_path)")
    checkpoint_load = run.index("torch.load(checkpoint_path")
    scoring = run.index("diagnostic_probe_row(model, probe)")
    assert consumed_write < first_hash < checkpoint_load < scoring
    assert '"diagnostic_attempt_consumed": True' in run
    assert "retries=0" in run
    assert "gpu=GPU_CLASS" in run
    assert "timeout=MAX_SECONDS" in run


def test_runner_writes_only_dedicated_diagnostic_artifacts() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '_atomic_write(Path(SOURCE_' not in source
    assert 'SOURCE_RESULT_PATH' in source
    assert 'SOURCE_CHECKPOINT_METADATA_PATH' in source
    assert '"ZERO_GPU_GATE.json"' in source
    assert '"DISPATCH_RESERVED.json"' in source
    assert '"ATTEMPT_CONSUMED.json"' in source
    assert '"ATTEMPT_FAILURE.json"' in source
    assert '"RAW_ROWS.json"' in source
    assert '"SUMMARY.json"' in source
    assert '"RESULT.json"' in source


def test_scientific_workflow_requires_exact_issue_authority_and_one_shot_phases() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert "[modal-chm-v1-100m-stage-c-postmortem-1008-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V1_100M_STAGE_C_POSTMORTEM_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "authority_comment_id" in source
    assert "checkpoint_sha256" in source
    assert "checkpoint_bytes" in source
    assert "modal billing rates --json" in source
    assert "worst = hourly * 2.0" in source
    assert "worst <= 3.00" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "automatic_retry_authorized=false" in source
    assert "new_scientific_seed_authorized=false" in source
    assert "stage_d_authorized=false" in source
    assert "workflow_sha" in source
    assert "runner_sha" in source
    assert "863bd038e60da5511503adb0c8e1046a680ed3bd" in source
    assert "863bd038e60da551150fd23b0a3143c68e0b60333" not in source


def test_authority_audit_workflow_is_unique_cpu_read_only_inspection() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-postmortem-1008-authority-audit-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "modal billing rates --json" in source
    assert "worst = hourly * 2.0" in source
    assert "worst <= 3.00" in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "--phase state" not in source
    assert "diagnostic_result_namespace_unused" in source
    assert "gpu_allocated" in source
    assert "diagnostic_attempt_consumed" in source
    assert "trigger_authorized=false" in source
    assert "CHM_V1_100M_STAGE_C_POSTMORTEM_AUTHORITY_AUDIT_V1_PASS" in source


def test_all_embedded_workflow_python_is_syntax_valid() -> None:
    for path in (WORKFLOW, AUDIT_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        blocks = _python_heredocs(source)
        assert blocks, f"no embedded Python blocks in {path.name}"
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} embedded Python block {index} is invalid: {exc}"
                ) from exc


def test_workflows_do_not_define_a_new_scientific_seed() -> None:
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert '"seed":' not in scientific
    assert '"seed":' not in audit
    assert "source_seed_977001_consumed_historical_only=true" in audit
