from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import pytest
import torch

import tam_research.chm_v2_100m_qva_stage_c_execution as execution


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_qva_stage_c_1182_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v2-100m-qva-stage-c-1182-v1.yml"
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v2-100m-qva-stage-c-1182-authority-audit-v2.yml"
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
            assert node.end_lineno is not None
            return "\n".join(lines[start : node.end_lineno])
    raise AssertionError(f"function {name!r} not found")


def test_execution_contract_is_frozen_and_never_self_authorizes() -> None:
    manifest = execution.validate_execution_contract()
    assert manifest["classification"] == "CHM_V2_100M_QVA_STAGE_C_RUN_CONTROL_TRIGGER_WITHHELD"
    assert manifest["control_issue"] == 1182
    assert manifest["preregistration_issue"] == 1176
    assert manifest["hypothesis_issue"] == 1147
    assert manifest["systems_issue"] == 1155
    assert manifest["scientific_seed_reserved"] == 2_011_761
    assert manifest["train_stream_generator_seed"] == 2_021_761
    assert manifest["result_root"] == "/vol/chm-v2/100m-qva-stage-c/issue-1182/seed-2011761-v1"
    assert manifest["trigger_title"] == "[modal-chm-v2-100m-qva-stage-c-1182-seed-2011761-v1]"
    assert manifest["audit_title"] == "[modal-chm-v2-100m-qva-stage-c-1182-authority-audit-v2]"
    assert manifest["training_tokens_per_model"] == 33_554_432
    assert manifest["optimizer_steps_per_model"] == 2_048
    assert manifest["warmup_steps"] == 40
    assert manifest["tokens_per_optimizer_step"] == 16_384
    assert manifest["gpu_class"] == "L4"
    assert manifest["cpu_cores"] == 4
    assert manifest["ram_gib"] == 16
    assert manifest["max_gpu_seconds"] == 9_600
    assert manifest["max_billed_compute_usd"] == 3.0
    assert manifest["trigger_authorized_by_module"] is False
    assert manifest["gpu_allocation_authorized_by_module"] is False
    assert manifest["scientific_seed_consumed_by_module"] is False
    assert manifest["replication_authorized_by_module"] is False
    assert manifest["stage_d_authorized_by_module"] is False


def test_training_plan_is_exact_deterministic_three_way_stream() -> None:
    plan_a = execution.build_training_start_plan(2_000_000_000)
    plan_b = execution.build_training_start_plan(2_000_000_000)
    assert tuple(plan_a.shape) == (2_048, 4, 4)
    assert plan_a.numel() == 32_768
    assert plan_a.dtype == torch.int64
    assert torch.equal(plan_a, plan_b)
    assert execution.start_plan_sha256(plan_a) == execution.start_plan_sha256(plan_b)
    assert int(plan_a.min()) >= 0


def test_live_rate_cap_is_exact_9600_second_envelope() -> None:
    at_cap = execution.validate_live_rate_cap(1.125)
    assert at_cap["worst_case_usd"] == pytest.approx(3.0)
    assert at_cap["max_gpu_seconds"] == 9_600.0
    with pytest.raises(RuntimeError, match="exceeds"):
        execution.validate_live_rate_cap(1.125001)
    with pytest.raises(RuntimeError):
        execution.validate_live_rate_cap(0.0)


def test_runner_is_syntax_valid_and_bound_to_frozen_qva_protocol() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        'CONTROL_ISSUE = 1182',
        'PREREG_ISSUE = 1176',
        'SCIENTIFIC_SEED = 2_011_761',
        'PHASE = "chm-v2-100m-qva-stage-c-1182-seed-2011761-v1"',
        'RESULT_ROOT = "/vol/chm-v2/100m-qva-stage-c/issue-1182/seed-2011761-v1"',
        'GPU_CLASS = "L4"',
        'MAX_GPU_SECONDS = 9_600',
        'MAX_BILLED_COMPUTE_USD = 3.00',
        'QVA_BLOB = "a34a8dffc2c2c1702a909782c3aba2593e90947c"',
        'STAGE_C_PROTOCOL_BLOB = "05e905937ff3a39276462a47fe93e8a499682d1a"',
        'MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"',
        'EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"',
    )
    for item in required:
        assert item in source


def test_runner_uses_host_staged_transport_and_final_checkpoint_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    train = _function_source(source, "_train_model")
    run = _function_source(source, "run_scientific")

    assert "host_staged_gather" in train
    assert "qva_flat_training_session_logits" in train
    assert "eiem_flat_training_session_logits" in train
    assert "local_session_logits" in train
    assert "CHECKPOINT_STEPS" not in source
    assert 'target = checkpoint_dir / "step-2048.pt"' in source
    assert "step-0512" not in source
    assert "step-1024" not in source
    assert "step-1536" not in source

    assert "_device_tokens(" not in source
    assert "train_data._device_cache" in run
    assert "val_data._device_cache" in run
    assert "train_source = train_data.data" in run
    assert "val_source = val_data.data" in run


def test_gpu_attempt_is_consumed_before_any_post_allocation_validation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_scientific")
    consumed_write = run.index("_atomic_write(consumed_path, raw_entry)")
    consumed_commit = run.index("volume.commit()", consumed_write)
    try_start = run.index("    try:", consumed_commit)
    binding = run.index("_validate_bindings(", try_start)
    account = run.index("_validate_account_binding(", try_start)
    cuda = run.index("torch.cuda.is_available()", try_start)
    model = run.index("CHMV1100MLocalLM()", try_start)

    assert consumed_write < consumed_commit < try_start < binding < account < cuda < model
    assert '"scientific_seed_consumed": True' in run
    assert "failure_evidence = evidence" in run
    assert '"classification": "CHM_V2_100M_QVA_STAGE_C_INFRASTRUCTURE_OR_RUNTIME_FAILURE"' in run
    assert "_atomic_write(failure_path, failure)" in run


def test_runner_has_no_resume_or_automatic_retry_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "retries=RETRIES" in source
    assert "RETRIES = 0" in source
    assert "resume_from" not in source
    assert '"resume_authorized": False' in source
    assert '"automatic_retry_authorized": False' in source
    assert "scientific attempt already consumed; no retry/resume/redispatch" in source


def test_runner_qva_probe_rows_match_frozen_classifier_schema() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    rows = _function_source(source, "_three_way_rows")
    for field in (
        "family",
        "case_id",
        "generator_version",
        "evidence_distance",
        "local_correct",
        "raw_correct",
        "qva_correct",
        "local_candidate_nll",
        "raw_candidate_nll",
        "qva_candidate_nll",
        "local_stale_choice",
        "raw_stale_choice",
        "qva_stale_choice",
    ):
        assert f'"{field}"' in rows
    assert "qva_retrieval_traces" in rows
    assert "qva_integration_diagnostics" in rows


def test_scientific_workflow_is_issue_only_final_authority_one_shot() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert "[modal-chm-v2-100m-qva-stage-c-1182-seed-2011761-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V2_1182_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "authority_comment_id" in source
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "account_selection_evidence_sha256" in source
    assert "scientific_seed=2011761" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "automatic_retry_authorized=false" in source
    assert "checkpoint_resume_authorized=false" in source
    assert "replication_authorized=false" in source
    assert "stage_d_authorized=false" in source
    assert "modal token info | tee" in source
    assert "modal.Workspace.from_context" not in source
    assert "(?:Workspace|User)" in source


def test_scientific_workflow_rechecks_rates_after_reservation_before_launch() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    reserve = source.index("Reserve one-shot scientific dispatch")
    rates = source.index("Re-read live rates immediately before L4 allocation")
    launch = source.index("Launch QVA Stage-C scientific attempt exactly once")
    assert reserve < rates < launch
    rate_block = source[rates:launch]
    assert "modal billing rates --json" in rate_block
    assert "worst = hourly * (9600.0 / 3600.0)" in rate_block
    assert "worst <= 3.00" in rate_block


def test_authority_audit_is_cpu_only_account_binding_inspection() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v2-100m-qva-stage-c-1182-authority-audit-v2]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "scripts/modal_select_account_v3.py" in source
    assert "--required-volume tam-research-data" in source
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "account_selection_evidence_sha256" in source
    assert "modal billing rates --json" in source
    assert "worst = hourly * (9600.0 / 3600.0)" in source
    assert "worst <= 3.00" in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "--phase state" not in source
    assert "gpu_allocated=false" in source
    assert "scientific_seed_consumed=false" in source
    assert "trigger_authorized=false" in source
    assert "modal token info | tee" in source
    assert "modal.Workspace.from_context" not in source
    assert "selected_workspace_name" not in source
    assert "(?:Workspace|User)" in source
    assert "37063049449" in source
    assert "modal-chm-v2-100m-qva-stage-c-1182-authority-audit-v1.yml" in source


def test_all_embedded_workflow_python_is_syntax_valid() -> None:
    for path in (WORKFLOW, AUDIT_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        blocks = _python_heredocs(source)
        assert blocks, f"no Python heredocs in {path.name}"
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} embedded Python block {index} invalid: {exc}"
                ) from exc


def test_static_files_do_not_reference_retired_branch_identities() -> None:
    joined = "\n".join(
        [
            RUNNER.read_text(encoding="utf-8"),
            WORKFLOW.read_text(encoding="utf-8"),
            AUDIT_WORKFLOW.read_text(encoding="utf-8"),
            inspect.getsource(execution),
        ]
    )
    assert "2011371" not in joined
    assert "2011121" not in joined
    assert "977001" not in joined
    assert "research/chm-v2-qva-stage-c-run-control-1182-v2" not in joined
