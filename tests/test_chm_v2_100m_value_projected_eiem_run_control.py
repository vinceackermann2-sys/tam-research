from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import pytest
import torch

from tam_research.chm_v2_100m_value_projected_eiem_run_control import (
    AUDIT_TITLE,
    MAX_COMPUTE_USD,
    MAX_GPU_SECONDS,
    PHASE,
    RESULT_ROOT,
    SCIENTIFIC_SEED,
    TRIGGER_TITLE,
    assert_no_execution_authority,
    plan_digest,
    projected_worst_case_compute_usd,
    protocol_manifest,
    summarize_three_way_probe_rows,
    training_plan_for_tokens,
    validate_live_rate_cap,
    validate_seed_for_preparation,
    validation_plan_for_tokens,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_value_projected_eiem_1115_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v2-100m-value-projected-eiem-1115-v1.yml"
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v2-100m-value-projected-eiem-1115-authority-audit-v1.yml"
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
            start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
            assert node.end_lineno is not None
            return "\n".join(lines[start : node.end_lineno])
    raise AssertionError(f"function {name!r} not found")


def test_run_control_manifest_freezes_single_use_identity_without_execution_authority() -> None:
    manifest = protocol_manifest()
    assert manifest["control_issue"] == 1115
    assert manifest["prereg_issue"] == 1112
    assert SCIENTIFIC_SEED == 2_011_121
    assert PHASE == "chm-v2-100m-value-projected-eiem-1115-seed-2011121-v1"
    assert RESULT_ROOT == "/vol/chm-v2/100m-value-projected-eiem/issue-1115/seed-2011121-v1"
    assert TRIGGER_TITLE == "[modal-chm-v2-100m-value-projected-eiem-1115-seed-2011121-v1]"
    assert AUDIT_TITLE == "[modal-chm-v2-100m-value-projected-eiem-1115-authority-audit-v1]"
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["trigger_authorized_by_module"] is False
    assert manifest["gpu_allocation_authorized_by_module"] is False
    assert manifest["scientific_seed_consumed_by_module"] is False
    assert manifest["stage_d_authorized"] is False
    assert manifest["scale_up_authorized"] is False
    assert manifest["multi_seed_replication_authorized"] is False


def test_seed_is_reserved_only_and_consumed_chm_v1_seed_is_refused() -> None:
    assert validate_seed_for_preparation(2_011_121) == 2_011_121
    with pytest.raises(RuntimeError):
        validate_seed_for_preparation(2_011_121, request_execution=True)
    with pytest.raises(RuntimeError):
        validate_seed_for_preparation(977_001)
    with pytest.raises(RuntimeError):
        validate_seed_for_preparation(2_011_122)

    assert_no_execution_authority()
    for kwargs in (
        {"gpu": True},
        {"training": True},
        {"trigger_creation": True},
        {"scientific_seed_consumption": True},
        {"stage_d": True},
    ):
        with pytest.raises(RuntimeError):
            assert_no_execution_authority(**kwargs)


def test_three_way_training_and_validation_plans_are_frozen_deterministic() -> None:
    train_a = training_plan_for_tokens(2_000_000_000)
    train_b = training_plan_for_tokens(2_000_000_000)
    assert tuple(train_a.shape) == (2048, 4, 4)
    assert torch.equal(train_a, train_b)
    assert plan_digest(train_a) == plan_digest(train_b)

    val_a = validation_plan_for_tokens(10_000_000)
    val_b = validation_plan_for_tokens(10_000_000)
    assert tuple(val_a.shape) == (128, 1, 8)
    assert torch.equal(val_a, val_b)
    assert plan_digest(val_a) == plan_digest(val_b)


def test_live_rate_cap_is_exactly_four_hour_six_dollar_envelope() -> None:
    assert MAX_GPU_SECONDS == 14_400
    assert MAX_COMPUTE_USD == 6.00
    assert projected_worst_case_compute_usd(1.25) == pytest.approx(5.0)
    result = validate_live_rate_cap(1.25)
    assert result["within_cap"] is True
    assert result["worst_case_compute_usd"] == pytest.approx(5.0)
    with pytest.raises(RuntimeError):
        validate_live_rate_cap(1.500001)


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    families = ("rare_fact", "overwrite", "two_hop", "local_negative")
    for family_index, family in enumerate(families):
        for case_id in range(128):
            local_correct = case_id < 32
            raw_correct = case_id < 40
            vp_correct = case_id < 48
            rows.append(
                {
                    "family": family,
                    "case_id": family_index * 1000 + case_id,
                    "local_correct": local_correct,
                    "raw_correct": raw_correct,
                    "vp_correct": vp_correct,
                    "local_candidate_nll": 1.0,
                    "raw_candidate_nll": 0.95,
                    "vp_candidate_nll": 0.85,
                    "local_stale_choice": family == "overwrite" and case_id < 20,
                    "raw_stale_choice": family == "overwrite" and case_id < 18,
                    "vp_stale_choice": family == "overwrite" and case_id < 10,
                }
            )
    return rows


def test_three_way_summary_reports_vp_vs_local_and_vp_vs_raw_without_gate_mutation() -> None:
    summary = summarize_three_way_probe_rows(_synthetic_rows())
    agg = summary["aggregate_long_range"]
    assert agg["vp_vs_local_accuracy_gain"] == pytest.approx(0.125)
    assert agg["vp_vs_raw_accuracy_gain"] == pytest.approx(0.0625)
    assert agg["vp_vs_local_nll_benefit"] == pytest.approx(0.15)
    assert agg["vp_vs_raw_nll_benefit"] == pytest.approx(0.10)
    assert summary["overwrite"]["vp_stale_rate"] < summary["overwrite"]["local_stale_rate"]


def test_runner_is_host_import_safe_and_scientific_path_is_from_scratch() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    prefix = source.split("app = modal.App", 1)[0]
    assert "import torch" not in prefix
    assert "chm_v2_100m_value_projected_eiem_run_control import" in source
    assert "torch.load(" not in source
    assert "CHMV1100MLocalLM()" in source
    assert "CHMV1100MEIEMLM()" in source
    assert "CHMV2100MValueProjectedEIEMLM()" in source
    assert "_seed_all(SCIENTIFIC_SEED)" in source
    assert 'kind not in {"local", "raw_eiem", "vp_eiem"}' in source
    assert "vp_eiem_flat_training_session_logits" in source
    assert "vp_eiem_exact_flat_two_chunk_logits" in source
    assert "torch.compile" not in source


def test_runner_consumes_seed_before_any_scientific_training_and_never_resumes() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_scientific")
    consumed = run.index("_atomic_write(consumed_path, consumed)")
    local_construct = run.index("CHMV1100MLocalLM()")
    train_call = run.index("_train_model(")
    assert consumed < local_construct < train_call
    assert '"scientific_seed_consumed": True' in run
    assert "no retry/resume/redispatch" in run
    assert "checkpoint_resume_authorized" in run
    assert "automatic_retry_authorized" in run
    assert "retries=RETRIES" in source
    assert "RETRIES = 0" in source


def test_runner_enforces_paired_initialization_and_vp_zero_up_projection() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_scientific")
    assert '_assert_state_equal(local.backbone, raw.backbone, "LOCAL/RAW backbone")' in run
    assert '_assert_state_equal(local.backbone, vp.backbone, "LOCAL/VP backbone")' in run
    assert '_assert_state_equal(raw.query_address, vp.query_address, "RAW/VP query")' in run
    assert '_assert_state_equal(raw.key_address, vp.key_address, "RAW/VP key")' in run
    assert "torch.equal(raw.memory_gate_logit, vp.memory_gate_logit)" in run
    assert "torch.count_nonzero(vp.value_up.weight)" in run
    assert "torch.count_nonzero(vp.value_up.bias)" in run


def test_runner_records_required_projection_and_label_direction_diagnostics() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "projection_correction_ratios" in source
    assert "raw_to_label_direction_cosine" in source
    assert "projected_to_label_direction_cosine" in source
    assert "vp_projection" in source
    assert "gate_statistics" in source
    assert "retrieval_traces" in source
    assert "classification" in source
    assert "stop_reasons" in source


def test_scientific_workflow_has_exact_authority_account_and_one_shot_guards() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "types: [opened]" in source
    assert TRIGGER_TITLE in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V2_100M_VALUE_PROJECTED_EIEM_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "account_selection_evidence_sha256" in source
    assert "modal_select_account_v3.py" in source
    assert "modal_runtime_admission_probe_1067_v1.py" in source
    assert "modal billing rates --json" in source
    assert "worst=hourly*4.0" in source.replace(" ", "")
    assert "worst<=6.00" in source.replace(" ", "")
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "seed_2011121_consumed=false" in source
    assert "automatic_retry_authorized=false" in source
    assert "stage_d_authorized=false" in source


def test_authority_audit_is_preallocation_only_and_binds_runtime_eligible_account() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "types: [opened]" in source
    assert AUDIT_TITLE in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "modal_select_account_v3.py" in source
    assert "--required-volume tam-research-data" in source
    assert "--runtime-probe-path modal_runtime_admission_probe_1067_v1.py" in source
    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "--phase state" not in source
    assert "result_namespace_unused" in source
    assert "seed_2011121_consumed" in source
    assert "gpu_allocated" in source
    assert "trigger_authorized=false" in source
    assert "runtime_admission_verified=true" in source
    assert "live_4h_worst_case_usd" in source


def test_all_embedded_workflow_python_blocks_compile() -> None:
    for path in (WORKFLOW, AUDIT_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        blocks = _python_heredocs(source)
        assert blocks, path.name
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(f"{path.name} Python heredoc {index} invalid: {exc}") from exc


def test_only_scientific_workflow_can_reach_reservation_or_gpu_phase() -> None:
    scientific = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "--phase reserve" in scientific and "--phase run" in scientific
    assert "--phase reserve" not in audit and "--phase run" not in audit
    assert "scientific_seed=2011121" in audit
    assert "seed_2011121_consumed=false" in audit
