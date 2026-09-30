from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import pytest
import torch

from tam_research.chm_v2_100m_stage_b import (
    ENGINEERING_SEED,
    MAX_PROJECTED_PAIR_COMPUTE_USD,
    MAX_PROJECTED_PAIR_SECONDS,
    MEASURED_TOKENS_PER_MODEL,
    MIN_QVA_TOKENS_PER_SECOND,
    MIN_QVA_V1_THROUGHPUT_RATIO,
    MIN_V1_TOKENS_PER_SECOND,
    RESULT_ROOT,
    STREAM_SEED,
    TRIGGER_TITLE,
    build_start_plan,
    classify_stage_b,
    project_from_throughput,
    protocol_manifest,
    start_plan_sha256,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_qva_stage_b_1155_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v2-100m-qva-stage-b-1155-v1.yml"
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v2-100m-qva-stage-b-1155-authority-audit-v1.yml"
)


def _row(
    *,
    kind: str,
    params: int,
    tps: float,
    peak_gib: float = 14.0,
    digest: str,
    finite: bool = True,
) -> dict[str, object]:
    return {
        "kind": kind,
        "trainable_parameters": params,
        "measured_tokens": MEASURED_TOKENS_PER_MODEL,
        "tokens_per_second": tps,
        "peak_vram_bytes": int(peak_gib * 1024**3),
        "finite_loss": finite,
        "finite_parameters": finite,
        "start_plan_sha256": digest,
    }


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
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            assert node.end_lineno is not None
            return "\n".join(lines[start : node.end_lineno])
    raise AssertionError(f"function {name!r} not found")


def test_stage_b_contract_is_exact_non_scientific_envelope() -> None:
    manifest = validate_contract()
    assert manifest == protocol_manifest()
    assert manifest["classification"] == "CHM_V2_100M_QVA_STAGE_B_L4_SYSTEMS_PREFLIGHT_TRIGGER_WITHHELD"
    assert ENGINEERING_SEED == 1_147_201
    assert STREAM_SEED == 1_157_201
    assert manifest["historical_consumed_chm_v1_seed_not_reusable"] == 977001
    assert manifest["models"]["chm_v1_eiem_parameters"] == 101_836_800
    assert manifest["models"]["chm_v2_qva_parameters"] == 101_885_985
    assert manifest["resource_plan"]["gpu"] == "1x NVIDIA L4"
    assert manifest["resource_plan"]["max_gpu_seconds"] == 1200
    assert manifest["resource_plan"]["max_stage_b_compute_usd"] == 0.50
    assert manifest["resource_plan"]["retries"] == 0
    assert manifest["training_geometry"]["warmup_steps"] == 2
    assert manifest["training_geometry"]["measured_steps"] == 8
    assert manifest["training_geometry"]["measured_tokens_per_model"] == 131_072
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["scientific_execution_authorized"] is False
    assert RESULT_ROOT == "/vol/chm-v2/100m-qva-stage-b/parent-1147/seed-1147201-v1"
    assert TRIGGER_TITLE == "[modal-chm-v2-100m-qva-stage-b-parent-1147-v1]"


def test_start_plan_is_regenerated_deterministically_from_same_cpu_seed() -> None:
    first = build_start_plan()
    second = build_start_plan()
    assert first.dtype == torch.int64
    assert tuple(first.shape) == (10, 4, 4)
    assert first.numel() == 160
    assert torch.equal(first, second)
    digest = start_plan_sha256(first)
    assert digest == start_plan_sha256(second)
    assert digest == protocol_manifest()["stream_plan"]["sha256"]
    assert int(first.min()) >= 0
    assert int(first.max()) < 2_000_000_000 - 1025


def test_projection_and_classifier_pass_at_frozen_gates() -> None:
    digest = start_plan_sha256(build_start_plan())
    v1 = _row(
        kind="v1",
        params=101_836_800,
        tps=max(MIN_V1_TOKENS_PER_SECOND, 20_000.0),
        digest=digest,
    )
    qva = _row(
        kind="qva",
        params=101_885_985,
        tps=max(MIN_QVA_TOKENS_PER_SECOND, 18_000.0),
        digest=digest,
    )
    decision = classify_stage_b(v1=v1, qva=qva, live_hourly_resource_usd=1.1172)
    assert decision["classification"] == "CHM_V2_100M_QVA_STAGE_B_SYSTEMS_PASS"
    assert decision["passed"] is True
    assert decision["stop_reasons"] == []
    assert decision["projection"]["qva_v1_throughput_ratio"] >= MIN_QVA_V1_THROUGHPUT_RATIO
    assert decision["projection"]["pair_projected_seconds"] <= MAX_PROJECTED_PAIR_SECONDS
    assert decision["projection"]["pair_projected_compute_usd"] <= MAX_PROJECTED_PAIR_COMPUTE_USD
    assert decision["scientific_seed_authorized"] is False
    assert decision["scientific_execution"] is False


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda v1, qva: qva.update(tokens_per_second=9_999.0), "qva_throughput_below_gate"),
        (lambda v1, qva: qva.update(peak_vram_bytes=int(22.01 * 1024**3)), "qva_peak_vram_above_gate"),
        (lambda v1, qva: qva.update(start_plan_sha256="0" * 64), "qva_start_plan_digest_mismatch"),
        (lambda v1, qva: qva.update(trainable_parameters=1), "qva_parameter_count_mismatch"),
        (lambda v1, qva: qva.update(finite_loss=False), "qva_nonfinite_loss"),
    ],
)
def test_classifier_stops_on_integrity_or_resource_failures(mutator, reason: str) -> None:
    digest = start_plan_sha256(build_start_plan())
    v1 = _row(kind="v1", params=101_836_800, tps=20_000.0, digest=digest)
    qva = _row(kind="qva", params=101_885_985, tps=18_000.0, digest=digest)
    mutator(v1, qva)
    decision = classify_stage_b(v1=v1, qva=qva, live_hourly_resource_usd=1.1172)
    assert decision["classification"] == "CHM_V2_100M_QVA_STAGE_B_SYSTEMS_STOP"
    assert decision["passed"] is False
    assert reason in decision["stop_reasons"]


def test_projection_formula_is_pair_sum_and_live_rate_bound() -> None:
    p = project_from_throughput(
        v1_tokens_per_second=20_000.0,
        qva_tokens_per_second=18_000.0,
        live_hourly_resource_usd=1.1172,
    )
    expected = 33_554_432 / 20_000.0 + 33_554_432 / 18_000.0
    assert p["pair_projected_seconds"] == pytest.approx(expected)
    assert p["qva_v1_throughput_ratio"] == pytest.approx(0.9)
    assert p["pair_projected_compute_usd"] == pytest.approx(expected / 3600.0 * 1.1172)


def test_runner_has_one_shot_gpu_boundary_and_no_scientific_seed_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'ENGINEERING_SEED = 1_147_201' in source
    assert 'HISTORICAL_CONSUMED_SCIENTIFIC_SEED = 977_001' in source
    assert 'GPU_CLASS = "L4"' in source
    assert 'MAX_SECONDS = 1_200' in source
    assert 'MAX_BILLED_COMPUTE_USD = 0.50' in source
    assert 'retries=0' in source
    assert '"scientific_seed_authorized": False' in source
    assert '"scientific_execution": False' in source
    assert '"automatic_retry_authorized": False' in source
    assert "torch.compile" not in source
    assert "977_001" in source
    assert "scientific_seed =" not in source.lower()


def test_runner_regenerates_cpu_plan_for_each_model_and_requires_identity() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_stage_b")
    assert "v1_start_plan = build_start_plan()" in run
    assert "qva_start_plan = build_start_plan()" in run
    assert "v1_plan_digest = start_plan_sha256(v1_start_plan)" in run
    assert "qva_plan_digest = start_plan_sha256(qva_start_plan)" in run
    assert "torch.equal(v1_start_plan, qva_start_plan)" in run
    assert "paired_start_plan_identical" in run


def test_attempt_consumption_is_durable_before_measurement() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_stage_b")
    consumed = run.index("_atomic_write(consumed_path, consumed)")
    fingerprint = run.index("actual_fingerprint = fingerprint_frozen_corpus")
    first_benchmark = run.index("_benchmark_model(")
    assert consumed < fingerprint < first_benchmark
    assert '"engineering_attempt_consumed": True' in run
    assert '"gpu_allocation_started": True' in run


def test_runner_checks_paired_inherited_initialization() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_stage_b")
    assert "v1.backbone.state_dict()" in run
    assert "qva.backbone.state_dict()" in run
    assert '("query_address.weight", "key_address.weight", "memory_gate_logit")' in run
    assert "inherited backbone init mismatch" in run
    assert "inherited EIEM init mismatch" in run


def test_launch_workflow_requires_exact_authority_unique_trigger_and_live_rate() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert "[modal-chm-v2-100m-qva-stage-b-parent-1147-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V2_1155_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "authority_comment_id" in source
    assert "engineering_seed=1147201" in source
    assert "scientific_seed_authorized=false" in source
    assert "modal billing rates --json" in source
    assert "worst <= 0.50" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase inspect" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "stage_c_authorized_automatically=false" in source


def test_authority_audit_is_cpu_inspect_only_and_cannot_launch() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v2-100m-qva-stage-b-1155-authority-audit-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "modal billing rates --json" in source
    assert "worst <= 0.50" in source
    assert "--phase inspect" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "result_namespace_unused=true" in source
    assert "gpu_allocated=false" in source
    assert "engineering_attempt_consumed=false" in source
    assert "trigger_authorized=false" in source
    assert "scientific_seed_authorized=false" in source


def test_all_embedded_workflow_python_is_syntax_valid() -> None:
    for path in (WORKFLOW, AUDIT_WORKFLOW):
        blocks = _python_heredocs(path.read_text(encoding="utf-8"))
        assert blocks, f"no Python heredocs found in {path.name}"
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} embedded Python block {index} syntax error: {exc}"
                ) from exc
