from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import pytest
import torch

from tam_research.chm_v3_100m_daec_stable_stage_b import (
    AUDIT_TITLE,
    ENGINEERING_SEED,
    MAX_PROJECTED_THREE_MODEL_COMPUTE_USD,
    MAX_PROJECTED_THREE_MODEL_SECONDS,
    MAX_STAGE_B_COMPUTE_USD,
    MAX_GPU_SECONDS,
    MEASURED_TOKENS_PER_MODEL,
    MIN_DAEC_RAW_THROUGHPUT_RATIO,
    RESULT_ROOT,
    STREAM_SEED,
    TRIGGER_TITLE,
    build_start_plan,
    classify_stage_b,
    project_from_throughput,
    protocol_manifest,
    start_plan_sha256,
    validate_contract,
    validate_engineering_seed,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v3_100m_daec_stable_stage_b_1302_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v3-100m-daec-stable-stage-b-1302-v1.yml"
AUDIT_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "modal-chm-v3-100m-daec-stable-stage-b-1302-authority-audit-v1.yml"
)


def _row(*, params: int, tps: float, digest: str, peak_gib: float = 12.0) -> dict[str, object]:
    return {
        "trainable_parameters": params,
        "measured_tokens": MEASURED_TOKENS_PER_MODEL,
        "finite_loss": True,
        "finite_parameters": True,
        "start_plan_sha256": digest,
        "tokens_per_second": tps,
        "peak_vram_bytes": int(peak_gib * 1024**3),
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            assert node.end_lineno is not None
            return "\n".join(lines[start : node.end_lineno])
    raise AssertionError(f"function {name!r} not found")


def test_contract_is_exact_non_scientific_three_model_envelope() -> None:
    manifest = validate_contract()
    assert manifest == protocol_manifest()
    assert manifest["classification"] == "CHM_V3_100M_DAEC_STABLE_TARGET_NLL_STAGE_B_RECHECK_TRIGGER_WITHHELD"
    assert ENGINEERING_SEED == 1_288_201
    assert STREAM_SEED == 1_298_201
    assert manifest["models"] == {
        "local_parameters": 101_803_520,
        "raw_eiem_parameters": 101_836_800,
        "daec_parameters": 101_853_697,
    }
    assert manifest["resource_plan"]["gpu"] == "1x NVIDIA L4"
    assert MAX_GPU_SECONDS == 1200
    assert MAX_STAGE_B_COMPUTE_USD == 0.50
    assert manifest["resource_plan"]["retries"] == 0
    assert manifest["training_geometry"]["warmup_steps"] == 2
    assert manifest["training_geometry"]["measured_steps"] == 8
    assert MEASURED_TOKENS_PER_MODEL == 131_072
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["scientific_execution_authorized"] is False
    assert manifest["stage_c_authorized"] is False
    assert RESULT_ROOT == "/vol/chm-v3/100m-daec-stage-b-stable/parent-1288/seed-1288201-v1"
    assert TRIGGER_TITLE == "[modal-chm-v3-100m-daec-stage-b-stable-1288-seed-1288201-v1]"
    assert AUDIT_TITLE == "[modal-chm-v3-100m-daec-stage-b-stable-1288-authority-audit-v1]"


def test_engineering_seed_refuses_historical_and_stage_a_identities() -> None:
    assert validate_engineering_seed(1_288_201) == 1_288_201
    for seed in (977_001, 2_011_121, 2_011_371, 2_011_431, 2_011_761, 1_234_001, 1_234_201, 1_262_201):
        with pytest.raises(RuntimeError):
            validate_engineering_seed(seed)
    with pytest.raises(RuntimeError):
        validate_engineering_seed(1_234_202)


def test_start_plan_is_deterministic_and_exact_shape() -> None:
    first = build_start_plan()
    second = build_start_plan()
    assert first.dtype == torch.int64
    assert tuple(first.shape) == (10, 4, 4)
    assert first.numel() == 160
    assert torch.equal(first, second)
    assert start_plan_sha256(first) == start_plan_sha256(second)
    assert start_plan_sha256(first) == protocol_manifest()["stream_plan"]["sha256"]
    assert int(first.min()) >= 0
    assert int(first.max()) < 2_000_000_000 - 1025


def test_three_model_projection_formula_and_pass_gate() -> None:
    digest = start_plan_sha256(build_start_plan())
    local = _row(params=101_803_520, tps=25_000.0, digest=digest)
    raw = _row(params=101_836_800, tps=24_000.0, digest=digest)
    daec = _row(params=101_853_697, tps=20_000.0, digest=digest)
    decision = classify_stage_b(
        local=local,
        raw=raw,
        daec=daec,
        live_hourly_resource_usd=1.1172,
    )
    assert decision["classification"] == "CHM_V3_100M_DAEC_STABLE_TARGET_NLL_STAGE_B_SYSTEMS_PASS"
    assert decision["passed"] is True
    assert decision["stop_reasons"] == []
    projection = decision["projection"]
    assert projection is not None
    expected = 33_554_432 / 25_000 + 33_554_432 / 24_000 + 33_554_432 / 20_000
    assert projection["three_model_projected_seconds"] == pytest.approx(expected)
    assert projection["daec_raw_throughput_ratio"] == pytest.approx(20_000 / 24_000)
    assert projection["three_model_projected_compute_usd"] == pytest.approx(
        expected / 3600 * 1.1172
    )
    assert projection["three_model_projected_seconds"] <= MAX_PROJECTED_THREE_MODEL_SECONDS
    assert projection["three_model_projected_compute_usd"] <= MAX_PROJECTED_THREE_MODEL_COMPUTE_USD
    assert decision["scientific_seed_authorized"] is False
    assert decision["stage_c_authorized_automatically"] is False


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda l, r, d: d.update(tokens_per_second=7_999.0), "daec_throughput_below_gate"),
        (lambda l, r, d: r.update(tokens_per_second=7_999.0), "raw_throughput_below_gate"),
        (lambda l, r, d: l.update(tokens_per_second=7_999.0), "local_throughput_below_gate"),
        (lambda l, r, d: d.update(peak_vram_bytes=int(22.01 * 1024**3)), "daec_peak_vram_above_gate"),
        (lambda l, r, d: d.update(start_plan_sha256="0" * 64), "daec_start_plan_digest_mismatch"),
        (lambda l, r, d: d.update(trainable_parameters=1), "daec_parameter_count_mismatch"),
        (lambda l, r, d: d.update(finite_loss=False), "daec_nonfinite_loss"),
    ],
)
def test_classifier_stops_on_integrity_or_resource_failures(mutator, reason: str) -> None:
    digest = start_plan_sha256(build_start_plan())
    local = _row(params=101_803_520, tps=25_000.0, digest=digest)
    raw = _row(params=101_836_800, tps=24_000.0, digest=digest)
    daec = _row(params=101_853_697, tps=20_000.0, digest=digest)
    mutator(local, raw, daec)
    decision = classify_stage_b(
        local=local,
        raw=raw,
        daec=daec,
        live_hourly_resource_usd=1.1172,
    )
    assert decision["classification"] == "CHM_V3_100M_DAEC_STABLE_TARGET_NLL_STAGE_B_SYSTEMS_STOP"
    assert decision["passed"] is False
    assert reason in decision["stop_reasons"]


def test_classifier_stops_on_ratio_walltime_and_compute_gates() -> None:
    digest = start_plan_sha256(build_start_plan())
    local = _row(params=101_803_520, tps=8_000.0, digest=digest)
    raw = _row(params=101_836_800, tps=15_000.0, digest=digest)
    daec = _row(params=101_853_697, tps=8_000.0, digest=digest)
    decision = classify_stage_b(
        local=local,
        raw=raw,
        daec=daec,
        live_hourly_resource_usd=2.0,
    )
    assert decision["passed"] is False
    assert "daec_raw_throughput_ratio_below_gate" in decision["stop_reasons"]
    assert "projected_three_model_compute_cost_above_gate" in decision["stop_reasons"]


def test_projection_rejects_nonpositive_inputs_without_crashing_classifier() -> None:
    with pytest.raises(ValueError):
        project_from_throughput(
            local_tokens_per_second=0,
            raw_tokens_per_second=1,
            daec_tokens_per_second=1,
            live_hourly_resource_usd=1,
        )
    digest = start_plan_sha256(build_start_plan())
    local = _row(params=101_803_520, tps=0.0, digest=digest)
    raw = _row(params=101_836_800, tps=20_000.0, digest=digest)
    daec = _row(params=101_853_697, tps=20_000.0, digest=digest)
    decision = classify_stage_b(local=local, raw=raw, daec=daec, live_hourly_resource_usd=1.0)
    assert "invalid_projection_inputs" in decision["stop_reasons"]


def test_runner_has_one_shot_gpu_boundary_and_no_scientific_seed_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'ENGINEERING_SEED = 1_288_201' in source
    assert 'GPU_CLASS = "L4"' in source
    assert 'MAX_SECONDS = 1_200' in source
    assert 'MAX_BILLED_COMPUTE_USD = 0.50' in source
    assert "retries=0" in source
    assert '"scientific_seed_authorized": False' in source
    assert '"scientific_execution": False' in source
    assert 'STABLE_NLL_BLOB = "482e2c930fa02448e45327747f10571315fd41dc"' in source
    assert '"stage_c_authorized_automatically": False' in source
    assert "torch.compile" not in source

    tree = ast.parse(source)
    assigned_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
    assert "scientific_seed" not in assigned_names
    assert "SCIENTIFIC_SEED" not in assigned_names


def test_attempt_consumption_precedes_corpus_access_and_benchmarking() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_stage_b")
    consumed = run.index("_atomic_write(consumed_path, consumed)")
    corpus = run.index("actual_fingerprint = fingerprint_frozen_corpus")
    first_benchmark = run.index("_benchmark_model(")
    assert consumed < corpus < first_benchmark
    assert '"engineering_attempt_consumed": True' in run
    assert '"gpu_allocation_started": True' in run


def test_runner_regenerates_same_plan_for_all_three_models() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_stage_b")
    for name in ("local_plan", "raw_plan", "daec_plan"):
        assert f"{name} = build_start_plan()" in run
    assert "torch.equal(local_plan, raw_plan)" in run
    assert "torch.equal(local_plan, daec_plan)" in run
    assert "paired_start_plan_identical" in run


def test_runner_checks_paired_initialization_and_uses_correct_loss_surfaces() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    run = _function_source(source, "run_stage_b")
    assert "local.backbone.state_dict()" in run
    assert "raw.backbone.state_dict()" in run
    assert "daec.backbone.state_dict()" in run
    assert '("query_address.weight", "key_address.weight", "memory_gate_logit")' in run

    bench = _function_source(source, "_benchmark_model")
    assert "local_session_logits(model, x)" in bench
    assert "eiem_flat_training_session_logits(model, x)" in bench
    assert "daec_flat_training_session_nll_stable(model, x, y)" in bench
    assert "daec_flat_training_session_log_probs" not in bench
    assert "F.cross_entropy(" in bench
    assert "F.nll_loss(" not in bench
    assert "daec_flat_training_session_nll_stable(model, x, y)" in bench
    assert "daec_flat_training_session_nll(model, x, y)" not in bench


def test_audit_workflow_is_dynamic_account_cpu_inspect_only() -> None:
    source = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v3-100m-daec-stage-b-stable-1288-authority-audit-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "actions: read" in source
    assert "modal_select_account_v3.py" in source
    assert "--required-volume tam-research-data" in source
    assert "--runtime-probe-path modal_runtime_admission_probe_1067_v1.py" in source
    assert "--force-account" not in source
    assert "modal token info" in source
    assert r'^(?:Workspace|User):\s*([^\s(]+)' in source
    assert "MODAL_TOKEN_ID_2" in source and "MODAL_TOKEN_SECRET_2" in source
    assert "modal billing rates --json" in source
    assert "worst <= 0.50" in source
    assert "tam_research/chm_v3_100m_daec_target_nll_stable.py" in source
    assert "482e2c930fa02448e45327747f10571315fd41dc" in source
    assert "--phase inspect" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "result_namespace_unused=true" in source
    assert "gpu_allocated=false" in source
    assert "engineering_attempt_consumed=false" in source
    assert "scientific_seed_authorized=false" in source
    assert "stage_c_authorized=false" in source


def test_launch_workflow_requires_exact_authority_account_and_one_shot_phases() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert TRIGGER_TITLE in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V3_1302_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "authority_comment_id" in source
    assert "selected_modal_account" in source
    assert "selected_modal_workspace" in source
    assert "account_selection_evidence_sha256" in source
    assert "modal token info" in source
    assert r'^(?:Workspace|User):\s*([^\s(]+)' in source
    assert "modal billing rates --json" in source
    assert "worst <= 0.50" in source
    assert "tam_research/chm_v3_100m_daec_target_nll_stable.py" in source
    assert "482e2c930fa02448e45327747f10571315fd41dc" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase inspect" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "engineering_seed=1288201" in source
    assert "scientific_seed_authorized=false" in source
    assert "stage_c_authorized=false" in source
    assert "automatic_retry_authorized=false" in source
    assert "MODAL_TOKEN_ID_2" in source and "MODAL_TOKEN_SECRET_2" in source


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


def test_stage_b_files_do_not_modify_or_embed_stage_a_architecture() -> None:
    contract = (ROOT / "tam_research" / "chm_v3_100m_daec_stage_b.py").read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "class CHMV3100MDAECLM" not in contract
    assert "class CHMV3100MDAECLM" not in runner
    assert "COPY_GATE_INIT" not in contract
    assert "nn.Linear" not in runner


def test_target_nll_stage_b_preserves_original_ratio_gate_and_helper_identity() -> None:
    manifest = validate_contract()
    assert MIN_DAEC_RAW_THROUGHPUT_RATIO == 0.60
    assert manifest["resource_plan"]["max_stage_b_compute_usd"] == 0.50

    runner = RUNNER.read_text(encoding="utf-8")
    launch = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    helper_blob = "482e2c930fa02448e45327747f10571315fd41dc"

    assert helper_blob in runner
    assert helper_blob in launch
    assert helper_blob in audit
    assert "tam_research/chm_v3_100m_daec_target_nll_stable.py" in launch
    assert "tam_research/chm_v3_100m_daec_target_nll_stable.py" in audit
    assert "stable_nll_blob" in launch
    assert "stable_nll_blob" in audit
    assert "1234201" not in launch
    assert "1234201" not in audit
    assert "CHM_V3_1237_FINAL_LAUNCHER_AUTHORITY_V1" not in launch
