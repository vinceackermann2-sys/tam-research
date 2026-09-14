from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_chm_v1_eiem_seed19591_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-eiem-optimized-seed19591-v1.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_seed19591_launcher_parses_and_freezes_exact_authority() -> None:
    source = _text(LAUNCHER)
    assert ast.parse(source) is not None

    required = (
        'PHASE = "chm-v1-eiem-optimized-seed-19591-v1"',
        'TRIGGER_TITLE = "[modal-chm-v1-eiem-optimized-seed-19591-v1]"',
        "RESEARCH_ISSUE = 959",
        "RUN_CONTROL_ISSUE = 966",
        "AUTHORIZATION_COMMENT_ID = 5_655_031_401",
        'AUTHORIZATION_REF = "issue-959-comment-5655031401"',
        'SCIENTIFIC_AUTHORITY_SHA = "22da370b335727973e76fd20d0a3f87f6bb2d320"',
        'SCIENTIFIC_AUTHORITY_TREE = "a217d3f88e6784b491c7f3dc4333a22283c2583f"',
        "SCIENTIFIC_SEED = 19_591",
        'RESULT_ROOT = "/vol/chm-v1/eiem-optimized-replication/issue-966/seed-19591-v1"',
        'GPU_CLASS = "L4"',
        "CPU_CORES = 4",
        "RAM_MIB = 8192",
        "MAX_SECONDS_PER_SEED = 3600",
        "MAX_BILLED_COMPUTE_USD = 1.25",
        "LONG_MEMORY_EVALUATOR_SEED = 8_540_911",
        "VALIDATION_SAMPLING_SEED = 8_540_912",
        "SYSTEMS_TIMING_SEED = 8_540_913",
        "VALIDATION_BATCHES = 64",
        "VALIDATION_BATCH_SIZE = 8",
        "SESSION_TOKENS = 1024",
    )
    for needle in required:
        assert needle in source, needle

    one_seed = 0.80000 + 4 * 0.04730 + 8 * 0.00800
    assert abs(one_seed - 1.0532) < 1e-12
    assert one_seed <= 1.25


def test_only_seed19591_function_can_request_gpu_and_consumes_before_training() -> None:
    source = _text(LAUNCHER)
    tree = ast.parse(source)
    gpu_decorated: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            for keyword in decorator.keywords:
                if keyword.arg == "gpu":
                    gpu_decorated.append((node.name, ast.unparse(keyword.value)))

    assert gpu_decorated == [("run_seed_19591", "GPU_CLASS")]
    run = _function_source(source, "run_seed_19591")
    assert "SEED_CONSUMED.json" in run
    assert "ATTEMPT_FAILURE.json" in run
    assert '"automatic_retry_authorized": False' in run
    assert '"next_seed_authorized_automatically": False' in run
    assert run.index("_atomic_write(consumed_path, consumed)") < run.index("train_one(")
    assert "validate_replication_seed(SCIENTIFIC_SEED, paid_run_authorized=True)" in run
    assert "gpu_allocation_started" in run
    assert "retries=0" in source
    assert "timeout=MAX_SECONDS_PER_SEED" in source
    assert "def run_seed_19592" not in source
    assert "def run_seed_19593" not in source
    assert "def run_seed_8611" not in source
    assert "def run_seed_8612" not in source
    assert "def run_seed_8613" not in source


def test_indexed_inference_uses_only_validated_optimized_state() -> None:
    source = _text(LAUNCHER)
    batch = _function_source(source, "_run_eiem_batch")
    probes = _function_source(source, "_evaluate_probes")
    exactness = _function_source(source, "_ordinary_indexed_exactness_audit")
    timing = _function_source(source, "_systems_timing")

    assert "OptimizedEpisodicState if mode == \"indexed\" else EpisodicState" in batch
    assert "optimized_wrapper_seconds_total" in batch
    assert "optimized_sidecar_build_seconds_total" in batch
    assert "OptimizedEpisodicState" in probes
    assert 'mode="flat"' in exactness
    assert 'mode="indexed"' in exactness
    assert "verify_indexed_exactness=True" in exactness
    assert "torch.allclose" in exactness
    assert "atol=1e-5" in exactness
    assert '"eiem_optimized_indexed"' in timing

    run = _function_source(source, "run_seed_19591")
    assert "evaluate_eiem_optimized_language" in run
    assert "optimized_minus_flat_nll" in run
    assert "optimized_indexed_flat_mismatch" in run
    assert "optimized_indexed_flat_logit_mismatch" in run
    assert "optimized_indexed_flat_nll_mismatch" in run
    assert 'immediate_stop_reasons.append("sparse_read_fraction_gt_0.25")' in run
    assert 'immediate_stop_reasons.append("index_overhead_erases_practical_advantage")' in run
    assert '"SEED_19591_STOP_BEFORE_19592"' in run
    assert '"SEED_19591_COMPLETE_REVIEW_BEFORE_19592"' in run


def test_zero_gpu_preflight_refuses_existing_namespace_and_binds_replication_guard() -> None:
    source = _text(LAUNCHER)
    preflight = _function_source(source, "verify_zero_gpu")
    reserve = _function_source(source, "reserve_dispatch")

    assert "validate_replication_seed(SCIENTIFIC_SEED, paid_run_authorized=True)" in preflight
    assert "replication_preflight()" in preflight
    assert "protocol_preflight(DATA_DIR)" in preflight
    assert "fingerprint_frozen_corpus(DATA_DIR)" in preflight
    assert "if root.exists():" in preflight
    assert "scientific_seed_consumed\": False" in preflight
    assert "marker_path.exists()" in reserve
    assert "consumed_path.exists()" in reserve
    assert "result_path.exists()" in reserve
    assert "failure_path.exists()" in reserve


def test_workflow_is_owner_only_source_bound_and_budget_bound() -> None:
    workflow = _text(WORKFLOW)
    required = (
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-chm-v1-eiem-optimized-seed-19591-v1]'",
        'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"',
        "git -C source merge-base --is-ancestor 22da370b335727973e76fd20d0a3f87f6bb2d320",
        'test "$(git -C source rev-parse 22da370b335727973e76fd20d0a3f87f6bb2d320^{tree})" = "a217d3f88e6784b491c7f3dc4333a22283c2583f"',
        'git -C source diff --exit-code 22da370b335727973e76fd20d0a3f87f6bb2d320 "$SOURCE_SHA" -- tam_research architectures/chm_v1',
        'test "$(git -C source rev-parse "$SOURCE_SHA:modal_chm_v1_eiem_seed19591_v1.py")" = "$HARNESS_SHA"',
        "5655031401",
        "CHM_V1_959_EXPLICIT_OWNER_AUTHORIZATION_SEED_19591",
        "CHM_V1_966_FINAL_LAUNCHER_AUTHORITY_V1",
        "scientific_seed=19591",
        "max_billed_compute_usd=1.25",
        "modal billing rates --json",
        "one_seed <= 1.25",
        "modal>=1.5.4,<1.6",
    )
    for needle in required:
        assert needle in workflow, needle


def test_workflow_orders_preflight_reservation_and_single_scientific_dispatch() -> None:
    workflow = _text(WORKFLOW)
    preflight = workflow.index("--phase preflight")
    reserve = workflow.index("--phase reserve")
    run = workflow.index("--phase run")
    assert preflight < reserve < run
    assert workflow.count("--phase run") == 1
    assert "--phase run-19592" not in workflow
    assert "--phase run-19593" not in workflow
    assert "run_seed_19592" not in workflow
    assert "run_seed_19593" not in workflow
    assert "Do not retry or redispatch automatically" in workflow
    assert "SEED_CONSUMED.json" in workflow
    assert "ATTEMPT_FAILURE.json" in workflow
    assert "RESULT.json" in workflow
    assert "Seed 19592 remains blocked" in workflow
