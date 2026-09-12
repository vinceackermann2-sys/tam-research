from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_chm_v1_eiem_seed8611_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-eiem-seed8611-v1.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_seed8611_launcher_parses_and_freezes_authorized_envelope() -> None:
    source = _text(LAUNCHER)
    tree = ast.parse(source)
    assert tree is not None

    required = (
        'PHASE = "chm-v1-eiem-seed-8611-v1"',
        'TRIGGER_TITLE = "[modal-chm-v1-eiem-seed-8611-v1]"',
        "RESEARCH_ISSUE = 854",
        "RUN_CONTROL_ISSUE = 903",
        "AUTHORIZATION_COMMENT_ID = 5_644_581_046",
        'SCIENTIFIC_AUTHORITY_SHA = "e86ffd453be3efc3026576f5ebbab6b76ab54b95"',
        "SCIENTIFIC_SEED = 8611",
        'RESULT_ROOT = "/vol/chm-v1/eiem-small-lm/issue-903/seed-8611-v1"',
        'GPU_CLASS = "L4"',
        "CPU_CORES = 4",
        "RAM_MIB = 8192",
        "MAX_SECONDS_PER_SEED = 3600",
        "MAX_AGGREGATE_BILLED_COMPUTE_USD = 4.0",
        "LONG_MEMORY_EVALUATOR_SEED = 8_540_911",
        "VALIDATION_SAMPLING_SEED = 8_540_912",
        "SYSTEMS_TIMING_SEED = 8_540_913",
        "VALIDATION_BATCHES = 64",
        "VALIDATION_BATCH_SIZE = 8",
        "SESSION_TOKENS = 1024",
        "TIMING_BATCH1_SESSIONS = 8",
        "TIMING_THROUGHPUT_BATCHES = 8",
        "TIMING_THROUGHPUT_BATCH_SIZE = 8",
    )
    for needle in required:
        assert needle in source, needle

    one_seed = 3600 * (0.000222 + 4 * 0.0000131 + 8 * 0.00000222)
    assert abs(one_seed - 1.051776) < 1e-12
    assert abs(3 * one_seed - 3.155328) < 1e-12
    assert 3 * one_seed <= 4.0


def test_only_seed8611_scientific_function_can_request_gpu() -> None:
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

    assert gpu_decorated == [("run_seed_8611", "GPU_CLASS")]

    run = _function_source(source, "run_seed_8611")
    assert "SEED_CONSUMED.json" in run
    assert "ATTEMPT_FAILURE.json" in run
    assert '"automatic_retry_authorized": False' in run
    assert '"next_seed_authorized_automatically": False' in run
    assert run.index("_atomic_write(consumed_path, consumed)") < run.index("train_one(")
    assert "gpu_allocation_started" in run
    assert "retries=0" in source
    assert "timeout=MAX_SECONDS_PER_SEED" in source
    assert "run_seed_8612" not in source
    assert "run_seed_8613" not in source


def test_frozen_scientific_measurement_envelope_is_present() -> None:
    source = _text(LAUNCHER)

    for needle in (
        "_ordinary_indexed_exactness_audit",
        "_systems_timing",
        'mode="flat"',
        'mode="indexed"',
        "verify_indexed_exactness=True",
        "verify_indexed_exactness=False",
        "torch.allclose",
        "atol=1e-5",
        "524_288",
        "TIMING_BATCH1_SESSIONS",
        "TIMING_THROUGHPUT_BATCHES",
        "TIMING_THROUGHPUT_BATCH_SIZE",
        'immediate_stop_reasons.append("sparse_read_fraction_gt_0.25")',
        "combined_exact_match_rate",
        "ordinary_max_abs_logit_delta",
        'torch.autocast(device_type="cuda", dtype=torch.bfloat16)',
    ):
        assert needle in source, needle

    exactness = _function_source(source, "_ordinary_indexed_exactness_audit")
    assert "VALIDATION_SAMPLING_SEED" in exactness
    assert "VALIDATION_BATCH_SIZE" in exactness
    assert "SESSION_TOKENS" in exactness
    assert 'mode="flat"' in exactness
    assert 'mode="indexed"' in exactness
    assert "verify_indexed_exactness=True" in exactness

    timing = _function_source(source, "_systems_timing")
    assert "SYSTEMS_TIMING_SEED" in timing
    assert "TIMING_BATCH1_SESSIONS" in timing
    assert "TIMING_THROUGHPUT_BATCHES" in timing
    assert "TIMING_THROUGHPUT_BATCH_SIZE" in timing
    assert 'mode="flat"' in timing
    assert 'mode="indexed"' in timing


def test_workflow_is_owner_only_source_bound_and_post_merge_authority_bound() -> None:
    workflow = _text(WORKFLOW)

    required = (
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-chm-v1-eiem-seed-8611-v1]'",
        'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"',
        "git -C source merge-base --is-ancestor e86ffd453be3efc3026576f5ebbab6b76ab54b95",
        'git -C source diff --exit-code e86ffd453be3efc3026576f5ebbab6b76ab54b95 "$SOURCE_SHA" -- tam_research architectures/chm_v1',
        'test "$(git -C source rev-parse "$SOURCE_SHA:modal_chm_v1_eiem_seed8611_v1.py")" = "$HARNESS_SHA"',
        "CHM_V1_FINAL_LAUNCHER_AUTHORITY_V1",
        'f"source_sha={source_sha}"',
        'f"source_tree={source_tree}"',
        'f"harness_blob={harness_sha}"',
        '"ci_status=PASS"',
        '"ci_url=https://github.com/"',
        'assert trigger["created_at"] > authority["created_at"]',
        "modal billing rates --json",
        "modal>=1.5.4,<1.6",
        "max $4.00 aggregate billed compute",
    )
    for needle in required:
        assert needle in workflow, needle


def test_workflow_orders_preflight_reservation_and_single_scientific_dispatch() -> None:
    workflow = _text(WORKFLOW)

    preflight = workflow.index("--phase preflight")
    reserve = workflow.index("--phase reserve")
    run = workflow.index("--phase run")
    assert preflight < reserve < run

    # There is only one paid scientific phase and it is seed 8611.  No subsequent
    # seed can be chained from this workflow.
    assert workflow.count("--phase run") == 1
    assert "8612" not in workflow
    assert "8613" not in workflow
    assert "Do not retry or redispatch automatically" in workflow
    assert "SEED_CONSUMED.json" in workflow
    assert "ATTEMPT_FAILURE.json" in workflow
    assert "RESULT.json" in workflow
