from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_LAUNCHER = ROOT / "modal_chm_v1_eiem_seed8611_v1.py"
V2_LAUNCHER = ROOT / "modal_chm_v1_eiem_seed8611_v2.py"
V2_WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-eiem-seed8611-v2.yml"


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


def test_v2_is_fresh_namespace_with_unit_correct_reference_cost() -> None:
    source = _text(V2_LAUNCHER)
    ast.parse(source)

    required = (
        'PHASE = "chm-v1-eiem-seed-8611-v2"',
        'TRIGGER_TITLE = "[modal-chm-v1-eiem-seed-8611-v2]"',
        'RESULT_ROOT = "/vol/chm-v1/eiem-small-lm/issue-903/seed-8611-v2"',
        'APP_NAME = "chm-v1-eiem-small-lm-seed-8611-v2"',
        "SCIENTIFIC_SEED = 8611",
        'GPU_CLASS = "L4"',
        "CPU_CORES = 4",
        "RAM_MIB = 8192",
        "MAX_SECONDS_PER_SEED = 3600",
        "MAX_AGGREGATE_BILLED_COMPUTE_USD = 4.0",
        "L4_USD_PER_HOUR = 0.80000",
        "PHYSICAL_CPU_USD_PER_CORE_HOUR = 0.04730",
        "RAM_USD_PER_GIB_HOUR = 0.00800",
        '"v1_preallocation_failure_run": 34686481894',
    )
    for needle in required:
        assert needle in source, needle

    one_seed = 0.8 + 4 * 0.0473 + 8 * 0.008
    assert abs(one_seed - 1.0532) < 1e-12
    assert abs(3 * one_seed - 3.1596) < 1e-12
    assert 3 * one_seed <= 4.0


def test_v2_preserves_frozen_scientific_helper_implementation() -> None:
    v1 = _text(V1_LAUNCHER)
    v2 = _text(V2_LAUNCHER)
    frozen_helpers = (
        "_finite_model",
        "_autocast",
        "_empty_stats",
        "_merge_stats",
        "_evaluate_probes",
        "_run_eiem_batch",
        "_evaluate_eiem_language_batched",
        "_ordinary_indexed_exactness_audit",
        "_time_local_inputs",
        "_time_eiem_inputs",
        "_systems_timing",
    )
    for name in frozen_helpers:
        assert _function_source(v2, name) == _function_source(v1, name), name


def test_v2_only_seed8611_can_request_gpu_and_consumption_precedes_training() -> None:
    source = _text(V2_LAUNCHER)
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
    assert run.index("_atomic_write(consumed_path, consumed)") < run.index("train_one(")
    assert '"automatic_retry_authorized": False' in run
    assert '"next_seed_authorized_automatically": False' in run
    assert "ATTEMPT_FAILURE.json" in run
    assert "run_seed_8612" not in source
    assert "run_seed_8613" not in source


def test_v2_workflow_parses_live_hourly_rates_instead_of_per_second_literals() -> None:
    workflow = _text(V2_WORKFLOW)
    for needle in (
        "modal billing rates --json > modal-rates.json",
        'rates["gpu_hour_cost_l4"]',
        'rates["cpu_hour_cost"]',
        'rates["mem_gib_hour_cost"]',
        "one_seed = hours_per_seed * (l4 + 4 * cpu + 8 * ram)",
        "three_seed = 3 * one_seed",
        "assert three_seed <= 4.0",
        'Path("modal-rate-envelope.json").write_text',
    ):
        assert needle in workflow, needle

    for stale_literal in ("0.000222", "0.0000131", "0.00000222"):
        assert stale_literal not in workflow


def test_v2_workflow_requires_abandoned_v1_and_unique_v2_authority() -> None:
    workflow = _text(V2_WORKFLOW)
    required = (
        "github.event.issue.user.login == github.repository_owner",
        "github.event.issue.title == '[modal-chm-v1-eiem-seed-8611-v2]'",
        '"abandoned_trigger_issue": 919',
        '"abandoned_workflow_run": 34686481894',
        "CHM_V1_SEED_8611_V1_PREALLOCATION_FAILURE",
        "CHM_V1_FINAL_LAUNCHER_AUTHORITY_V2",
        '"workflow_run=34686481894"',
        '"zero_gpu_preflight_started=false"',
        '"dispatch_reserved=false"',
        '"gpu_allocated=false"',
        '"scientific_seed_8611_consumed=false"',
        '"v1_trigger_reuse=false"',
        'f"source_sha={source_sha}"',
        'f"source_tree={source_tree}"',
        'f"harness_blob={harness_sha}"',
        '"ci_status=PASS"',
        '"v1_preallocation_failure_run=34686481894"',
        'assert abandoned["state"] == "closed"',
    )
    for needle in required:
        assert needle in workflow, needle


def test_v2_workflow_orders_single_use_phases_and_quotes_failure_comment_safely() -> None:
    workflow = _text(V2_WORKFLOW)
    preflight = workflow.index("--phase preflight")
    reserve = workflow.index("--phase reserve")
    run = workflow.index("--phase run")
    assert preflight < reserve < run
    assert workflow.count("--phase run") == 1
    assert "--phase run-8612" not in workflow
    assert "--phase run-8613" not in workflow
    assert "run_seed_8612" not in workflow
    assert "run_seed_8613" not in workflow

    failure = workflow[workflow.index("Record failed V2 workflow without retrying") :]
    assert "cat > failure-comment.md <<'EOF'" in failure
    assert "`/vol/chm-v1/eiem-small-lm/issue-903/seed-8611-v2`" in failure
    assert "`SEED_CONSUMED.json`" in failure
    assert "`ATTEMPT_FAILURE.json`" in failure
    assert "`RESULT.json`" in failure
    assert '-f body="$(cat failure-comment.md)"' in failure
