from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import tam_research.chm_v1_exact_repo_timing_973 as timing973


def test_issue973_protocol_constants_and_seed_guards() -> None:
    assert timing973.ISSUE == 973
    assert timing973.DIAGNOSTIC_SEEDS == (973_001, 973_002)
    assert timing973.SMOKE_SEED == 973_099

    for seed in timing973.BLOCKED_SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            timing973.validate_seed(seed)

    for seed in timing973.BLOCKED_CONSUMED_DIAGNOSTIC_SEEDS:
        with pytest.raises(RuntimeError, match="consumed #971 diagnostic seed refused"):
            timing973.validate_seed(seed)

    assert timing973.validate_seed(973_001, protocol=True) == 973_001
    assert timing973.validate_seed(973_002, protocol=True) == 973_002
    with pytest.raises(RuntimeError, match="unfrozen #973 protocol seed refused"):
        timing973.validate_seed(timing973.SMOKE_SEED, protocol=True)


def _synthetic_rows(*, warm_near: float, cold_near: float):
    rows = []
    for seed in timing973.DIAGNOSTIC_SEEDS:
        for memory_size in timing973.MEMORY_SIZES:
            for geometry in timing973.GEOMETRIES:
                warm_ratio = warm_near if geometry == "near" else 1.0
                cold_ratio = cold_near if geometry == "near" else 1.0
                rows.append(
                    {
                        "seed": seed,
                        "memory_size": memory_size,
                        "geometry": geometry,
                        "correctness": {
                            "corrected_frozen_full_result_parity": True,
                            "corrected_frozen_value_parity": True,
                            "flat_answer_match_rate": 1.0,
                        },
                        "warm": {"corrected_over_frozen_wall_ratio": warm_ratio},
                        "cold": {"corrected_over_frozen_wall_ratio": cold_ratio},
                    }
                )
    return rows


def test_issue973_decision_logic_pass_amortization_and_stop() -> None:
    passed = timing973.classify_exact_repo_protocol(
        _synthetic_rows(warm_near=0.80, cold_near=1.00)
    )
    assert passed["classification"] == "EXACT_REPO_CORRECTED_BATCHED_SYSTEMS_PASS"
    assert passed["warm_primary_pass"] is True
    assert passed["cold_secondary_pass"] is True

    amortized = timing973.classify_exact_repo_protocol(
        _synthetic_rows(warm_near=0.80, cold_near=1.20)
    )
    assert (
        amortized["classification"]
        == "EXACT_REPO_CORRECTED_BATCHED_AMORTIZATION_REQUIRED"
    )
    assert amortized["warm_primary_pass"] is True
    assert amortized["cold_secondary_pass"] is False

    stopped = timing973.classify_exact_repo_protocol(
        _synthetic_rows(warm_near=0.95, cold_near=0.80)
    )
    assert stopped["classification"] == "EXACT_REPO_CORRECTED_BATCHED_SYSTEMS_STOP"
    assert stopped["warm_primary_pass"] is False


def test_issue973_correctness_failure_overrides_timing() -> None:
    rows = _synthetic_rows(warm_near=0.50, cold_near=0.50)
    rows[0]["correctness"]["flat_answer_match_rate"] = 0.0
    decision = timing973.classify_exact_repo_protocol(rows)
    assert decision["classification"] == "CORRECTNESS_FAIL"
    assert decision["correctness_failures"]


def test_issue973_smoke_uses_only_nonprotocol_seed() -> None:
    row = timing973.benchmark_configuration(
        seed=timing973.SMOKE_SEED,
        memory_size=512,
        geometry="near",
    )
    assert row["seed"] == timing973.SMOKE_SEED
    assert row["correctness"]["corrected_frozen_full_result_parity"] is True
    assert row["correctness"]["corrected_frozen_value_parity"] is True
    assert row["correctness"]["flat_answer_match_rate"] == 1.0


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_issue973_runner_has_no_gpu_modal_or_training_execution_path() -> None:
    source = inspect.getsource(timing973)
    tree = ast.parse(source)

    imported_roots: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name:
                calls.add(name)

    assert "modal" not in imported_roots
    assert all(not name.startswith("modal.") for name in calls)
    assert all(not name.endswith(".backward") and name != "backward" for name in calls)
    assert all("optimizer.step" not in name for name in calls)
    assert all("train_one" not in name for name in calls)


def test_issue973_workflow_is_cpu_one_shot_and_issue_scoped() -> None:
    workflow = Path(".github/workflows/chm-v1-issue973-exact-cpu-timing.yml").read_text(
        encoding="utf-8"
    )
    assert "runs-on: ubuntu-latest" in workflow
    assert "github.event.issue.title == '[chm-v1-exact-cpu-timing-973-v1]'" in workflow
    assert "#973 refuses GitHub Actions rerun attempts" in workflow
    assert "actions/checkout@v6" in workflow
    assert "ref: ${{ steps.authority.outputs.source_sha }}" in workflow
    assert "git rev-parse HEAD" in workflow
    assert "torch==2.10.0" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "modal" not in workflow.lower()
    assert "cuda" not in workflow.lower()
