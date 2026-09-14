from __future__ import annotations

import ast
import inspect

import pytest

import tam_research.chm_v1_corrected_batched_timing as timing


def test_issue_971_protocol_constants_are_frozen() -> None:
    assert timing.ISSUE == 971
    assert timing.DIAGNOSTIC_SEEDS == (971_001, 971_002)
    assert timing.MEMORY_SIZES == (512, 1_024, 4_096)
    assert timing.GEOMETRIES == ("exact", "near", "random")
    assert timing.ADDRESS_WIDTH == 32
    assert timing.VALUE_WIDTH == 256
    assert timing.QUERY_COUNT == 32
    assert timing.NEAR_NOISE_STD == 0.05
    assert timing.REPEATS == 7


def test_issue_971_seed_guard_refuses_all_scientific_seeds() -> None:
    for seed in timing.BLOCKED_SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            timing.validate_systems_seed(seed)
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            timing.validate_systems_seed(seed, protocol=True)

    for seed in timing.DIAGNOSTIC_SEEDS:
        assert timing.validate_systems_seed(seed, protocol=True) == seed

    assert timing.validate_systems_seed(timing.SMOKE_SEED) == timing.SMOKE_SEED
    with pytest.raises(RuntimeError, match="unfrozen #971 protocol seed refused"):
        timing.validate_systems_seed(timing.SMOKE_SEED, protocol=True)


def test_smoke_configuration_gates_correctness_before_descriptive_timing() -> None:
    row = timing.benchmark_configuration(
        seed=timing.SMOKE_SEED,
        memory_size=512,
        geometry="near",
    )

    correctness = row["correctness"]
    assert correctness["corrected_frozen_full_result_parity"] is True
    assert correctness["corrected_frozen_value_parity"] is True
    assert correctness["flat_answer_match_rate"] == 1.0

    for regime in ("warm", "cold"):
        result = row[regime]
        assert result["repeats"] == 7
        assert result["alternating_order"] is True
        assert len(result["frozen_samples"]) == 7
        assert len(result["corrected_samples"]) == 7
        assert result["frozen_median"]["wall_ns"] > 0
        assert result["corrected_median"]["wall_ns"] > 0
        assert result["corrected_over_frozen_wall_ratio"] > 0.0


def _fake_rows(*, warm_near: float, cold_near: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in timing.DIAGNOSTIC_SEEDS:
        for memory_size in timing.MEMORY_SIZES:
            for geometry in timing.GEOMETRIES:
                warm_ratio = warm_near if geometry == "near" else 1.0
                cold_ratio = cold_near if geometry == "near" else 1.0
                rows.append(
                    {
                        "seed": seed,
                        "memory_size": memory_size,
                        "geometry": geometry,
                        "warm": {"corrected_over_frozen_wall_ratio": warm_ratio},
                        "cold": {"corrected_over_frozen_wall_ratio": cold_ratio},
                    }
                )
    return rows


def test_classification_pass_requires_warm_and_cold_gates() -> None:
    rows = _fake_rows(warm_near=0.80, cold_near=1.00)
    decision = timing.classify_protocol(rows)
    assert decision["classification"] == "CORRECTED_BATCHED_SYSTEMS_PASS"
    assert decision["warm_primary_pass"] is True
    assert decision["cold_secondary_pass"] is True


def test_classification_amortization_when_only_cold_gate_fails() -> None:
    rows = _fake_rows(warm_near=0.80, cold_near=1.20)
    decision = timing.classify_protocol(rows)
    assert decision["classification"] == "CORRECTED_BATCHED_AMORTIZATION_REQUIRED"
    assert decision["warm_primary_pass"] is True
    assert decision["cold_secondary_pass"] is False


def test_classification_stop_when_primary_warm_gate_fails() -> None:
    rows = _fake_rows(warm_near=0.95, cold_near=1.00)
    decision = timing.classify_protocol(rows)
    assert decision["classification"] == "CORRECTED_BATCHED_SYSTEMS_STOP"
    assert decision["warm_primary_pass"] is False


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_issue_971_module_has_no_paid_or_training_execution_path() -> None:
    source = inspect.getsource(timing)
    tree = ast.parse(source)

    imported_roots: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name:
                called_names.add(name)

    assert "modal" not in imported_roots
    assert all(not name.startswith("modal.") for name in called_names)
    assert all(not name.endswith(".backward") and name != "backward" for name in called_names)
    assert all("optimizer.step" not in name for name in called_names)
    assert all("train_one" not in name for name in called_names)
    assert all(not name.endswith(".to") for name in called_names)
