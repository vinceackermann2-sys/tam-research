from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import tam_research.chm_v1_clipped_bound_prototype as clipped
from tam_research.chm_v1_exact_index import ExactEpisodicIndex, _lower_bound_sq
from tam_research.chm_v1_post_materialization_profiler import (
    build_query_batch,
    synthetic_state_payload,
)


def _index_and_queries(
    *,
    geometry: str,
    seed: int = clipped.SYNTHETIC_PROTOTYPE_SEEDS[0],
    session_length: int = 96,
    query_count: int = 5,
) -> tuple[ExactEpisodicIndex, list[np.ndarray]]:
    keys, _values = synthetic_state_payload(
        session_length=session_length,
        key_width=8,
        value_width=1,
        seed=seed,
    )
    batch = build_query_batch(
        keys,
        query_count=query_count,
        geometry=geometry,
        noise_std=0.05,
        seed=seed + 1,
    )
    points = keys.detach().float().cpu().numpy().astype(np.float32, copy=False)
    index = ExactEpisodicIndex(
        points,
        np.arange(session_length, dtype=np.int64),
        leaf_size=16,
    )
    queries = [
        row.detach().float().cpu().numpy().astype(np.float32, copy=False)
        for row in batch.queries
    ]
    return index, queries


def test_clipped_bound_is_exactly_frozen_on_random_and_boundary_cases() -> None:
    rng = np.random.default_rng(957_777)
    cases: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(1000):
        a = rng.standard_normal(16).astype(np.float32)
        b = rng.standard_normal(16).astype(np.float32)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        q = rng.standard_normal(16).astype(np.float32)
        cases.append((q, lo, hi))

    cases.extend(
        [
            (
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            ),
            (
                np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
                np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            ),
            (
                np.array([3.0, -3.0, 0.5, -0.5], dtype=np.float32),
                np.array([-2.0, -2.0, 0.5, -0.5], dtype=np.float32),
                np.array([2.0, 2.0, 0.5, -0.5], dtype=np.float32),
            ),
            (
                np.array([np.float32(-0.0), 1.0, -1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, -2.0, -1.0], dtype=np.float32),
                np.array([0.0, 2.0, -1.0, 1.0], dtype=np.float32),
            ),
        ]
    )

    for q, lo, hi in cases:
        q64 = q.astype(np.float64)
        lo64 = lo.astype(np.float64)
        hi64 = hi.astype(np.float64)
        scratch = np.empty_like(q64)
        observed = clipped.clipped_lower_bound_f64(q64, lo64, hi64, scratch)
        expected = _lower_bound_sq(q, lo, hi)
        assert observed == expected


def test_sidecars_are_exact_casts_of_frozen_node_bounds() -> None:
    index, _queries = _index_and_queries(geometry="near")
    prototype = clipped.ClippedBoundPrototype(index)
    prototype.assert_sidecars_exact()
    assert prototype.node_count > 0


def test_full_searchresult_parity_across_query_geometries() -> None:
    for geometry in ("exact", "near", "random"):
        index, queries = _index_and_queries(geometry=geometry)
        prototype = clipped.ClippedBoundPrototype(index)
        for query in queries:
            frozen = index.indexed_search(query)
            result = prototype.indexed_search(query)
            flat = index.flat_search(query)
            assert result == frozen
            assert (result.item_id, result.position) == (flat.item_id, flat.position)


def test_duplicate_and_equidistant_tie_order_is_unchanged() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    item_ids = np.asarray([9, 3, 7, 2, 8, 5], dtype=np.int64)
    index = ExactEpisodicIndex(points, item_ids, leaf_size=2)
    prototype = clipped.ClippedBoundPrototype(index)

    for query in (
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 0.0], dtype=np.float32),
    ):
        assert prototype.indexed_search(query) == index.indexed_search(query)
        flat = index.flat_search(query)
        result = prototype.indexed_search(query)
        assert (result.item_id, result.position) == (flat.item_id, flat.position)


def test_benchmark_reports_accounting_parity_without_timing_gate() -> None:
    result = clipped.benchmark_clipped_bound_once(
        session_length=64,
        query_count=4,
        key_width=8,
        geometry="near",
        noise_std=0.05,
        seed=clipped.SYNTHETIC_PROTOTYPE_SEEDS[0],
        repeats=1,
    )
    assert result["measurement_kind"] == "zero_credit_clipped_bound_systems_prototype_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["timing_is_ci_gate"] is False
    assert result["correctness"] == {
        "flat_answer_parity": True,
        "frozen_indexed_full_accounting_parity": True,
        "precomputed_bounds_exact": True,
    }
    timing = result["timing_ns"]
    assert isinstance(timing["frozen_indexed_median"], int)
    assert isinstance(timing["prototype_indexed_median"], int)
    assert timing["frozen_indexed_median"] >= 0
    assert timing["prototype_indexed_median"] >= 0
    assert isinstance(timing["prototype_over_frozen_ratio"], float)


def test_refuses_every_frozen_scientific_seed() -> None:
    for seed in clipped.SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            clipped.benchmark_clipped_bound_once(
                session_length=32,
                query_count=2,
                key_width=4,
                geometry="near",
                seed=seed,
                repeats=1,
            )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_prototype_has_no_executable_gpu_modal_or_training_path() -> None:
    source = inspect.getsource(clipped)
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
    assert all(".cuda" not in name and not name.startswith("cuda") for name in called_names)
    assert all(not name.endswith(".backward") and name != "backward" for name in called_names)
    assert all("optimizer" not in name.lower() for name in called_names)
