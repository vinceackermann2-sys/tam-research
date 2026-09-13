from __future__ import annotations

import inspect

import numpy as np
import pytest

import tam_research.chm_v1_f64_bound_prototype as prototype_module
from tam_research.chm_v1_exact_index import ExactEpisodicIndex, _lower_bound_sq
from tam_research.chm_v1_f64_bound_prototype import (
    PrecomputedF64BoundPrototype,
    benchmark_f64_bound_once,
    lower_bound_precomputed_f64,
)


def _normalized_points(seed: int, rows: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((rows, width)).astype(np.float32)
    norms = np.linalg.norm(points.astype(np.float64), axis=1, keepdims=True)
    return (points / norms.astype(np.float32)).astype(np.float32)


def _query_sets(points: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    exact = points[:8].copy()

    near = exact + rng.normal(0.0, 0.05, exact.shape).astype(np.float32)
    near_norms = np.linalg.norm(near.astype(np.float64), axis=1, keepdims=True)
    near = (near / near_norms.astype(np.float32)).astype(np.float32)

    random = rng.standard_normal(exact.shape).astype(np.float32)
    random_norms = np.linalg.norm(random.astype(np.float64), axis=1, keepdims=True)
    random = (random / random_norms.astype(np.float32)).astype(np.float32)

    return {"exact": exact, "near": near, "random": random}


def test_precomputed_storage_is_exact_float64_cast_of_frozen_bounds() -> None:
    points = _normalized_points(1, 97, 12)
    frozen = ExactEpisodicIndex(points, np.arange(97, dtype=np.int64), leaf_size=5)
    prototype = PrecomputedF64BoundPrototype(frozen)

    assert prototype.node_count > 1
    prototype.assert_storage_exact()
    for node in prototype.iter_nodes():
        lo64, hi64 = prototype.bounds_for_node(node)
        assert lo64.dtype == np.float64
        assert hi64.dtype == np.float64
        np.testing.assert_array_equal(lo64, node.lo.astype(np.float64))
        np.testing.assert_array_equal(hi64, node.hi.astype(np.float64))


def test_precomputed_lower_bounds_match_frozen_scalar_exactly() -> None:
    points = _normalized_points(2, 128, 16)
    frozen = ExactEpisodicIndex(points, np.arange(128, dtype=np.int64), leaf_size=4)
    prototype = PrecomputedF64BoundPrototype(frozen)
    query_sets = _query_sets(points, seed=3)

    nodes = prototype.iter_nodes()
    assert nodes
    for queries in query_sets.values():
        for query in queries:
            q = frozen._query(query)
            q64 = q.astype(np.float64)
            for node in nodes[:32]:
                lo64, hi64 = prototype.bounds_for_node(node)
                expected = _lower_bound_sq(q, node.lo, node.hi)
                actual = lower_bound_precomputed_f64(q64, lo64, hi64)
                assert actual == expected


def test_boundary_and_equidistant_lower_bounds_preserve_exact_values() -> None:
    points = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    frozen = ExactEpisodicIndex(points, np.arange(5, dtype=np.int64), leaf_size=1)
    prototype = PrecomputedF64BoundPrototype(frozen)
    root = frozen.root
    lo64, hi64 = prototype.bounds_for_node(root)

    queries = (
        root.lo.copy(),
        root.hi.copy(),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([2.0, 0.0], dtype=np.float32),
        np.asarray([0.0, -2.0], dtype=np.float32),
    )
    for query in queries:
        q = frozen._query(query)
        assert lower_bound_precomputed_f64(q.astype(np.float64), lo64, hi64) == _lower_bound_sq(
            q, root.lo, root.hi
        )


def test_exact_near_random_queries_match_frozen_full_search_result() -> None:
    points = _normalized_points(4, 192, 24)
    item_ids = np.arange(5000, 5192, dtype=np.int64)
    frozen = ExactEpisodicIndex(points, item_ids, leaf_size=8)
    prototype = PrecomputedF64BoundPrototype(frozen)

    for queries in _query_sets(points, seed=5).values():
        for query in queries:
            flat = frozen.flat_search(query)
            frozen_indexed = frozen.indexed_search(query)
            result = prototype.indexed_search(query)

            assert result == frozen_indexed
            assert (result.item_id, result.position) == (flat.item_id, flat.position)
            assert result.squared_distance == frozen_indexed.squared_distance
            assert result.address_vector_reads == frozen_indexed.address_vector_reads
            assert result.directory_nodes_visited == frozen_indexed.directory_nodes_visited


def test_duplicate_and_global_equidistant_ties_preserve_smallest_item_id() -> None:
    points = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    item_ids = np.asarray([20, 3, 11, 7, 5], dtype=np.int64)
    frozen = ExactEpisodicIndex(points, item_ids, leaf_size=1)
    prototype = PrecomputedF64BoundPrototype(frozen)

    duplicate = prototype.assert_frozen_parity(np.asarray([1.0, 0.0], dtype=np.float32))
    equidistant = prototype.assert_frozen_parity(np.asarray([0.0, 0.0], dtype=np.float32))
    assert duplicate.item_id == 3
    assert equidistant.item_id == 3


def test_benchmark_reports_parity_and_keeps_timing_descriptive_only() -> None:
    result = benchmark_f64_bound_once(
        session_length=96,
        query_count=12,
        key_width=8,
        geometry="near",
        noise_std=0.05,
        seed=946_101,
        repeats=2,
    )

    assert result["measurement_kind"] == "zero_credit_systems_prototype_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["timing_is_ci_gate"] is False
    assert result["correctness"] == {
        "flat_answer_parity": True,
        "frozen_indexed_full_accounting_parity": True,
        "precomputed_bounds_exact": True,
    }

    accounting = result["accounting"]
    assert 0 < accounting["address_vector_reads"] <= accounting["flat_address_vector_reads"]
    assert 0.0 < accounting["address_read_fraction"] <= 1.0
    assert accounting["directory_nodes_visited"] > 0
    assert accounting["directory_nodes_per_query"] > 0
    assert accounting["precomputed_bound_nodes"] > 0

    timing = result["timing_ns"]
    assert isinstance(timing["frozen_indexed_median"], int)
    assert isinstance(timing["prototype_indexed_median"], int)
    assert timing["frozen_indexed_median"] >= 0
    assert timing["prototype_indexed_median"] >= 0
    assert isinstance(timing["prototype_over_frozen_ratio"], float)
    assert timing["prototype_over_frozen_ratio"] >= 0.0


def test_benchmark_refuses_all_frozen_scientific_seeds() -> None:
    for seed in prototype_module.SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            benchmark_f64_bound_once(
                session_length=16,
                query_count=2,
                key_width=4,
                geometry="exact",
                seed=seed,
                repeats=1,
            )


def test_issue_946_synthetic_seed_family_is_fixed_and_non_scientific() -> None:
    assert prototype_module.SYNTHETIC_PROTOTYPE_SEEDS == (946_001, 946_002)
    assert not set(prototype_module.SYNTHETIC_PROTOTYPE_SEEDS).intersection(
        prototype_module.SCIENTIFIC_SEEDS
    )


def test_prototype_source_exposes_no_gpu_paid_or_training_execution_path() -> None:
    source = inspect.getsource(prototype_module).lower()
    assert "import modal" not in source
    assert "modal." not in source
    assert "torch.cuda" not in source
    assert ".cuda(" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source
    assert "scientific_credit\": true" not in source
    assert "modal_trigger\": true" not in source
