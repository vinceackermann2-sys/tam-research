from __future__ import annotations

import inspect

import numpy as np
import pytest

import tam_research.chm_v1_array_traversal_prototype as prototype_module
from tam_research.chm_v1_array_traversal_prototype import (
    ArrayBackedExactTraversalPrototype,
    benchmark_array_traversal_once,
    make_queries,
)
from tam_research.chm_v1_exact_index import ExactEpisodicIndex, _lower_bound_sq


def _normalized_points(seed: int, rows: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((rows, width)).astype(np.float32)
    norms = np.linalg.norm(points.astype(np.float64), axis=1, keepdims=True)
    return (points / norms.astype(np.float32)).astype(np.float32)


def test_flattened_tree_preserves_frozen_leaf_partition() -> None:
    points = _normalized_points(1, 73, 8)
    frozen = ExactEpisodicIndex(points, np.arange(73, dtype=np.int64), leaf_size=5)
    prototype = ArrayBackedExactTraversalPrototype(frozen)

    assert prototype.node_count >= prototype.leaf_count >= 1
    assert prototype.node_lo.dtype == np.float32
    assert prototype.node_hi.dtype == np.float32
    assert prototype.left.dtype == np.int64
    assert prototype.right.dtype == np.int64
    assert prototype.leaf_positions.dtype == np.int64
    assert sorted(prototype.leaf_positions.tolist()) == list(range(73))
    assert int(prototype.leaf_length.sum()) == 73


def test_vectorized_sibling_bounds_match_frozen_scalar_reduction_exactly() -> None:
    points = _normalized_points(2, 96, 12)
    frozen = ExactEpisodicIndex(points, np.arange(96, dtype=np.int64), leaf_size=4)
    prototype = ArrayBackedExactTraversalPrototype(frozen)
    queries = _normalized_points(3, 7, 12)

    internal_nodes = np.flatnonzero(prototype.leaf_length == 0).tolist()
    assert internal_nodes
    for node_index in internal_nodes[:20]:
        left_index = int(prototype.left[node_index])
        right_index = int(prototype.right[node_index])
        for query in queries:
            left_bound, right_bound = prototype.child_bounds(query, node_index)
            expected_left = _lower_bound_sq(
                query, prototype.node_lo[left_index], prototype.node_hi[left_index]
            )
            expected_right = _lower_bound_sq(
                query, prototype.node_lo[right_index], prototype.node_hi[right_index]
            )
            assert left_bound == expected_left
            assert right_bound == expected_right


def test_exact_near_random_queries_match_frozen_index_and_accounting() -> None:
    points = _normalized_points(4, 128, 16)
    item_ids = np.arange(1000, 1128, dtype=np.int64)
    frozen = ExactEpisodicIndex(points, item_ids, leaf_size=8)
    prototype = ArrayBackedExactTraversalPrototype(frozen)

    for geometry in ("exact", "near", "random"):
        queries = make_queries(
            points,
            query_count=16,
            geometry=geometry,
            noise=0.05,
            seed=936_100,
        )
        for query in queries:
            flat = frozen.flat_search(query)
            frozen_indexed = frozen.indexed_search(query)
            array_result, diagnostics = prototype.indexed_search_with_diagnostics(query)

            assert (array_result.item_id, array_result.position) == (
                flat.item_id,
                flat.position,
            )
            assert array_result == frozen_indexed
            assert diagnostics.node_visits == array_result.directory_nodes_visited
            assert diagnostics.child_bounds_evaluated == 2 * diagnostics.child_bound_batches
            assert diagnostics.heap_pushes >= diagnostics.node_visits
            assert diagnostics.heap_pops >= diagnostics.node_visits


def test_duplicate_and_equidistant_ties_preserve_smallest_item_id_rule() -> None:
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
    prototype = ArrayBackedExactTraversalPrototype(frozen)

    duplicate_query = np.asarray([1.0, 0.0], dtype=np.float32)
    duplicate_result = prototype.assert_frozen_parity(duplicate_query)
    assert duplicate_result.item_id == 3

    # The origin is equidistant from every point. The shared ordering contract
    # must therefore select the globally smallest item ID.
    equidistant_query = np.asarray([0.0, 0.0], dtype=np.float32)
    equidistant_result = prototype.assert_frozen_parity(equidistant_query)
    assert equidistant_result.item_id == 3


def test_query_generation_is_deterministic_and_normalized() -> None:
    points = _normalized_points(5, 64, 8)
    for geometry in ("exact", "near", "random"):
        first = make_queries(
            points,
            query_count=8,
            geometry=geometry,
            seed=936_200,
            noise=0.05,
        )
        second = make_queries(
            points,
            query_count=8,
            geometry=geometry,
            seed=936_200,
            noise=0.05,
        )
        np.testing.assert_array_equal(first, second)
        norms = np.linalg.norm(first.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-6, atol=1e-6)


def test_benchmark_reports_correctness_accounting_without_timing_gate() -> None:
    result = benchmark_array_traversal_once(
        session_length=96,
        query_count=12,
        key_width=8,
        geometry="near",
        noise=0.05,
        seed=936_300,
        leaf_size=4,
    )

    assert result["measurement_kind"] == "engineering_only_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["correctness"]["flat_answer_parity"] is True
    assert result["correctness"]["frozen_indexed_full_accounting_parity"] is True

    accounting = result["accounting"]
    assert 0 < accounting["address_vector_reads"] <= accounting["flat_address_vector_reads"]
    assert 0.0 < accounting["address_read_fraction"] <= 1.0
    assert accounting["directory_nodes_visited"] > 0
    assert accounting["child_bounds_evaluated"] == 2 * accounting["child_bound_batches"]
    assert accounting["array_node_count"] >= accounting["array_leaf_count"] >= 1

    # Timing exists for descriptive measurement but correctness never depends
    # on a speed threshold.
    for elapsed_ns in result["timing_ns"].values():
        assert isinstance(elapsed_ns, int)
        assert elapsed_ns >= 0


def test_benchmark_refuses_all_frozen_scientific_seeds() -> None:
    for seed in prototype_module.SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            benchmark_array_traversal_once(
                session_length=16,
                query_count=2,
                key_width=4,
                geometry="exact",
                seed=seed,
                leaf_size=2,
            )


def test_prototype_source_has_no_cuda_modal_or_model_training_path() -> None:
    source = inspect.getsource(prototype_module).lower()
    assert "torch.cuda" not in source
    assert ".cuda(" not in source
    assert "modal." not in source
    assert "optimizer" not in source
    assert "backward(" not in source
    assert "not scientific evidence" in prototype_module.benchmark_array_traversal_once.__doc__.lower() if prototype_module.benchmark_array_traversal_once.__doc__ else True
