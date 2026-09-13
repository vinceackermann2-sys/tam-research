from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pytest

import tam_research.chm_v1_f64_scratch_prototype as prototype_module
from tam_research.chm_v1_exact_index import ExactEpisodicIndex, _lower_bound_sq
from tam_research.chm_v1_f64_scratch_prototype import (
    PreallocatedF64ScratchPrototype,
    benchmark_f64_scratch_once,
    lower_bound_preallocated_f64,
)
from tam_research.chm_v1_post_materialization_profiler import build_query_batch, synthetic_state_payload


def _normalized_points(seed: int, rows: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((rows, width)).astype(np.float32)
    norms = np.linalg.norm(points.astype(np.float64), axis=1, keepdims=True)
    return (points / norms.astype(np.float32)).astype(np.float32)


def test_precomputed_bounds_are_exact_float64_casts_of_frozen_nodes() -> None:
    points = _normalized_points(1, 97, 12)
    frozen = ExactEpisodicIndex(points, np.arange(97, dtype=np.int64), leaf_size=5)
    prototype = PreallocatedF64ScratchPrototype(frozen)

    assert prototype.node_count >= 1
    for node in prototype.iter_nodes():
        lo64, hi64 = prototype.bounds_for_node(node)
        assert lo64.dtype == np.float64
        assert hi64.dtype == np.float64
        np.testing.assert_array_equal(lo64, node.lo.astype(np.float64))
        np.testing.assert_array_equal(hi64, node.hi.astype(np.float64))


def test_scratch_kernel_matches_frozen_bound_exactly() -> None:
    rng = np.random.default_rng(948_100)
    for _ in range(256):
        query = rng.standard_normal(16).astype(np.float32)
        lo = rng.standard_normal(16).astype(np.float32)
        hi = lo + np.abs(rng.standard_normal(16).astype(np.float32))
        q64 = query.astype(np.float64)
        lo64 = lo.astype(np.float64)
        hi64 = hi.astype(np.float64)
        scratch_lo = np.empty_like(q64)
        scratch_hi = np.empty_like(q64)
        observed = lower_bound_preallocated_f64(q64, lo64, hi64, scratch_lo, scratch_hi)
        expected = _lower_bound_sq(query, lo, hi)
        assert observed == expected


def test_exact_near_random_queries_preserve_full_frozen_search_result() -> None:
    keys, _ = synthetic_state_payload(
        session_length=160,
        key_width=16,
        value_width=1,
        seed=948_200,
    )
    points = keys.numpy().astype(np.float32, copy=False)
    frozen = ExactEpisodicIndex(points, np.arange(160, dtype=np.int64), leaf_size=8)
    prototype = PreallocatedF64ScratchPrototype(frozen)

    for geometry in ("exact", "near", "random"):
        batch = build_query_batch(
            keys,
            query_count=16,
            geometry=geometry,
            noise_std=0.05,
            seed=948_201,
        )
        for query_tensor in batch.queries:
            query = query_tensor.numpy().astype(np.float32, copy=False)
            flat = frozen.flat_search(query)
            frozen_indexed = frozen.indexed_search(query)
            prototype_result = prototype.indexed_search(query)
            assert prototype_result == frozen_indexed
            assert (prototype_result.item_id, prototype_result.position) == (
                flat.item_id,
                flat.position,
            )


def test_duplicate_and_equidistant_ties_preserve_frozen_rule() -> None:
    points = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    item_ids = np.asarray([20, 3, 11, 7, 5], dtype=np.int64)
    frozen = ExactEpisodicIndex(points, item_ids, leaf_size=1)
    prototype = PreallocatedF64ScratchPrototype(frozen)

    assert prototype.assert_frozen_parity(np.asarray([1.0, 0.0], dtype=np.float32)).item_id == 3
    assert prototype.assert_frozen_parity(np.asarray([0.0, 0.0], dtype=np.float32)).item_id == 3


def test_one_search_reuses_the_same_two_scratch_arrays_for_all_bounds() -> None:
    points = _normalized_points(4, 128, 16)
    frozen = ExactEpisodicIndex(points, np.arange(128, dtype=np.int64), leaf_size=4)
    prototype = PreallocatedF64ScratchPrototype(frozen)
    query = points[17].copy()

    original = prototype_module.lower_bound_preallocated_f64
    seen_lo: list[int] = []
    seen_hi: list[int] = []

    def recording_kernel(query_f64, lo_f64, hi_f64, scratch_lo, scratch_hi):
        seen_lo.append(id(scratch_lo))
        seen_hi.append(id(scratch_hi))
        return original(query_f64, lo_f64, hi_f64, scratch_lo, scratch_hi)

    with patch.object(prototype_module, "lower_bound_preallocated_f64", recording_kernel):
        result, diagnostics = prototype.indexed_search_with_diagnostics(query)

    assert result == frozen.indexed_search(query)
    assert diagnostics.lower_bound_calls == len(seen_lo) == len(seen_hi)
    assert diagnostics.lower_bound_calls > 1
    assert set(seen_lo) == {diagnostics.scratch_lo_id}
    assert set(seen_hi) == {diagnostics.scratch_hi_id}
    assert diagnostics.scratch_lo_id != diagnostics.scratch_hi_id


def test_benchmark_reports_parity_without_timing_gate() -> None:
    result = benchmark_f64_scratch_once(
        session_length=96,
        query_count=12,
        key_width=8,
        geometry="near",
        noise_std=0.05,
        seed=948_300,
        repeats=1,
    )
    assert result["measurement_kind"] == "zero_credit_systems_prototype_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["timing_is_ci_gate"] is False
    assert result["correctness"]["flat_answer_parity"] is True
    assert result["correctness"]["frozen_indexed_full_accounting_parity"] is True
    assert result["correctness"]["precomputed_bounds_exact"] is True
    accounting = result["accounting"]
    assert 0 < accounting["address_vector_reads"] <= accounting["flat_address_vector_reads"]
    assert accounting["directory_nodes_visited"] > 0
    for key, value in result["timing_ns"].items():
        if key.endswith("_median"):
            assert isinstance(value, int) and value >= 0


def test_refuses_all_frozen_scientific_seeds() -> None:
    for seed in prototype_module.SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            benchmark_f64_scratch_once(
                session_length=16,
                query_count=2,
                key_width=4,
                geometry="exact",
                seed=seed,
                repeats=1,
            )


def test_source_has_no_cuda_modal_or_training_path() -> None:
    source = inspect.getsource(prototype_module).lower()
    assert "torch.cuda" not in source
    assert ".cuda(" not in source
    assert "modal." not in source
    assert "optimizer" not in source
    assert "backward(" not in source
    assert "not scientific evidence" in source
