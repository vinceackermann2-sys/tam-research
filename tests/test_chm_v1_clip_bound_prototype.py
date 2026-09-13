from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

import tam_research.chm_v1_clip_bound_prototype as prototype
from tam_research.chm_v1_exact_index import ExactEpisodicIndex, _lower_bound_sq
from tam_research.chm_v1_post_materialization_profiler import (
    build_query_batch,
    synthetic_state_payload,
)


def _index_and_queries(
    *,
    geometry: str,
    seed: int = prototype.SYNTHETIC_PROTOTYPE_SEEDS[0],
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


def test_clip_bound_is_bit_exact_to_frozen_random_and_boundaries() -> None:
    rng = np.random.default_rng(955_777)
    cases: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(500):
        a = rng.standard_normal(32).astype(np.float32)
        b = rng.standard_normal(32).astype(np.float32)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        q = rng.standard_normal(32).astype(np.float32)
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
        ]
    )

    for q, lo, hi in cases:
        q64 = q.astype(np.float64)
        lo64 = lo.astype(np.float64)
        hi64 = hi.astype(np.float64)
        scratch = np.empty_like(q64)
        observed = prototype.clip_subtract_dot_lower_bound(
            q64, lo64, hi64, scratch
        )
        expected = _lower_bound_sq(q, lo, hi)
        assert observed == expected


def test_clip_search_matches_full_frozen_result_across_geometries() -> None:
    for geometry in ("exact", "near", "random"):
        index, queries = _index_and_queries(geometry=geometry)
        traversal = prototype.ClipBoundTraversal(index)
        traversal.assert_sidecars_exact()
        for query in queries:
            prototype.assert_node_bound_exactness(traversal, query)
            flat = index.flat_search(query)
            frozen = index.indexed_search(query)
            clip = traversal.indexed_search(query)
            assert clip == frozen
            assert (clip.item_id, clip.position) == (flat.item_id, flat.position)


def test_clip_search_preserves_duplicate_and_equidistant_ties() -> None:
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
    traversal = prototype.ClipBoundTraversal(index)

    for query in (
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 0.0], dtype=np.float32),
    ):
        assert traversal.indexed_search(query) == index.indexed_search(query)
        flat = index.flat_search(query)
        clip = traversal.indexed_search(query)
        assert (clip.item_id, clip.position) == (flat.item_id, flat.position)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_indexed_search_has_one_scratch_allocation_and_clip_kernel_has_none() -> None:
    search_tree = ast.parse(
        textwrap.dedent(inspect.getsource(prototype.ClipBoundTraversal.indexed_search))
    )
    search_calls = [
        _call_name(node.func)
        for node in ast.walk(search_tree)
        if isinstance(node, ast.Call)
    ]
    assert search_calls.count("np.empty_like") == 1

    kernel_tree = ast.parse(
        textwrap.dedent(inspect.getsource(prototype.clip_subtract_dot_lower_bound))
    )
    kernel_calls = [
        _call_name(node.func)
        for node in ast.walk(kernel_tree)
        if isinstance(node, ast.Call)
    ]
    assert "np.empty_like" not in kernel_calls
    assert kernel_calls.count("np.clip") == 1
    assert kernel_calls.count("np.subtract") == 1
    assert "np.maximum" not in kernel_calls


def test_small_benchmark_reports_parity_without_timing_thresholds() -> None:
    result = prototype.benchmark_clip_bound_once(
        session_length=96,
        query_count=5,
        key_width=8,
        geometry="near",
        noise_std=0.05,
        seed=955_123,
        repeats=3,
    )
    assert result["measurement_kind"] == "zero_credit_clip_bound_systems_prototype_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["timing_is_ci_gate"] is False
    assert result["correctness"] == {
        "flat_answer_parity": True,
        "scratch_frozen_full_searchresult_parity": True,
        "clip_frozen_full_searchresult_parity": True,
        "precomputed_sidecars_exact": True,
    }
    assert result["accounting"]["lower_bound_scratch_vectors_per_search"] == 1
    assert result["accounting"]["address_vector_reads"] <= result["accounting"]["flat_address_vector_reads"]
    assert result["accounting"]["directory_nodes_visited"] > 0
    for value in (
        result["timing_ns"]["frozen_indexed_median"],
        result["timing_ns"]["scratch_indexed_median"],
        result["timing_ns"]["clip_indexed_median"],
    ):
        assert isinstance(value, int)
        assert value >= 0


def test_prototype_refuses_all_frozen_scientific_seeds() -> None:
    for seed in prototype.SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            prototype.benchmark_clip_bound_once(
                session_length=32,
                query_count=2,
                key_width=4,
                geometry="near",
                seed=seed,
                repeats=3,
            )


def test_prototype_has_no_executable_gpu_modal_or_training_path() -> None:
    source = inspect.getsource(prototype)
    tree = ast.parse(source)

    imported_roots: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                called_names.add(name)

    assert "modal" not in imported_roots
    assert all(not name.startswith("modal.") for name in called_names)
    assert all(".cuda" not in name and not name.startswith("cuda") for name in called_names)
    assert all(not name.endswith(".backward") and name != "backward" for name in called_names)
    assert all("optimizer" not in name.lower() for name in called_names)
