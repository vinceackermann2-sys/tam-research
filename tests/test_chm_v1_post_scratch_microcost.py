from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import tam_research.chm_v1_post_scratch_microcost as microcost
from tam_research.chm_v1_exact_index import ExactEpisodicIndex, _lower_bound_sq
from tam_research.chm_v1_post_materialization_profiler import (
    build_query_batch,
    synthetic_state_payload,
)


def _synthetic_index_and_queries(
    *,
    geometry: str,
    seed: int = microcost.SYNTHETIC_DIAGNOSTIC_SEEDS[0],
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


def test_scratch_bound_arithmetic_is_exactly_frozen_for_random_and_boundaries() -> None:
    rng = np.random.default_rng(953_777)
    cases: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(200):
        a = rng.standard_normal(12).astype(np.float32)
        b = rng.standard_normal(12).astype(np.float32)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        q = rng.standard_normal(12).astype(np.float32)
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
        scratch_lo = np.empty_like(q64)
        scratch_hi = np.empty_like(q64)
        observed = microcost._scratch_bound_arithmetic(
            q64, lo64, hi64, scratch_lo, scratch_hi
        )
        expected = _lower_bound_sq(q, lo, hi)
        assert observed == expected


def test_post_scratch_search_matches_full_frozen_result_across_geometries() -> None:
    for geometry in ("exact", "near", "random"):
        index, queries = _synthetic_index_and_queries(geometry=geometry)
        traversal = microcost.PostScratchTraversal(index)
        for query in queries:
            flat = index.flat_search(query)
            frozen = index.indexed_search(query)
            scratch = traversal.indexed_search(query)
            assert scratch == frozen
            assert (scratch.item_id, scratch.position) == (flat.item_id, flat.position)


def test_post_scratch_preserves_duplicate_and_equidistant_tie_order() -> None:
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
    traversal = microcost.PostScratchTraversal(index)

    for query in (
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 0.0], dtype=np.float32),
    ):
        assert traversal.indexed_search(query) == index.indexed_search(query)
        flat = index.flat_search(query)
        scratch = traversal.indexed_search(query)
        assert (scratch.item_id, scratch.position) == (flat.item_id, flat.position)


def test_replay_counts_are_deterministic_and_result_exact() -> None:
    index, queries = _synthetic_index_and_queries(geometry="near", query_count=4)
    traversal = microcost.PostScratchTraversal(index)

    first = [traversal.replay_with_attribution(query) for query in queries]
    second = [traversal.replay_with_attribution(query) for query in queries]

    for query, a, b in zip(queries, first, second):
        frozen = index.indexed_search(query)
        assert a.result == frozen
        assert b.result == frozen
        assert a.counts == b.counts
        assert a.counts.address_vector_reads == frozen.address_vector_reads
        assert a.counts.directory_nodes_visited == frozen.directory_nodes_visited
        assert a.counts.lower_bound_calls > 0
        assert a.counts.heap_pops > 0
        assert a.counts.leaf_calls > 0
        assert a.counts.candidate_iterations == a.counts.address_vector_reads
        assert set(a.bucket_ns) == {
            "bound_arithmetic",
            "bound_lookup_dispatch",
            "heap_push",
            "heap_pop",
            "leaf_distance_indexing",
            "candidate_iteration",
            "residual_control",
        }
        assert isinstance(a.total_ns, int)
        assert a.total_ns >= 0


def test_cprofile_call_counts_match_semantic_replay_counts() -> None:
    result = microcost.profile_post_scratch_microcost(
        session_length=96,
        query_count=5,
        key_width=8,
        geometry="near",
        noise_std=0.05,
        seed=microcost.SYNTHETIC_DIAGNOSTIC_SEEDS[0],
    )

    assert result["correctness"]["scratch_frozen_full_searchresult_parity"] is True
    assert result["correctness"]["flat_answer_parity"] is True
    counts = result["counts"]
    expected = result["cprofile_expected_counts"]
    observed = {name: int(values["calls"]) for name, values in result["cprofile"].items()}

    assert expected["_scratch_bound_arithmetic"] == counts["lower_bound_calls"]
    assert expected["_squared_distances"] == counts["leaf_calls"]
    assert expected["_heap_push"] == counts["heap_pushes"]
    assert expected["_heap_pop"] == counts["heap_pops"]
    assert observed == expected
    assert result["timing_is_ci_gate"] is False


def test_deterministic_signature_excludes_all_timing_noise() -> None:
    kwargs = dict(
        session_length=64,
        query_count=4,
        key_width=8,
        geometry="exact",
        noise_std=0.05,
        seed=microcost.SYNTHETIC_DIAGNOSTIC_SEEDS[0],
    )
    first = microcost.profile_post_scratch_microcost(**kwargs)
    second = microcost.profile_post_scratch_microcost(**kwargs)
    assert microcost.deterministic_signature(first) == microcost.deterministic_signature(second)


def test_profiler_refuses_all_frozen_scientific_seeds() -> None:
    for seed in microcost.SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            microcost.profile_post_scratch_microcost(
                session_length=32,
                query_count=2,
                key_width=4,
                geometry="near",
                seed=seed,
            )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_profiler_has_no_executable_gpu_modal_or_training_path() -> None:
    source = inspect.getsource(microcost)
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
