from __future__ import annotations

"""Zero-credit post-scratch traversal attribution for CHM-v1 successor #953.

This module recreates the exact conceptual #948 scratch traversal internally for
measurement only.  It does not import the closed/unmerged #948 prototype and it
does not modify the frozen :class:`ExactEpisodicIndex` implementation.

The profiler separates scratch lower-bound NumPy arithmetic from sidecar
lookup/dispatch, heap operations, leaf distance work, candidate iteration, and
residual Python control.  All timings are descriptive engineering evidence;
correctness and deterministic call/accounting parity are the CI gates.
"""

import cProfile
from dataclasses import dataclass
import heapq
from itertools import count
import pstats
import time
from typing import Any, Literal

import numpy as np

from .chm_v1_exact_index import ExactEpisodicIndex, KDNode, SearchResult, _squared_distances
from .chm_v1_post_materialization_profiler import build_query_batch, synthetic_state_payload

SCIENTIFIC_SEEDS = (8611, 8612, 8613)
SYNTHETIC_DIAGNOSTIC_SEEDS = (953_001, 953_002)
QueryGeometry = Literal["exact", "near", "random"]

_BUCKET_NAMES = (
    "bound_arithmetic",
    "bound_lookup_dispatch",
    "heap_push",
    "heap_pop",
    "leaf_distance_indexing",
    "candidate_iteration",
    "residual_control",
)


@dataclass(frozen=True)
class TraversalCounts:
    lower_bound_calls: int
    heap_pushes: int
    heap_pops: int
    leaf_calls: int
    candidate_iterations: int
    address_vector_reads: int
    directory_nodes_visited: int


@dataclass(frozen=True)
class ReplayResult:
    result: SearchResult
    counts: TraversalCounts
    total_ns: int
    bucket_ns: dict[str, int]


def validate_diagnostic_seed(seed: int) -> int:
    seed = int(seed)
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError(
            f"scientific seed refused by zero-credit post-scratch profiler: {seed}"
        )
    return seed


def _scratch_bound_arithmetic(
    query_f64: np.ndarray,
    lo_f64: np.ndarray,
    hi_f64: np.ndarray,
    scratch_lo: np.ndarray,
    scratch_hi: np.ndarray,
) -> float:
    """Exact #948 lower-bound arithmetic with no per-node temporary arrays."""

    np.subtract(lo_f64, query_f64, out=scratch_lo)
    np.subtract(query_f64, hi_f64, out=scratch_hi)
    np.maximum(scratch_lo, scratch_hi, out=scratch_lo)
    np.maximum(scratch_lo, 0.0, out=scratch_lo)
    return float(scratch_lo @ scratch_lo)


def _heap_push(queue: list[tuple[float, int, KDNode]], item: tuple[float, int, KDNode]) -> None:
    """Named exact heap wrapper so cProfile call counts are easy to audit."""

    heapq.heappush(queue, item)


def _heap_pop(queue: list[tuple[float, int, KDNode]]) -> tuple[float, int, KDNode]:
    """Named exact heap wrapper so cProfile call counts are easy to audit."""

    return heapq.heappop(queue)


class PostScratchTraversal:
    """Measurement-only recreation of #948 semantics over a frozen exact index."""

    def __init__(self, frozen: ExactEpisodicIndex):
        self.frozen = frozen
        self.points = frozen.points
        self.item_ids = frozen.item_ids
        self._bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def record(node: KDNode) -> None:
            node_id = id(node)
            if node_id in self._bounds:
                raise AssertionError("frozen tree unexpectedly reuses a node object")
            self._bounds[node_id] = (
                np.asarray(node.lo, dtype=np.float64).copy(),
                np.asarray(node.hi, dtype=np.float64).copy(),
            )
            if node.item_positions is None:
                if node.left is None or node.right is None:
                    raise RuntimeError("malformed exact-index node")
                record(node.left)
                record(node.right)

        record(frozen.root)
        self.assert_sidecars_exact()

    def assert_sidecars_exact(self) -> None:
        stack = [self.frozen.root]
        seen = 0
        while stack:
            node = stack.pop()
            lo64, hi64 = self._bounds[id(node)]
            if not np.array_equal(lo64, node.lo.astype(np.float64)):
                raise AssertionError("lo sidecar differs from exact frozen cast")
            if not np.array_equal(hi64, node.hi.astype(np.float64)):
                raise AssertionError("hi sidecar differs from exact frozen cast")
            seen += 1
            if node.item_positions is None:
                if node.left is None or node.right is None:
                    raise RuntimeError("malformed exact-index node")
                stack.extend((node.left, node.right))
        if seen != len(self._bounds):
            raise AssertionError("sidecar/tree node accounting mismatch")

    def _bound(
        self,
        query_f64: np.ndarray,
        node: KDNode,
        scratch_lo: np.ndarray,
        scratch_hi: np.ndarray,
    ) -> float:
        lo64, hi64 = self._bounds[id(node)]
        return _scratch_bound_arithmetic(
            query_f64, lo64, hi64, scratch_lo, scratch_hi
        )

    def indexed_search(self, query: np.ndarray) -> SearchResult:
        q = self.frozen._query(query)
        q64 = np.asarray(q, dtype=np.float64)
        scratch_lo = np.empty_like(q64)
        scratch_hi = np.empty_like(q64)

        best_distance = float("inf")
        best_item_id = np.iinfo(np.int64).max
        best_position = -1
        vector_reads = 0
        node_visits = 0
        serial = count()

        queue: list[tuple[float, int, KDNode]] = [
            (self._bound(q64, self.frozen.root, scratch_lo, scratch_hi), next(serial), self.frozen.root)
        ]

        while queue:
            lower_bound, _, node = _heap_pop(queue)
            if lower_bound > best_distance:
                break
            node_visits += 1

            if node.item_positions is not None:
                positions = node.item_positions
                distances = _squared_distances(self.points[positions], q)
                vector_reads += len(positions)
                for position, distance in zip(positions.tolist(), distances.tolist()):
                    item_id = int(self.item_ids[position])
                    candidate = (float(distance), item_id)
                    if candidate < (best_distance, best_item_id):
                        best_distance, best_item_id = candidate
                        best_position = int(position)
                continue

            if node.left is None or node.right is None:
                raise RuntimeError("malformed exact-index node")
            for child in (node.left, node.right):
                bound = self._bound(q64, child, scratch_lo, scratch_hi)
                if bound <= best_distance:
                    _heap_push(queue, (bound, next(serial), child))

        if best_position < 0:
            raise RuntimeError("post-scratch exact traversal returned no candidate")
        return SearchResult(
            item_id=int(best_item_id),
            position=int(best_position),
            squared_distance=float(best_distance),
            address_vector_reads=int(vector_reads),
            directory_nodes_visited=int(node_visits),
        )

    def replay_with_attribution(self, query: np.ndarray) -> ReplayResult:
        """Replay one exact traversal with disjoint coarse timing buckets."""

        total_started = time.perf_counter_ns()
        q = self.frozen._query(query)
        q64 = np.asarray(q, dtype=np.float64)
        scratch_lo = np.empty_like(q64)
        scratch_hi = np.empty_like(q64)

        best_distance = float("inf")
        best_item_id = np.iinfo(np.int64).max
        best_position = -1
        vector_reads = 0
        node_visits = 0
        serial = count()

        bucket = {name: 0 for name in _BUCKET_NAMES}
        lower_bound_calls = 0
        heap_pushes = 0
        heap_pops = 0
        leaf_calls = 0
        candidate_iterations = 0

        def timed_bound(node: KDNode) -> float:
            nonlocal lower_bound_calls
            bound_started = time.perf_counter_ns()
            lo64, hi64 = self._bounds[id(node)]
            arithmetic_started = time.perf_counter_ns()
            value = _scratch_bound_arithmetic(
                q64, lo64, hi64, scratch_lo, scratch_hi
            )
            arithmetic_elapsed = time.perf_counter_ns() - arithmetic_started
            bound_elapsed = time.perf_counter_ns() - bound_started
            bucket["bound_arithmetic"] += arithmetic_elapsed
            bucket["bound_lookup_dispatch"] += bound_elapsed - arithmetic_elapsed
            lower_bound_calls += 1
            return value

        queue: list[tuple[float, int, KDNode]] = [
            (timed_bound(self.frozen.root), next(serial), self.frozen.root)
        ]

        while queue:
            started = time.perf_counter_ns()
            lower_bound, _, node = _heap_pop(queue)
            bucket["heap_pop"] += time.perf_counter_ns() - started
            heap_pops += 1

            if lower_bound > best_distance:
                break
            node_visits += 1

            if node.item_positions is not None:
                positions = node.item_positions
                started = time.perf_counter_ns()
                block = self.points[positions]
                distances = _squared_distances(block, q)
                bucket["leaf_distance_indexing"] += time.perf_counter_ns() - started
                leaf_calls += 1
                vector_reads += len(positions)

                started = time.perf_counter_ns()
                positions_list = positions.tolist()
                distances_list = distances.tolist()
                for position, distance in zip(positions_list, distances_list):
                    candidate_iterations += 1
                    item_id = int(self.item_ids[position])
                    candidate = (float(distance), item_id)
                    if candidate < (best_distance, best_item_id):
                        best_distance, best_item_id = candidate
                        best_position = int(position)
                bucket["candidate_iteration"] += time.perf_counter_ns() - started
                continue

            if node.left is None or node.right is None:
                raise RuntimeError("malformed exact-index node")
            for child in (node.left, node.right):
                bound = timed_bound(child)
                if bound <= best_distance:
                    started = time.perf_counter_ns()
                    _heap_push(queue, (bound, next(serial), child))
                    bucket["heap_push"] += time.perf_counter_ns() - started
                    heap_pushes += 1

        if best_position < 0:
            raise RuntimeError("post-scratch replay returned no candidate")

        total_elapsed = time.perf_counter_ns() - total_started
        measured = sum(
            bucket[name] for name in _BUCKET_NAMES if name != "residual_control"
        )
        bucket["residual_control"] = int(total_elapsed - measured)

        result = SearchResult(
            item_id=int(best_item_id),
            position=int(best_position),
            squared_distance=float(best_distance),
            address_vector_reads=int(vector_reads),
            directory_nodes_visited=int(node_visits),
        )
        return ReplayResult(
            result=result,
            counts=TraversalCounts(
                lower_bound_calls=int(lower_bound_calls),
                heap_pushes=int(heap_pushes),
                heap_pops=int(heap_pops),
                leaf_calls=int(leaf_calls),
                candidate_iterations=int(candidate_iterations),
                address_vector_reads=int(vector_reads),
                directory_nodes_visited=int(node_visits),
            ),
            total_ns=int(total_elapsed),
            bucket_ns={name: int(bucket[name]) for name in _BUCKET_NAMES},
        )


def _profile_named_calls(
    traversal: PostScratchTraversal,
    queries: list[np.ndarray],
) -> dict[str, Any]:
    profiler = cProfile.Profile()
    profiler.enable()
    for query in queries:
        traversal.indexed_search(query)
    profiler.disable()

    stats = pstats.Stats(profiler).stats
    wanted = {
        "_scratch_bound_arithmetic": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
        "_squared_distances": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
        "_heap_push": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
        "_heap_pop": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
    }
    for (_filename, _line, name), values in stats.items():
        if name not in wanted:
            continue
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = values
        wanted[name]["calls"] += int(total_calls)
        wanted[name]["primitive_calls"] = int(
            wanted[name].get("primitive_calls", 0) + primitive_calls
        )
        wanted[name]["self_seconds"] += float(self_seconds)
        wanted[name]["cumulative_seconds"] += float(cumulative_seconds)
    return wanted


def _sum_counts(counts: list[TraversalCounts]) -> TraversalCounts:
    fields = TraversalCounts.__dataclass_fields__
    return TraversalCounts(
        **{
            field: int(sum(getattr(item, field) for item in counts))
            for field in fields
        }
    )


def profile_post_scratch_microcost(
    *,
    session_length: int,
    query_count: int = 32,
    key_width: int = 32,
    geometry: QueryGeometry = "near",
    noise_std: float = 0.05,
    seed: int = SYNTHETIC_DIAGNOSTIC_SEEDS[0],
) -> dict[str, Any]:
    """Run one deterministic CPU-only post-scratch attribution configuration."""

    seed = validate_diagnostic_seed(seed)
    if session_length < 1 or query_count < 1 or query_count > session_length:
        raise ValueError("invalid session_length/query_count")
    if key_width < 1:
        raise ValueError("key_width must be positive")

    keys, _values = synthetic_state_payload(
        session_length=session_length,
        key_width=key_width,
        value_width=1,
        seed=seed,
    )
    batch = build_query_batch(
        keys,
        query_count=query_count,
        geometry=geometry,
        noise_std=noise_std,
        seed=seed + 1,
    )
    points = keys.detach().float().cpu().numpy().astype(np.float32, copy=False)
    frozen = ExactEpisodicIndex(
        points,
        np.arange(session_length, dtype=np.int64),
        leaf_size=16,
    )
    traversal = PostScratchTraversal(frozen)
    queries = [
        row.detach().float().cpu().numpy().astype(np.float32, copy=False)
        for row in batch.queries
    ]

    frozen_results: list[SearchResult] = []
    scratch_results: list[SearchResult] = []
    flat_results: list[SearchResult] = []
    for query in queries:
        flat = frozen.flat_search(query)
        frozen_result = frozen.indexed_search(query)
        scratch_result = traversal.indexed_search(query)
        if scratch_result != frozen_result:
            raise AssertionError(
                f"scratch/frozen SearchResult mismatch: scratch={scratch_result} frozen={frozen_result}"
            )
        if (scratch_result.item_id, scratch_result.position) != (
            flat.item_id,
            flat.position,
        ):
            raise AssertionError(
                f"scratch/flat answer mismatch: scratch={scratch_result} flat={flat}"
            )
        flat_results.append(flat)
        frozen_results.append(frozen_result)
        scratch_results.append(scratch_result)

    baseline_started = time.perf_counter_ns()
    for query in queries:
        traversal.indexed_search(query)
    unprofiled_scratch_ns = time.perf_counter_ns() - baseline_started

    replays = [traversal.replay_with_attribution(query) for query in queries]
    for replay, frozen_result in zip(replays, frozen_results):
        if replay.result != frozen_result:
            raise AssertionError(
                f"timed replay/frozen mismatch: replay={replay.result} frozen={frozen_result}"
            )

    counts = _sum_counts([replay.counts for replay in replays])
    total_replay_ns = int(sum(replay.total_ns for replay in replays))
    bucket_ns = {
        name: int(sum(replay.bucket_ns[name] for replay in replays))
        for name in _BUCKET_NAMES
    }
    bucket_shares = {
        name: float(bucket_ns[name] / max(total_replay_ns, 1))
        for name in _BUCKET_NAMES
    }

    cprofile = _profile_named_calls(traversal, queries)
    expected_profile_counts = {
        "_scratch_bound_arithmetic": counts.lower_bound_calls,
        "_squared_distances": counts.leaf_calls,
        "_heap_push": counts.heap_pushes,
        "_heap_pop": counts.heap_pops,
    }
    for name, expected in expected_profile_counts.items():
        observed = int(cprofile[name]["calls"])
        if observed != expected:
            raise AssertionError(
                f"cProfile call-count mismatch for {name}: observed={observed} expected={expected}"
            )

    return {
        "measurement_kind": "zero_credit_post_scratch_microcost_cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "timing_is_ci_gate": False,
        "device": "cpu",
        "seed": int(seed),
        "config": {
            "session_length": int(session_length),
            "query_count": int(query_count),
            "key_width": int(key_width),
            "geometry": geometry,
            "noise_std": float(batch.noise_std),
            "leaf_size": 16,
        },
        "correctness": {
            "scratch_frozen_full_searchresult_parity": True,
            "flat_answer_parity": True,
            "queries_checked": int(query_count),
        },
        "counts": {
            field: int(getattr(counts, field))
            for field in TraversalCounts.__dataclass_fields__
        },
        "unprofiled_scratch_ns": int(unprofiled_scratch_ns),
        "semantic_replay_total_ns": int(total_replay_ns),
        "bucket_ns": bucket_ns,
        "bucket_shares": bucket_shares,
        "cprofile": cprofile,
        "cprofile_expected_counts": expected_profile_counts,
        "interpretation_ceiling": (
            "Zero-credit software attribution only; not scientific evidence and not "
            "authority for seed 8612/8613, paid/GPU work, or model training."
        ),
    }


def deterministic_signature(result: dict[str, Any]) -> dict[str, Any]:
    """Strip all timing/profile-duration fields for deterministic CI checks."""

    return {
        "measurement_kind": result["measurement_kind"],
        "scientific_credit": result["scientific_credit"],
        "paid_compute": result["paid_compute"],
        "modal_trigger": result["modal_trigger"],
        "timing_is_ci_gate": result["timing_is_ci_gate"],
        "device": result["device"],
        "seed": result["seed"],
        "config": result["config"],
        "correctness": result["correctness"],
        "counts": result["counts"],
        "cprofile_call_counts": {
            name: int(values["calls"])
            for name, values in result["cprofile"].items()
        },
        "cprofile_expected_counts": result["cprofile_expected_counts"],
        "interpretation_ceiling": result["interpretation_ceiling"],
    }
