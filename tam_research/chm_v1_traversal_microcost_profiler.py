from __future__ import annotations

"""Zero-credit traversal micro-cost profiler for CHM-v1 successor #938.

This module is additive diagnostics only. It does not modify the frozen exact
index or the negative #936 prototype. The frozen algorithm is replayed with
explicit component timers to obtain deterministic operation counts, while
cProfile and isolated replay microbenchmarks provide descriptive timing views.
No scientific seed, GPU path, paid compute, or model training is reachable here.
"""

import cProfile
from dataclasses import asdict, dataclass
import heapq
from itertools import count
import pstats
import time
from typing import Any, Literal

import numpy as np

from .chm_v1_array_traversal_prototype import ArrayBackedExactTraversalPrototype
from .chm_v1_exact_index import (
    ExactEpisodicIndex,
    KDNode,
    SearchResult,
    _lower_bound_sq,
    _squared_distances,
)
from .chm_v1_post_materialization_profiler import (
    build_query_batch,
    synthetic_state_payload,
)

SCIENTIFIC_SEEDS = (8611, 8612, 8613)
SYNTHETIC_MICROCOST_SEED = 938_001
QueryGeometry = Literal["exact", "near", "random"]


@dataclass(frozen=True)
class TraceCounts:
    lower_bound_calls: int
    heap_push_calls: int
    heap_pop_calls: int
    leaf_visits: int
    vector_reads: int
    candidate_iterations: int
    node_visits: int
    internal_node_visits: int


@dataclass(frozen=True)
class TraceTiming:
    total_ns: int
    lower_bound_ns: int
    heap_push_ns: int
    heap_pop_ns: int
    leaf_distance_and_index_ns: int
    candidate_loop_ns: int
    residual_ns: int


@dataclass(frozen=True)
class TraceReplay:
    result: SearchResult
    counts: TraceCounts
    timing: TraceTiming
    bound_values: tuple[float, ...]
    first_leaf_positions: tuple[int, ...]


def validate_microcost_seed(seed: int) -> int:
    seed = int(seed)
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError(f"scientific seed refused by zero-credit micro-cost profiler: {seed}")
    return seed


def _identity(result: SearchResult) -> tuple[int, int, float, int, int]:
    return (
        int(result.item_id),
        int(result.position),
        float(result.squared_distance),
        int(result.address_vector_reads),
        int(result.directory_nodes_visited),
    )


def trace_indexed_search(
    index: ExactEpisodicIndex,
    query: np.ndarray,
    *,
    max_bound_samples: int = 512,
) -> TraceReplay:
    """Replay frozen indexed_search semantics with explicit component timers.

    The result and accounting must match the frozen implementation exactly.
    Timings are descriptive because the timer calls perturb the replay.
    """

    started_total = time.perf_counter_ns()
    q = index._query(query)
    best_distance = float("inf")
    best_item_id = np.iinfo(np.int64).max
    best_position = -1
    vector_reads = 0
    node_visits = 0
    internal_node_visits = 0
    leaf_visits = 0
    candidate_iterations = 0
    lower_bound_calls = 0
    heap_push_calls = 0
    heap_pop_calls = 0

    lower_bound_ns = 0
    heap_push_ns = 0
    heap_pop_ns = 0
    leaf_distance_and_index_ns = 0
    candidate_loop_ns = 0

    bound_values: list[float] = []
    first_leaf_positions: tuple[int, ...] = ()
    serial = count()

    started = time.perf_counter_ns()
    root_bound = _lower_bound_sq(q, index.root.lo, index.root.hi)
    lower_bound_ns += time.perf_counter_ns() - started
    lower_bound_calls += 1
    bound_values.append(root_bound)
    queue: list[tuple[float, int, KDNode]] = [(root_bound, next(serial), index.root)]

    while queue:
        started = time.perf_counter_ns()
        lower_bound, _, node = heapq.heappop(queue)
        heap_pop_ns += time.perf_counter_ns() - started
        heap_pop_calls += 1

        if lower_bound > best_distance:
            break

        node_visits += 1
        if node.item_positions is not None:
            leaf_visits += 1
            positions = node.item_positions
            if not first_leaf_positions:
                first_leaf_positions = tuple(int(x) for x in positions.tolist())

            started = time.perf_counter_ns()
            distances = _squared_distances(index.points[positions], q)
            leaf_distance_and_index_ns += time.perf_counter_ns() - started
            vector_reads += len(positions)

            started = time.perf_counter_ns()
            position_list = positions.tolist()
            distance_list = distances.tolist()
            candidate_iterations += len(position_list)
            for position, distance in zip(position_list, distance_list):
                item_id = int(index.item_ids[position])
                candidate = (float(distance), item_id)
                if candidate < (best_distance, best_item_id):
                    best_distance, best_item_id = candidate
                    best_position = int(position)
            candidate_loop_ns += time.perf_counter_ns() - started
            continue

        internal_node_visits += 1
        if node.left is None or node.right is None:
            raise RuntimeError("malformed exact index node during trace")
        for child in (node.left, node.right):
            started = time.perf_counter_ns()
            bound = _lower_bound_sq(q, child.lo, child.hi)
            lower_bound_ns += time.perf_counter_ns() - started
            lower_bound_calls += 1
            if len(bound_values) < max_bound_samples:
                bound_values.append(bound)
            if bound <= best_distance:
                started = time.perf_counter_ns()
                heapq.heappush(queue, (bound, next(serial), child))
                heap_push_ns += time.perf_counter_ns() - started
                heap_push_calls += 1

    if best_position < 0:
        raise RuntimeError("micro-cost trace returned no candidate")

    result = SearchResult(
        item_id=int(best_item_id),
        position=int(best_position),
        squared_distance=float(best_distance),
        address_vector_reads=int(vector_reads),
        directory_nodes_visited=int(node_visits),
    )
    total_ns = time.perf_counter_ns() - started_total
    measured_component_ns = (
        lower_bound_ns
        + heap_push_ns
        + heap_pop_ns
        + leaf_distance_and_index_ns
        + candidate_loop_ns
    )
    return TraceReplay(
        result=result,
        counts=TraceCounts(
            lower_bound_calls=int(lower_bound_calls),
            heap_push_calls=int(heap_push_calls),
            heap_pop_calls=int(heap_pop_calls),
            leaf_visits=int(leaf_visits),
            vector_reads=int(vector_reads),
            candidate_iterations=int(candidate_iterations),
            node_visits=int(node_visits),
            internal_node_visits=int(internal_node_visits),
        ),
        timing=TraceTiming(
            total_ns=int(total_ns),
            lower_bound_ns=int(lower_bound_ns),
            heap_push_ns=int(heap_push_ns),
            heap_pop_ns=int(heap_pop_ns),
            leaf_distance_and_index_ns=int(leaf_distance_and_index_ns),
            candidate_loop_ns=int(candidate_loop_ns),
            residual_ns=int(max(0, total_ns - measured_component_ns)),
        ),
        bound_values=tuple(float(x) for x in bound_values),
        first_leaf_positions=first_leaf_positions,
    )


def _profile_summary(index: ExactEpisodicIndex, queries: list[np.ndarray]) -> dict[str, Any]:
    profiler = cProfile.Profile()
    profiler.enable()
    for query in queries:
        index.indexed_search(query)
    profiler.disable()
    stats = pstats.Stats(profiler)

    wanted = {
        "indexed_search": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
        "_lower_bound_sq": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
        "_squared_distances": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
        "heapq.heappush": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
        "heapq.heappop": {"calls": 0, "self_seconds": 0.0, "cumulative_seconds": 0.0},
    }

    for (filename, _line, funcname), (_cc, nc, tt, ct, _callers) in stats.stats.items():
        target: str | None = None
        if funcname == "indexed_search" and filename.endswith("chm_v1_exact_index.py"):
            target = "indexed_search"
        elif funcname == "_lower_bound_sq" and filename.endswith("chm_v1_exact_index.py"):
            target = "_lower_bound_sq"
        elif funcname == "_squared_distances" and filename.endswith("chm_v1_exact_index.py"):
            target = "_squared_distances"
        elif "heappush" in funcname:
            target = "heapq.heappush"
        elif "heappop" in funcname:
            target = "heapq.heappop"
        if target is not None:
            wanted[target]["calls"] += int(nc)
            wanted[target]["self_seconds"] += float(tt)
            wanted[target]["cumulative_seconds"] += float(ct)

    total_profiled_seconds = float(stats.total_tt)
    return {
        "profiled_total_seconds": total_profiled_seconds,
        "functions": wanted,
        "note": "cProfile timing is descriptive and includes profiling overhead.",
    }


def _median_unprofiled_indexed_ns(
    index: ExactEpisodicIndex,
    queries: list[np.ndarray],
    *,
    repeats: int,
) -> int:
    samples: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for query in queries:
            index.indexed_search(query)
        samples.append(time.perf_counter_ns() - started)
    return int(np.median(np.asarray(samples, dtype=np.int64)))


def _time_repeated(callable_obj: Any, repeats: int) -> int:
    started = time.perf_counter_ns()
    for _ in range(repeats):
        callable_obj()
    return int(time.perf_counter_ns() - started)


def _isolated_microbenchmarks(
    index: ExactEpisodicIndex,
    query: np.ndarray,
    replay: TraceReplay,
    *,
    repeats: int,
) -> dict[str, Any]:
    q = index._query(query)
    root = index.root
    if root.left is None or root.right is None:
        raise ValueError("microbenchmarks require an internal root")

    scalar_pair = lambda: (
        _lower_bound_sq(q, root.left.lo, root.left.hi),
        _lower_bound_sq(q, root.right.lo, root.right.hi),
    )
    scalar_pair_expected = scalar_pair()
    scalar_pair_ns = _time_repeated(scalar_pair, repeats)

    array_prototype = ArrayBackedExactTraversalPrototype(index)
    array_pair = lambda: array_prototype.child_bounds(q, 0)
    array_pair_expected = array_pair()
    array_pair_ns = _time_repeated(array_pair, repeats)
    if array_pair_expected != scalar_pair_expected:
        raise AssertionError("#936 child-bound pair differs from frozen scalar pair")

    if not replay.first_leaf_positions:
        raise RuntimeError("trace did not record a representative leaf")
    positions = np.asarray(replay.first_leaf_positions, dtype=np.int64)

    def leaf_distance_call() -> np.ndarray:
        return _squared_distances(index.points[positions], q)

    representative_distances = leaf_distance_call()
    leaf_distance_ns = _time_repeated(leaf_distance_call, repeats)

    position_list = positions.tolist()
    distance_list = representative_distances.tolist()

    def candidate_loop_call() -> tuple[float, int, int]:
        best_distance = float("inf")
        best_item_id = np.iinfo(np.int64).max
        best_position = -1
        for position, distance in zip(position_list, distance_list):
            item_id = int(index.item_ids[position])
            candidate = (float(distance), item_id)
            if candidate < (best_distance, best_item_id):
                best_distance, best_item_id = candidate
                best_position = int(position)
        return best_distance, int(best_item_id), best_position

    candidate_loop_call()
    candidate_loop_ns = _time_repeated(candidate_loop_call, repeats)

    heap_values = list(replay.bound_values[:64])
    if not heap_values:
        heap_values = [0.0]
    heap: list[tuple[float, int, int]] = [
        (float(value), serial, serial) for serial, value in enumerate(heap_values)
    ]
    heapq.heapify(heap)
    serial_counter = len(heap)

    started = time.perf_counter_ns()
    for _ in range(repeats):
        item = heapq.heappop(heap)
        heapq.heappush(heap, (item[0], serial_counter, item[2]))
        serial_counter += 1
    heap_pair_ns = int(time.perf_counter_ns() - started)

    return {
        "repeats": int(repeats),
        "frozen_two_scalar_bounds_total_ns": scalar_pair_ns,
        "array_backed_child_bounds_total_ns": array_pair_ns,
        "array_over_scalar_bound_pair_ratio": float(array_pair_ns / max(scalar_pair_ns, 1)),
        "array_pair_matches_scalar_exactly": True,
        "heap_pop_push_pair_total_ns": heap_pair_ns,
        "leaf_distance_and_index_total_ns": leaf_distance_ns,
        "candidate_loop_total_ns": candidate_loop_ns,
        "representative_leaf_items": int(len(position_list)),
        "note": "Isolated replay timings are descriptive only and are not CI gates.",
    }


def _sum_counts(replays: list[TraceReplay]) -> dict[str, int]:
    fields = TraceCounts.__dataclass_fields__.keys()
    return {
        field: int(sum(getattr(replay.counts, field) for replay in replays))
        for field in fields
    }


def _sum_timing(replays: list[TraceReplay]) -> dict[str, int]:
    fields = TraceTiming.__dataclass_fields__.keys()
    return {
        field: int(sum(getattr(replay.timing, field) for replay in replays))
        for field in fields
    }


def profile_traversal_microcost_once(
    *,
    session_length: int,
    query_count: int,
    geometry: QueryGeometry = "near",
    noise_std: float = 0.05,
    key_width: int = 32,
    seed: int = SYNTHETIC_MICROCOST_SEED,
    leaf_size: int = 16,
    baseline_repeats: int = 3,
    micro_repeats: int = 128,
) -> dict[str, Any]:
    """Run one deterministic zero-credit micro-cost decomposition.

    Correctness, deterministic counts, accounting, and seed refusal are valid CI
    gates. Every timing field is descriptive only.
    """

    seed = validate_microcost_seed(seed)
    if session_length < 2:
        raise ValueError("session_length must be at least two")
    if query_count < 1 or query_count > session_length:
        raise ValueError("query_count must be in [1, session_length]")
    if key_width < 1 or leaf_size < 1:
        raise ValueError("key_width and leaf_size must be positive")
    if baseline_repeats < 1 or micro_repeats < 1:
        raise ValueError("repeat counts must be positive")

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
    item_ids = np.arange(session_length, dtype=np.int64)
    index = ExactEpisodicIndex(points, item_ids, leaf_size=leaf_size)
    queries = [
        query.detach().float().cpu().numpy().astype(np.float32, copy=False)
        for query in batch.queries
    ]

    flat_results = [index.flat_search(query) for query in queries]
    frozen_results = [index.indexed_search(query) for query in queries]
    replays = [trace_indexed_search(index, query) for query in queries]

    flat_answer_parity = all(
        (flat.item_id, flat.position) == (frozen.item_id, frozen.position)
        for flat, frozen in zip(flat_results, frozen_results)
    )
    replay_full_parity = all(
        _identity(frozen) == _identity(replay.result)
        for frozen, replay in zip(frozen_results, replays)
    )
    if not flat_answer_parity or not replay_full_parity:
        raise AssertionError("micro-cost profiler observed exactness/accounting mismatch")

    counts = _sum_counts(replays)
    timing = _sum_timing(replays)
    frozen_reads = int(sum(result.address_vector_reads for result in frozen_results))
    frozen_nodes = int(sum(result.directory_nodes_visited for result in frozen_results))
    if counts["vector_reads"] != frozen_reads or counts["node_visits"] != frozen_nodes:
        raise AssertionError("trace accounting differs from frozen indexed accounting")
    if counts["candidate_iterations"] != counts["vector_reads"]:
        raise AssertionError("every vector read must participate in one candidate iteration")

    unprofiled_ns = _median_unprofiled_indexed_ns(
        index,
        queries,
        repeats=baseline_repeats,
    )
    cprofile = _profile_summary(index, queries)
    micro = _isolated_microbenchmarks(
        index,
        queries[0],
        replays[0],
        repeats=micro_repeats,
    )

    traced_total = max(timing["total_ns"], 1)
    trace_fractions = {
        "lower_bound": timing["lower_bound_ns"] / traced_total,
        "heap_push": timing["heap_push_ns"] / traced_total,
        "heap_pop": timing["heap_pop_ns"] / traced_total,
        "leaf_distance_and_index": timing["leaf_distance_and_index_ns"] / traced_total,
        "candidate_loop": timing["candidate_loop_ns"] / traced_total,
        "residual": timing["residual_ns"] / traced_total,
    }

    return {
        "measurement_kind": "zero_credit_traversal_microcost_cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "seed": int(seed),
        "session_length": int(session_length),
        "query_count": int(query_count),
        "geometry": geometry,
        "noise_std": float(batch.noise_std),
        "key_width": int(key_width),
        "leaf_size": int(leaf_size),
        "correctness": {
            "flat_answer_parity": bool(flat_answer_parity),
            "trace_full_frozen_accounting_parity": bool(replay_full_parity),
        },
        "operation_counts": counts,
        "frozen_accounting": {
            "address_vector_reads": frozen_reads,
            "flat_address_vector_reads": int(session_length * query_count),
            "address_read_fraction": float(
                frozen_reads / max(session_length * query_count, 1)
            ),
            "directory_nodes_visited": frozen_nodes,
            "directory_nodes_per_query": float(frozen_nodes / query_count),
        },
        "unprofiled_frozen_indexed_ns": int(unprofiled_ns),
        "traced_timing_ns": timing,
        "traced_component_fractions": trace_fractions,
        "cprofile": cprofile,
        "isolated_replay": micro,
        "timing_is_ci_gate": False,
        "interpretation_ceiling": (
            "Zero-credit software micro-cost attribution only; not scientific evidence, "
            "not authority for seed 8612/8613, and not a scaling/novelty claim."
        ),
    }


def deterministic_signature(result: dict[str, Any]) -> dict[str, Any]:
    """Return fields appropriate for deterministic correctness CI assertions."""

    return {
        "measurement_kind": result["measurement_kind"],
        "scientific_credit": result["scientific_credit"],
        "paid_compute": result["paid_compute"],
        "modal_trigger": result["modal_trigger"],
        "seed": result["seed"],
        "session_length": result["session_length"],
        "query_count": result["query_count"],
        "geometry": result["geometry"],
        "noise_std": result["noise_std"],
        "key_width": result["key_width"],
        "leaf_size": result["leaf_size"],
        "correctness": result["correctness"],
        "operation_counts": result["operation_counts"],
        "frozen_accounting": result["frozen_accounting"],
        "timing_is_ci_gate": result["timing_is_ci_gate"],
        "interpretation_ceiling": result["interpretation_ceiling"],
    }
