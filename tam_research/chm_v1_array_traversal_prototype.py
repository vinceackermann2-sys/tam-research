from __future__ import annotations

"""Zero-credit array-backed traversal prototype for CHM-v1 successor #936.

The frozen :class:`ExactEpisodicIndex` still owns tree construction, points,
item IDs, leaf partitioning, distance arithmetic, and exact tie semantics. This
module only converts the already-built KDNode tree into contiguous node arrays
and evaluates sibling lower-bound deltas together before preserving the frozen
left/right heap insertion order.

No scientific seeds, GPU path, Modal execution, or model training live here.
"""

from dataclasses import dataclass
import heapq
from itertools import count
import time
from typing import Literal

import numpy as np

from .chm_v1_exact_index import (
    ExactEpisodicIndex,
    KDNode,
    SearchResult,
    _lower_bound_sq,
    _squared_distances,
)

SCIENTIFIC_SEEDS = (8611, 8612, 8613)
SYNTHETIC_PROTOTYPE_SEED = 936_001
QueryGeometry = Literal["exact", "near", "random"]


@dataclass(frozen=True)
class TraversalDiagnostics:
    node_visits: int
    leaf_visits: int
    child_bound_batches: int
    child_bounds_evaluated: int
    heap_pushes: int
    heap_pops: int


class ArrayBackedExactTraversalPrototype:
    """Array-backed traversal over an already-built frozen exact KD tree."""

    def __init__(self, frozen: ExactEpisodicIndex):
        self.frozen = frozen
        self.points = frozen.points
        self.item_ids = frozen.item_ids
        self.leaf_size = frozen.leaf_size

        lo_rows: list[np.ndarray] = []
        hi_rows: list[np.ndarray] = []
        left: list[int] = []
        right: list[int] = []
        leaf_offset: list[int] = []
        leaf_length: list[int] = []
        leaf_positions: list[int] = []

        def flatten(node: KDNode) -> int:
            node_index = len(lo_rows)
            lo_rows.append(np.asarray(node.lo, dtype=np.float32).copy())
            hi_rows.append(np.asarray(node.hi, dtype=np.float32).copy())
            left.append(-1)
            right.append(-1)
            leaf_offset.append(-1)
            leaf_length.append(0)

            if node.item_positions is not None:
                positions = np.asarray(node.item_positions, dtype=np.int64)
                leaf_offset[node_index] = len(leaf_positions)
                leaf_length[node_index] = int(positions.size)
                leaf_positions.extend(int(position) for position in positions.tolist())
                return node_index

            if node.left is None or node.right is None:
                raise RuntimeError("malformed frozen exact-index node")
            left_index = flatten(node.left)
            right_index = flatten(node.right)
            left[node_index] = left_index
            right[node_index] = right_index
            return node_index

        root_index = flatten(frozen.root)
        if root_index != 0:
            raise AssertionError("prototype flattening expected root at index zero")

        self.node_lo = np.stack(lo_rows, axis=0).astype(np.float32, copy=False)
        self.node_hi = np.stack(hi_rows, axis=0).astype(np.float32, copy=False)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.leaf_offset = np.asarray(leaf_offset, dtype=np.int64)
        self.leaf_length = np.asarray(leaf_length, dtype=np.int64)
        self.leaf_positions = np.asarray(leaf_positions, dtype=np.int64)

        if self.node_lo.shape != self.node_hi.shape:
            raise AssertionError("node bound arrays must align")
        if self.node_lo.shape[1] != self.points.shape[1]:
            raise AssertionError("node/address dimensions must align")
        if int(self.leaf_length.sum()) != int(self.points.shape[0]):
            raise AssertionError("flattened leaves must cover every frozen point once")

    @property
    def node_count(self) -> int:
        return int(self.left.size)

    @property
    def leaf_count(self) -> int:
        return int(np.count_nonzero(self.leaf_length > 0))

    def child_bounds(self, query: np.ndarray, node_index: int) -> tuple[float, float]:
        """Evaluate the two frozen child bounds with shared vectorized deltas.

        Delta construction is vectorized over the sibling pair. Each final dot
        product is kept row-by-row so its reduction order exactly matches the
        frozen scalar ``delta @ delta`` implementation.
        """

        node_index = int(node_index)
        left_index = int(self.left[node_index])
        right_index = int(self.right[node_index])
        if left_index < 0 or right_index < 0:
            raise ValueError("child_bounds requires an internal node")

        q = self.frozen._query(query)  # reuse frozen validation/coercion exactly
        children = np.asarray((left_index, right_index), dtype=np.int64)
        q64 = np.asarray(q, dtype=np.float64)
        lo64 = self.node_lo[children].astype(np.float64, copy=False)
        hi64 = self.node_hi[children].astype(np.float64, copy=False)
        delta = np.maximum(0.0, np.maximum(lo64 - q64[None, :], q64[None, :] - hi64))
        # Keep the frozen reduction order for each child.
        return float(delta[0] @ delta[0]), float(delta[1] @ delta[1])

    def indexed_search_with_diagnostics(
        self, query: np.ndarray
    ) -> tuple[SearchResult, TraversalDiagnostics]:
        q = self.frozen._query(query)
        best_distance = float("inf")
        best_item_id = np.iinfo(np.int64).max
        best_position = -1
        vector_reads = 0
        node_visits = 0
        leaf_visits = 0
        child_bound_batches = 0
        child_bounds_evaluated = 0
        heap_pushes = 1
        heap_pops = 0
        serial = count()

        root_bound = _lower_bound_sq(q, self.node_lo[0], self.node_hi[0])
        queue: list[tuple[float, int, int]] = [(root_bound, next(serial), 0)]

        while queue:
            lower_bound, _, node_index = heapq.heappop(queue)
            heap_pops += 1
            # Preserve the frozen strict-greater pruning rule. Equality remains
            # searchable because it can contain a smaller item_id tie winner.
            if lower_bound > best_distance:
                break

            node_visits += 1
            length = int(self.leaf_length[node_index])
            if length > 0:
                leaf_visits += 1
                offset = int(self.leaf_offset[node_index])
                positions = self.leaf_positions[offset : offset + length]
                # Frozen leaf kernel and Python candidate iteration are reused
                # unchanged: this prototype targets traversal only.
                distances = _squared_distances(self.points[positions], q)
                vector_reads += len(positions)
                for position, distance in zip(positions.tolist(), distances.tolist()):
                    item_id = int(self.item_ids[position])
                    candidate = (float(distance), item_id)
                    if candidate < (best_distance, best_item_id):
                        best_distance, best_item_id = candidate
                        best_position = int(position)
                continue

            left_index = int(self.left[node_index])
            right_index = int(self.right[node_index])
            if left_index < 0 or right_index < 0:
                raise RuntimeError("malformed array-backed traversal node")

            left_bound, right_bound = self.child_bounds(q, node_index)
            child_bound_batches += 1
            child_bounds_evaluated += 2
            # Push in the same left-then-right order as the frozen index so
            # serial tie ordering and best-first behavior remain identical.
            for child_index, bound in (
                (left_index, left_bound),
                (right_index, right_bound),
            ):
                if bound <= best_distance:
                    heapq.heappush(queue, (bound, next(serial), child_index))
                    heap_pushes += 1

        if best_position < 0:
            raise RuntimeError("array-backed exact traversal returned no candidate")

        result = SearchResult(
            item_id=best_item_id,
            position=best_position,
            squared_distance=best_distance,
            address_vector_reads=vector_reads,
            directory_nodes_visited=node_visits,
        )
        diagnostics = TraversalDiagnostics(
            node_visits=node_visits,
            leaf_visits=leaf_visits,
            child_bound_batches=child_bound_batches,
            child_bounds_evaluated=child_bounds_evaluated,
            heap_pushes=heap_pushes,
            heap_pops=heap_pops,
        )
        return result, diagnostics

    def indexed_search(self, query: np.ndarray) -> SearchResult:
        return self.indexed_search_with_diagnostics(query)[0]

    def assert_frozen_parity(self, query: np.ndarray) -> SearchResult:
        flat = self.frozen.flat_search(query)
        frozen_indexed = self.frozen.indexed_search(query)
        prototype = self.indexed_search(query)
        identity = lambda result: (
            result.item_id,
            result.position,
            result.squared_distance,
            result.address_vector_reads,
            result.directory_nodes_visited,
        )
        if identity(prototype) != identity(frozen_indexed):
            raise AssertionError(
                f"array-backed/frozen indexed mismatch: prototype={prototype} "
                f"frozen={frozen_indexed}"
            )
        if (prototype.item_id, prototype.position) != (flat.item_id, flat.position):
            raise AssertionError(
                f"array-backed/flat exactness mismatch: prototype={prototype} flat={flat}"
            )
        return prototype


def validate_benchmark_seed(seed: int) -> int:
    seed = int(seed)
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError(f"scientific seed refused by zero-credit traversal prototype: {seed}")
    return seed


def _normalized_rows(rng: np.random.Generator, rows: int, width: int) -> np.ndarray:
    values = rng.standard_normal((rows, width)).astype(np.float32)
    norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    return (values / norms.astype(np.float32)).astype(np.float32)


def make_queries(
    points: np.ndarray,
    *,
    query_count: int,
    geometry: QueryGeometry,
    seed: int,
    noise: float = 0.05,
) -> np.ndarray:
    seed = validate_benchmark_seed(seed)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError("points must be a non-empty matrix")
    if query_count < 1:
        raise ValueError("query_count must be positive")
    rng = np.random.default_rng(seed)
    positions = np.linspace(0, points.shape[0] - 1, num=query_count, dtype=np.int64)

    if geometry == "exact":
        return points[positions].copy()
    if geometry == "near":
        perturbation = _normalized_rows(rng, query_count, points.shape[1])
        queries = points[positions] + np.float32(noise) * perturbation
        norms = np.linalg.norm(queries.astype(np.float64), axis=1, keepdims=True)
        return (queries / norms.astype(np.float32)).astype(np.float32)
    if geometry == "random":
        return _normalized_rows(rng, query_count, points.shape[1])
    raise ValueError(f"unknown query geometry {geometry!r}")


def benchmark_array_traversal_once(
    *,
    session_length: int,
    query_count: int = 32,
    key_width: int = 32,
    geometry: QueryGeometry = "near",
    noise: float = 0.05,
    seed: int = SYNTHETIC_PROTOTYPE_SEED,
    leaf_size: int = 16,
) -> dict[str, object]:
    """Descriptive CPU-only comparison; timing is never a correctness gate."""

    seed = validate_benchmark_seed(seed)
    if session_length < 1 or query_count < 1:
        raise ValueError("session_length and query_count must be positive")
    rng = np.random.default_rng(seed)
    points = _normalized_rows(rng, session_length, key_width)
    item_ids = np.arange(session_length, dtype=np.int64)
    frozen = ExactEpisodicIndex(points, item_ids, leaf_size=leaf_size)
    prototype = ArrayBackedExactTraversalPrototype(frozen)
    queries = make_queries(
        points,
        query_count=query_count,
        geometry=geometry,
        noise=noise,
        seed=seed + 1,
    )

    flat_results: list[SearchResult] = []
    frozen_results: list[SearchResult] = []
    prototype_results: list[SearchResult] = []
    diagnostics: list[TraversalDiagnostics] = []

    flat_started = time.perf_counter_ns()
    for query in queries:
        flat_results.append(frozen.flat_search(query))
    flat_ns = time.perf_counter_ns() - flat_started

    frozen_started = time.perf_counter_ns()
    for query in queries:
        frozen_results.append(frozen.indexed_search(query))
    frozen_ns = time.perf_counter_ns() - frozen_started

    prototype_started = time.perf_counter_ns()
    for query in queries:
        result, diag = prototype.indexed_search_with_diagnostics(query)
        prototype_results.append(result)
        diagnostics.append(diag)
    prototype_ns = time.perf_counter_ns() - prototype_started

    def identity(result: SearchResult) -> tuple[int, int, float, int, int]:
        return (
            result.item_id,
            result.position,
            result.squared_distance,
            result.address_vector_reads,
            result.directory_nodes_visited,
        )

    indexed_parity = all(
        identity(frozen_result) == identity(prototype_result)
        for frozen_result, prototype_result in zip(frozen_results, prototype_results)
    )
    flat_answer_parity = all(
        (flat.item_id, flat.position) == (prototype.item_id, prototype.position)
        for flat, prototype in zip(flat_results, prototype_results)
    )
    if not indexed_parity or not flat_answer_parity:
        raise AssertionError("zero-credit traversal benchmark observed exactness mismatch")

    reads = int(sum(result.address_vector_reads for result in prototype_results))
    nodes = int(sum(result.directory_nodes_visited for result in prototype_results))
    flat_reads = int(session_length * query_count)
    bound_batches = int(sum(diag.child_bound_batches for diag in diagnostics))
    bounds_evaluated = int(sum(diag.child_bounds_evaluated for diag in diagnostics))

    return {
        "measurement_kind": "engineering_only_cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "seed": seed,
        "session_length": int(session_length),
        "query_count": int(query_count),
        "key_width": int(key_width),
        "geometry": geometry,
        "noise": float(noise),
        "leaf_size": int(leaf_size),
        "correctness": {
            "flat_answer_parity": flat_answer_parity,
            "frozen_indexed_full_accounting_parity": indexed_parity,
        },
        "accounting": {
            "address_vector_reads": reads,
            "flat_address_vector_reads": flat_reads,
            "address_read_fraction": reads / max(flat_reads, 1),
            "directory_nodes_visited": nodes,
            "directory_nodes_per_query": nodes / query_count,
            "child_bound_batches": bound_batches,
            "child_bounds_evaluated": bounds_evaluated,
            "array_node_count": prototype.node_count,
            "array_leaf_count": prototype.leaf_count,
        },
        "timing_ns": {
            "flat_query_only": int(flat_ns),
            "frozen_indexed_query_only": int(frozen_ns),
            "array_backed_indexed_query_only": int(prototype_ns),
        },
        "interpretation_ceiling": (
            "Zero-credit traversal systems prototype only; not scientific evidence, "
            "not authority for seed 8612/8613, and not a scaling/novelty claim."
        ),
    }
