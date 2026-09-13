from __future__ import annotations

"""Zero-credit clipped-query lower-bound prototype for CHM-v1 successor #957.

The frozen :class:`ExactEpisodicIndex` still owns tree construction, float32
addresses, item IDs, leaf membership, leaf-distance arithmetic, heap ordering,
pruning semantics, candidate iteration, and accounting. This module changes one
thing only: exact node-box lower-bound arithmetic uses one reusable float64
scratch buffer via ``clip(query, lo, hi) - query`` before the same dot-product
reduction.

No scientific seed, CUDA/Modal path, paid compute, optimizer, backward pass, or
model training is reachable here.
"""

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
from .chm_v1_post_materialization_profiler import build_query_batch, synthetic_state_payload

SCIENTIFIC_SEEDS = (8611, 8612, 8613)
SYNTHETIC_PROTOTYPE_SEEDS = (957_001, 957_002)
QueryGeometry = Literal["exact", "near", "random"]


def validate_prototype_seed(seed: int) -> int:
    seed = int(seed)
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError(
            f"scientific seed refused by zero-credit clipped-bound prototype: {seed}"
        )
    return seed


def clipped_lower_bound_f64(
    query_f64: np.ndarray,
    lo_f64: np.ndarray,
    hi_f64: np.ndarray,
    scratch: np.ndarray,
) -> float:
    """Exact squared distance from ``query`` to an axis-aligned node box.

    The magnitude of each non-zero coordinate equals the frozen
    ``max(max(lo-q, q-hi), 0)`` delta; the upper-side case differs only by sign,
    which vanishes before the unchanged dot-product reduction.
    """

    np.clip(query_f64, lo_f64, hi_f64, out=scratch)
    np.subtract(scratch, query_f64, out=scratch)
    return float(scratch @ scratch)


class ClippedBoundPrototype:
    """Frozen exact traversal with single-scratch clipped-query bounds."""

    def __init__(self, frozen: ExactEpisodicIndex):
        self.frozen = frozen
        self.points = frozen.points
        self.item_ids = frozen.item_ids
        self._bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._nodes: dict[int, KDNode] = {}

        def record(node: KDNode) -> None:
            node_id = id(node)
            if node_id in self._bounds:
                raise AssertionError("frozen tree unexpectedly reuses a KDNode object")
            self._bounds[node_id] = (
                np.asarray(node.lo, dtype=np.float64).copy(),
                np.asarray(node.hi, dtype=np.float64).copy(),
            )
            self._nodes[node_id] = node
            if node.item_positions is None:
                if node.left is None or node.right is None:
                    raise RuntimeError("malformed frozen exact-index node")
                record(node.left)
                record(node.right)

        record(frozen.root)
        self.assert_sidecars_exact()

    @property
    def node_count(self) -> int:
        return len(self._bounds)

    def assert_sidecars_exact(self) -> None:
        for node_id, node in self._nodes.items():
            lo64, hi64 = self._bounds[node_id]
            if lo64.dtype != np.float64 or hi64.dtype != np.float64:
                raise AssertionError("sidecar bounds must be float64")
            if not np.array_equal(lo64, node.lo.astype(np.float64)):
                raise AssertionError("lo sidecar differs from exact frozen float32 cast")
            if not np.array_equal(hi64, node.hi.astype(np.float64)):
                raise AssertionError("hi sidecar differs from exact frozen float32 cast")

    def bounds_for_node(self, node: KDNode) -> tuple[np.ndarray, np.ndarray]:
        try:
            return self._bounds[id(node)]
        except KeyError as exc:
            raise ValueError("node does not belong to this frozen index") from exc

    def lower_bound(self, query: np.ndarray, node: KDNode) -> float:
        q = self.frozen._query(query)
        q64 = np.asarray(q, dtype=np.float64)
        scratch = np.empty_like(q64)
        lo64, hi64 = self.bounds_for_node(node)
        return clipped_lower_bound_f64(q64, lo64, hi64, scratch)

    def indexed_search(self, query: np.ndarray) -> SearchResult:
        q = self.frozen._query(query)
        q64 = np.asarray(q, dtype=np.float64)
        scratch = np.empty_like(q64)

        best_distance = float("inf")
        best_item_id = np.iinfo(np.int64).max
        best_position = -1
        vector_reads = 0
        node_visits = 0
        serial = count()

        root_lo64, root_hi64 = self.bounds_for_node(self.frozen.root)
        queue: list[tuple[float, int, KDNode]] = [
            (
                clipped_lower_bound_f64(q64, root_lo64, root_hi64, scratch),
                next(serial),
                self.frozen.root,
            )
        ]

        while queue:
            lower_bound, _, node = heapq.heappop(queue)
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
                raise RuntimeError("malformed frozen exact-index node")
            for child in (node.left, node.right):
                lo64, hi64 = self.bounds_for_node(child)
                bound = clipped_lower_bound_f64(q64, lo64, hi64, scratch)
                if bound <= best_distance:
                    heapq.heappush(queue, (bound, next(serial), child))

        if best_position < 0:
            raise RuntimeError("clipped-bound exact traversal returned no candidate")

        return SearchResult(
            item_id=int(best_item_id),
            position=int(best_position),
            squared_distance=float(best_distance),
            address_vector_reads=int(vector_reads),
            directory_nodes_visited=int(node_visits),
        )

    def assert_frozen_parity(self, query: np.ndarray) -> SearchResult:
        flat = self.frozen.flat_search(query)
        frozen = self.frozen.indexed_search(query)
        prototype = self.indexed_search(query)
        if prototype != frozen:
            raise AssertionError(
                f"clipped/frozen mismatch: prototype={prototype} frozen={frozen}"
            )
        if (prototype.item_id, prototype.position) != (flat.item_id, flat.position):
            raise AssertionError(
                f"clipped/flat mismatch: prototype={prototype} flat={flat}"
            )
        return prototype


def benchmark_clipped_bound_once(
    *,
    session_length: int,
    query_count: int = 32,
    key_width: int = 32,
    geometry: QueryGeometry = "near",
    noise_std: float = 0.05,
    seed: int = SYNTHETIC_PROTOTYPE_SEEDS[0],
    repeats: int = 7,
) -> dict[str, object]:
    """Descriptive CPU timing helper; timings are never correctness/CI gates."""

    seed = validate_prototype_seed(seed)
    if session_length < 1 or query_count < 1 or query_count > session_length:
        raise ValueError("invalid session_length/query_count")
    if key_width < 1 or repeats < 1:
        raise ValueError("key_width and repeats must be positive")

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
    prototype = ClippedBoundPrototype(frozen)
    queries = [
        row.detach().float().cpu().numpy().astype(np.float32, copy=False)
        for row in batch.queries
    ]

    frozen_results = [frozen.indexed_search(query) for query in queries]
    prototype_results = [prototype.indexed_search(query) for query in queries]
    flat_results = [frozen.flat_search(query) for query in queries]
    if prototype_results != frozen_results:
        raise AssertionError("prototype/frozen full-accounting parity failure")
    if any(
        (prototype_result.item_id, prototype_result.position)
        != (flat_result.item_id, flat_result.position)
        for prototype_result, flat_result in zip(prototype_results, flat_results)
    ):
        raise AssertionError("prototype/flat answer parity failure")

    frozen_samples: list[int] = []
    prototype_samples: list[int] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            started = time.perf_counter_ns()
            for query in queries:
                frozen.indexed_search(query)
            frozen_samples.append(time.perf_counter_ns() - started)

            started = time.perf_counter_ns()
            for query in queries:
                prototype.indexed_search(query)
            prototype_samples.append(time.perf_counter_ns() - started)
        else:
            started = time.perf_counter_ns()
            for query in queries:
                prototype.indexed_search(query)
            prototype_samples.append(time.perf_counter_ns() - started)

            started = time.perf_counter_ns()
            for query in queries:
                frozen.indexed_search(query)
            frozen_samples.append(time.perf_counter_ns() - started)

    frozen_median = int(np.median(np.asarray(frozen_samples, dtype=np.int64)))
    prototype_median = int(np.median(np.asarray(prototype_samples, dtype=np.int64)))
    reads = int(sum(result.address_vector_reads for result in frozen_results))
    nodes = int(sum(result.directory_nodes_visited for result in frozen_results))

    return {
        "measurement_kind": "zero_credit_clipped_bound_systems_prototype_cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "timing_is_ci_gate": False,
        "seed": int(seed),
        "session_length": int(session_length),
        "query_count": int(query_count),
        "key_width": int(key_width),
        "geometry": geometry,
        "noise_std": float(batch.noise_std),
        "correctness": {
            "flat_answer_parity": True,
            "frozen_indexed_full_accounting_parity": True,
            "precomputed_bounds_exact": True,
        },
        "accounting": {
            "address_vector_reads": reads,
            "flat_address_vector_reads": int(session_length * query_count),
            "address_read_fraction": reads / max(session_length * query_count, 1),
            "directory_nodes_visited": nodes,
            "directory_nodes_per_query": nodes / query_count,
            "precomputed_bound_nodes": prototype.node_count,
        },
        "timing_ns": {
            "frozen_indexed_median": frozen_median,
            "prototype_indexed_median": prototype_median,
            "prototype_over_frozen_ratio": prototype_median / max(frozen_median, 1),
        },
        "interpretation_ceiling": (
            "Zero-credit exact lower-bound systems prototype only; not scientific "
            "evidence and not authority for seed 8612/8613 or paid/GPU training."
        ),
    }
