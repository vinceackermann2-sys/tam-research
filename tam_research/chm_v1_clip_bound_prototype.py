from __future__ import annotations

"""Zero-credit clip-fused lower-bound prototype for CHM-v1 successor #955.

The frozen exact index still owns tree construction, float32 learned addresses,
item IDs, leaf membership, leaf-distance arithmetic, heap ordering, pruning,
candidate iteration, and accounting. This module changes lower-bound arithmetic
only: precomputed exact float64 node-bound sidecars plus one float64 query cast
and one reusable float64 scratch vector per search, evaluated as
``clip -> subtract -> dot``.

No scientific seed, CUDA/Modal path, paid compute, optimizer, backward pass, or
model training is reachable here.
"""

import heapq
from itertools import count
import time
from typing import Any, Literal

import numpy as np

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
from .chm_v1_post_scratch_microcost import PostScratchTraversal

SCIENTIFIC_SEEDS = (8611, 8612, 8613)
SYNTHETIC_PROTOTYPE_SEEDS = (955_001, 955_002)
QueryGeometry = Literal["exact", "near", "random"]


def validate_prototype_seed(seed: int) -> int:
    seed = int(seed)
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError(
            f"scientific seed refused by zero-credit clip-bound prototype: {seed}"
        )
    return seed


def clip_subtract_dot_lower_bound(
    query_f64: np.ndarray,
    lo_f64: np.ndarray,
    hi_f64: np.ndarray,
    scratch: np.ndarray,
) -> float:
    """Exact interval-distance lower bound with one reusable scratch vector."""

    q64 = np.asarray(query_f64, dtype=np.float64)
    lo64 = np.asarray(lo_f64, dtype=np.float64)
    hi64 = np.asarray(hi_f64, dtype=np.float64)
    if scratch.dtype != np.float64 or scratch.shape != q64.shape:
        raise ValueError("scratch must be a float64 vector matching the query shape")
    np.clip(q64, lo64, hi64, out=scratch)
    np.subtract(q64, scratch, out=scratch)
    return float(scratch @ scratch)


class ClipBoundTraversal:
    """Frozen exact traversal with only lower-bound arithmetic replaced."""

    def __init__(self, frozen: ExactEpisodicIndex):
        self.frozen = frozen
        self.points = frozen.points
        self.item_ids = frozen.item_ids
        self._bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def record(node: KDNode) -> None:
            node_id = id(node)
            if node_id in self._bounds:
                raise AssertionError("frozen tree unexpectedly reuses a KDNode object")
            self._bounds[node_id] = (
                np.asarray(node.lo, dtype=np.float64).copy(),
                np.asarray(node.hi, dtype=np.float64).copy(),
            )
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
        stack = [self.frozen.root]
        seen = 0
        while stack:
            node = stack.pop()
            lo64, hi64 = self._bounds[id(node)]
            if not np.array_equal(lo64, node.lo.astype(np.float64)):
                raise AssertionError("lo sidecar differs from exact frozen float32 cast")
            if not np.array_equal(hi64, node.hi.astype(np.float64)):
                raise AssertionError("hi sidecar differs from exact frozen float32 cast")
            seen += 1
            if node.item_positions is None:
                if node.left is None or node.right is None:
                    raise RuntimeError("malformed frozen exact-index node")
                stack.extend((node.left, node.right))
        if seen != len(self._bounds):
            raise AssertionError("sidecar/tree node accounting mismatch")

    def lower_bound(self, query: np.ndarray, node: KDNode) -> float:
        q = self.frozen._query(query)
        q64 = np.asarray(q, dtype=np.float64)
        scratch = np.empty_like(q64)
        lo64, hi64 = self._bounds[id(node)]
        return clip_subtract_dot_lower_bound(q64, lo64, hi64, scratch)

    def indexed_search(self, query: np.ndarray) -> SearchResult:
        q = self.frozen._query(query)
        q64 = np.asarray(q, dtype=np.float64)
        # The only per-search scratch allocation in the lower-bound prototype.
        scratch = np.empty_like(q64)

        best_distance = float("inf")
        best_item_id = np.iinfo(np.int64).max
        best_position = -1
        vector_reads = 0
        node_visits = 0
        serial = count()

        root_lo, root_hi = self._bounds[id(self.frozen.root)]
        queue: list[tuple[float, int, KDNode]] = [
            (
                clip_subtract_dot_lower_bound(q64, root_lo, root_hi, scratch),
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
                lo64, hi64 = self._bounds[id(child)]
                bound = clip_subtract_dot_lower_bound(q64, lo64, hi64, scratch)
                if bound <= best_distance:
                    heapq.heappush(queue, (bound, next(serial), child))

        if best_position < 0:
            raise RuntimeError("clip-bound exact traversal returned no candidate")

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
                f"clip/frozen indexed mismatch: prototype={prototype} frozen={frozen}"
            )
        if (prototype.item_id, prototype.position) != (flat.item_id, flat.position):
            raise AssertionError(
                f"clip/flat answer mismatch: prototype={prototype} flat={flat}"
            )
        return prototype


def assert_node_bound_exactness(
    traversal: ClipBoundTraversal,
    query: np.ndarray,
) -> None:
    q = traversal.frozen._query(query)
    q64 = np.asarray(q, dtype=np.float64)
    scratch = np.empty_like(q64)
    stack = [traversal.frozen.root]
    while stack:
        node = stack.pop()
        lo64, hi64 = traversal._bounds[id(node)]
        observed = clip_subtract_dot_lower_bound(q64, lo64, hi64, scratch)
        expected = _lower_bound_sq(q, node.lo, node.hi)
        if observed != expected:
            raise AssertionError(
                f"clip/frozen bound mismatch: observed={observed} expected={expected}"
            )
        if node.item_positions is None:
            if node.left is None or node.right is None:
                raise RuntimeError("malformed frozen exact-index node")
            stack.extend((node.left, node.right))


def benchmark_clip_bound_once(
    *,
    session_length: int,
    query_count: int = 32,
    key_width: int = 32,
    geometry: QueryGeometry = "near",
    noise_std: float = 0.05,
    seed: int = SYNTHETIC_PROTOTYPE_SEEDS[0],
    repeats: int = 9,
) -> dict[str, Any]:
    """Descriptive CPU median comparison; timings are never CI correctness gates."""

    seed = validate_prototype_seed(seed)
    if session_length < 1 or query_count < 1 or query_count > session_length:
        raise ValueError("invalid session_length/query_count")
    if key_width < 1 or repeats < 3 or repeats % 3 != 0:
        raise ValueError("key_width must be positive and repeats a positive multiple of 3")

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
    scratch_baseline = PostScratchTraversal(frozen)
    prototype = ClipBoundTraversal(frozen)
    queries = [
        row.detach().float().cpu().numpy().astype(np.float32, copy=False)
        for row in batch.queries
    ]

    frozen_results = [frozen.indexed_search(query) for query in queries]
    scratch_results = [scratch_baseline.indexed_search(query) for query in queries]
    prototype_results = [prototype.indexed_search(query) for query in queries]
    flat_results = [frozen.flat_search(query) for query in queries]
    if scratch_results != frozen_results or prototype_results != frozen_results:
        raise AssertionError("prototype/scratch/frozen full SearchResult parity failure")
    if any(
        (p.item_id, p.position) != (f.item_id, f.position)
        for p, f in zip(prototype_results, flat_results)
    ):
        raise AssertionError("clip/flat answer parity failure")

    implementations = {
        "frozen": frozen.indexed_search,
        "scratch": scratch_baseline.indexed_search,
        "clip": prototype.indexed_search,
    }
    samples: dict[str, list[int]] = {name: [] for name in implementations}
    orders = (
        ("frozen", "scratch", "clip"),
        ("scratch", "clip", "frozen"),
        ("clip", "frozen", "scratch"),
    )
    for repeat in range(repeats):
        for name in orders[repeat % len(orders)]:
            search = implementations[name]
            started = time.perf_counter_ns()
            for query in queries:
                search(query)
            samples[name].append(time.perf_counter_ns() - started)

    medians = {
        name: int(np.median(np.asarray(values, dtype=np.int64)))
        for name, values in samples.items()
    }
    reads = int(sum(result.address_vector_reads for result in frozen_results))
    nodes = int(sum(result.directory_nodes_visited for result in frozen_results))

    return {
        "measurement_kind": "zero_credit_clip_bound_systems_prototype_cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "timing_is_ci_gate": False,
        "seed": int(seed),
        "config": {
            "session_length": int(session_length),
            "query_count": int(query_count),
            "key_width": int(key_width),
            "geometry": geometry,
            "noise_std": float(batch.noise_std),
            "leaf_size": 16,
            "repeats": int(repeats),
        },
        "correctness": {
            "flat_answer_parity": True,
            "scratch_frozen_full_searchresult_parity": True,
            "clip_frozen_full_searchresult_parity": True,
            "precomputed_sidecars_exact": True,
        },
        "accounting": {
            "address_vector_reads": reads,
            "flat_address_vector_reads": int(session_length * query_count),
            "address_read_fraction": reads / max(session_length * query_count, 1),
            "directory_nodes_visited": nodes,
            "directory_nodes_per_query": nodes / query_count,
            "sidecar_nodes": prototype.node_count,
            "lower_bound_scratch_vectors_per_search": 1,
        },
        "timing_ns": {
            "frozen_indexed_median": medians["frozen"],
            "scratch_indexed_median": medians["scratch"],
            "clip_indexed_median": medians["clip"],
            "clip_over_frozen_ratio": medians["clip"] / max(medians["frozen"], 1),
            "clip_over_scratch_ratio": medians["clip"] / max(medians["scratch"], 1),
        },
        "interpretation_ceiling": (
            "Zero-credit exact-index lower-bound systems prototype only; not scientific "
            "evidence and not authority for seed 8612/8613 or paid/GPU training."
        ),
    }
