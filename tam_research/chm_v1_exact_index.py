from __future__ import annotations

"""Zero-parameter exact branch-and-bound index for CHM-v1 / EIEM.

This module is implementation plumbing for preregistration #854.  It does not
run scientific seeds or authorize GPU work.  Stored ``item_id`` values are
monotonic insertion IDs owned by the memory store; they are never task labels
or oracle generator identifiers.

The flat and indexed paths deliberately share the exact ordering contract
``(squared_distance, item_id)``.  Branches whose lower bound equals the current
best distance are still searched, so duplicate/equidistant keys cannot create
an indexed-vs-flat mismatch merely through tie handling.
"""

from dataclasses import dataclass
import heapq
from itertools import count

import numpy as np


@dataclass
class KDNode:
    lo: np.ndarray
    hi: np.ndarray
    item_positions: np.ndarray | None = None
    left: "KDNode | None" = None
    right: "KDNode | None" = None


@dataclass(frozen=True)
class SearchResult:
    item_id: int
    position: int
    squared_distance: float
    address_vector_reads: int
    directory_nodes_visited: int


def _validate(points: np.ndarray, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    item_ids = np.asarray(item_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty [items, address_dim] matrix")
    if item_ids.shape != (points.shape[0],):
        raise ValueError("item_ids must contain exactly one ID per point")
    if len(np.unique(item_ids)) != len(item_ids):
        raise ValueError("item_ids must be unique")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    return points, item_ids


def _build(points: np.ndarray, positions: np.ndarray, leaf_size: int) -> KDNode:
    block = points[positions]
    lo = block.min(axis=0)
    hi = block.max(axis=0)
    if len(positions) <= leaf_size:
        return KDNode(lo=lo, hi=hi, item_positions=positions.copy())
    axis = int(np.argmax(hi - lo))
    # Stable sort matters when many keys share the split coordinate.
    order = np.argsort(block[:, axis], kind="mergesort")
    ordered = positions[order]
    mid = len(ordered) // 2
    return KDNode(
        lo=lo,
        hi=hi,
        left=_build(points, ordered[:mid], leaf_size),
        right=_build(points, ordered[mid:], leaf_size),
    )


def _lower_bound_sq(query: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(lo - query, query - hi))
    return float(delta @ delta)


class ExactEpisodicIndex:
    """Exact nearest-neighbour lookup with explicit vector-read accounting."""

    def __init__(self, points: np.ndarray, item_ids: np.ndarray, *, leaf_size: int = 16):
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        self.points, self.item_ids = _validate(points, item_ids)
        self.leaf_size = int(leaf_size)
        self.root = _build(
            self.points,
            np.arange(self.points.shape[0], dtype=np.int64),
            self.leaf_size,
        )

    def _query(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.points.shape[1],):
            raise ValueError(f"query must have shape {(self.points.shape[1],)}")
        if not np.isfinite(q).all():
            raise ValueError("query must be finite")
        return q

    def flat_search(self, query: np.ndarray) -> SearchResult:
        q = self._query(query)
        distances = ((self.points - q) ** 2).sum(axis=1)
        # np.lexsort uses the last key as primary: distance first, stable ID second.
        position = int(np.lexsort((self.item_ids, distances))[0])
        return SearchResult(
            item_id=int(self.item_ids[position]),
            position=position,
            squared_distance=float(distances[position]),
            address_vector_reads=int(self.points.shape[0]),
            directory_nodes_visited=0,
        )

    def indexed_search(self, query: np.ndarray) -> SearchResult:
        q = self._query(query)
        best_distance = float("inf")
        best_item_id = np.iinfo(np.int64).max
        best_position = -1
        vector_reads = 0
        node_visits = 0
        serial = count()
        queue: list[tuple[float, int, KDNode]] = [
            (_lower_bound_sq(q, self.root.lo, self.root.hi), next(serial), self.root)
        ]

        while queue:
            lower_bound, _, node = heapq.heappop(queue)
            # Strictly greater only.  Equality can contain a candidate with a
            # smaller item_id and must remain searchable under the shared tie rule.
            if lower_bound > best_distance:
                break
            node_visits += 1
            if node.item_positions is not None:
                positions = node.item_positions
                block = self.points[positions]
                distances = ((block - q) ** 2).sum(axis=1)
                vector_reads += len(positions)
                for position, distance in zip(positions.tolist(), distances.tolist()):
                    item_id = int(self.item_ids[position])
                    candidate = (float(distance), item_id)
                    if candidate < (best_distance, best_item_id):
                        best_distance, best_item_id = candidate
                        best_position = int(position)
                continue

            if node.left is None or node.right is None:
                raise RuntimeError("malformed exact index node")
            for child in (node.left, node.right):
                bound = _lower_bound_sq(q, child.lo, child.hi)
                if bound <= best_distance:
                    heapq.heappush(queue, (bound, next(serial), child))

        if best_position < 0:
            raise RuntimeError("exact indexed search returned no candidate")
        return SearchResult(
            item_id=best_item_id,
            position=best_position,
            squared_distance=best_distance,
            address_vector_reads=vector_reads,
            directory_nodes_visited=node_visits,
        )

    def assert_exact(self, query: np.ndarray) -> tuple[SearchResult, SearchResult]:
        flat = self.flat_search(query)
        indexed = self.indexed_search(query)
        if indexed.item_id != flat.item_id or indexed.position != flat.position:
            raise AssertionError(
                "EIEM exactness violation: "
                f"flat=(id={flat.item_id},pos={flat.position},d={flat.squared_distance}) "
                f"indexed=(id={indexed.item_id},pos={indexed.position},d={indexed.squared_distance})"
            )
        return flat, indexed
