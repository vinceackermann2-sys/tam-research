from __future__ import annotations

"""Zero-credit contiguous-state prototype for CHM-v1 successor issue #930.

This module is intentionally additive. It does not modify or replace the frozen
CHM-v1 scientific implementation. It reuses the frozen ExactEpisodicIndex but
keeps episodic keys, values, and insertion IDs in persistent contiguous NumPy
arrays so warm retrieval does not restack the entire state on every query.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Literal

import numpy as np
import torch

from .chm_v1_exact_index import ExactEpisodicIndex, SearchResult

LEAF_SIZE = 16
SCIENTIFIC_SEEDS = (8611, 8612, 8613)
SYNTHETIC_PROTOTYPE_SEED = 930_001
RetrievalMode = Literal["flat", "indexed"]


@dataclass
class ContiguousEpisodicStatePrototype:
    session_id: str
    next_item_id: int = 0
    index_build_seconds_total: float = 0.0
    flat_search_seconds_total: float = 0.0
    indexed_search_seconds_total: float = 0.0
    verification_seconds_total: float = 0.0
    write_seconds_total: float = 0.0
    index_build_count: int = 0
    _keys: np.ndarray | None = field(default=None, init=False, repr=False)
    _values: np.ndarray | None = field(default=None, init=False, repr=False)
    _item_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64),
        init=False,
        repr=False,
    )
    _cached_index: ExactEpisodicIndex | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return int(self._item_ids.shape[0])

    @property
    def keys(self) -> np.ndarray:
        if self._keys is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._keys

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._values

    @property
    def item_ids(self) -> np.ndarray:
        return self._item_ids

    def reset(self) -> None:
        self.next_item_id = 0
        self.index_build_seconds_total = 0.0
        self.flat_search_seconds_total = 0.0
        self.indexed_search_seconds_total = 0.0
        self.verification_seconds_total = 0.0
        self.write_seconds_total = 0.0
        self.index_build_count = 0
        self._keys = None
        self._values = None
        self._item_ids = np.empty((0,), dtype=np.int64)
        self._cached_index = None

    def payload_bytes(self) -> int:
        key_bytes = 0 if self._keys is None else int(self._keys.nbytes)
        value_bytes = 0 if self._values is None else int(self._values.nbytes)
        return key_bytes + value_bytes + int(self._item_ids.nbytes)

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        started = time.perf_counter()
        try:
            if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
                raise ValueError("keys/values must be aligned [items, dim] matrices")
            if keys.shape[0] == 0:
                return
            k = keys.detach().float().cpu().numpy().astype(np.float32, copy=True)
            v = values.detach().float().cpu().numpy().astype(np.float32, copy=True)
            if not np.isfinite(k).all() or not np.isfinite(v).all():
                raise ValueError("episodic writes must be finite")
            if self._keys is not None and k.shape[1] != self._keys.shape[1]:
                raise ValueError("key width changed within one episodic session")
            if self._values is not None and v.shape[1] != self._values.shape[1]:
                raise ValueError("value width changed within one episodic session")

            new_ids = np.arange(
                self.next_item_id,
                self.next_item_id + k.shape[0],
                dtype=np.int64,
            )
            if self._keys is None:
                self._keys = k
                self._values = v
                self._item_ids = new_ids
            else:
                # O(n) work is paid at write/update time, never on a warm query.
                self._keys = np.concatenate((self._keys, k), axis=0)
                assert self._values is not None
                self._values = np.concatenate((self._values, v), axis=0)
                self._item_ids = np.concatenate((self._item_ids, new_ids), axis=0)
            self.next_item_id += int(k.shape[0])
            self._cached_index = None
        finally:
            self.write_seconds_total += time.perf_counter() - started

    def index(self) -> ExactEpisodicIndex:
        if self._keys is None or len(self) == 0:
            raise RuntimeError("cannot retrieve from empty episodic state")
        if self._cached_index is None:
            started = time.perf_counter()
            self._cached_index = ExactEpisodicIndex(
                self._keys,
                self._item_ids,
                leaf_size=LEAF_SIZE,
            )
            self.index_build_seconds_total += time.perf_counter() - started
            self.index_build_count += 1
        return self._cached_index

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        mode: RetrievalMode,
        verify_indexed_exactness: bool = True,
    ) -> tuple[torch.Tensor, SearchResult, bool, float, float, float]:
        if query.ndim != 1:
            raise ValueError("query must be one address vector")
        if self._values is None or self._keys is None or len(self) == 0:
            raise RuntimeError("cannot retrieve from empty episodic state")

        q = query.detach().float().cpu().numpy().astype(np.float32, copy=False)
        build_before = self.index_build_seconds_total
        index = self.index()
        build_delta = self.index_build_seconds_total - build_before
        verification_elapsed = 0.0

        if mode == "flat":
            started = time.perf_counter()
            result = index.flat_search(q)
            search_elapsed = time.perf_counter() - started
            self.flat_search_seconds_total += search_elapsed
            exact_match = True
        elif mode == "indexed":
            started = time.perf_counter()
            result = index.indexed_search(q)
            search_elapsed = time.perf_counter() - started
            self.indexed_search_seconds_total += search_elapsed
            if verify_indexed_exactness:
                verify_started = time.perf_counter()
                flat = index.flat_search(q)
                verification_elapsed = time.perf_counter() - verify_started
                self.verification_seconds_total += verification_elapsed
                exact_match = (
                    result.item_id == flat.item_id
                    and result.position == flat.position
                )
                if not exact_match:
                    raise AssertionError(
                        f"prototype indexed/flat mismatch in session {self.session_id}: "
                        f"indexed={result} flat={flat}"
                    )
            else:
                exact_match = True
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")

        value = torch.as_tensor(
            self._values[result.position],
            device=query.device,
            dtype=query.dtype,
        )
        return (
            value,
            result,
            exact_match,
            build_delta,
            search_elapsed,
            verification_elapsed,
        )


def prototype_accounting(state: ContiguousEpisodicStatePrototype) -> dict[str, int]:
    """Deterministic accounting; timing is deliberately excluded."""

    return {
        "items": len(state),
        "key_elements_persistent": 0 if state._keys is None else int(state._keys.size),
        "value_elements_persistent": 0 if state._values is None else int(state._values.size),
        "item_id_scalars_persistent": int(state._item_ids.size),
        "warm_retrieve_np_stack_calls": 0,
        "warm_retrieve_full_state_materializations": 0,
        "payload_bytes": state.payload_bytes(),
        "index_build_count": int(state.index_build_count),
    }


def validate_benchmark_seed(seed: int) -> int:
    seed = int(seed)
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError(f"scientific seed refused by zero-credit prototype: {seed}")
    return seed


def benchmark_prototype_once(
    *,
    session_length: int,
    query_count: int,
    key_width: int = 32,
    value_width: int = 256,
    seed: int = SYNTHETIC_PROTOTYPE_SEED,
) -> dict[str, Any]:
    """CPU-only descriptive benchmark. Never use timing as a CI gate."""

    seed = validate_benchmark_seed(seed)
    if session_length < 1 or query_count < 1 or query_count > session_length:
        raise ValueError("invalid session/query count")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.nn.functional.normalize(
        torch.randn(session_length, key_width, generator=generator, device="cpu"),
        dim=-1,
    ).float()
    values = torch.randn(
        session_length,
        value_width,
        generator=generator,
        device="cpu",
    ).float()
    state = ContiguousEpisodicStatePrototype(f"prototype-{seed}")
    state.write(keys, values)
    state.index()

    positions = np.linspace(0, session_length - 1, num=query_count, dtype=np.int64)
    started = time.perf_counter_ns()
    reads = 0
    nodes = 0
    for position in positions.tolist():
        _, result, exact, _, _, _ = state.retrieve(
            keys[position],
            mode="indexed",
            verify_indexed_exactness=True,
        )
        if not exact:
            raise AssertionError("prototype exactness verification failed")
        reads += int(result.address_vector_reads)
        nodes += int(result.directory_nodes_visited)
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "measurement_kind": "engineering_only_cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "seed": seed,
        "session_length": int(session_length),
        "query_count": int(query_count),
        "elapsed_ns": int(elapsed_ns),
        "indexed_address_vector_reads": int(reads),
        "directory_nodes_visited": int(nodes),
        "accounting": prototype_accounting(state),
        "interpretation_ceiling": (
            "Zero-credit systems prototype only; not scientific evidence and not "
            "authority for seed 8612/8613 or any paid/GPU execution."
        ),
    }
