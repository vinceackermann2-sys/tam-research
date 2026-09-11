from __future__ import annotations

"""Systems-only batched transport for CHM-v1 / EIEM exact evaluation.

The merged scalar path in :mod:`tam_research.chm_v1_small_lm` remains the
correctness reference.  This module changes only host/device transfer
frequency: all model-side query and integration operations retain their scalar
per-token shapes, while a chunk's query vectors are transferred to CPU once per
retrieval hop and the selected exact values are transferred back once per hop.

Exact flat/index search itself is still executed query-by-query on CPU in the
same order and with the same ``(squared_distance, insertion_id)`` contract.
Nothing here launches a scientific run, uses a reserved seed, or authorizes GPU
spend.
"""

from dataclasses import dataclass
import time

import numpy as np
import torch

from .chm_v1_exact_index import SearchResult
from .chm_v1_small_lm import (
    RETRIEVAL_HOPS,
    CHMV1EIEMLM,
    EpisodicState,
    RetrievalMode,
    RetrievalStats,
    _hidden,
)


@dataclass(frozen=True)
class BatchedRetrieval:
    values: torch.Tensor
    results: tuple[SearchResult, ...]
    exact_matches: tuple[bool, ...]
    index_build_seconds: float
    search_seconds: float
    verification_seconds: float


def retrieve_many_exact(
    state: EpisodicState,
    queries: torch.Tensor,
    *,
    mode: RetrievalMode,
    verify_indexed_exactness: bool = True,
) -> BatchedRetrieval:
    """Retrieve a matrix of exact queries with one host/device transfer pair.

    Search remains scalar and deterministic on CPU.  Only the tensor transfer is
    coalesced.  This deliberately avoids vectorizing model-side projections,
    integration, or exact-search arithmetic, because those could change floating
    point operation shapes relative to the scalar reference.
    """
    if queries.ndim != 2:
        raise ValueError("queries must be [query_count, address_dim]")
    if queries.shape[0] == 0:
        raise ValueError("queries must contain at least one address vector")

    _, stored_values, _ = state._arrays()
    query_matrix = (
        queries.detach().float().cpu().numpy().astype(np.float32, copy=False)
    )
    build_before = state.index_build_seconds_total
    index = state.index()
    build_delta = state.index_build_seconds_total - build_before

    if mode == "flat":
        started = time.perf_counter()
        results = tuple(index.flat_search(query) for query in query_matrix)
        search_elapsed = time.perf_counter() - started
        state.flat_search_seconds_total += search_elapsed
        exact_matches = (True,) * len(results)
        verification_elapsed = 0.0
    elif mode == "indexed":
        started = time.perf_counter()
        results = tuple(index.indexed_search(query) for query in query_matrix)
        search_elapsed = time.perf_counter() - started
        state.indexed_search_seconds_total += search_elapsed

        if verify_indexed_exactness:
            verify_started = time.perf_counter()
            flat_results = tuple(index.flat_search(query) for query in query_matrix)
            verification_elapsed = time.perf_counter() - verify_started
            state.verification_seconds_total += verification_elapsed
            exact_matches = tuple(
                indexed.item_id == flat.item_id
                and indexed.position == flat.position
                for indexed, flat in zip(results, flat_results)
            )
            if not all(exact_matches):
                mismatch = next(i for i, match in enumerate(exact_matches) if not match)
                raise AssertionError(
                    f"indexed/flat mismatch in session {state.session_id} at batch query "
                    f"{mismatch}: indexed={results[mismatch]} flat={flat_results[mismatch]}"
                )
        else:
            verification_elapsed = 0.0
            exact_matches = (True,) * len(results)
    else:
        raise ValueError(f"unknown retrieval mode {mode!r}")

    positions = np.asarray([result.position for result in results], dtype=np.int64)
    # Force one contiguous host buffer so a GPU destination performs one value
    # transfer for the complete hop rather than one transfer per token.
    selected = np.ascontiguousarray(stored_values[positions])
    values = torch.as_tensor(
        selected,
        device=queries.device,
        dtype=queries.dtype,
    )
    return BatchedRetrieval(
        values=values,
        results=results,
        exact_matches=exact_matches,
        index_build_seconds=build_delta,
        search_seconds=search_elapsed,
        verification_seconds=verification_elapsed,
    )


def _scalar_query_matrix(model: CHMV1EIEMLM, states: torch.Tensor) -> torch.Tensor:
    """Preserve the reference per-token Linear+normalize operation shape."""
    return torch.stack([model.query_for(states[index]) for index in range(states.shape[0])])


def _scalar_integrate_matrix(
    model: CHMV1EIEMLM,
    states: torch.Tensor,
    memories: torch.Tensor,
) -> torch.Tensor:
    """Preserve the reference per-token gate/add operation shape."""
    return torch.stack(
        [model._integrate(states[index], memories[index]) for index in range(states.shape[0])]
    )


@torch.no_grad()
def forward_session_chunk_batched_transport(
    model: CHMV1EIEMLM,
    tokens: torch.Tensor,
    states: list[EpisodicState],
    *,
    mode: RetrievalMode,
    update_memory: bool = True,
    verify_indexed_exactness: bool = True,
) -> tuple[torch.Tensor, RetrievalStats]:
    """Reference-equivalent chunk evaluation with coalesced CPU/GPU transport.

    Causality is unchanged: only state that existed before this call may be read;
    current-chunk writes occur after all logits are complete.  Hop 2 still uses
    the representation after hop-1 integration.  Per-token model math keeps the
    same shapes as ``CHMV1EIEMLM.forward_session_chunk``.
    """
    if tokens.shape[0] != len(states):
        raise ValueError("one EpisodicState is required per batch element")

    hidden = _hidden(model.backbone, tokens)
    keys = model.key_for(hidden)
    fused = hidden.clone()
    stats = RetrievalStats()

    for batch_index, state in enumerate(states):
        if len(state) == 0:
            continue
        memory_size = len(state)
        stats.state_payload_bytes = max(stats.state_payload_bytes, state.payload_bytes())
        query_state = hidden[batch_index]

        for _ in range(RETRIEVAL_HOPS):
            queries = _scalar_query_matrix(model, query_state)
            batch = retrieve_many_exact(
                state,
                queries,
                mode=mode,
                verify_indexed_exactness=verify_indexed_exactness,
            )
            for result_index, (result, exact_match) in enumerate(
                zip(batch.results, batch.exact_matches)
            ):
                # Batch-level timings are accounted once; per-query algorithmic
                # counters remain exact and additive.
                stats.add(
                    result,
                    memory_size,
                    exact_match=exact_match,
                    index_build_seconds=(
                        batch.index_build_seconds if result_index == 0 else 0.0
                    ),
                    search_seconds=batch.search_seconds if result_index == 0 else 0.0,
                    verification_seconds=(
                        batch.verification_seconds if result_index == 0 else 0.0
                    ),
                )
            query_state = _scalar_integrate_matrix(model, query_state, batch.values)

        fused[batch_index] = query_state

    logits = model.backbone.lm_head(fused)
    if update_memory:
        for batch_index, state in enumerate(states):
            write_before = state.write_seconds_total
            state.write(keys[batch_index], hidden[batch_index])
            stats.write_seconds += state.write_seconds_total - write_before
            stats.state_payload_bytes = max(stats.state_payload_bytes, state.payload_bytes())
    return logits, stats
