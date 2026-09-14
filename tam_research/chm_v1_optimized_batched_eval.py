from __future__ import annotations

"""CPU-safe optimized batched retrieval integration for CHM-v1 issue #969.

The generic batched transport in :mod:`tam_research.chm_v1_batched_eval` is a
reference-equivalent transport helper whose indexed branch resolves through
``EpisodicState.index()``.  ``OptimizedEpisodicState`` intentionally keeps that
frozen index and exposes the #958 clipped traversal separately through
``optimized_index()``.  Therefore a caller that wants the clipped traversal
must dispatch to it explicitly rather than relying on the generic helper.

This module provides that explicit integration without changing the generic
reference helper, the frozen scientific/model/training implementation, any
launcher/workflow, or any paid/GPU path.
"""

import time

import numpy as np
import torch

from .chm_v1_batched_eval import (
    BatchedRetrieval,
    _scalar_integrate_matrix,
    _scalar_query_matrix,
)
from .chm_v1_optimized_replication import OptimizedEpisodicState
from .chm_v1_small_lm import (
    RETRIEVAL_HOPS,
    CHMV1EIEMLM,
    RetrievalStats,
    _hidden,
)

ISSUE = 969
DIAGNOSTIC_SEEDS = (969_001, 969_002)
BLOCKED_SCIENTIFIC_SEEDS = (19_591, 19_592, 19_593, 8_611, 8_612, 8_613)


def validate_diagnostic_seed(seed: int) -> int:
    """Refuse every scientific seed known to the predecessor protocols."""

    seed = int(seed)
    if seed in BLOCKED_SCIENTIFIC_SEEDS:
        raise RuntimeError(
            f"scientific seed refused by #969 zero-GPU systems diagnostic: {seed}"
        )
    return seed


def retrieve_many_optimized_exact(
    state: OptimizedEpisodicState,
    queries: torch.Tensor,
    *,
    verify_indexed_exactness: bool = True,
) -> BatchedRetrieval:
    """Run the #958 clipped exact traversal with one batched transport pair.

    Search remains scalar and deterministic on CPU, exactly like the generic
    batched transport.  The only routing difference is explicit use of
    ``state.optimized_index()`` rather than inherited ``state.index()``.
    """

    if not isinstance(state, OptimizedEpisodicState):
        raise TypeError("optimized batched retrieval requires OptimizedEpisodicState")
    if queries.ndim != 2:
        raise ValueError("queries must be [query_count, address_dim]")
    if queries.shape[0] == 0:
        raise ValueError("queries must contain at least one address vector")

    call_started = time.perf_counter()
    _, stored_values, _ = state._arrays()
    query_matrix = queries.detach().float().cpu().numpy().astype(np.float32, copy=False)

    build_before = state.index_build_seconds_total
    optimized = state.optimized_index()
    build_delta = state.index_build_seconds_total - build_before

    started = time.perf_counter()
    results = tuple(optimized.indexed_search(query) for query in query_matrix)
    search_elapsed = time.perf_counter() - started
    state.indexed_search_seconds_total += search_elapsed

    if verify_indexed_exactness:
        verify_started = time.perf_counter()
        flat_results = tuple(optimized.frozen.flat_search(query) for query in query_matrix)
        verification_elapsed = time.perf_counter() - verify_started
        state.verification_seconds_total += verification_elapsed
        exact_matches = tuple(
            indexed.item_id == flat.item_id
            and indexed.position == flat.position
            and indexed.squared_distance == flat.squared_distance
            for indexed, flat in zip(results, flat_results)
        )
        if not all(exact_matches):
            mismatch = next(i for i, match in enumerate(exact_matches) if not match)
            raise AssertionError(
                f"optimized/flat mismatch in session {state.session_id} at batch query "
                f"{mismatch}: optimized={results[mismatch]} flat={flat_results[mismatch]}"
            )
    else:
        verification_elapsed = 0.0
        exact_matches = (True,) * len(results)

    positions = np.asarray([result.position for result in results], dtype=np.int64)
    selected = np.ascontiguousarray(stored_values[positions])
    values = torch.as_tensor(selected, device=queries.device, dtype=queries.dtype)

    total_elapsed = time.perf_counter() - call_started
    wrapper_elapsed = max(
        0.0,
        total_elapsed - build_delta - search_elapsed - verification_elapsed,
    )
    state.optimized_wrapper_seconds_total += wrapper_elapsed

    return BatchedRetrieval(
        values=values,
        results=results,
        exact_matches=exact_matches,
        index_build_seconds=build_delta,
        search_seconds=search_elapsed,
        verification_seconds=verification_elapsed,
    )


@torch.no_grad()
def forward_session_chunk_optimized_batched_transport(
    model: CHMV1EIEMLM,
    tokens: torch.Tensor,
    states: list[OptimizedEpisodicState],
    *,
    update_memory: bool = True,
    verify_indexed_exactness: bool = True,
) -> tuple[torch.Tensor, RetrievalStats]:
    """Reference-equivalent chunk evaluation using the clipped exact traversal."""

    if tokens.shape[0] != len(states):
        raise ValueError("one OptimizedEpisodicState is required per batch element")
    if any(not isinstance(state, OptimizedEpisodicState) for state in states):
        raise TypeError("all states must be OptimizedEpisodicState instances")

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
            batch = retrieve_many_optimized_exact(
                state,
                queries,
                verify_indexed_exactness=verify_indexed_exactness,
            )
            for result_index, (result, exact_match) in enumerate(
                zip(batch.results, batch.exact_matches)
            ):
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
