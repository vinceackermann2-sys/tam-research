from __future__ import annotations

import inspect

import numpy as np
import pytest

import tam_research.chm_v1_traversal_microcost_profiler as profiler_module
from tam_research.chm_v1_exact_index import ExactEpisodicIndex
from tam_research.chm_v1_post_materialization_profiler import (
    build_query_batch,
    synthetic_state_payload,
)
from tam_research.chm_v1_traversal_microcost_profiler import (
    deterministic_signature,
    profile_traversal_microcost_once,
    trace_indexed_search,
)


def _case(
    *,
    seed: int,
    rows: int = 96,
    width: int = 8,
    query_count: int = 8,
    geometry: str = "near",
    leaf_size: int = 8,
) -> tuple[ExactEpisodicIndex, list[np.ndarray]]:
    keys, _ = synthetic_state_payload(
        session_length=rows,
        key_width=width,
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
        np.arange(rows, dtype=np.int64),
        leaf_size=leaf_size,
    )
    queries = [
        query.detach().float().cpu().numpy().astype(np.float32, copy=False)
        for query in batch.queries
    ]
    return index, queries


def _identity(result: object) -> tuple[int, int, float, int, int]:
    return (
        int(result.item_id),
        int(result.position),
        float(result.squared_distance),
        int(result.address_vector_reads),
        int(result.directory_nodes_visited),
    )


@pytest.mark.parametrize("geometry", ["exact", "near", "random"])
def test_trace_replays_frozen_semantics_and_accounting_exactly(geometry: str) -> None:
    index, queries = _case(seed=938_100, geometry=geometry)

    for query in queries:
        flat = index.flat_search(query)
        frozen = index.indexed_search(query)
        replay = trace_indexed_search(index, query)

        assert (flat.item_id, flat.position) == (frozen.item_id, frozen.position)
        assert _identity(replay.result) == _identity(frozen)
        assert replay.counts.vector_reads == frozen.address_vector_reads
        assert replay.counts.node_visits == frozen.directory_nodes_visited
        assert replay.counts.candidate_iterations == replay.counts.vector_reads
        assert replay.counts.lower_bound_calls == 1 + 2 * replay.counts.internal_node_visits
        assert replay.counts.heap_pop_calls in (
            replay.counts.node_visits,
            replay.counts.node_visits + 1,
        )
        assert replay.counts.leaf_visits >= 1
        assert replay.first_leaf_positions
        assert replay.bound_values


def test_profile_counts_match_cprofile_call_counts_and_frozen_accounting() -> None:
    result = profile_traversal_microcost_once(
        session_length=128,
        query_count=12,
        geometry="near",
        noise_std=0.05,
        key_width=12,
        seed=938_200,
        leaf_size=8,
        baseline_repeats=1,
        micro_repeats=8,
    )

    assert result["measurement_kind"] == "zero_credit_traversal_microcost_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["timing_is_ci_gate"] is False
    assert result["correctness"]["flat_answer_parity"] is True
    assert result["correctness"]["trace_full_frozen_accounting_parity"] is True

    counts = result["operation_counts"]
    accounting = result["frozen_accounting"]
    assert counts["vector_reads"] == accounting["address_vector_reads"]
    assert counts["node_visits"] == accounting["directory_nodes_visited"]
    assert counts["candidate_iterations"] == counts["vector_reads"]
    assert counts["lower_bound_calls"] == 1 * result["query_count"] + 2 * counts["internal_node_visits"]
    assert 0.0 < accounting["address_read_fraction"] <= 1.0

    functions = result["cprofile"]["functions"]
    assert functions["indexed_search"]["calls"] == result["query_count"]
    assert functions["_lower_bound_sq"]["calls"] == counts["lower_bound_calls"]
    assert functions["_squared_distances"]["calls"] == counts["leaf_visits"]
    assert functions["heapq.heappush"]["calls"] == counts["heap_push_calls"]
    assert functions["heapq.heappop"]["calls"] == counts["heap_pop_calls"]


def test_descriptive_timing_fields_exist_but_have_no_threshold_gate() -> None:
    result = profile_traversal_microcost_once(
        session_length=96,
        query_count=6,
        geometry="random",
        key_width=8,
        seed=938_300,
        leaf_size=8,
        baseline_repeats=1,
        micro_repeats=4,
    )

    assert isinstance(result["unprofiled_frozen_indexed_ns"], int)
    assert result["unprofiled_frozen_indexed_ns"] >= 0
    for value in result["traced_timing_ns"].values():
        assert isinstance(value, int)
        assert value >= 0
    for value in result["traced_component_fractions"].values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    micro = result["isolated_replay"]
    assert micro["array_pair_matches_scalar_exactly"] is True
    assert micro["repeats"] == 4
    assert micro["representative_leaf_items"] >= 1
    for key in (
        "frozen_two_scalar_bounds_total_ns",
        "array_backed_child_bounds_total_ns",
        "heap_pop_push_pair_total_ns",
        "leaf_distance_and_index_total_ns",
        "candidate_loop_total_ns",
    ):
        assert isinstance(micro[key], int)
        assert micro[key] >= 0
    assert isinstance(micro["array_over_scalar_bound_pair_ratio"], float)
    assert micro["array_over_scalar_bound_pair_ratio"] >= 0.0


def test_deterministic_signature_excludes_all_descriptive_timing() -> None:
    kwargs = dict(
        session_length=64,
        query_count=4,
        geometry="near",
        noise_std=0.05,
        key_width=8,
        seed=938_400,
        leaf_size=4,
        baseline_repeats=1,
        micro_repeats=2,
    )
    first = profile_traversal_microcost_once(**kwargs)
    second = profile_traversal_microcost_once(**kwargs)

    assert deterministic_signature(first) == deterministic_signature(second)
    signature = deterministic_signature(first)
    assert "unprofiled_frozen_indexed_ns" not in signature
    assert "traced_timing_ns" not in signature
    assert "traced_component_fractions" not in signature
    assert "cprofile" not in signature
    assert "isolated_replay" not in signature


def test_profiler_refuses_all_frozen_scientific_seeds() -> None:
    for seed in profiler_module.SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            profile_traversal_microcost_once(
                session_length=32,
                query_count=2,
                geometry="exact",
                key_width=4,
                seed=seed,
                leaf_size=4,
                baseline_repeats=1,
                micro_repeats=1,
            )


def test_profiler_source_has_no_gpu_modal_or_training_execution_path() -> None:
    source = inspect.getsource(profiler_module).lower()
    assert "torch.cuda" not in source
    assert ".cuda(" not in source
    assert "modal." not in source
    assert "optimizer" not in source
    assert "backward(" not in source
    assert "8612" in source and "8613" in source
    assert "not scientific evidence" in source
