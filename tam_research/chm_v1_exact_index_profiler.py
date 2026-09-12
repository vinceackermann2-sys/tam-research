from __future__ import annotations

"""CPU-only engineering profiler for the frozen CHM-v1 exact index.

This module is additive diagnostic plumbing for follow-up issue #927. It does
not run models, allocate a GPU, call Modal, consume scientific seeds, or change
the frozen CHM-v1 retrieval semantics. Timing values are engineering
measurements only and are never pass/fail thresholds.
"""

import argparse
import json
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .chm_v1_exact_index import ExactEpisodicIndex, KDNode, SearchResult
from .chm_v1_small_lm import ADDRESS_DIM, LEAF_SIZE, SCIENTIFIC_SEEDS, EpisodicState

SYNTHETIC_PROFILER_SEED = 927_001
DEFAULT_VALUE_WIDTH = 256
MEASUREMENT_KIND = "engineering_only_cpu"
INTERPRETATION_CEILING = (
    "Engineering measurement of current CHM-v1 exact-index implementation "
    "overhead only; not scientific evidence, not a scalability claim, and not "
    "authority to execute another scientific seed."
)


def _validate_inputs(
    *,
    session_length: int,
    query_count: int,
    key_width: int,
    value_width: int,
    seed: int,
) -> None:
    if int(seed) in SCIENTIFIC_SEEDS:
        raise RuntimeError(f"scientific seed refused by CPU-only profiler: {seed}")
    if session_length < 1:
        raise ValueError("session_length must be positive")
    if query_count < 1 or query_count > session_length:
        raise ValueError("query_count must be in [1, session_length]")
    if key_width < 1:
        raise ValueError("key_width must be positive")
    if value_width < 1:
        raise ValueError("value_width must be positive")


def _tree_build_accounting(root: KDNode, key_width: int) -> dict[str, int]:
    """Reconstruct deterministic logical materialization counts from the tree.

    ``chm_v1_exact_index._build`` materializes ``points[positions]`` at every
    node, computes two bounds over that block, and, for internal nodes, stable
    sorts one coordinate and materializes a reordered positions array. These
    counts describe logical implementation work; they are not wall-clock
    estimates.
    """

    totals = {
        "tree_nodes": 0,
        "tree_internal_nodes": 0,
        "tree_leaf_nodes": 0,
        "build_block_rows_materialized": 0,
        "build_block_elements_materialized": 0,
        "build_bound_reduction_input_elements": 0,
        "build_sort_items": 0,
        "build_ordered_position_scalars_materialized": 0,
        "build_leaf_position_scalars_copied": 0,
        "tree_bound_scalars_stored": 0,
    }

    def walk(node: KDNode) -> int:
        totals["tree_nodes"] += 1
        totals["tree_bound_scalars_stored"] += 2 * key_width

        if node.item_positions is not None:
            rows = int(len(node.item_positions))
            totals["tree_leaf_nodes"] += 1
            totals["build_leaf_position_scalars_copied"] += rows
        else:
            if node.left is None or node.right is None:
                raise RuntimeError("malformed exact index node in profiler")
            totals["tree_internal_nodes"] += 1
            rows = walk(node.left) + walk(node.right)
            totals["build_sort_items"] += rows
            totals["build_ordered_position_scalars_materialized"] += rows

        elements = rows * key_width
        totals["build_block_rows_materialized"] += rows
        totals["build_block_elements_materialized"] += elements
        # _build calls block.min(axis=0) and block.max(axis=0).
        totals["build_bound_reduction_input_elements"] += 2 * elements
        return rows

    root_rows = walk(root)
    totals["tree_root_rows"] = root_rows
    return totals


def _sum_result_field(results: list[SearchResult], field: str) -> int:
    return int(sum(int(getattr(result, field)) for result in results))


def _time_flat_queries(
    index: ExactEpisodicIndex, queries: np.ndarray
) -> tuple[int, list[SearchResult]]:
    started = time.perf_counter_ns()
    outputs = [index.flat_search(query) for query in queries]
    return time.perf_counter_ns() - started, outputs


def _time_indexed_queries(
    index: ExactEpisodicIndex, queries: np.ndarray
) -> tuple[int, list[SearchResult]]:
    started = time.perf_counter_ns()
    outputs = [index.indexed_search(query) for query in queries]
    return time.perf_counter_ns() - started, outputs


def _time_state_retrievals(
    state: EpisodicState,
    queries: list[torch.Tensor],
    *,
    mode: str,
    verify_indexed_exactness: bool,
) -> int:
    started = time.perf_counter_ns()
    for query in queries:
        state.retrieve(
            query,
            mode=mode,  # type: ignore[arg-type]
            verify_indexed_exactness=verify_indexed_exactness,
        )
    return time.perf_counter_ns() - started


def profile_exact_index_overhead(
    *,
    session_length: int,
    query_count: int = 8,
    key_width: int = ADDRESS_DIM,
    value_width: int = DEFAULT_VALUE_WIDTH,
    seed: int = SYNTHETIC_PROFILER_SEED,
) -> dict[str, Any]:
    """Profile the current exact-index implementation on deterministic CPU data."""

    session_length = int(session_length)
    query_count = int(query_count)
    key_width = int(key_width)
    value_width = int(value_width)
    seed = int(seed)
    _validate_inputs(
        session_length=session_length,
        query_count=query_count,
        key_width=key_width,
        value_width=value_width,
        seed=seed,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    keys = F.normalize(
        torch.randn(session_length, key_width, generator=generator, device="cpu"),
        dim=-1,
    ).to(dtype=torch.float32)
    values = torch.randn(
        session_length,
        value_width,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )

    state = EpisodicState(session_id=f"chm-v1-index-profiler-{seed}")
    prefill_started = time.perf_counter_ns()
    state.write(keys, values)
    prefill_ns = time.perf_counter_ns() - prefill_started

    arrays_started = time.perf_counter_ns()
    keys_np, values_np, item_ids_np = state._arrays()
    state_arrays_materialization_ns = time.perf_counter_ns() - arrays_started

    tree_only_started = time.perf_counter_ns()
    standalone_index = ExactEpisodicIndex(keys_np, item_ids_np, leaf_size=LEAF_SIZE)
    index_tree_only_build_ns = time.perf_counter_ns() - tree_only_started

    state_index_started = time.perf_counter_ns()
    index = state.index()
    state_index_build_with_arrays_ns = time.perf_counter_ns() - state_index_started
    if state.index_build_count != 1:
        raise AssertionError("profiler expected exactly one cached index build")

    query_positions = np.linspace(
        0, session_length - 1, num=query_count, dtype=np.int64
    )
    query_tensors = [keys[int(position)].clone() for position in query_positions]
    queries_np = np.stack(
        [query.detach().cpu().numpy().astype(np.float32, copy=False) for query in query_tensors],
        axis=0,
    )

    flat_query_only_ns, flat_results = _time_flat_queries(index, queries_np)
    indexed_query_only_ns, indexed_results = _time_indexed_queries(index, queries_np)

    parity = all(
        indexed.item_id == flat.item_id
        and indexed.position == flat.position
        and indexed.squared_distance == flat.squared_distance
        for flat, indexed in zip(flat_results, indexed_results)
    )
    if not parity:
        raise AssertionError("CPU profiler observed indexed/flat exactness divergence")

    # The cache is already populated. These timings therefore expose the warm
    # wrapper path, including EpisodicState._arrays() on every retrieval.
    flat_retrieve_warm_wrapper_ns = _time_state_retrievals(
        state,
        query_tensors,
        mode="flat",
        verify_indexed_exactness=False,
    )
    indexed_retrieve_warm_wrapper_no_verify_ns = _time_state_retrievals(
        state,
        query_tensors,
        mode="indexed",
        verify_indexed_exactness=False,
    )
    indexed_retrieve_warm_wrapper_with_verify_ns = _time_state_retrievals(
        state,
        query_tensors,
        mode="indexed",
        verify_indexed_exactness=True,
    )

    flat_reads = _sum_result_field(flat_results, "address_vector_reads")
    indexed_reads = _sum_result_field(indexed_results, "address_vector_reads")
    indexed_nodes = _sum_result_field(indexed_results, "directory_nodes_visited")
    expected_flat_reads = session_length * query_count
    if flat_reads != expected_flat_reads:
        raise AssertionError(
            f"flat accounting mismatch: observed={flat_reads} expected={expected_flat_reads}"
        )

    tree_accounting = _tree_build_accounting(standalone_index.root, key_width)
    if tree_accounting["tree_root_rows"] != session_length:
        raise AssertionError("tree accounting did not recover the full synthetic session")

    per_arrays_key_elements = session_length * key_width
    per_arrays_value_elements = session_length * value_width
    per_arrays_id_scalars = session_length
    per_arrays_total_elements = (
        per_arrays_key_elements + per_arrays_value_elements + per_arrays_id_scalars
    )

    implementation = {
        # EpisodicState.retrieve calls _arrays once even with a warm cached index.
        "warm_retrieve_state_array_materializations_per_query": 1,
        "cold_first_indexed_retrieve_state_array_materializations": 2,
        "per_state_arrays_key_rows_stacked": session_length,
        "per_state_arrays_value_rows_stacked": session_length,
        "per_state_arrays_key_elements_stacked": per_arrays_key_elements,
        "per_state_arrays_value_elements_stacked": per_arrays_value_elements,
        "per_state_arrays_item_id_scalars_materialized": per_arrays_id_scalars,
        "per_state_arrays_total_numpy_elements_materialized": per_arrays_total_elements,
        "measured_warm_indexed_wrapper_key_elements_stacked": (
            query_count * per_arrays_key_elements
        ),
        "measured_warm_indexed_wrapper_value_elements_stacked": (
            query_count * per_arrays_value_elements
        ),
        "measured_warm_indexed_wrapper_item_id_scalars_materialized": (
            query_count * per_arrays_id_scalars
        ),
        "prefill_python_list_appends": 3 * session_length,
        # _validate performs np.unique(item_ids) and a finite check over points.
        "index_constructor_unique_id_inputs": session_length,
        "index_constructor_finiteness_elements_checked": per_arrays_key_elements,
        # Every visited leaf materializes self.points[positions], then converts
        # positions and distances to Python lists for the candidate loop.
        "indexed_candidate_rows_materialized": indexed_reads,
        "indexed_candidate_elements_materialized": indexed_reads * key_width,
        "indexed_candidate_position_tolist_scalars": indexed_reads,
        "indexed_candidate_distance_tolist_scalars": indexed_reads,
        "indexed_python_candidate_iterations": indexed_reads,
        # Exhaustive flat work used for answer parity and optional verification.
        "flat_distance_rows_evaluated": flat_reads,
        "flat_distance_elements_evaluated": flat_reads * key_width,
        "flat_lexsort_items": flat_reads,
        "correctness_audit_flat_address_vector_reads": flat_reads,
        "verify_exact_flat_address_vector_reads_for_measured_queries": flat_reads,
        # This frozen implementation has no torch.isclose/nonzero visibility
        # scan in the active retrieval path. Keep explicit zeroes so a later
        # implementation cannot silently inherit a false accounting assumption.
        "torch_isclose_element_checks": 0,
        "torch_nonzero_calls": 0,
        "visibility_hit_check_rows_scanned": 0,
        **tree_accounting,
    }

    return {
        "measurement_kind": MEASUREMENT_KIND,
        "device": "cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "seed": seed,
        "scientific_seed": False,
        "config": {
            "session_length": session_length,
            "query_count": query_count,
            "key_width": key_width,
            "value_width": value_width,
            "leaf_size": LEAF_SIZE,
        },
        "timing_ns": {
            "prefill_state_write_ns": int(prefill_ns),
            "state_arrays_materialization_ns": int(state_arrays_materialization_ns),
            "index_tree_only_build_ns": int(index_tree_only_build_ns),
            "state_index_build_with_arrays_ns": int(state_index_build_with_arrays_ns),
            "flat_query_only_ns": int(flat_query_only_ns),
            "indexed_query_only_ns": int(indexed_query_only_ns),
            "flat_retrieve_warm_wrapper_ns": int(flat_retrieve_warm_wrapper_ns),
            "indexed_retrieve_warm_wrapper_no_verify_ns": int(
                indexed_retrieve_warm_wrapper_no_verify_ns
            ),
            "indexed_retrieve_warm_wrapper_with_verify_ns": int(
                indexed_retrieve_warm_wrapper_with_verify_ns
            ),
        },
        "algorithmic_accounting": {
            "flat_exact_address_vector_reads": flat_reads,
            "indexed_address_vector_reads": indexed_reads,
            "indexed_directory_nodes_visited": indexed_nodes,
            "indexed_read_fraction_of_flat": indexed_reads / max(flat_reads, 1),
        },
        "implementation_accounting": implementation,
        "correctness": {
            "flat_indexed_exact_answer_parity": parity,
            "queries_checked": query_count,
        },
        "interpretation_ceiling": INTERPRETATION_CEILING,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only CHM-v1 exact-index implementation overhead profiler"
    )
    parser.add_argument(
        "--session-lengths",
        nargs="+",
        type=int,
        default=[64, 256, 1024],
    )
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--key-width", type=int, default=ADDRESS_DIM)
    parser.add_argument("--value-width", type=int, default=DEFAULT_VALUE_WIDTH)
    parser.add_argument("--seed", type=int, default=SYNTHETIC_PROFILER_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profiles = [
        profile_exact_index_overhead(
            session_length=session_length,
            query_count=args.queries,
            key_width=args.key_width,
            value_width=args.value_width,
            seed=args.seed,
        )
        for session_length in args.session_lengths
    ]
    print(
        json.dumps(
            {
                "measurement_kind": MEASUREMENT_KIND,
                "scientific_credit": False,
                "paid_compute": False,
                "profiles": profiles,
                "interpretation_ceiling": INTERPRETATION_CEILING,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
