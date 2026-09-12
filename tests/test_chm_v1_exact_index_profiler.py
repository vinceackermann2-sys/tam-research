from __future__ import annotations

import inspect
import json

import pytest

import tam_research.chm_v1_exact_index_profiler as profiler
from tam_research.chm_v1_small_lm import SCIENTIFIC_SEEDS


def _profile(session_length: int) -> dict:
    return profiler.profile_exact_index_overhead(
        session_length=session_length,
        query_count=4,
        key_width=8,
        value_width=16,
        seed=profiler.SYNTHETIC_PROFILER_SEED,
    )


def test_profiler_exact_parity_and_separates_sparse_from_linear_work() -> None:
    result = _profile(32)

    assert result["measurement_kind"] == "engineering_only_cpu"
    assert result["device"] == "cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["scientific_seed"] is False
    assert result["correctness"]["flat_indexed_exact_answer_parity"] is True
    assert result["correctness"]["queries_checked"] == 4

    algorithmic = result["algorithmic_accounting"]
    assert algorithmic["flat_exact_address_vector_reads"] == 32 * 4
    assert 0 < algorithmic["indexed_address_vector_reads"] <= 32 * 4
    assert algorithmic["indexed_directory_nodes_visited"] > 0
    assert algorithmic["indexed_read_fraction_of_flat"] <= 1.0

    implementation = result["implementation_accounting"]
    assert implementation["warm_retrieve_state_array_materializations_per_query"] == 1
    assert implementation["cold_first_indexed_retrieve_state_array_materializations"] == 2
    assert implementation["per_state_arrays_key_rows_stacked"] == 32
    assert implementation["per_state_arrays_value_rows_stacked"] == 32
    assert implementation["per_state_arrays_key_elements_stacked"] == 32 * 8
    assert implementation["per_state_arrays_value_elements_stacked"] == 32 * 16
    assert implementation["per_state_arrays_item_id_scalars_materialized"] == 32
    assert implementation["per_state_arrays_total_numpy_elements_materialized"] == (
        32 * 8 + 32 * 16 + 32
    )
    assert implementation["measured_warm_indexed_wrapper_key_elements_stacked"] == 4 * 32 * 8
    assert implementation["measured_warm_indexed_wrapper_value_elements_stacked"] == 4 * 32 * 16
    assert implementation["indexed_candidate_rows_materialized"] == algorithmic[
        "indexed_address_vector_reads"
    ]
    assert implementation["indexed_candidate_position_tolist_scalars"] == algorithmic[
        "indexed_address_vector_reads"
    ]
    assert implementation["indexed_candidate_distance_tolist_scalars"] == algorithmic[
        "indexed_address_vector_reads"
    ]
    assert implementation["tree_root_rows"] == 32
    assert implementation["tree_nodes"] >= 1

    # The frozen branch does not use torch.isclose/nonzero as a visibility scan.
    assert implementation["torch_isclose_element_checks"] == 0
    assert implementation["torch_nonzero_calls"] == 0
    assert implementation["visibility_hit_check_rows_scanned"] == 0

    # Timing is recorded, but correctness never depends on a speed threshold.
    for elapsed_ns in result["timing_ns"].values():
        assert isinstance(elapsed_ns, int)
        assert elapsed_ns >= 0

    json.dumps(result, sort_keys=True)


def test_profiler_linear_accounting_is_monotone_without_timing_assertions() -> None:
    small = _profile(16)
    large = _profile(32)

    small_algorithmic = small["algorithmic_accounting"]
    large_algorithmic = large["algorithmic_accounting"]
    assert small_algorithmic["flat_exact_address_vector_reads"] == 16 * 4
    assert large_algorithmic["flat_exact_address_vector_reads"] == 32 * 4

    small_impl = small["implementation_accounting"]
    large_impl = large["implementation_accounting"]
    assert large_impl["per_state_arrays_key_elements_stacked"] > small_impl[
        "per_state_arrays_key_elements_stacked"
    ]
    assert large_impl["per_state_arrays_value_elements_stacked"] > small_impl[
        "per_state_arrays_value_elements_stacked"
    ]
    assert large_impl["per_state_arrays_item_id_scalars_materialized"] > small_impl[
        "per_state_arrays_item_id_scalars_materialized"
    ]
    assert large_impl["build_block_rows_materialized"] > small_impl[
        "build_block_rows_materialized"
    ]
    assert large_impl["flat_lexsort_items"] > small_impl["flat_lexsort_items"]


def test_profiler_refuses_all_frozen_scientific_seeds() -> None:
    for seed in SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            profiler.profile_exact_index_overhead(
                session_length=8,
                query_count=2,
                key_width=4,
                value_width=8,
                seed=seed,
            )


def test_profiler_source_has_no_cuda_path_and_keeps_interpretation_ceiling() -> None:
    source = inspect.getsource(profiler).lower()
    assert "torch.cuda" not in source
    assert ".cuda(" not in source
    assert "not scientific evidence" in profiler.INTERPRETATION_CEILING.lower()
    assert "not a scalability claim" in profiler.INTERPRETATION_CEILING.lower()
