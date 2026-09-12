from __future__ import annotations

import pytest
import torch

from tam_research.chm_v1_post_materialization_profiler import (
    SCIENTIFIC_SEEDS,
    SYNTHETIC_PROFILE_SEED,
    build_query_batch,
    deterministic_signature,
    profile_query_kernel_once,
    synthetic_state_payload,
    validate_profile_seed,
)


def test_scientific_seeds_are_refused() -> None:
    for seed in SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            validate_profile_seed(seed)
        with pytest.raises(RuntimeError, match="scientific seed refused"):
            profile_query_kernel_once(
                session_length=32,
                query_count=4,
                geometry="exact",
                seed=seed,
            )


def test_payload_and_queries_are_cpu_deterministic() -> None:
    keys_a, values_a = synthetic_state_payload(
        session_length=64,
        key_width=16,
        value_width=24,
        seed=SYNTHETIC_PROFILE_SEED,
    )
    keys_b, values_b = synthetic_state_payload(
        session_length=64,
        key_width=16,
        value_width=24,
        seed=SYNTHETIC_PROFILE_SEED,
    )
    assert keys_a.device.type == "cpu"
    assert values_a.device.type == "cpu"
    assert torch.equal(keys_a, keys_b)
    assert torch.equal(values_a, values_b)

    for geometry, noise_std in (("exact", 0.0), ("near", 0.05), ("random", 0.0)):
        qa = build_query_batch(
            keys_a,
            query_count=8,
            geometry=geometry,
            noise_std=noise_std,
            seed=SYNTHETIC_PROFILE_SEED + 1,
        )
        qb = build_query_batch(
            keys_b,
            query_count=8,
            geometry=geometry,
            noise_std=noise_std,
            seed=SYNTHETIC_PROFILE_SEED + 1,
        )
        assert qa.anchor_positions == qb.anchor_positions
        assert qa.geometry == qb.geometry
        assert qa.noise_std == qb.noise_std
        assert torch.equal(qa.queries, qb.queries)
        assert qa.queries.device.type == "cpu"
        assert torch.isfinite(qa.queries).all()


def test_exact_geometry_is_exact_anchor_copy() -> None:
    keys, _ = synthetic_state_payload(session_length=17, key_width=8, value_width=4)
    batch = build_query_batch(keys, query_count=5, geometry="exact", noise_std=0.0)
    expected = keys[torch.tensor(batch.anchor_positions, dtype=torch.long)]
    assert torch.equal(batch.queries, expected)
    assert batch.noise_std == 0.0


def test_near_geometry_is_normalized_and_not_exact() -> None:
    keys, _ = synthetic_state_payload(session_length=32, key_width=16, value_width=4)
    batch = build_query_batch(
        keys,
        query_count=8,
        geometry="near",
        noise_std=0.10,
        seed=SYNTHETIC_PROFILE_SEED + 9,
    )
    anchors = keys[torch.tensor(batch.anchor_positions, dtype=torch.long)]
    assert not torch.equal(batch.queries, anchors)
    norms = torch.linalg.vector_norm(batch.queries, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)


def test_random_geometry_is_normalized() -> None:
    keys, _ = synthetic_state_payload(session_length=32, key_width=16, value_width=4)
    batch = build_query_batch(
        keys,
        query_count=8,
        geometry="random",
        noise_std=0.0,
        seed=SYNTHETIC_PROFILE_SEED + 11,
    )
    norms = torch.linalg.vector_norm(batch.queries, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("geometry", "noise_std"),
    [("exact", 0.0), ("near", 0.05), ("random", 0.0)],
)
def test_profiler_preserves_exact_answer_and_wrapper_parity(
    geometry: str,
    noise_std: float,
) -> None:
    result = profile_query_kernel_once(
        session_length=96,
        query_count=12,
        geometry=geometry,
        noise_std=noise_std,
        key_width=16,
        value_width=24,
        seed=SYNTHETIC_PROFILE_SEED,
    )
    assert result["measurement_kind"] == "zero_credit_systems_diagnostic_cpu"
    assert result["scientific_credit"] is False
    assert result["paid_compute"] is False
    assert result["modal_trigger"] is False
    assert result["device"] == "cpu"
    assert result["exact_match_count"] == 12
    assert result["wrapper_match_count"] == 12
    assert result["flat_item_ids"] == result["indexed_item_ids"]
    assert result["flat_address_vector_reads"] == 96 * 12
    assert 0 < result["indexed_address_vector_reads"] <= 96 * 12
    assert 0.0 < result["indexed_address_read_fraction"] <= 1.0
    assert result["directory_nodes_visited"] > 0
    assert result["directory_nodes_per_query"] > 0.0
    assert result["flat_query_ns"] > 0
    assert result["indexed_query_ns"] > 0
    assert result["contiguous_wrapper_ns"] > 0
    assert result["timing_is_ci_gate"] is False
    assert result["lower_bound_call_count"] is None
    assert "frozen ExactEpisodicIndex" in result["lower_bound_measurement"]
    assert result["accounting"]["warm_retrieve_full_state_materializations"] == 0
    assert result["accounting"]["warm_retrieve_np_stack_calls"] == 0


def test_non_timing_signature_is_deterministic() -> None:
    first = profile_query_kernel_once(
        session_length=80,
        query_count=10,
        geometry="near",
        noise_std=0.075,
        key_width=12,
        value_width=20,
        seed=SYNTHETIC_PROFILE_SEED + 20,
    )
    second = profile_query_kernel_once(
        session_length=80,
        query_count=10,
        geometry="near",
        noise_std=0.075,
        key_width=12,
        value_width=20,
        seed=SYNTHETIC_PROFILE_SEED + 20,
    )
    assert deterministic_signature(first) == deterministic_signature(second)


def test_query_count_accounting_scales_without_timing_assertions() -> None:
    small = profile_query_kernel_once(
        session_length=64,
        query_count=1,
        geometry="random",
        key_width=8,
        value_width=8,
        seed=SYNTHETIC_PROFILE_SEED + 30,
    )
    larger = profile_query_kernel_once(
        session_length=64,
        query_count=8,
        geometry="random",
        key_width=8,
        value_width=8,
        seed=SYNTHETIC_PROFILE_SEED + 30,
    )
    assert small["flat_address_vector_reads"] == 64
    assert larger["flat_address_vector_reads"] == 64 * 8
    assert small["exact_match_count"] == 1
    assert larger["exact_match_count"] == 8
    assert small["timing_is_ci_gate"] is False
    assert larger["timing_is_ci_gate"] is False


def test_invalid_configuration_is_rejected() -> None:
    keys, _ = synthetic_state_payload(session_length=8, key_width=4, value_width=4)
    with pytest.raises(ValueError, match="query_count"):
        build_query_batch(keys, query_count=0, geometry="exact")
    with pytest.raises(ValueError, match="query_count"):
        build_query_batch(keys, query_count=9, geometry="exact")
    with pytest.raises(ValueError, match="positive noise_std"):
        build_query_batch(keys, query_count=2, geometry="near", noise_std=0.0)
    with pytest.raises(ValueError, match="unknown query geometry"):
        build_query_batch(keys, query_count=2, geometry="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="session_length"):
        synthetic_state_payload(session_length=0)
