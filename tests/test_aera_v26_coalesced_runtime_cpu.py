from __future__ import annotations

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import diagnostic_config
from tam_research.aera import AERAState
from tam_research.aera_hardware_core_v23 import sparse_write_budget
from tam_research.aera_hardware_core_v24 import (
    EPISODIC_CAPACITY,
    READ_TOP_K,
    ContextualEpisodicMemoryState,
    _merge_epi_state,
    _select_epi_state,
)
from tam_research.aera_hardware_core_v25_1_nohost import (
    HardwareAwareAERATextLMV251NoHostTelemetry,
)
from tam_research.aera_hardware_core_v26 import (
    CoalescedFICEMMemory,
    HardwareAwareAERATextLMV26,
    coalesced_runtime_v26_protocol,
    merge_packed_epi_state,
    pack_ephemeral_epi_state,
    select_packed_epi_state,
    unpack_ephemeral_epi_state,
)


ATOL = 1e-6
RTOL = 1e-6


def _random_state(
    *,
    batch: int,
    d_model: int,
    memory_dim: int,
    seed: int,
    capacity: int = EPISODIC_CAPACITY,
) -> AERAState:
    g = torch.Generator().manual_seed(seed)
    return AERAState(
        stream=torch.randn(batch, d_model, generator=g),
        memory=ContextualEpisodicMemoryState(
            keys=torch.randn(batch, capacity, memory_dim, generator=g),
            values=torch.randn(batch, capacity, memory_dim, generator=g),
            strengths=torch.rand(batch, capacity, generator=g),
            valid=torch.rand(batch, capacity, generator=g) > 0.43,
        ),
    )


def _assert_epi_state_equal(
    left: AERAState,
    right: AERAState,
    *,
    exact: bool = False,
) -> None:
    assert isinstance(left.memory, ContextualEpisodicMemoryState)
    assert isinstance(right.memory, ContextualEpisodicMemoryState)
    pairs = (
        (left.stream, right.stream),
        (left.memory.keys, right.memory.keys),
        (left.memory.values, right.memory.values),
        (left.memory.strengths, right.memory.strengths),
    )
    for a, b in pairs:
        if exact:
            assert torch.equal(a, b)
        else:
            assert torch.allclose(a, b, atol=ATOL, rtol=RTOL)
    assert torch.equal(left.memory.valid, right.memory.valid)


def _paired_models(seed: int = 39801):
    cfg = diagnostic_config()
    torch.manual_seed(seed)
    source = HardwareAwareAERATextLMV251NoHostTelemetry(cfg).eval()
    torch.manual_seed(seed)
    candidate = HardwareAwareAERATextLMV26(cfg).eval()
    candidate.load_state_dict(source.state_dict(), strict=True)
    return source, candidate


def _force_all_stages_run(model) -> None:
    with torch.no_grad():
        for router in model.stage_routers:
            router.proj.weight.zero_()
            router.proj.bias.fill_(12.0)


def _assert_controls_equal(source_output: dict[str, object], candidate_output: dict[str, object]) -> None:
    source_routes = source_output["stage_routes"]
    candidate_routes = candidate_output["stage_routes"]
    assert isinstance(source_routes, list)
    assert isinstance(candidate_routes, list)
    assert len(source_routes) == len(candidate_routes)
    for source_chunk, candidate_chunk in zip(source_routes, candidate_routes):
        assert len(source_chunk) == len(candidate_chunk)
        for source_row, candidate_row in zip(source_chunk, candidate_chunk):
            assert torch.equal(
                source_row["stage_route_gate"], candidate_row["stage_route_gate"]
            )
            assert torch.allclose(
                source_row["stage_route_probability"],
                candidate_row["stage_route_probability"],
                atol=ATOL,
                rtol=RTOL,
            )
            assert source_row["executed_fraction"] == candidate_row["executed_fraction"]
            for boundary in ("start", "end"):
                source_boundary = source_row[boundary]
                candidate_boundary = candidate_row[boundary]
                assert (source_boundary is None) == (candidate_boundary is None)
                if source_boundary is None:
                    continue
                assert source_boundary.keys() == candidate_boundary.keys()
                for key in source_boundary:
                    assert torch.allclose(
                        source_boundary[key],
                        candidate_boundary[key],
                        atol=ATOL,
                        rtol=RTOL,
                    )


def _assert_discrete_stage_telemetry_equal(source, candidate) -> None:
    for source_stage, candidate_stage in zip(source.stages, candidate.stages):
        source_selected = source_stage.last_selected_indices
        candidate_selected = candidate_stage.last_selected_indices
        assert (source_selected is None) == (candidate_selected is None)
        if source_selected is not None:
            assert torch.equal(source_selected, candidate_selected)

        source_counts = source_stage.experts.last_counts
        candidate_counts = candidate_stage.experts.last_counts
        assert (source_counts is None) == (candidate_counts is None)
        if source_counts is not None:
            assert torch.equal(source_counts, candidate_counts)

        source_steps = source_stage.reasoner.last_steps
        candidate_steps = candidate_stage.reasoner.last_steps
        assert (source_steps is None) == (candidate_steps is None)
        if source_steps is not None:
            assert torch.equal(source_steps, candidate_steps)


def test_v26_pack_unpack_is_bit_exact_including_invalid_slots():
    state = _random_state(batch=5, d_model=24, memory_dim=16, seed=39810)
    restored = unpack_ephemeral_epi_state(pack_ephemeral_epi_state(state))
    _assert_epi_state_equal(state, restored, exact=True)


def test_v26_coalesced_select_and_merge_match_v25_1_componentwise_state_motion():
    base = _random_state(batch=6, d_model=24, memory_dim=16, seed=39811)
    idx = torch.tensor([0, 2, 5], dtype=torch.long)

    legacy_selected = _select_epi_state(base, idx)
    coalesced_selected = unpack_ephemeral_epi_state(
        select_packed_epi_state(pack_ephemeral_epi_state(base), idx)
    )
    _assert_epi_state_equal(legacy_selected, coalesced_selected, exact=True)

    update = _random_state(batch=3, d_model=24, memory_dim=16, seed=39812)
    legacy_merged = _merge_epi_state(base, update, idx)
    coalesced_merged = unpack_ephemeral_epi_state(
        merge_packed_epi_state(
            pack_ephemeral_epi_state(base),
            pack_ephemeral_epi_state(update),
            idx,
        )
    )
    _assert_epi_state_equal(legacy_merged, coalesced_merged, exact=True)


def test_v26_state_dict_and_parameter_geometry_are_identical_to_final_v25_1():
    source, candidate = _paired_models(39813)
    source_state = source.state_dict()
    candidate_state = candidate.state_dict()
    assert source_state.keys() == candidate_state.keys()
    assert sum(p.numel() for p in source.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )
    assert [name for name, _ in source.named_parameters()] == [
        name for name, _ in candidate.named_parameters()
    ]
    for name in source_state:
        assert torch.equal(source_state[name], candidate_state[name])
    assert not any("_execution_backend" in key for key in candidate_state)


def test_v26_ficem_reference_backend_matches_reads_for_empty_mixed_and_full_state():
    source, candidate = _paired_models(39814)
    source_memory = source.stages[0].memory
    candidate_memory = candidate.stages[0].memory
    assert isinstance(candidate_memory, CoalescedFICEMMemory)
    assert candidate_memory.execution_backend_name == "torch-reference-v25.1-exact"

    cfg = diagnostic_config()
    g = torch.Generator().manual_seed(39815)
    identity = torch.randn(3, cfg.chunk_size, cfg.d_model, generator=g)
    context = torch.randn(3, cfg.chunk_size, cfg.d_model, generator=g)

    for valid_kind in ("empty", "mixed", "full"):
        state = _random_state(
            batch=3,
            d_model=cfg.d_model,
            memory_dim=cfg.memory_dim,
            seed={"empty": 39816, "mixed": 39817, "full": 39818}[valid_kind],
        ).memory
        assert isinstance(state, ContextualEpisodicMemoryState)
        if valid_kind == "empty":
            state.valid.zero_()
        elif valid_kind == "full":
            state.valid.fill_(True)

        source_read = source_memory.read_with_reuse(identity, context, state)
        candidate_read = candidate_memory.read_with_reuse(identity, context, state)
        for source_tensor, candidate_tensor in zip(source_read, candidate_read):
            assert (source_tensor is None) == (candidate_tensor is None)
            if source_tensor is not None:
                assert torch.allclose(
                    source_tensor, candidate_tensor, atol=ATOL, rtol=RTOL
                )

    assert candidate_memory.backend_read_calls == 3


def test_v26_ficem_reference_backend_preserves_duplicate_heavy_k16_state_exactly():
    source, candidate = _paired_models(39819)
    source_memory = source.stages[0].memory
    candidate_memory = candidate.stages[0].memory
    cfg = diagnostic_config()

    assert source_memory.capacity == candidate_memory.capacity == EPISODIC_CAPACITY == 48
    assert READ_TOP_K == 4
    assert sparse_write_budget(255) == 16

    base = _random_state(
        batch=2,
        d_model=cfg.d_model,
        memory_dim=cfg.memory_dim,
        seed=39820,
    ).memory
    assert isinstance(base, ContextualEpisodicMemoryState)
    base.valid.fill_(True)
    base.strengths.clamp_(0.1, 1.0)

    g = torch.Generator().manual_seed(39821)
    projected = F.normalize(
        torch.randn(2, 16, cfg.memory_dim, generator=g), dim=-1
    )
    # Deliberately create incoming duplicate groups to exercise newest-wins logic.
    projected[:, 1:5] = projected[:, :1]
    projected[:, 8:12] = projected[:, 7:8]
    normalized_old = F.normalize(base.keys, dim=-1)
    payload = torch.randn(2, 16, cfg.d_model, generator=g)
    strength = torch.rand(2, 16, 1, generator=g).mul_(0.8).add_(0.2)

    source_next = source_memory.update_block_from_projected(
        projected, normalized_old, payload, strength, base
    )
    candidate_next = candidate_memory.update_block_from_projected(
        projected, normalized_old, payload, strength, base
    )
    source_state = AERAState(torch.zeros(2, cfg.d_model), source_next)
    candidate_state = AERAState(torch.zeros(2, cfg.d_model), candidate_next)
    _assert_epi_state_equal(source_state, candidate_state, exact=True)
    assert candidate_memory.backend_projected_update_calls == 1


def test_v26_hard_sparse_full_model_matches_final_v25_1_with_and_without_writes():
    cfg = diagnostic_config()
    g = torch.Generator().manual_seed(39822)
    tokens = torch.randint(0, cfg.vocab_size, (3, cfg.chunk_size * 2), generator=g)

    for update_memory in (False, True):
        source, candidate = _paired_models(39823 + int(update_memory))
        _force_all_stages_run(source)
        _force_all_stages_run(candidate)

        with torch.no_grad():
            source_output = source(
                tokens,
                hard=True,
                update_memory=update_memory,
                route_mode="hard_sparse",
            )
            candidate_output = candidate(
                tokens,
                hard=True,
                update_memory=update_memory,
                route_mode="hard_sparse",
            )

        for key in ("logits", "hidden", "next_event_prediction"):
            assert torch.allclose(
                source_output[key], candidate_output[key], atol=ATOL, rtol=RTOL
            )
        source_state = source_output["state"]
        candidate_state = candidate_output["state"]
        assert len(source_state.stages) == len(candidate_state.stages)
        for left, right in zip(source_state.stages, candidate_state.stages):
            _assert_epi_state_equal(left, right)

        _assert_controls_equal(source_output, candidate_output)
        _assert_discrete_stage_telemetry_equal(source, candidate)

        # Four stages, one mandatory foundation stage, two chunks: only the three
        # optional stages use the v26 coalesced selected-population path.
        expected_optional_calls = (cfg.n_stages - 1) * 2
        assert candidate.coalesced_float_state_select_calls == expected_optional_calls
        assert candidate.coalesced_valid_select_calls == expected_optional_calls
        assert candidate.coalesced_float_state_merge_calls == expected_optional_calls
        assert candidate.coalesced_valid_merge_calls == expected_optional_calls
        assert candidate.coalesced_pack_calls == 2 * expected_optional_calls
        assert (
            candidate.legacy_float_component_selects_avoided
            == 3 * expected_optional_calls
        )
        assert (
            candidate.legacy_float_component_merges_avoided
            == 3 * expected_optional_calls
        )


def _run_two_soft_chunks_for_gradients(model, tokens_a, tokens_b):
    model.train()
    model.set_memory_pretraining_mode(True)
    model.zero_grad(set_to_none=True)
    first = model(
        tokens_a,
        hard=False,
        update_memory=True,
        route_mode="soft",
    )
    second = model(
        tokens_b,
        state=first["state"],
        hard=False,
        update_memory=True,
        route_mode="soft",
    )
    loss = (
        first["logits"].float().square().mean() * 0.25
        + second["logits"].float().square().mean()
    )
    loss.backward()
    return first, second


def test_v26_soft_memory_pretraining_outputs_and_gradients_match_final_v25_1():
    cfg = diagnostic_config()
    g = torch.Generator().manual_seed(39825)
    tokens_a = torch.randint(0, cfg.vocab_size, (2, cfg.chunk_size), generator=g)
    tokens_b = torch.randint(0, cfg.vocab_size, (2, cfg.chunk_size), generator=g)

    source, candidate = _paired_models(39826)
    source_first, source_second = _run_two_soft_chunks_for_gradients(
        source, tokens_a, tokens_b
    )
    candidate_first, candidate_second = _run_two_soft_chunks_for_gradients(
        candidate, tokens_a, tokens_b
    )

    for source_output, candidate_output in (
        (source_first, candidate_first),
        (source_second, candidate_second),
    ):
        assert torch.allclose(
            source_output["logits"],
            candidate_output["logits"],
            atol=ATOL,
            rtol=RTOL,
        )

    source_grads = {name: p.grad for name, p in source.named_parameters()}
    candidate_grads = {name: p.grad for name, p in candidate.named_parameters()}
    assert source_grads.keys() == candidate_grads.keys()
    for name in source_grads:
        source_grad = source_grads[name]
        candidate_grad = candidate_grads[name]
        assert (source_grad is None) == (candidate_grad is None), name
        if source_grad is not None:
            assert torch.allclose(
                source_grad, candidate_grad, atol=ATOL, rtol=RTOL
            ), name

    # Soft/calibration paths intentionally remain the inherited dense training path.
    assert candidate.coalesced_float_state_select_calls == 0
    assert candidate.coalesced_float_state_merge_calls == 0


def test_v26_protocol_freezes_cpu_only_boundary_and_production_geometry():
    protocol = coalesced_runtime_v26_protocol()
    assert protocol["research_issue"] == 398
    assert protocol["learned_equations_changed"] is False
    assert protocol["routing_policy_changed"] is False
    assert protocol["optional_stage_skipping_changed"] is False
    assert protocol["expert_sparsity_changed"] is False
    assert protocol["reasoning_sparsity_changed"] is False
    assert protocol["coalesced_optional_state"] is True
    assert protocol["selected_population_float_state_index_selects_target"] == 1
    assert protocol["selected_population_validity_index_selects_target"] == 1
    assert protocol["selected_population_float_state_index_copies_target"] == 1
    assert protocol["selected_population_validity_index_copies_target"] == 1
    assert protocol["ficem_backend_interface"] is True
    assert protocol["real_language_selected_writes"] == 16
    assert (
        protocol["persistent_state_bytes_real_language_four_stage_memory_dim50"]
        == 77_760
    )
    assert protocol["persistent_runtime_pack_state"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False
