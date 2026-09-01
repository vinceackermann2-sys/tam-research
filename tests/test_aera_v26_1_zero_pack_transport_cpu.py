from __future__ import annotations

import inspect

import torch

from aera_v19_memory_necessity_cpu import diagnostic_config
from tam_research.aera import AERAState
from tam_research.aera_hardware_core_v24 import (
    EPISODIC_CAPACITY,
    ContextualEpisodicMemoryState,
    _merge_epi_state,
    _select_epi_state,
)
from tam_research.aera_hardware_core_v26 import HardwareAwareAERATextLMV26
from tam_research.aera_hardware_core_v26_1 import (
    HardwareAwareAERATextLMV261ZeroPackTransport,
    TorchComponentwiseStateTransport,
    zero_pack_transport_v26_1_protocol,
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
            valid=torch.rand(batch, capacity, generator=g) > 0.47,
        ),
    )


def _clone_state(state: AERAState) -> AERAState:
    assert isinstance(state.memory, ContextualEpisodicMemoryState)
    return AERAState(
        stream=state.stream.clone(),
        memory=ContextualEpisodicMemoryState(
            keys=state.memory.keys.clone(),
            values=state.memory.values.clone(),
            strengths=state.memory.strengths.clone(),
            valid=state.memory.valid.clone(),
        ),
    )


def _assert_state_equal(left: AERAState, right: AERAState, *, exact: bool = False) -> None:
    assert isinstance(left.memory, ContextualEpisodicMemoryState)
    assert isinstance(right.memory, ContextualEpisodicMemoryState)
    for a, b in (
        (left.stream, right.stream),
        (left.memory.keys, right.memory.keys),
        (left.memory.values, right.memory.values),
        (left.memory.strengths, right.memory.strengths),
    ):
        if exact:
            assert torch.equal(a, b)
        else:
            assert torch.allclose(a, b, atol=ATOL, rtol=RTOL)
    assert torch.equal(left.memory.valid, right.memory.valid)


def _paired_models(seed: int):
    cfg = diagnostic_config()
    torch.manual_seed(seed)
    source = HardwareAwareAERATextLMV26(cfg).eval()
    torch.manual_seed(seed)
    candidate = HardwareAwareAERATextLMV261ZeroPackTransport(cfg).eval()
    candidate.load_state_dict(source.state_dict(), strict=True)
    return source, candidate


def _force_all_stages_run(model) -> None:
    with torch.no_grad():
        for router in model.stage_routers:
            router.proj.weight.zero_()
            router.proj.bias.fill_(12.0)


def _force_optional_stages_skip(model) -> None:
    with torch.no_grad():
        for index, router in enumerate(model.stage_routers):
            router.proj.weight.zero_()
            router.proj.bias.fill_(12.0 if index == model.FOUNDATION_STAGE else -12.0)


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


def _assert_discrete_telemetry_equal(source, candidate) -> None:
    for source_stage, candidate_stage in zip(source.stages, candidate.stages):
        for attribute in ("last_selected_indices",):
            left = getattr(source_stage, attribute)
            right = getattr(candidate_stage, attribute)
            assert (left is None) == (right is None)
            if left is not None:
                assert torch.equal(left, right)

        left_counts = source_stage.experts.last_counts
        right_counts = candidate_stage.experts.last_counts
        assert (left_counts is None) == (right_counts is None)
        if left_counts is not None:
            assert torch.equal(left_counts, right_counts)

        left_steps = source_stage.reasoner.last_steps
        right_steps = candidate_stage.reasoner.last_steps
        assert (left_steps is None) == (right_steps is None)
        if left_steps is not None:
            assert torch.equal(left_steps, right_steps)


def test_v26_1_transport_reference_matches_componentwise_select_merge_randomized():
    backend = TorchComponentwiseStateTransport()
    assert backend.name == "torch-componentwise-exact-zero-pack"

    cases = (
        (1, torch.tensor([], dtype=torch.long), "empty"),
        (1, torch.tensor([0], dtype=torch.long), "full"),
        (8, torch.tensor([3], dtype=torch.long), "mixed"),
        (8, torch.tensor([0, 2, 5, 7], dtype=torch.long), "mixed"),
        (64, torch.arange(0, 64, 4, dtype=torch.long), "mixed"),
        (64, torch.arange(64, dtype=torch.long), "full"),
    )
    for case_index, (batch, idx, validity_kind) in enumerate(cases):
        base = _random_state(
            batch=batch,
            d_model=24,
            memory_dim=16,
            seed=40610 + case_index,
        )
        assert isinstance(base.memory, ContextualEpisodicMemoryState)
        if validity_kind == "empty":
            base.memory.valid.zero_()
        elif validity_kind == "full":
            base.memory.valid.fill_(True)
        # Invalid slots deliberately retain random/nonzero key/value/strength storage.
        snapshot = _clone_state(base)

        reference_selected = _select_epi_state(base, idx)
        backend_selected = backend.select(base, idx)
        _assert_state_equal(reference_selected, backend_selected, exact=True)
        _assert_state_equal(base, snapshot, exact=True)

        update = _random_state(
            batch=idx.numel(),
            d_model=24,
            memory_dim=16,
            seed=40630 + case_index,
        )
        update_snapshot = _clone_state(update)
        reference_merged = _merge_epi_state(base, update, idx)
        backend_merged = backend.merge(base, update, idx)
        _assert_state_equal(reference_merged, backend_merged, exact=True)
        _assert_state_equal(base, snapshot, exact=True)
        _assert_state_equal(update, update_snapshot, exact=True)


def test_v26_1_state_dict_parameters_and_buffers_are_identical_to_v26():
    source, candidate = _paired_models(40640)
    source_state = source.state_dict()
    candidate_state = candidate.state_dict()
    assert source_state.keys() == candidate_state.keys()
    assert [name for name, _ in source.named_parameters()] == [
        name for name, _ in candidate.named_parameters()
    ]
    assert [name for name, _ in source.named_buffers()] == [
        name for name, _ in candidate.named_buffers()
    ]
    assert sum(p.numel() for p in source.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )
    for name in source_state:
        assert torch.equal(source_state[name], candidate_state[name])
    assert not any("state_transport" in key for key in candidate_state)


def test_v26_1_hard_sparse_full_model_matches_v26_with_and_without_memory_writes():
    cfg = diagnostic_config()
    g = torch.Generator().manual_seed(40641)
    tokens = torch.randint(0, cfg.vocab_size, (3, cfg.chunk_size * 2), generator=g)

    for update_memory in (False, True):
        source, candidate = _paired_models(40642 + int(update_memory))
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
            _assert_state_equal(left, right)
        _assert_controls_equal(source_output, candidate_output)
        _assert_discrete_telemetry_equal(source, candidate)

        expected_optional_calls = (cfg.n_stages - 1) * 2
        assert candidate.zero_pack_transport_select_calls == expected_optional_calls
        assert candidate.zero_pack_transport_merge_calls == expected_optional_calls
        assert candidate.zero_pack_optional_stage_calls == expected_optional_calls
        # Inherited v26 pack counters must remain untouched in v26.1 optional routing.
        assert candidate.coalesced_pack_calls == 0
        assert candidate.coalesced_float_state_select_calls == 0
        assert candidate.coalesced_float_state_merge_calls == 0


def _run_calibration_gradient(model, tokens: torch.Tensor, route_mode: str):
    model.train()
    model.set_memory_pretraining_mode(True)
    model.zero_grad(set_to_none=True)
    output = model(
        tokens,
        hard=False,
        update_memory=True,
        route_mode=route_mode,
    )
    loss = output["logits"].float().square().mean()
    loss.backward()
    return output, {name: parameter.grad for name, parameter in model.named_parameters()}


def test_v26_1_soft_and_straight_through_outputs_and_gradients_match_v26():
    cfg = diagnostic_config()
    g = torch.Generator().manual_seed(40645)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.chunk_size), generator=g)

    for offset, route_mode in enumerate(("soft", "straight_through")):
        source, candidate = _paired_models(40646 + offset)
        source_output, source_grads = _run_calibration_gradient(
            source, tokens, route_mode
        )
        candidate_output, candidate_grads = _run_calibration_gradient(
            candidate, tokens, route_mode
        )
        assert torch.allclose(
            source_output["logits"], candidate_output["logits"], atol=ATOL, rtol=RTOL
        )
        source_state = source_output["state"]
        candidate_state = candidate_output["state"]
        for left, right in zip(source_state.stages, candidate_state.stages):
            _assert_state_equal(left, right)

        assert source_grads.keys() == candidate_grads.keys()
        for name in source_grads:
            source_grad = source_grads[name]
            candidate_grad = candidate_grads[name]
            assert (source_grad is None) == (candidate_grad is None), name
            if source_grad is not None:
                assert torch.allclose(
                    source_grad, candidate_grad, atol=ATOL, rtol=RTOL
                ), name

        assert candidate.zero_pack_transport_select_calls == 0
        assert candidate.zero_pack_transport_merge_calls == 0
        assert candidate.coalesced_pack_calls == 0


def test_v26_1_skipped_optional_stages_do_not_call_transport_backend():
    cfg = diagnostic_config()
    g = torch.Generator().manual_seed(40650)
    tokens = torch.randint(0, cfg.vocab_size, (4, cfg.chunk_size), generator=g)
    model = HardwareAwareAERATextLMV261ZeroPackTransport(cfg).eval()
    _force_optional_stages_skip(model)
    with torch.no_grad():
        output = model(
            tokens,
            hard=True,
            update_memory=True,
            route_mode="hard_sparse",
        )
    assert torch.isfinite(output["logits"]).all()
    assert model.zero_pack_transport_select_calls == 0
    assert model.zero_pack_transport_merge_calls == 0
    assert model.zero_pack_optional_stage_calls == 0
    assert model.coalesced_pack_calls == 0
    routes = output["stage_routes"]
    for chunk in routes:
        for stage_index, row in enumerate(chunk):
            if stage_index != model.FOUNDATION_STAGE:
                assert row["executed_fraction"] == 0.0


def test_v26_1_hard_sparse_route_has_zero_pack_or_state_cat_stack_calls():
    source = inspect.getsource(
        HardwareAwareAERATextLMV261ZeroPackTransport._route_one_stage
    )
    for forbidden in (
        "pack_ephemeral_epi_state",
        "select_packed_epi_state",
        "merge_packed_epi_state",
        "torch.cat",
        "torch.stack",
    ):
        assert forbidden not in source
    assert "_state_transport_backend.select" in source
    assert "_state_transport_backend.merge" in source
    assert ".nonzero(" in source
    assert "stage.forward_chunk" in source


def test_v26_1_protocol_freezes_zero_pack_cpu_only_boundary():
    protocol = zero_pack_transport_v26_1_protocol()
    assert protocol["research_issue"] == 406
    assert protocol["source_issue400_authoritative_decision"] == "FAIL"
    assert protocol["source_issue400_batch64_geomean_latency_ratio"] > 1.11
    assert protocol["source_issue400_kernel_event_ratio_each_row"] == 8.0 / 15.0
    assert protocol["learned_equations_changed"] is False
    assert protocol["state_dict_schema_changed"] is False
    assert protocol["routing_policy_changed"] is False
    assert protocol["optional_stage_skipping_changed"] is False
    assert protocol["expert_sparsity_changed"] is False
    assert protocol["reasoning_sparsity_changed"] is False
    assert protocol["ficem_equations_changed"] is False
    assert protocol["state_transport_backend_interface"] is True
    assert protocol["hard_sparse_ephemeral_pack_state"] is False
    assert protocol["hard_sparse_pack_helper_calls_target"] == 0
    assert protocol["hard_sparse_state_cat_calls_target"] == 0
    assert protocol["hard_sparse_state_stack_calls_target"] == 0
    assert protocol["persistent_state_format_changed"] is False
    assert protocol["persistent_state_extra_tensors"] == 0
    assert protocol["persistent_state_bytes_real_language_four_stage_memory_dim50"] == 77_760
    assert protocol["real_language_selected_writes"] == 16
    future = protocol["future_cuda_transport_target"]
    assert future["selected_population_gather_launches"] == 1
    assert future["selected_population_merge_launches"] == 1
    assert future["input_pack_required"] is False
    assert future["dense_masked_stage_execution"] is False
    assert protocol["cuda_backend_implemented"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False
