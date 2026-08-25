import torch

from aera_v19_memory_necessity_cpu import (
    CHUNK_SIZE,
    EVAL_SEED,
    _force_all_stages_run,
    diagnostic_config,
    make_batch,
)
from aera_v23_sparse_dual_delta_memory_cpu import (
    sparse_dual_delta_mechanism_probe,
    train_pair_with_v23_objective,
)
from tam_research.aera_hardware_core_v22 import (
    DualDeltaFastMemoryState,
    HardwareAwareAERATextLMV22,
)
from tam_research.aera_hardware_core_v23 import (
    HardwareAwareAERATextLMV23,
    budgeted_topk_indices,
    select_budgeted_event_pairs,
    sparse_dual_delta_memory_protocol,
    sparse_write_budget,
)


def test_v23_protocol_is_sparse_execution_only_over_v22_memory_semantics():
    protocol = sparse_dual_delta_memory_protocol()
    assert protocol["dual_delta_equation_changed"] is False
    assert protocol["inverse_covariance_update_changed"] is False
    assert protocol["qkvout_dimension_changed"] is False
    assert protocol["learned_parameter_count_changed"] is False
    assert protocol["read_path_changed"] is False
    assert protocol["routing_changed"] is False
    assert protocol["stream_changed"] is False
    assert protocol["controlled_selected_writes"] == 2
    assert protocol["real_language_selected_writes"] == 16
    assert protocol["extra_learned_parameters"] == 0
    assert protocol["gpu_authorized"] is False


def test_sparse_write_budget_is_frozen_two_of_five_and_sixteen_of_255():
    assert sparse_write_budget(0) == 0
    assert sparse_write_budget(1) == 1
    assert sparse_write_budget(5) == 2
    assert sparse_write_budget(15) == 2
    assert sparse_write_budget(16) == 2
    assert sparse_write_budget(32) == 2
    assert sparse_write_budget(33) == 3
    assert sparse_write_budget(255) == 16


def test_budgeted_topk_selection_restores_chronological_order():
    logits = torch.tensor([[[0.0], [9.0], [1.0], [2.0], [8.0]]])
    indices = budgeted_topk_indices(logits)
    assert indices.tolist() == [[1, 4]]


def test_straight_through_selector_is_hard_in_forward_and_dense_in_backward():
    address = torch.randn(1, 5, 3)
    payload = torch.randn(1, 5, 3)
    strength = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]])
    logits = torch.tensor(
        [[[0.0], [2.0], [1.0], [-1.0], [3.0]]], requires_grad=True
    )
    selected = select_budgeted_event_pairs(
        address,
        payload,
        strength,
        logits,
        differentiable_selector=True,
    )
    assert selected.indices.tolist() == [[1, 4]]
    # Straight-through multiplier is exactly one in the forward pass.
    torch.testing.assert_close(
        selected.strength.detach(),
        torch.tensor([[[0.2], [0.5]]]),
        atol=0.0,
        rtol=0.0,
    )
    selected.strength.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Unselected candidates participate in the soft-k-hot backward surrogate.
    unselected = torch.tensor([0, 2, 3])
    assert float(logits.grad[0, unselected, 0].abs().sum()) > 0.0


def test_v23_matches_v22_learned_parameters_except_frozen_pair_gate_reinit():
    torch.manual_seed(9240)
    v22 = HardwareAwareAERATextLMV22(diagnostic_config())
    _force_all_stages_run(v22)
    torch.manual_seed(9240)
    v23 = HardwareAwareAERATextLMV23(diagnostic_config())
    _force_all_stages_run(v23)
    assert sum(p.numel() for p in v22.parameters()) == sum(p.numel() for p in v23.parameters())
    assert v22.state_dict().keys() == v23.state_dict().keys()
    changed = []
    for key, value in v22.state_dict().items():
        if not torch.equal(v23.state_dict()[key], value):
            changed.append(key)
    assert changed
    assert all(".pair_write_gate.weight" in key for key in changed)


def test_v23_memory_off_logits_and_stream_are_bit_exact_v22():
    torch.manual_seed(9241)
    v22 = HardwareAwareAERATextLMV22(diagnostic_config())
    _force_all_stages_run(v22)
    torch.manual_seed(9241)
    v23 = HardwareAwareAERATextLMV23(diagnostic_config())
    _force_all_stages_run(v23)
    tokens = make_batch(2, EVAL_SEED + 940).tokens[:, : 2 * CHUNK_SIZE]
    out22 = v22(
        tokens,
        hard=True,
        route_mode="hard_sparse",
        update_memory=False,
        return_block_logits=False,
    )
    out23 = v23(
        tokens,
        hard=True,
        route_mode="hard_sparse",
        update_memory=False,
        return_block_logits=False,
    )
    torch.testing.assert_close(out23["logits"], out22["logits"], atol=0.0, rtol=0.0)
    for s22, s23 in zip(out22["state"].stages, out23["state"].stages):
        torch.testing.assert_close(s23.stream, s22.stream, atol=0.0, rtol=0.0)
        torch.testing.assert_close(s23.memory.matrix, s22.memory.matrix, atol=0.0, rtol=0.0)
        assert isinstance(s23.memory, DualDeltaFastMemoryState)
        torch.testing.assert_close(
            s23.memory.inverse_key_covariance,
            s22.memory.inverse_key_covariance,
            atol=0.0,
            rtol=0.0,
        )


def test_sparse_nonorthogonal_write_overwrite_mechanism_gate_passes():
    result = sparse_dual_delta_mechanism_probe()
    assert result["selected_count"] == 2.0
    assert result["candidate_count"] == 5.0
    assert result["selected_fraction"] == 0.4
    assert result["last_selected_indices"] == [1, 4]
    assert result["pass"] is True


def test_one_step_v23_training_is_finite_sparse_and_trains_selector_and_memory():
    full, stream, result = train_pair_with_v23_objective(steps=1)
    assert isinstance(full, HardwareAwareAERATextLMV23)
    assert isinstance(stream, HardwareAwareAERATextLMV23)
    assert len(result["history"]) == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in result["history"][0].values())
    stage = full.stages[0]
    assert stage.pair_write_gate.weight.grad is not None
    assert float(stage.pair_write_gate.weight.grad.abs().sum()) > 0.0
    for parameter in (
        stage.memory.q.weight,
        stage.memory.k.weight,
        stage.memory.v.weight,
        stage.memory.out.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0

    batch = make_batch(2, EVAL_SEED + 941)
    full.eval()
    full.set_memory_pretraining_mode(False)
    out = full(
        batch.tokens[:, :CHUNK_SIZE],
        hard=True,
        route_mode="hard_sparse",
        update_memory=True,
        return_block_logits=False,
    )
    for stage_module, state in zip(full.stages, out["state"].stages):
        assert stage_module.last_candidate_count == 5
        assert stage_module.last_selected_count == 2
        assert isinstance(state.memory, DualDeltaFastMemoryState)
        assert state.memory.matrix.grad_fn is None
        assert state.memory.inverse_key_covariance.grad_fn is None
