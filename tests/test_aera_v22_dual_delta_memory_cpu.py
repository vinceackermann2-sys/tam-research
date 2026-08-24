import torch

from aera_v19_memory_necessity_cpu import CHUNK_SIZE, EVAL_SEED, _force_all_stages_run, diagnostic_config, make_batch
from aera_v22_dual_delta_memory_cpu import (
    build_model,
    dual_delta_mechanism_probe,
    train_pair_with_v22_objective,
)
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21
from tam_research.aera_hardware_core_v22 import (
    DualDeltaFastMemoryState,
    HardwareAwareAERATextLMV22,
    dual_delta_memory_protocol,
    interference_corrected_dual_delta_update,
)


def test_v22_protocol_changes_only_session_memory_write_dynamics():
    protocol = dual_delta_memory_protocol()
    assert protocol["qkvout_dimension_changed"] is False
    assert protocol["learned_parameter_count_changed"] is False
    assert protocol["read_path_changed"] is False
    assert protocol["event_pair_candidates_changed"] is False
    assert protocol["routing_changed"] is False
    assert protocol["stream_changed"] is False
    assert protocol["blanket_matrix_decay"] is False
    assert protocol["gpu_authorized"] is False


def test_v22_learned_parameters_are_bit_exact_v21_at_matched_initialization():
    torch.manual_seed(9230)
    v21 = HardwareAwareAERATextLMV21(diagnostic_config())
    _force_all_stages_run(v21)
    torch.manual_seed(9230)
    v22 = HardwareAwareAERATextLMV22(diagnostic_config())
    _force_all_stages_run(v22)
    assert sum(p.numel() for p in v21.parameters()) == sum(p.numel() for p in v22.parameters())
    assert v21.state_dict().keys() == v22.state_dict().keys()
    for key, value in v21.state_dict().items():
        torch.testing.assert_close(v22.state_dict()[key], value, atol=0.0, rtol=0.0)


def test_v22_memory_off_logits_are_bit_exact_v21():
    torch.manual_seed(9231)
    v21 = HardwareAwareAERATextLMV21(diagnostic_config())
    _force_all_stages_run(v21)
    torch.manual_seed(9231)
    v22 = HardwareAwareAERATextLMV22(diagnostic_config())
    _force_all_stages_run(v22)
    tokens = make_batch(2, EVAL_SEED + 930).tokens[:, : 2 * CHUNK_SIZE]
    out21 = v21(tokens, hard=True, route_mode="hard_sparse", update_memory=False, return_block_logits=False)
    out22 = v22(tokens, hard=True, route_mode="hard_sparse", update_memory=False, return_block_logits=False)
    torch.testing.assert_close(out22["logits"], out21["logits"], atol=0.0, rtol=0.0)
    for s21, s22 in zip(out21["state"].stages, out22["state"].stages):
        torch.testing.assert_close(s22.stream, s21.stream, atol=0.0, rtol=0.0)
        torch.testing.assert_close(s22.memory.matrix, s21.memory.matrix, atol=0.0, rtol=0.0)
        assert isinstance(s22.memory, DualDeltaFastMemoryState)
        eye = torch.eye(diagnostic_config().memory_dim).expand(tokens.size(0), -1, -1)
        torch.testing.assert_close(s22.memory.inverse_key_covariance, eye, atol=0.0, rtol=0.0)


def test_strength_one_dual_delta_exactly_corrects_current_key():
    key = torch.tensor([[[0.8, 0.6]]], dtype=torch.float32)
    key = torch.nn.functional.normalize(key, dim=-1)
    target = torch.tensor([[[0.25, -0.75]]])
    matrix = torch.tensor([[[0.2, 0.1], [-0.4, 0.3]]])
    inverse = torch.eye(2).unsqueeze(0)
    strength = torch.ones(1, 1, 1)
    updated, next_inverse = interference_corrected_dual_delta_update(
        matrix,
        inverse,
        key,
        target,
        strength,
    )
    prediction = torch.einsum("bi,bij->bj", key[:, 0], updated)
    torch.testing.assert_close(prediction, target[:, 0], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(next_inverse).all()
    torch.testing.assert_close(next_inverse, next_inverse.transpose(-1, -2), atol=1e-6, rtol=1e-6)


def test_nonorthogonal_write_overwrite_mechanism_gate_passes():
    result = dual_delta_mechanism_probe()
    assert result["key_abs_offdiag_max"] > 0.0
    assert result["pass"] is True


def test_one_step_v22_conflict_free_training_runs_and_keeps_dual_state_local():
    full, stream, result = train_pair_with_v22_objective(steps=1)
    assert isinstance(full, HardwareAwareAERATextLMV22)
    assert isinstance(stream, HardwareAwareAERATextLMV22)
    assert len(result["history"]) == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in result["history"][0].values())
    batch = make_batch(2, EVAL_SEED + 931)
    full.eval()
    full.set_memory_pretraining_mode(False)
    out = full(batch.tokens[:, :CHUNK_SIZE], hard=True, route_mode="hard_sparse", update_memory=True, return_block_logits=False)
    for state in out["state"].stages:
        assert isinstance(state.memory, DualDeltaFastMemoryState)
        assert state.memory.matrix.grad_fn is None
        assert state.memory.inverse_key_covariance.grad_fn is None
