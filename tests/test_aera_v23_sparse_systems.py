import torch

import aera_v23_sparse_systems_l4 as bench
from tam_research import aera_real_language_v23_efficiency as eff
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v23 import (
    BudgetedSparseDualDeltaFastMemoryStage,
    HardwareAwareAERATextLMV23,
    sparse_write_budget,
)
from tam_research.aera_v23_sparse_systems_runtime_repair import _stage_probe_forward


def _tiny_model() -> HardwareAwareAERATextLMV23:
    cfg = HardwareAERAConfig(
        vocab_size=97,
        d_model=24,
        n_stages=4,
        n_heads=4,
        chunk_size=eff.CHUNK_SIZE,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=2,
        block_size=2,
    )
    return HardwareAwareAERATextLMV23(cfg)


def test_v23_systems_protocol_freezes_real_language_sparse_budget():
    protocol = eff.efficiency_protocol()
    assert eff.CHUNK_SIZE == 256
    assert sparse_write_budget(255) == 16
    assert protocol["real_language_candidates"] == 255
    assert protocol["real_language_selected_writes"] == 16
    assert protocol["memory_equations_changed"] is False
    assert protocol["routing_changed"] is False
    assert protocol["objective_weights_changed"] is False
    assert protocol["primary_runtime_requires_compile"] is False
    assert protocol["gpu_training_authorized"] is False
    assert protocol["fresh_real_language_seed_authorized"] is False
    assert protocol["100m_authorized"] is False


def test_v23_corrected_auxiliary_keeps_256_per_microbatch_cap():
    torch.manual_seed(8441)
    model = _tiny_model().train()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, eff.CHUNK_SIZE))
    terms = eff.memory_auxiliary_terms(model, tokens, step=3)
    assert int(terms["memory_aux_events"].item()) == 256
    try:
        eff.memory_auxiliary_terms(
            model,
            tokens,
            step=0,
            max_events=257,
        )
    except ValueError as exc:
        assert "per-microbatch budget" in str(exc)
    else:
        raise AssertionError("v23 accepted an over-budget memory auxiliary")


def test_v23_systems_benchmark_thresholds_match_issue_333():
    assert bench.MEMORY_DIM == 50
    assert bench.DENSE_CANDIDATES == 255
    assert bench.SPARSE_CANDIDATES == 16
    assert bench.RECURRENCE_BATCH == 8
    assert bench.MIN_RECURRENCE_SPEEDUP == 4.0
    assert bench.PROJECTION_MAX_SECONDS == 1500.0
    assert bench.EVAL_MARGIN_SECONDS == 150.0


def test_v23_production_cpu_preflight_is_sparse_and_memory_off_exact():
    result = eff.cpu_preflight()
    assert result["chunk_size"] == 256
    assert result["memory_dim"] == 50
    assert result["pair_candidates_per_chunk"] == 255
    assert result["selected_writes_per_chunk"] == 16
    assert result["learned_parameter_count_changed_from_v22"] is False
    assert result["memory_off_logits_and_state_bit_exact_v22"] is True
    assert result["memory_aux_events_per_microbatch"] == 256
    assert result["memory_aux_events_per_optimizer_step"] == 1024
    assert result["sparse_smoke"]["candidates"] == 255
    assert result["sparse_smoke"]["selected_writes"] == 16
    assert result["sparse_smoke"]["pair_gate_grad_norm"] > 0.0
    assert result["sparse_smoke"]["memory_k_grad_norm"] > 0.0


def test_post_335_stage_probe_enters_autocast_from_fp32_and_keeps_16_of_255():
    """Regression for the L4-only dtype mismatch that stopped #335 pre-measurement."""
    torch.manual_seed(8442)
    model = _tiny_model().train()
    stage = model.stages[0]
    assert isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage)
    stage.memory.set_differentiable_pretraining(True)
    events = torch.randn(
        1,
        eff.CHUNK_SIZE,
        model.cfg.d_model,
        dtype=torch.float32,
        requires_grad=True,
    )
    _, state, loss = _stage_probe_forward(stage, events)
    assert torch.isfinite(loss)
    loss.backward()
    assert events.grad is not None
    assert torch.isfinite(events.grad).all()
    assert torch.isfinite(state.memory.matrix).all()
    assert torch.isfinite(state.memory.inverse_key_covariance).all()
    assert stage.last_candidate_count == 255
    assert stage.last_selected_count == 16
    assert events.dtype == torch.float32
