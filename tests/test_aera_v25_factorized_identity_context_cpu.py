import torch

from aera_v19_memory_necessity_cpu import CHUNK_SIZE, make_batch
from aera_v25_factorized_identity_context_cpu import (
    ADDRESS_CONTRASTIVE_WEIGHT,
    EVAL_SEED,
    SEED,
    build_model,
    factorization_invariance_and_causality_probe,
    factorized_context_disambiguation_probe,
    factorized_mechanism_probe,
    heldout_factorized_address_and_selector_diagnostics,
    payload_memory_terms,
    primary_query_gradient_probe,
    train_pair_with_v25_objective,
)
from tam_research.aera_hardware_core_v25 import (
    FactorizedIdentityContextEpisodicMemoryStage,
    HardwareAwareAERATextLMV25,
    factorized_identity_context_protocol,
)
from tam_research.aera_hardware_core_v24 import episodic_state_bytes_per_session


def test_v25_uses_fresh_preregistered_seeds_and_v24_1_causal_objective():
    assert SEED == 8421
    assert EVAL_SEED == 8422
    assert ADDRESS_CONTRASTIVE_WEIGHT == 0.0


def test_v25_cross_operation_identity_is_exact_context_differs_and_future_is_causal():
    model = build_model(9580)
    result = factorization_invariance_and_causality_probe(model)
    assert result["write_query_identity_bit_identical"] is True
    assert result["write_query_context_differs"] is True
    assert result["write_query_context_l2"] > 0.0
    assert result["write_query_combined_differs"] is True
    assert result["future_value_identity_bit_identical"] is True
    assert result["future_value_context_bit_identical"] is True
    assert result["future_value_combined_bit_identical"] is True


def test_v25_factorized_memory_mechanism_overwrite_is_exact_without_training():
    result = factorized_mechanism_probe()
    assert result["current_accuracy"] >= 0.95
    assert result["overwrite_current_accuracy"] >= 0.95
    assert result["stale_error"] <= 0.05
    assert result["pass"] is True


def test_v25_factorization_disambiguates_same_identity_while_identity_only_does_not():
    result = factorized_context_disambiguation_probe()
    assert result["contextual_accuracy"] >= 0.95
    assert result["contextual_cross_context_error"] <= 0.05
    assert result["token_only_reference_accuracy"] <= 0.75
    assert result["contextual_valid_slots"] == 2.0
    assert result["token_only_valid_slots"] == 1.0
    assert result["pass"] is True


def test_v25_primary_query_ce_reaches_both_factor_projectors_payload_and_selector():
    result = primary_query_gradient_probe()
    assert result["loss_finite"] is True
    for name in ("identity_proj", "context_proj", "v", "out", "pair_write_gate"):
        assert result[f"{name}_grad_finite"] is True
        assert result[f"{name}_grad_nonzero"] is True
        assert result[f"{name}_grad_l1"] > 0.0


def test_v25_payload_auxiliary_is_finite_and_does_not_supervise_address_factors():
    model = build_model(9581)
    batch = make_batch(2, EVAL_SEED + 9581)
    terms = payload_memory_terms(model, batch.tokens)
    assert set(terms) == {"payload_token_loss", "payload_token_accuracy"}
    assert torch.isfinite(terms["payload_token_loss"])
    terms["payload_token_loss"].backward()
    stage = model.stages[0]
    assert stage.memory.v.weight.grad is not None
    assert stage.memory.out.weight.grad is not None
    assert float(stage.memory.v.weight.grad.abs().sum()) > 0.0
    assert float(stage.memory.out.weight.grad.abs().sum()) > 0.0
    assert stage.memory.identity_proj.weight.grad is None
    assert stage.memory.context_proj.weight.grad is None


def test_v25_one_training_step_keeps_sparse_budget_state_width_and_vectorized_update():
    full, stream_only, result = train_pair_with_v25_objective(steps=1)
    assert isinstance(full, HardwareAwareAERATextLMV25)
    assert type(full) is type(stream_only)
    assert len(result["history"]) == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in result["history"][0].values())
    stage = full.stages[0]
    assert isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage)
    assert stage.last_candidate_count == CHUNK_SIZE - 1
    assert stage.last_selected_count == 2
    assert stage.last_vectorized_update_calls == 1
    assert stage.memory.identity_dim + stage.memory.context_dim == stage.memory.memory_dim
    assert stage.memory.identity_dim == stage.memory.context_dim
    assert episodic_state_bytes_per_session(n_stages=4, memory_dim=50) == 77760
    protocol = factorized_identity_context_protocol()
    assert protocol["version"] == "aera-v25-factorized-identity-context-episodic-memory"
    assert protocol["combined_address_width_changed"] is False
    assert protocol["controlled_selected_writes"] == 2
    assert protocol["real_language_selected_writes"] == 16
    assert protocol["vectorized_update_calls_per_completed_stage_chunk"] == 1
    assert protocol["state_bytes_real_language_four_stage_memory_dim50"] == 77760


def test_v25_diagnostic_metrics_are_evaluation_only_and_well_formed():
    model = build_model(9582)
    batch = make_batch(2, EVAL_SEED + 9582)
    result = heldout_factorized_address_and_selector_diagnostics(model, batch)
    assert set(result) == {
        "identity_query_to_latest_write_top1",
        "identity_query_to_latest_write_margin",
        "combined_query_to_latest_write_top1",
        "combined_query_to_latest_write_margin",
        "context_margin_contribution",
        "true_key_value_selector_coverage",
        "write_payload_token_accuracy",
    }
    for name in (
        "identity_query_to_latest_write_top1",
        "combined_query_to_latest_write_top1",
        "true_key_value_selector_coverage",
        "write_payload_token_accuracy",
    ):
        assert 0.0 <= result[name] <= 1.0
    for name in (
        "identity_query_to_latest_write_margin",
        "combined_query_to_latest_write_margin",
        "context_margin_contribution",
    ):
        assert torch.isfinite(torch.tensor(result[name]))
