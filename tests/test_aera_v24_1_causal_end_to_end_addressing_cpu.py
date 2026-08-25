import torch

from aera_v19_memory_necessity_cpu import CHUNK_SIZE, make_batch
from aera_v24_1_causal_end_to_end_addressing_cpu import (
    ADDRESS_CONTRASTIVE_WEIGHT,
    EVAL_SEED,
    SEED,
    build_model,
    causal_objective_contradiction_probe,
    heldout_address_and_selector_diagnostics,
    payload_memory_terms,
    primary_query_gradient_probe,
    train_pair_with_v24_1_objective,
)
from tam_research.aera_hardware_core_v24 import (
    HardwareAwareAERATextLMV24,
    episodic_state_bytes_per_session,
    vectorized_contextual_episodic_protocol,
)


def test_v24_1_uses_fresh_preregistered_seeds_and_zero_address_auxiliary():
    assert SEED == 8411
    assert EVAL_SEED == 8412
    assert ADDRESS_CONTRASTIVE_WEIGHT == 0.0


def test_v24_1_proves_old_future_conditioned_address_objective_contradiction():
    model = build_model(9560)
    result = causal_objective_contradiction_probe(model)
    assert result["same_write_address_bit_identical"] is True
    assert result["old_transition_identities_differ"] is True
    assert result["write_and_query_views_differ"] is True
    assert result["write_query_view_l2"] > 0.0


def test_v24_1_primary_query_ce_reaches_memory_address_payload_and_selector():
    result = primary_query_gradient_probe()
    assert result["loss_finite"] is True
    for name in ("q", "k", "v", "out", "pair_write_gate"):
        assert result[f"{name}_grad_finite"] is True
        assert result[f"{name}_grad_nonzero"] is True
        assert result[f"{name}_grad_l1"] > 0.0


def test_v24_1_payload_auxiliary_is_finite_and_only_needs_v_out_path():
    torch.manual_seed(9562)
    model = build_model(9562)
    batch = make_batch(2, EVAL_SEED + 9562)
    terms = payload_memory_terms(model, batch.tokens)
    assert set(terms) == {"payload_token_loss", "payload_token_accuracy"}
    assert torch.isfinite(terms["payload_token_loss"])
    terms["payload_token_loss"].backward()
    stage = model.stages[0]
    assert stage.memory.v.weight.grad is not None
    assert stage.memory.out.weight.grad is not None
    assert float(stage.memory.v.weight.grad.abs().sum()) > 0.0
    assert float(stage.memory.out.weight.grad.abs().sum()) > 0.0
    assert stage.memory.q.weight.grad is None
    assert stage.memory.k.weight.grad is None


def test_v24_1_one_training_step_keeps_exact_v24_architecture_and_budget():
    full, stream_only, result = train_pair_with_v24_1_objective(steps=1)
    assert isinstance(full, HardwareAwareAERATextLMV24)
    assert type(full) is type(stream_only)
    assert len(result["history"]) == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in result["history"][0].values())
    stage = full.stages[0]
    assert stage.last_candidate_count == CHUNK_SIZE - 1
    assert stage.last_selected_count == 2
    assert stage.last_vectorized_update_calls == 1
    assert episodic_state_bytes_per_session(n_stages=4, memory_dim=50) == 77760
    protocol = vectorized_contextual_episodic_protocol()
    assert protocol["version"] == "aera-v24-vectorized-contextual-episodic-memory"
    assert protocol["vectorized_update_calls_per_completed_stage_chunk"] == 1
    assert protocol["controlled_selected_writes"] == 2


def test_v24_1_diagnostic_metrics_are_evaluation_only_and_well_formed():
    model = build_model(9563)
    batch = make_batch(2, EVAL_SEED + 9563)
    result = heldout_address_and_selector_diagnostics(model, batch)
    assert set(result) == {
        "query_to_latest_write_address_top1",
        "query_to_latest_write_address_margin",
        "true_key_value_selector_coverage",
        "write_payload_token_accuracy",
    }
    assert 0.0 <= result["query_to_latest_write_address_top1"] <= 1.0
    assert 0.0 <= result["true_key_value_selector_coverage"] <= 1.0
    assert 0.0 <= result["write_payload_token_accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(result["query_to_latest_write_address_margin"]))
