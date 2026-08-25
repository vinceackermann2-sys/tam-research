import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import CHUNK_SIZE, EVAL_SEED, make_batch
from aera_v24_vectorized_contextual_episodic_memory_cpu import (
    build_model,
    context_disambiguation_probe,
    contextual_memory_terms,
    direct_episodic_read_evaluation,
    episodic_mechanism_probe,
    heldout_local_contextual_code,
    train_pair_with_v24_objective,
)
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState


def test_v24_deterministic_episodic_mechanism_probe_passes():
    result = episodic_mechanism_probe()
    assert result["current_accuracy"] >= 0.95
    assert result["overwrite_current_accuracy"] >= 0.95
    assert result["stale_error"] <= 0.05
    assert result["pass"] is True


def test_v24_context_disambiguation_beats_same_geometry_token_only_reference():
    result = context_disambiguation_probe()
    assert result["contextual_accuracy"] >= 0.95
    assert result["contextual_cross_context_error"] <= 0.05
    assert result["token_only_reference_accuracy"] <= 0.75
    assert result["contextual_valid_slots"] == 2.0
    assert result["token_only_valid_slots"] == 1.0
    assert result["pass"] is True


def test_v24_newest_wins_within_same_incoming_vectorized_block():
    torch.manual_seed(9510)
    model = build_model(9510)
    memory = model.stages[0].memory
    memory.set_differentiable_pretraining(False)
    d_model = model.cfg.d_model
    address = torch.randn(1, 1, d_model)
    addresses = address.expand(1, 2, -1).clone()
    payloads = torch.randn(1, 2, d_model)
    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    state = memory.update_block(
        addresses,
        payloads,
        torch.ones(1, 2, 1),
        state,
    )
    assert isinstance(state, ContextualEpisodicMemoryState)
    assert int(state.valid.sum()) == 1
    expected = torch.tanh(memory.v(payloads[:, 1]))[0, 0]
    actual = state.values[0, state.valid[0]][0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_v24_contextual_auxiliary_is_finite_and_reaches_qkvout():
    torch.manual_seed(9511)
    model = build_model(9511)
    batch = make_batch(2, EVAL_SEED + 9511)
    terms = contextual_memory_terms(model, batch.tokens)
    loss = terms["address_contrastive_loss"] + terms["payload_token_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    stage = model.stages[0]
    for parameter in (
        stage.memory.q.weight,
        stage.memory.k.weight,
        stage.memory.v.weight,
        stage.memory.out.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_v24_one_training_step_is_finite_and_keeps_exact_sparse_budget():
    full, stream_only, result = train_pair_with_v24_objective(steps=1)
    assert len(result["history"]) == 1
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in result["history"][0].values()
    )
    assert type(full) is type(stream_only)
    stage = full.stages[0]
    assert stage.last_candidate_count == CHUNK_SIZE - 1
    assert stage.last_selected_count == 2
    assert stage.last_vectorized_update_calls == 1
    assert stage.pair_write_gate.weight.grad is not None
    assert float(stage.pair_write_gate.weight.grad.abs().sum()) > 0.0


def test_v24_direct_and_local_evaluators_run_on_actual_episdodic_state():
    model = build_model(9512)
    batch = make_batch(2, EVAL_SEED + 9512)
    direct = direct_episodic_read_evaluation(model, batch)
    local = heldout_local_contextual_code(model, batch.tokens)
    assert set(direct) == {
        "overall_accuracy",
        "overwrite_current_value_accuracy",
        "overwrite_stale_value_error",
    }
    assert set(local) == {
        "write_transition_address_identity_top1",
        "write_transition_address_margin",
        "write_payload_token_accuracy",
    }
    for value in direct.values():
        assert 0.0 <= value <= 1.0
    assert 0.0 <= local["write_transition_address_identity_top1"] <= 1.0
    assert 0.0 <= local["write_payload_token_accuracy"] <= 1.0
