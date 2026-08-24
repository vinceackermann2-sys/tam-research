import math

import torch

from aera_v19_memory_necessity_cpu import EVAL_SEED, make_batch
from aera_v21_memory_code_objective_cpu import (
    PAYLOAD_LEGAL_ACCURACY_MIN,
    PAYLOAD_RECONSTRUCTION_WEIGHT,
    QK_ALIGNMENT_WEIGHT,
    QK_IMPROVEMENT_MIN,
    evaluate_memory_code,
    memory_code_terms,
    train_pair_with_memory_code_objective,
)
from aera_v21_memory_necessity_cpu import build_model


def _grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def test_frozen_objective_constants():
    assert QK_ALIGNMENT_WEIGHT == 1.0
    assert PAYLOAD_RECONSTRUCTION_WEIGHT == 1.0
    assert QK_IMPROVEMENT_MIN == 1e-3
    assert PAYLOAD_LEGAL_ACCURACY_MIN == 0.90


def test_memory_code_auxiliary_gradients_are_isolated_from_backbone_and_decoder():
    model = build_model(9501)
    batch = make_batch(2, EVAL_SEED + 401)
    model.zero_grad(set_to_none=True)
    terms = memory_code_terms(model, batch.tokens)
    loss = terms["qk_alignment_loss"] + terms["payload_reconstruction_loss"]
    loss.backward()

    memory = model.stages[0].memory
    assert _grad_norm(memory.q.weight) > 0.0
    assert _grad_norm(memory.k.weight) > 0.0
    assert _grad_norm(memory.v.weight) > 0.0
    assert _grad_norm(memory.out.weight) > 0.0

    # Detached event representations and frozen-copy decoder weights prevent the
    # auxiliary from succeeding by rewriting the backbone or LM decoder.
    assert _grad_norm(model.token_emb.weight) == 0.0
    assert _grad_norm(model.local_pos.weight) == 0.0
    assert _grad_norm(model.stages[0].norm.weight) == 0.0
    assert _grad_norm(model.norm.weight) == 0.0
    assert _grad_norm(model.lm_head.weight) == 0.0


def test_memory_code_terms_are_finite_and_sequence_derived():
    model = build_model(9502)
    batch = make_batch(2, EVAL_SEED + 402)
    terms = memory_code_terms(model, batch.tokens)
    assert set(terms) == {
        "qk_alignment_loss",
        "qk_cosine",
        "payload_reconstruction_loss",
        "payload_reconstruction_accuracy",
    }
    for value in terms.values():
        assert torch.isfinite(value)
    assert -1.0 <= float(terms["qk_cosine"].detach()) <= 1.0
    assert 0.0 <= float(terms["payload_reconstruction_accuracy"].detach()) <= 1.0


def test_heldout_code_diagnostic_is_bounded_and_finite():
    model = build_model(9503)
    batch = make_batch(3, EVAL_SEED + 403)
    row = evaluate_memory_code(model, batch)
    assert -1.0 <= row["qk_cosine"] <= 1.0
    assert 0.0 <= row["payload_legal_value_accuracy"] <= 1.0
    assert 0.0 <= row["payload_full_vocab_accuracy"] <= 1.0
    assert all(math.isfinite(v) for v in row.values())


def test_one_step_training_keeps_pair_checkpoints_identical_at_start_and_runs():
    full, stream, result = train_pair_with_memory_code_objective(steps=1)
    assert full is not None and stream is not None
    assert len(result["history"]) == 1
    assert "initial_memory_code" in result
    assert "final_memory_code" in result
    row = result["history"][0]
    for value in row.values():
        assert math.isfinite(value)
