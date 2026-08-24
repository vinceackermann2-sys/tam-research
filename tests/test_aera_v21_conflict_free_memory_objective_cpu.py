import math

import torch

from aera_v19_memory_necessity_cpu import EVAL_SEED, make_batch
from aera_v21_conflict_free_memory_objective_cpu import (
    ADDRESS_CONTRASTIVE_WEIGHT,
    LATENT_PAYLOAD_WEIGHT,
    PAYLOAD_TOKEN_WEIGHT,
    conflict_free_memory_terms,
    train_pair_with_conflict_free_objective,
)
from aera_v21_memory_necessity_cpu import build_model


def _grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def test_frozen_conflict_free_weights():
    assert ADDRESS_CONTRASTIVE_WEIGHT == 1.0
    assert PAYLOAD_TOKEN_WEIGHT == 1.0
    assert LATENT_PAYLOAD_WEIGHT == 0.0


def test_terms_contain_no_latent_payload_objective():
    model = build_model(9901)
    batch = make_batch(2, EVAL_SEED + 701)
    terms = conflict_free_memory_terms(model, batch.tokens)
    assert set(terms) == {
        "address_contrastive_loss",
        "payload_token_loss",
        "payload_token_accuracy",
    }


def test_auxiliary_gradients_reach_only_existing_memory_code():
    model = build_model(9902)
    batch = make_batch(2, EVAL_SEED + 702)
    model.zero_grad(set_to_none=True)
    terms = conflict_free_memory_terms(model, batch.tokens)
    (terms["address_contrastive_loss"] + terms["payload_token_loss"]).backward()
    memory = model.stages[0].memory
    assert _grad_norm(memory.q.weight) > 0.0
    assert _grad_norm(memory.k.weight) > 0.0
    assert _grad_norm(memory.v.weight) > 0.0
    assert _grad_norm(memory.out.weight) > 0.0
    assert _grad_norm(model.token_emb.weight) == 0.0
    assert _grad_norm(model.local_pos.weight) == 0.0
    assert _grad_norm(model.stages[0].norm.weight) == 0.0
    assert _grad_norm(model.norm.weight) == 0.0
    assert _grad_norm(model.lm_head.weight) == 0.0


def test_one_step_conflict_free_training_runs():
    full, stream, result = train_pair_with_conflict_free_objective(steps=1)
    assert full is not None and stream is not None
    assert len(result["history"]) == 1
    row = result["history"][0]
    assert all(math.isfinite(v) for v in row.values())
    assert "initial_local_code" in result
    assert "final_local_code" in result
