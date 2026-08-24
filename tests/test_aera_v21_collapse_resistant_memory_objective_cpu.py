import math

import torch

from aera_v19_memory_necessity_cpu import EVAL_SEED, make_batch
from aera_v21_collapse_resistant_memory_objective_cpu import (
    ADDRESS_CONTRASTIVE_WEIGHT,
    ADDRESS_MARGIN_MIN,
    ADDRESS_TEMPERATURE,
    PAYLOAD_LATENT_WEIGHT,
    PAYLOAD_LEGAL_ACCURACY_MIN,
    PAYLOAD_TOKEN_WEIGHT,
    collapse_resistant_memory_terms,
    evaluate_local_memory_code,
    multi_positive_contrastive_loss,
    train_pair_with_collapse_resistant_objective,
)
from aera_v21_memory_necessity_cpu import build_model


def _grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def test_frozen_objective_constants():
    assert ADDRESS_TEMPERATURE == 0.10
    assert ADDRESS_CONTRASTIVE_WEIGHT == 1.0
    assert PAYLOAD_LATENT_WEIGHT == 1.0
    assert PAYLOAD_TOKEN_WEIGHT == 1.0
    assert ADDRESS_MARGIN_MIN == 0.05
    assert PAYLOAD_LEGAL_ACCURACY_MIN == 0.90


def test_contrastive_loss_penalizes_collapsed_addresses():
    identity = torch.arange(4)
    collapsed = torch.ones(4, 4)
    separated = torch.eye(4)
    collapsed_loss = multi_positive_contrastive_loss(collapsed, collapsed, identity)
    separated_loss = multi_positive_contrastive_loss(separated, separated, identity)
    assert float(separated_loss) < float(collapsed_loss) - 0.5


def test_auxiliary_gradients_are_isolated_to_existing_memory_code():
    model = build_model(9801)
    batch = make_batch(2, EVAL_SEED + 501)
    model.zero_grad(set_to_none=True)
    terms = collapse_resistant_memory_terms(model, batch.tokens)
    loss = (
        terms["address_contrastive_loss"]
        + terms["payload_latent_loss"]
        + terms["payload_token_loss"]
    )
    loss.backward()

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


def test_local_code_metrics_are_bounded_and_finite():
    model = build_model(9802)
    row = evaluate_local_memory_code(model)
    assert 0.0 <= row["address_top1_accuracy"] <= 1.0
    assert math.isfinite(row["address_mean_diag_minus_best_other_margin"])
    assert 0.0 <= row["payload_legal_value_accuracy"] <= 1.0


def test_one_step_training_runs_with_frozen_objective():
    full, stream, result = train_pair_with_collapse_resistant_objective(steps=1)
    assert full is not None and stream is not None
    assert len(result["history"]) == 1
    assert "initial_local_code" in result
    assert "final_local_code" in result
    assert all(math.isfinite(v) for v in result["history"][0].values())
