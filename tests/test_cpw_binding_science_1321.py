"""CPW #1321 zero-GPU five-arm model/data integrity.

All experimental seeds below are ONLY CPU test fixtures, not paid-run seeds.
"""
from __future__ import annotations

import gc

import pytest
import torch

from tam_research.cpw_binding_science.evaluation import (
    cpu_train_integrity_step,
    evaluate_counterfactual_groups,
    evaluate_distances,
)
from tam_research.cpw_binding_science.models import ARMS, PARAMETERS, build_model, query_logits
from tam_research.cpw_binding_v2.task import (
    QUERY_POSITION,
    SCORED_DELAYS,
    VALUE_DELAYS,
    explicit_key_lookup,
)
from tam_research.models import parameter_count


@pytest.fixture(scope="module", autouse=True)
def _cpu_only_threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def _perfect_oracle(tokens: torch.Tensor) -> torch.Tensor:
    target = explicit_key_lookup(tokens)
    logits = torch.full((tokens.size(0), 4_200), -10.0)
    logits.scatter_(1, target[:, None], 10.0)
    return logits


def _query_blind(tokens: torch.Tensor) -> torch.Tensor:
    # Never inspect query token or source key; fixed value token at d=32.
    target = tokens[:, QUERY_POSITION - SCORED_DELAYS[0]]
    logits = torch.full((tokens.size(0), 4_200), -10.0)
    logits.scatter_(1, target[:, None], 10.0)
    return logits


def test_data_fingerprint_replays_across_arms_but_differs_across_seeds():
    a = evaluate_distances(
        _perfect_oracle, eval_seed=1321, examples_per_distance=8, batch_size=4
    )
    b = evaluate_distances(
        _query_blind, eval_seed=1321, examples_per_distance=8, batch_size=4
    )
    c = evaluate_distances(
        _perfect_oracle, eval_seed=1322, examples_per_distance=8, batch_size=4
    )
    assert a["data_sha256_by_distance"] == b["data_sha256_by_distance"]
    assert a["data_sha256_by_distance"] != c["data_sha256_by_distance"]
    assert a["evaluation_examples"] == 32
    assert all(a["by_distance"][str(d)]["accuracy"] == 1.0 for d in SCORED_DELAYS)
    assert b["by_distance"]["32"]["accuracy"] == 1.0
    assert all(b["by_distance"][str(d)]["accuracy"] == 0.0 for d in (64, 128, 256))
    assert b["long_mean_accuracy"] == 0.0
    assert a["long_mean_accuracy"] == 1.0


def test_all_four_counterfactuals_reject_key_blind_guess():
    oracle = evaluate_counterfactual_groups(
        _perfect_oracle, eval_seed=1321, source_groups=8, group_batch_size=4
    )
    blind = evaluate_counterfactual_groups(
        _query_blind, eval_seed=1321, source_groups=8, group_batch_size=4
    )
    assert oracle["data_sha256"] == blind["data_sha256"]
    assert oracle["query_examples"] == 32
    assert oracle["all_four_group_accuracy"] == 1.0
    assert oracle["per_query_accuracy"] == 1.0
    assert blind["per_query_accuracy"] == 0.25
    assert blind["all_four_group_accuracy"] == 0.0


def test_invalid_eval_shapes_are_rejected():
    with pytest.raises(ValueError):
        evaluate_counterfactual_groups(_perfect_oracle, eval_seed=1, source_groups=0)
    with pytest.raises(ValueError):
        evaluate_distances(_perfect_oracle, eval_seed=1, examples_per_distance=0)
    with pytest.raises(ValueError, match="final-query logits"):
        evaluate_counterfactual_groups(
            lambda x: torch.zeros((x.size(0), 20, 4_200)),
            eval_seed=1, source_groups=1
        )


@pytest.mark.parametrize("arm", ARMS)
def test_exact_frozen_model_factory_and_strict_future_invariance(arm: str):
    torch.manual_seed(1321)
    model = build_model(arm)
    assert parameter_count(model) == PARAMETERS[arm]
    model.eval()
    tokens = torch.randint(0, 4_200, (1, 16))
    future = tokens.clone()
    future[:, 9:] = (future[:, 9:] + 1) % 4_200
    with torch.no_grad():
        early = model(tokens)
        modified = model(future)
        assert early.shape == (1, 16, 50_257)
        torch.testing.assert_close(early[:, :9], modified[:, :9], atol=5e-5, rtol=0)
        logits = query_logits(model, tokens)
        torch.testing.assert_close(logits, early[:, -1], atol=5e-5, rtol=0)
    del model
    gc.collect()


@pytest.mark.parametrize("arm", ARMS)
def test_single_cpu_optimizer_step_reaches_real_model_parameters(arm: str):
    torch.manual_seed(1322)
    model = build_model(arm)
    result = cpu_train_integrity_step(model, train_seed=1323, batch_size=1)
    assert result["cpu_only"] == 1.0
    assert result["query_loss"] > 0
    assert result["gradient_norm"] > 0
    del model
    gc.collect()


def test_frozen_task_slots_are_consistent():
    assert set(SCORED_DELAYS).issubset(set(VALUE_DELAYS))
    assert QUERY_POSITION == 319
    assert len(ARMS) == len(PARAMETERS) == 5
    assert PARAMETERS["afm_first1"] == PARAMETERS["afm_last1"]
    assert PARAMETERS["r1_final"] == PARAMETERS["sequence_only"]
    assert PARAMETERS["transformer"] > PARAMETERS["r1_final"]
