from __future__ import annotations

import ast
import inspect
import math

import pytest
import torch

import tam_research.cpw_query_gradient_matrix_1452 as probe
from tam_research.cpw_binding_science.harness import ARMS, PARAMETERS
from tam_research.cpw_binding_v2.task import (
    QUERY_POSITION, SCORED_DELAYS, VALUE_DELAYS,
    explicit_key_lookup,
)


def test_four_counterfactual_queries_share_one_source_prefix_and_bind_four_values():
    rows, sha = probe._counterfactual_source(probe.ENGINEERING_SEED)
    assert len(sha) == 64 and len(rows) == 4
    assert [delay for delay,_,_ in rows] == list(SCORED_DELAYS)
    assert len({int(tokens[0,-1]) for _,tokens,_ in rows}) == 4
    assert len({int(target[0]) for _,_,target in rows}) == 4
    assert len({tuple(tokens[0,:QUERY_POSITION].tolist()) for _,tokens,_ in rows}) == 1
    assert all(torch.equal(explicit_key_lookup(tokens), y)
               for _, tokens, y in rows)
    assert sum(int(int(rows[0][2][0]) == int(y[0])) for _,_,y in rows) == 1


def test_gradient_matrix_coordinates_and_zero_norm_semantics():
    grad = torch.zeros(1, QUERY_POSITION + 1, 3)
    zero = probe._row_metrics(grad, 256)
    assert zero["key_gradient_l2"] == [0.0]*8
    assert zero["value_gradient_l2"] == [0.0]*8
    assert zero["key_concentration"]["zero_norm"] is True
    assert zero["key_concentration"]["normalized_eight_positions"] is None
    assert zero["value_concentration"]["target_fraction_of_all_eight"] is None
    selected = VALUE_DELAYS.index(256)
    grad[:,QUERY_POSITION-256-1,:] = 4
    grad[:,QUERY_POSITION-256,:] = 2
    one = probe._row_metrics(grad,256)
    assert one["selected_source_key_position"] == QUERY_POSITION-257
    assert one["selected_source_value_position"] == QUERY_POSITION-256
    assert one["key_concentration"]["target_fraction_of_all_eight"] == 1.0
    assert one["value_concentration"]["target_fraction_of_all_eight"] == 1.0
    assert one["key_concentration"]["distractor_sum_grad_l2"] == 0.0


def test_nonlearning_local_sequence_cannot_receive_far_source_gradients():
    result = probe.audit(arms=("sequence_only",),seed=probe.ENGINEERING_SEED)
    assert result["classification"] == probe.CLASSIFICATION
    assert result["key_blind_accuracy"] == .25
    assert result["training_performed"] is False
    assert result["optimizer_steps"] == 0
    assert result["scientific_attempt"] is False
    assert result["gpu_allocated"] is False
    assert result["breakthrough_claim"] is False
    row = result["results"][0]
    assert row["parameter_count"] == PARAMETERS["sequence_only"]
    assert len(row["source_gradients_key_l2_4x8"]) == 4
    assert len(row["source_gradients_value_l2_4x8"]) == 4
    assert all(len(x)==8 for x in row["source_gradients_key_l2_4x8"])
    assert all(all(abs(v)<1e-12 for v in x)
               for x in row["source_gradients_key_l2_4x8"])
    assert all(all(abs(v)<1e-12 for v in x)
               for x in row["source_gradients_value_l2_4x8"])
    assert row["key_routing_change"]["comparable_query_pairs"] == 0
    assert all(r["key_concentration"]["zero_norm"] for r in row["rows"])
    assert len({r["query_token"] for r in row["rows"]}) == 4


def test_exact_seed_contract_and_five_frozen_names():
    assert probe.ENGINEERING_SEED == 1452001
    assert set(ARMS) == {
        "transformer","sequence_only","afm_last1","afm_first1","r1_final"
    }
    for invalid in (0, 1452002, True, "1452001"):
        with pytest.raises(ValueError,match="preregistered"):
            probe.audit(arms=("sequence_only",),seed=invalid)
    for invalid in ((),("sequence_only","sequence_only"),("unknown",)):
        with pytest.raises(ValueError,match="unique frozen"):
            probe.audit(arms=invalid)


def test_no_training_or_paid_compute_calls_and_fixed_dependency_shas():
    source = inspect.getsource(probe)
    tree = ast.parse(source)
    called = {ast.unparse(n.func)
              for n in ast.walk(tree) if isinstance(n,ast.Call)}
    prohibited = {
        "optimizer.step","loss.backward", "model.train", "torch.cuda",
        "modal.App", "modal.run", "torch.save", "torch.load",
        "train_arm","evaluate_arm", "torch.optim.AdamW",
    }
    assert prohibited.isdisjoint(called)
    assert "loss.backward" in called
    assert "build_arm" in called
    assert "model.eval" not in called  # the model is constructed in eval mode
    assert ".cpu().eval()" in source
    assert probe.NEW_SCIENTIFIC_ATTEMPT is False
    assert probe.GPU_AUTHORIZED is False
    assert probe.FROZEN_DEPENDENCY_BLOBS == {
        "task":"79dc0ceda617985eba40f119343bdb2992aa395d",
        "harness":"63ab35fc82ea06d91ce2a9b5cbd1d2d9c56baf3f",
        "prior_single_query_audit":"49c6154e48fce7edace97ad630783153d39a6b0d",
    }
    expected_key_positions=[QUERY_POSITION-d-1 for d in VALUE_DELAYS]
    assert expected_key_positions[0]==286 and expected_key_positions[-1]==62
