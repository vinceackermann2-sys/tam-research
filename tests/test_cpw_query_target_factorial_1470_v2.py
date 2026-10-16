from __future__ import annotations
import ast
import inspect
import math
import pytest
import torch
from tam_research.cpw_binding_science.harness import ARMS, PARAMETERS
import tam_research.cpw_query_target_factorial_1470_v2 as probe

def test_zero_norm_and_query_label_factorial_contrasts():
    grid = [[{
        "key_concentration": {"normalized_eight_positions": None},
        "key_gradient_l2": [0.] * 8
    } for _ in range(4)] for _ in range(4)]
    zero = probe._stats(grid, "key")
    assert zero["query_pair_count"] == 0 and zero["label_pair_count"] == 0
    assert zero["query_fixed_label_mean_tv"] is None
    assert zero["zero_norm_cells"] == 16
    for i in range(4):
        for j in range(4):
            p = [0.] * 8
            p[(0, 1, 3, 7)[i]] = 1.
            grid[i][j]["key_concentration"]["normalized_eight_positions"] = p
    result = probe._stats(grid, "key")
    assert result["query_fixed_label_mean_tv"] == 1.
    assert result["label_fixed_query_mean_tv"] == 0.
    assert result["query_pair_count"] == result["label_pair_count"] == 24

def test_signed_telescoping_is_algebraic():
    torch.manual_seed(43)
    signed = [[torch.randn(16, 6) for _ in range(4)] for _ in range(4)]
    d = probe._signed_contrasts(signed)
    assert d["interaction_count"] == 36
    assert d["max_diagonal_telescoping_residual"] < 1e-5

def test_four_by_four_cpu_sequence_only_no_training():
    r = probe.audit(arms=("sequence_only",), seed=probe.ENGINEERING_SEED)
    assert r["classification"] == "CPW_1470_V2_RANDOM_INIT_FACTORIAL_QUERY_TARGET_ONLY"
    assert r["engineering_seed_not_scientific"] == 1453001
    assert r["training_performed"] is False and r["optimizer_steps"] == 0
    assert r["scientific_attempt"] is False and r["gpu_allocated"] is False
    a = r["results"][0]
    assert a["parameter_count"] == PARAMETERS["sequence_only"]
    assert len(a["grid_4x4"]) == 4
    assert all(len(row) == 4 for row in a["grid_4x4"])
    assert a["signed_key_differences"]["interaction_count"] == 36
    assert a["signed_value_differences"]["interaction_count"] == 36
    assert a["signed_key_differences"]["query_fixed_label_signed_mean_l2"] == 0.
    assert a["signed_value_differences"]["label_fixed_query_signed_mean_l2"] == 0.
    assert a["key_metrics"]["zero_norm_cells"] == 16
    assert a["value_metrics"]["zero_norm_cells"] == 16
    assert all(not any(x != 0. for x in cell["key_gradient_l2"] + cell["value_gradient_l2"])
               for row in a["grid_4x4"] for cell in row)
    assert all(cell["correct_binding_label"] == (i == j)
               for i, row in enumerate(a["grid_4x4"]) for j, cell in enumerate(row))
    assert a["max_fixed_query_logit_identity_residual"] < 1e-4

def test_seed_five_frozen_arms_and_no_training():
    assert set(ARMS) == {"transformer","sequence_only","afm_last1","afm_first1","r1_final"}
    assert probe.ENGINEERING_SEED == 1453001
    for bad in (1452001,1453002,True,"1453001"):
        with pytest.raises(ValueError,match="preregistered"):
            probe.audit(arms=("sequence_only",),seed=bad)
    for bad in ((),("sequence_only","sequence_only"),("unknown",)):
        with pytest.raises(ValueError,match="unique frozen"):
            probe.audit(arms=bad)
    calls = {ast.unparse(n.func) for n in ast.walk(ast.parse(inspect.getsource(probe)))
             if isinstance(n, ast.Call)}
    assert not {"optimizer.step","model.train","torch.cuda","torch.optim.AdamW",
                "train_arm","evaluate_arm","torch.save","torch.load"} & calls
    assert "torch.autograd.grad" in calls

def test_separate_signed_key_value_metrics_are_not_conflated():
    # Key gradients vary only with query; value gradients only with CE label.
    key = [[torch.full((8, 3), float(i)) for j in range(4)] for i in range(4)]
    val = [[torch.full((8, 3), float(j)) for j in range(4)] for i in range(4)]
    km = probe._signed_contrasts(key)
    vm = probe._signed_contrasts(val)
    assert km["query_fixed_label_signed_mean_l2"] > 0.
    assert km["label_fixed_query_signed_mean_l2"] == 0.
    assert vm["query_fixed_label_signed_mean_l2"] == 0.
    assert vm["label_fixed_query_signed_mean_l2"] > 0.
    assert km["query_label_interaction_signed_mean_l2"] < 1e-6
    assert vm["query_label_interaction_signed_mean_l2"] < 1e-6
    assert km["max_diagonal_telescoping_residual"] < 1e-6
    assert vm["max_diagonal_telescoping_residual"] < 1e-6
