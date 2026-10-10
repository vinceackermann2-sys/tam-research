from __future__ import annotations

import math

import pytest
import torch

from experiments.rlt.cpu_seed_sensitivity_av import (
    CASES, DISTANCES, EASY_SECONDS, EVAL_PER_CLASS,
    EXPECTED_PARAMETERS, LONG_SECONDS, MODELS,
    balanced_evaluate, gate_diagnostics, run_case,
)
from experiments.rlt.cpu_lastwrite_curriculum_at import (
    balanced_batch_for_distance, timed_phase,
)
from experiments.rlt.cpu_lastwrite_probe_ar import (
    make_models, last_write_labels,
)
from experiments.rlt.model import parameter_count


def test_three_fresh_seed_cases_are_disjoint_and_unique() -> None:
    assert [row["case"] for row in CASES] == [0, 1, 2]
    seeds = [v for row in CASES for k, v in row.items() if k.endswith("_seed")]
    assert len(seeds) == 12 and len(set(seeds)) == 12
    assert not (set(seeds) & {58232, 58233})
    assert MODELS == ("residual_scan", "adaptive_scan")
    assert DISTANCES == (1, 4, 16, 40, 63)
    assert EASY_SECONDS == 20.0 and LONG_SECONDS == 90.0


@pytest.mark.parametrize("case", CASES)
def test_balanced_heldout_for_each_fresh_case(case: dict[str, int]) -> None:
    for distance in DISTANCES:
        x, y = balanced_batch_for_distance(
            distance, 64, case["eval_seed"] + distance,
        )
        assert x.shape == (2 * EVAL_PER_CLASS, 64)
        assert int(y.sum()) == EVAL_PER_CLASS
        assert torch.equal(last_write_labels(x), y)


@pytest.mark.parametrize("name", MODELS)
def test_equal_parameters_and_finite_gate_stats(name: str) -> None:
    torch.manual_seed(CASES[0]["model_seed"])
    model = make_models()[name]
    assert parameter_count(model) == EXPECTED_PARAMETERS
    stats = gate_diagnostics(model, CASES[0]["eval_seed"])
    assert all(math.isfinite(value) for value in stats.values())
    assert 0 <= stats["retention_mean"] <= 1
    assert 0 <= stats["gain_mean"] <= 2
    assert 0 <= stats["gain_near_zero_frac"] <= 1


def test_cpu_timed_phase_nonzero_without_step_cap() -> None:
    torch.manual_seed(CASES[0]["model_seed"])
    model = make_models()["adaptive_scan"]
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    stats = timed_phase(
        model, opt, seconds=0.015, seed=CASES[0]["easy_seed"],
        seq_len=8, distances=(1,),
    )
    assert stats["train_seconds"] >= 0.015
    assert stats["steps"] >= 1
    assert stats["tokens_seen"] == stats["steps"] * 8 * 8


def test_unregistered_case_refused_without_training() -> None:
    with pytest.raises(ValueError, match="unregistered AV case"):
        run_case(3)
    with pytest.raises(ValueError, match="unregistered AV case"):
        run_case(-1)
