import torch

from aera_v19_memory_necessity_cpu import EVAL_SEED, make_batch
from aera_v21_conflict_free_memory_objective_cpu import train_pair_with_conflict_free_objective
from aera_v21_residual_raw_recall_audit_cpu import (
    DIAGNOSTIC_MEMORY_LR,
    NO_DECAY,
    PRODUCTION_DIAGNOSTIC_DECAY,
    RAW_RECALL_MIN,
    _temporary_memory_decay,
    learned_geometry,
    sequential_raw_recall,
)
from aera_v21_residual_raw_recall_audit_cpu_v2 import _least_squares_matrix
from aera_v21_write_kinetics_audit_cpu import _evaluate_mode


def test_frozen_residual_raw_recall_constants():
    assert RAW_RECALL_MIN == 0.95
    assert DIAGNOSTIC_MEMORY_LR == 1.0
    assert PRODUCTION_DIAGNOSTIC_DECAY == 0.999
    assert NO_DECAY == 1.0


def test_least_squares_matrix_reconstructs_independent_bindings():
    # Deliberately underdetermined, matching the audit geometry: fewer live
    # bindings than memory dimensions.  The matrix-capacity oracle must preserve
    # every exactly representable binding instead of depending on LAPACK's
    # default underdetermined lstsq rank choice.
    k = torch.eye(8, 12)
    v = torch.randn(8, 12, generator=torch.Generator().manual_seed(9921))
    matrix = _least_squares_matrix(k, v)
    torch.testing.assert_close(k @ matrix, v, atol=1e-5, rtol=1e-5)


def test_temporary_memory_decay_restores_all_stages():
    full, _, _ = train_pair_with_conflict_free_objective(steps=1)
    before = [float(stage.memory.decay) for stage in full.stages]
    with _temporary_memory_decay(full, 1.0):
        assert all(float(stage.memory.decay) == 1.0 for stage in full.stages)
    assert [float(stage.memory.decay) for stage in full.stages] == before


def test_new_sequential_q_baseline_matches_prior_strict_raw_evaluator():
    full, _, _ = train_pair_with_conflict_free_objective(steps=1)
    batch = make_batch(3, EVAL_SEED + 921)
    prior = _evaluate_mode(
        full,
        batch,
        write_chunks_only=True,
        keep_candidate_one=True,
        memory_lr=1.0,
    )["raw_memory_decode"]
    current = sequential_raw_recall(full, batch, read="q", decay=0.999)
    assert abs(prior["overall_accuracy"] - current["overall_accuracy"]) < 1e-7
    assert abs(
        prior["overwrite_current_value_accuracy"]
        - current["overwrite_current_value_accuracy"]
    ) < 1e-7


def test_learned_geometry_is_finite_and_has_expected_key_count():
    full, _, _ = train_pair_with_conflict_free_objective(steps=1)
    geometry = learned_geometry(full)
    assert 0.0 <= geometry["q_to_k_top1_accuracy"] <= 1.0
    assert len(geometry["k_singular_values"]) == 12
    assert 0 <= geometry["k_effective_rank"] <= 12
    assert torch.isfinite(torch.tensor(geometry["k_condition_number_effective"]))
