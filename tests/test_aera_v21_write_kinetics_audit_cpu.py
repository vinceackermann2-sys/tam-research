import torch

from aera_v19_memory_necessity_cpu import CHUNK_SIZE, EVAL_SEED, _evaluate, make_batch
from aera_v21_conflict_free_memory_objective_cpu import train_pair_with_conflict_free_objective
from aera_v21_memory_necessity_cpu import build_model
from aera_v21_write_kinetics_audit_cpu import (
    DIAGNOSTIC_ONE_SHOT_LR,
    MATERIAL_GAIN,
    RAW_RECALL_STRONG,
    _evaluate_mode,
    _keep_candidate_one_preserve_gate,
    _temporary_memory_lr,
    trace_write_kinetics,
)


def test_frozen_write_kinetics_thresholds():
    assert MATERIAL_GAIN == 0.15
    assert RAW_RECALL_STRONG == 0.95
    assert DIAGNOSTIC_ONE_SHOT_LR == 1.0


def test_candidate_selector_preserves_position_one_gate_and_suppresses_others():
    model = build_model(9911)
    batch = make_batch(2, EVAL_SEED + 811)
    chunk = batch.tokens[:, :CHUNK_SIZE]
    pos = torch.arange(CHUNK_SIZE)
    events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
    stage = model.stages[0]
    base_h = stage.norm(events)
    features = torch.cat((base_h[:, :-1], base_h[:, 1:]), dim=-1)
    original = stage.pair_write_gate(features).detach()
    with _keep_candidate_one_preserve_gate(model):
        masked = stage.pair_write_gate(features).detach()
    torch.testing.assert_close(masked[:, 1], original[:, 1], atol=0.0, rtol=0.0)
    assert bool((masked[:, 0] <= -29.0).all())
    assert bool((masked[:, 2:] <= -29.0).all())


def test_temporary_memory_lr_restores_all_stages():
    model = build_model(9912)
    before = [float(stage.memory.lr) for stage in model.stages]
    with _temporary_memory_lr(model, 1.0):
        assert all(float(stage.memory.lr) == 1.0 for stage in model.stages)
    assert [float(stage.memory.lr) for stage in model.stages] == before


def test_chunkwise_normal_matches_reference_after_one_step_training():
    full, _, _ = train_pair_with_conflict_free_objective(steps=1)
    batch = make_batch(4, EVAL_SEED + 812)
    reference = _evaluate(full, batch, memory=True)
    audited = _evaluate_mode(full, batch)["final"]
    assert abs(reference["query_accuracy"] - audited["overall_accuracy"]) < 1e-7
    assert abs(reference["query_nll"] - audited["query_nll"]) < 1e-6


def test_write_kinetics_trace_reports_effective_eta():
    full, _, _ = train_pair_with_conflict_free_objective(steps=1)
    batch = make_batch(2, EVAL_SEED + 813)
    trace = trace_write_kinetics(full, batch)
    assert 0.0 <= trace["initial_target_strength_mean"] <= 1.0
    assert 0.0 <= trace["overwrite_target_strength_mean"] <= 1.0
    assert 0.0 <= trace["initial_target_eta_mean"] <= 0.2
    assert 0.0 <= trace["overwrite_target_eta_mean"] <= 0.2
