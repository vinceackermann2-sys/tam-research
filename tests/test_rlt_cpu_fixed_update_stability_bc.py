"""BC CPU preflight: no training attempt, checkpoint saving or job mutation."""
import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.cpu_fixed_update_stability_bc import (
    BATCH, DISTANCES, EASY_UPDATES, EVAL_SEED, LONG_UPDATES, MILESTONES,
    MODEL_SEEDS, PARAMETERS, SEQ_LEN, build, binary_logits, holdouts,
    train_updates,
)
from experiments.rlt.cpu_lastwrite_probe_ar import SET0, SET1, last_write_labels, make_batch
from experiments.rlt.model import parameter_count


def test_immutable_protocol_and_new_seeds():
    assert len(set(MODEL_SEEDS)) == 3
    assert all(s not in {20301081, 20441081, 20541081} for s in MODEL_SEEDS)
    assert (EASY_UPDATES, LONG_UPDATES, MILESTONES) == (2100, 1200, (400,800,1200))


def test_model_parameter_count_and_carry():
    for seed in MODEL_SEEDS:
        m = build(seed)
        assert m.carry_between_chunks and m.chunk_size == 8
        assert parameter_count(m) == PARAMETERS


def test_eval_balanced_lastwrite_no_oracle_label_leakage():
    for gap,(x,y) in holdouts().items():
        assert x.shape == (256,SEQ_LEN)
        assert int(y.sum()) == 128
        assert torch.equal(last_write_labels(x), y)
        assert torch.equal(x[:, SEQ_LEN-1-gap], y+SET0)


def test_pair_flip_past_final_chunk_changes_answer_only():
    for gap in (16,64,127):
        x,y=holdouts()[gap]
        flipped=x.clone()
        flipped[:,SEQ_LEN-1-gap]=(1-y)+SET0
        assert torch.equal(x[:,-8:],flipped[:,-8:])
        assert torch.equal(last_write_labels(flipped),1-y)


def test_forward_gradient_cpu_only_without_optimizer_step():
    torch.set_num_threads(2)
    model=build(MODEL_SEEDS[0])
    x,y,_=make_batch(BATCH,(16,),20711081,seq_len=SEQ_LEN)
    loss=F.cross_entropy(binary_logits(model,x),y)
    loss.backward()
    assert loss.isfinite()
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_preflight_does_not_execute_training_job():
    assert len(MODEL_SEEDS)==3
    assert callable(train_updates)
