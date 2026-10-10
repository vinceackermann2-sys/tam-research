"""BA diagnostics preflight, zero training/zero GPU; never calls run()."""
import torch
import torch.nn.functional as F

from experiments.rlt.cpu_carry_intervention_ba import (
    build, binary_logits, counterfactual_pairs, compare_carry_states,
    EVAL_SEED, EXPECTED_PARAMETERS, MODEL_SEED,
)
from experiments.rlt.cpu_bounded_streaming_aw import balanced_holdout
from experiments.rlt.model import parameter_count


def test_model_seed_parameter_count_and_single_carry_channel():
    model = build()
    assert parameter_count(model) == EXPECTED_PARAMETERS
    assert model.carry_between_chunks and model.chunk_size == 8


def test_counterfactual_oracle_and_identical_suffix():
    for gap in (1, 4, 16, 64, 127):
        a, b, target = counterfactual_pairs(gap, 20561081 + gap)
        assert len(target) == 16 and int(target.sum()) == 8
        index = a.size(1) - 1 - gap
        assert torch.equal(a[:, index+1:], b[:, index+1:])
        if gap >= 8:
            assert torch.equal(a[:, -8:], b[:, -8:])
        else:
            assert not torch.equal(a[:, -8:], b[:, -8:])


def test_diagnostic_state_pair_has_valid_lengths_and_finite_sensitivities():
    torch.set_num_threads(2)
    model = build()
    report = compare_carry_states(model, 64, seed=20561081 + 64)
    assert report["identical_final_chunk"]
    assert report["write_index"] == 63
    assert len(report["state_l2_by_chunk"]) == 16
    assert len(report["state_relative_l2_by_chunk"]) == 16
    assert report["class_balance_each"] == 8
    assert report["state_l2_by_chunk"][7] > 0
    assert all(v >= 0 and torch.isfinite(torch.tensor(v)) for v in report["state_l2_by_chunk"])
    assert report["readout_logit_mean_absolute_difference"] >= 0


def test_fresh_model_can_backpropagate_final_binary_loss():
    torch.set_num_threads(2)
    model = build()
    x, y = balanced_holdout(16, seed=EVAL_SEED + 16)
    logits = binary_logits(model, x[:8])
    assert logits.shape == (8, 2)
    loss = F.cross_entropy(logits, y[:8])
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_seed_not_reused_from_AY():
    assert MODEL_SEED not in (20441081, 20301081)
