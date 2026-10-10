"""Pure preflight AY tests: NEVER invoke AY run() or consume its job."""
import torch
import torch.nn.functional as F

from experiments.rlt.cpu_bounded_threeway_ay import (
    build_models, binary_logits, DISTANCES, EXPECTED_PARAMETERS, CANDIDATES,
    EVAL_SEED, BATCH_SIZE,
)
from experiments.rlt.cpu_bounded_streaming_aw import balanced_holdout
from experiments.rlt.cpu_lastwrite_probe_ar import last_write_labels
from experiments.rlt.model import parameter_count


def test_models_params_initial_copy_and_structure():
    models = build_models()
    assert tuple(models) == CANDIDATES
    assert {n: parameter_count(m) for n, m in models.items()} == EXPECTED_PARAMETERS
    assert models["carry"].carry_between_chunks
    assert not models["reset"].carry_between_chunks
    for k, v in models["carry"].state_dict().items():
        torch.testing.assert_close(v, models["reset"].state_dict()[k], atol=0, rtol=0)
    assert models["sliding_kv"].receptive_distance == 14


def test_model_output_backward_and_finite_on_distinct_examples():
    torch.set_num_threads(2)
    x, y = balanced_holdout(16, seed=EVAL_SEED + 16)
    assert torch.equal(last_write_labels(x), y)
    for model in build_models().values():
        logits = binary_logits(model, x[:BATCH_SIZE])
        assert logits.shape == (BATCH_SIZE, 2)
        loss = F.cross_entropy(logits, y[:BATCH_SIZE])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_all_eval_distances_are_balanced_and_oracle_valid():
    for gap in DISTANCES:
        x, y = balanced_holdout(gap, seed=EVAL_SEED + gap)
        assert x.shape == (256, 128)
        assert int(y.sum()) == 128
        assert torch.equal(last_write_labels(x), y)
