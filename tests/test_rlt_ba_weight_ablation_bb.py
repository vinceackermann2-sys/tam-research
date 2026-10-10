"""BB correctness preflight loads frozen BA weights; never trains or reruns BA."""
import torch

from experiments.rlt.cpu_ba_weight_ablation_bb import (
    load_and_verify, counterfactual_test, SEEDS, EXPECTED_CHECKPOINT_SHA256,
)
from experiments.rlt.cpu_carry_intervention_ba import counterfactual_pairs


def test_ba_weights_are_sha256_pinned_and_same_across_all_ablations():
    models = load_and_verify()
    assert list(models) == ["carry", "full_reset", "erase_t120"]
    assert len(SEEDS) == 2 and EXPECTED_CHECKPOINT_SHA256
    assert models["carry"].carry_between_chunks
    assert not models["full_reset"].carry_between_chunks


def test_out_of_window_counterfactual_equal_logits_when_state_erased():
    torch.set_num_threads(2)
    models = load_and_verify()
    for gap in (16, 64, 127):
        a, b, y = counterfactual_pairs(gap, seed=20661081 + gap)
        assert torch.equal(a[:, -8:], b[:, -8:])
        for key in ("full_reset", "erase_t120"):
            with torch.no_grad():
                first = models[key](a)[:, -1]
                flipped = models[key](b)[:, -1]
            torch.testing.assert_close(first, flipped, atol=0, rtol=0)


def test_carry_and_ablations_share_all_trainable_weights():
    models = load_and_verify()
    counts = {n: sum(p.numel() for p in model.parameters()) for n,model in models.items()}
    assert set(counts.values()) == {27712}
