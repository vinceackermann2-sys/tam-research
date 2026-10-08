from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tam_research.cpw_binding_science.harness import (
    ARMS,
    PARAMETERS,
    build_arm,
    evaluate_arm,
    query_logits,
)
from tam_research.cpw_binding_v2.task import (
    SCORED_DELAYS,
    counterfactual_query,
    explicit_key_lookup,
    make_binding_batch,
    positional_keyblind_guess,
)
from tam_research.models import parameter_count


def test_exact_five_arm_parameter_contracts() -> None:
    for arm in ARMS:
        model = build_arm(arm)
        assert parameter_count(model) == PARAMETERS[arm]
    assert PARAMETERS["afm_last1"] == PARAMETERS["afm_first1"]
    assert PARAMETERS["r1_final"] == PARAMETERS["sequence_only"]
    assert PARAMETERS["transformer"] > PARAMETERS["r1_final"]


def test_deconfounded_oracle_and_keyblind_counterfactual_exact_ceiling() -> None:
    g = torch.Generator().manual_seed(1321)
    base = make_binding_batch(batch_size=7, generator=g, delay=32)
    prefix = base.tokens[:, :-1].clone()
    fixed_guess = positional_keyblind_guess(base.tokens)
    oracle = 0
    blind = 0
    all_targets = []
    for delay in SCORED_DELAYS:
        tokens, targets = counterfactual_query(base, new_delay=delay)
        assert torch.equal(tokens[:, :-1], prefix)
        assert torch.equal(explicit_key_lookup(tokens), targets)
        blind += int((fixed_guess == targets).sum())
        oracle += len(targets)
        all_targets.append(targets)
    assert oracle == 4 * len(base.targets)
    assert blind == len(base.targets)
    for i in range(4):
        for j in range(i + 1, 4):
            assert torch.all(all_targets[i] != all_targets[j])


def test_identical_train_and_eval_streams_replay() -> None:
    for delay in (None, 32, 128):
        g1 = torch.Generator().manual_seed(3307)
        g2 = torch.Generator().manual_seed(3307)
        a = make_binding_batch(batch_size=5, generator=g1, delay=delay)
        b = make_binding_batch(batch_size=5, generator=g2, delay=delay)
        assert torch.equal(a.tokens, b.tokens)
        assert torch.equal(a.targets, b.targets)


@pytest.mark.parametrize("arm", ["sequence_only", "afm_first1", "r1_final"])
def test_corrected_query_logits_are_finite_and_causal(arm: str) -> None:
    torch.manual_seed(1321)
    model = build_arm(arm).eval()
    batch = make_binding_batch(
        batch_size=1,
        generator=torch.Generator().manual_seed(1322),
        delay=256,
    )
    altered = batch.tokens.clone()
    altered[:, -1] = (altered[:, -1] + 1) % model.cfg.vocab_size
    with torch.no_grad():
        original = query_logits(model, batch.tokens)
        changed = query_logits(model, altered)
    assert original.shape == (1, 50_257)
    assert torch.isfinite(original).all()
    assert torch.isfinite(changed).all()
    # Same prefix, different query key should be able to change the output.
    # Random untrained logits are NOT evidence of learned binding.
    assert float((original - changed).abs().max()) > 0.0


def test_single_cpu_backward_reaches_early_afm() -> None:
    torch.manual_seed(1323)
    model = build_arm("afm_first1")
    batch = make_binding_batch(
        batch_size=1,
        generator=torch.Generator().manual_seed(1324),
        delay=64,
    )
    logits = query_logits(model, batch.tokens)
    loss = F.cross_entropy(logits.float(), batch.targets)
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    afm = model.blocks[0].afm
    assert afm is not None
    for p in (afm.key_proj.weight, afm.value_proj.weight, afm.out_proj.weight):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_full_evaluator_reports_keyblind_baseline_and_four_query_accuracy() -> None:
    model = build_arm("sequence_only")
    result = evaluate_arm(
        model, seed=1321, examples_per_distance=1,
        counterfactual_source_groups=1, eval_batch_size=1,
    )
    assert result["counterfactual_groups"] == 1
    assert result["keyblind_fixed_query_accuracy"] == 0.25
    assert 0.0 <= result["counterfactual_all_four_accuracy"] <= 1.0
    assert set(result["by_distance"]) == {str(v) for v in SCORED_DELAYS}
    for record in result["by_distance"].values():
        assert math.isfinite(float(record["nll"]))
