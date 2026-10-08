"""Independent adversarial audit of merged #1321 evaluator; ZERO GPU.

Only CPU fixtures, never scientific seeds or learned capability claims.
"""
from __future__ import annotations

import gc
import hashlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.cpw_binding_science import harness
from tam_research.cpw_binding_v2.task import QUERY_POSITION, explicit_key_lookup


class _AuditPredictor(nn.Module):
    def __init__(self, *, keyblind: bool = False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.keyblind = keyblind
        self.input_sha256 = hashlib.sha256()

    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        # This is an AUDIT ORACLE, never a trainable benchmark model.
        self.input_sha256.update(tokens.cpu().contiguous().numpy().tobytes())
        if self.keyblind:
            target = tokens[:, QUERY_POSITION - 32]
        else:
            target = explicit_key_lookup(tokens)
        logits = torch.full((tokens.size(0), 4_200), -10.0)
        logits.scatter_(1, target[:, None], 10.0)
        return logits


def _evaluate_fake(monkeypatch, *, seed: int, keyblind: bool = False):
    predictor = _AuditPredictor(keyblind=keyblind)
    with monkeypatch.context() as patch:
        patch.setattr(harness, "query_logits", lambda m, tokens: m.score(tokens))
        result = harness.evaluate_arm(
            predictor, seed=seed, examples_per_distance=8,
            counterfactual_source_groups=8, eval_batch_size=4,
        )
    return result, predictor.input_sha256.hexdigest()


def test_true_oracle_and_keyblind_controls_end_to_end(monkeypatch):
    oracle, oracle_sha = _evaluate_fake(monkeypatch, seed=1326)
    blind, blind_sha = _evaluate_fake(monkeypatch, seed=1326, keyblind=True)
    assert oracle_sha == blind_sha
    assert oracle["long_mean_accuracy"] == 1.0
    assert oracle["counterfactual_all_four_accuracy"] == 1.0
    assert oracle["counterfactual_individual_accuracy"] == 1.0
    assert oracle["keyblind_fixed_query_accuracy"] == 0.25
    assert all(oracle["by_distance"][str(d)]["accuracy"] == 1.0 for d in (32, 64, 128, 256))
    assert blind["by_distance"]["32"]["accuracy"] == 1.0
    assert all(blind["by_distance"][str(d)]["accuracy"] == 0.0 for d in (64, 128, 256))
    assert blind["long_mean_accuracy"] == 0.0
    assert blind["counterfactual_individual_accuracy"] == 0.25
    assert blind["counterfactual_all_four_accuracy"] == 0.0


def test_paired_evaluator_uses_identical_inputs_for_each_arm(monkeypatch):
    a, sha_a = _evaluate_fake(monkeypatch, seed=1326)
    b, sha_b = _evaluate_fake(monkeypatch, seed=1326)
    c, sha_c = _evaluate_fake(monkeypatch, seed=1327)
    assert sha_a == sha_b
    assert sha_a != sha_c
    assert a["by_distance"] == b["by_distance"]
    assert a["counterfactual_groups"] == b["counterfactual_groups"]


def test_non_cpu_gpu_request_rejected_before_other_actions(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("preauthority rejection must occur before side effects")
    monkeypatch.setattr(harness, "seed_all", _forbidden)
    monkeypatch.setattr(harness, "build_arm", _forbidden)
    kwargs = dict(
        arm="sequence_only", seed=1326, steps=1,
        micro_batch_size=1, grad_accum_steps=1,
        eval_examples_per_distance=1, counterfactual_source_groups=1,
    )
    for device in ("cuda", "cuda:0", "mps"):
        with pytest.raises(PermissionError, match="CPU-only"):
            harness.train_arm(**kwargs, device=device)


@pytest.mark.parametrize("arm", harness.ARMS)
def test_all_five_frozen_models_respect_strict_future_token_causality(arm):
    torch.manual_seed(1326)
    torch.set_num_threads(2)
    model = harness.build_arm(arm).eval()
    original = torch.randint(0, 4_000, (1, 16))
    changed = original.clone()
    changed[:, 9:] = (changed[:, 9:] + 19) % 4_000
    with torch.no_grad():
        a = model(original)
        b = model(changed)
        q = harness.query_logits(model, original)
    torch.testing.assert_close(a[:, :9], b[:, :9], atol=5e-5, rtol=0)
    torch.testing.assert_close(a[:, -1], q, atol=5e-5, rtol=0)
    del model
    gc.collect()


@pytest.mark.parametrize("arm", harness.ARMS)
def test_all_five_cpu_one_step_real_gradient_path(arm):
    torch.manual_seed(1327)
    torch.set_num_threads(2)
    model = harness.build_arm(arm).train()
    from tam_research.cpw_binding_v2.task import make_binding_batch
    data = make_binding_batch(
        batch_size=1, generator=torch.Generator().manual_seed(1328),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(
        harness.query_logits(model, data.tokens).float(), data.targets
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(bool(g.abs().sum() > 0) for g in grads)
    optimizer.step()
    del optimizer, model
    gc.collect()
