"""Causality and integrity tests for original SI surprise router v2 (CPU-only)."""
import torch
from torch.nn import functional as F
from scripts.si_original_surprise_router_v2 import (
    WorldModel, causal_surprise_mask, world, loss_fn
)


def test_no_future_observation_leakage():
    torch.manual_seed(96300)
    obs, action, _ = world(4, 24, 96300, mode="switch")
    modified = obs.clone()
    modified[:, 14:] += 5
    mask1, flags1 = causal_surprise_mask(obs, action)
    mask2, flags2 = causal_surprise_mask(modified, action)
    assert torch.equal(mask1[:, :, :14, :14], mask2[:, :, :14, :14])
    assert torch.equal(flags1[:, :14], flags2[:, :14])
    model = WorldModel().eval()
    with torch.no_grad():
        pred1 = model(obs, action, "surprise")["prediction"][:, :14]
        pred2 = model(modified, action, "surprise")["prediction"][:, :14]
    assert torch.allclose(pred1, pred2, atol=1e-6)


def test_finite_backward_through_predictor():
    torch.manual_seed(96300)
    obs, action, future = world(3, 24, 96300, mode="mixed")
    model = WorldModel()
    pred = model(obs, action, "surprise")
    loss, mse = loss_fn(pred, future)
    loss.backward()
    assert torch.isfinite(mse)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.blocks[0]["attn"].qkv.weight.grad is not None


def test_same_parameter_count_all_attention_modes():
    model = WorldModel().eval()
    obs, action, _ = world(2, 24, 96300, "mixed")
    counts = set()
    for mode in ("no_history", "local", "global", "surprise"):
        counts.add(sum(p.numel() for p in model.parameters()))
        with torch.no_grad():
            out = model(obs, action, mode)
        assert out["prediction"].shape == (2, 24, 3)
        assert torch.isfinite(out["prediction"]).all()
    assert len(counts) == 1


def test_surprise_attention_uses_one_sdpa_per_layer(monkeypatch):
    original = F.scaled_dot_product_attention
    count = [0]
    def wrapped(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", wrapped)
    obs, action, _ = world(2, 24, 96300, mode="mixed")
    model = WorldModel()
    model(obs, action, "surprise")
    assert count[0] == len(model.blocks)


def test_worlds_deterministic_and_distinct():
    a = world(3, 24, 96300, "switch", "delayed")
    b = world(3, 24, 96300, "switch", "delayed")
    c = world(3, 24, 96300, "switch", "base")
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[2], c[2])
