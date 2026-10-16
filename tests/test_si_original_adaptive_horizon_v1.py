"""Tests for isolated SI adaptive horizon (CPU only)."""
import torch

from scripts.si_original_adaptive_horizon_v1 import (
    Experiment, HorizonWorldModel, make_batch, objective,
)


def test_causal_prefix_invariance():
    torch.manual_seed(95001)
    model = HorizonWorldModel(Experiment()).eval()
    observations, actions, _ = make_batch(3, 24, 95001, "mixed")
    modified = observations.clone()
    modified[:, 12:] += 8.0
    with torch.no_grad():
        original = model(observations, actions, "adaptive")["prediction"][:, :12]
        altered = model(modified, actions, "adaptive")["prediction"][:, :12]
    assert torch.allclose(original, altered, atol=1e-6)


def test_gate_gradient_and_numerical_safety():
    torch.manual_seed(95001)
    config = Experiment()
    model = HorizonWorldModel(config)
    observations, actions, targets = make_batch(3, 24, 95001, "mixed")
    output = model(observations, actions, "adaptive")
    loss, _ = objective(output, targets, "adaptive", config)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    gate_grad = model.blocks[0].attn.gate[0].weight.grad
    assert gate_grad is not None and gate_grad.abs().sum() > 0


def test_equal_instantiated_parameter_budget_and_fixed_gate():
    config = Experiment()
    model = HorizonWorldModel(config).eval()
    observations, actions, _ = make_batch(2, 24, 95003, "stable")
    counts = set()
    with torch.no_grad():
        for mode in ("local", "global", "adaptive"):
            counts.add(sum(p.numel() for p in model.parameters()))
            output = model(observations, actions, mode)
            assert output["prediction"].shape == (2, 24, 3)
            assert torch.isfinite(output["prediction"]).all()
    assert len(counts) == 1


def test_world_stream_reproducibility():
    x = make_batch(2, 24, 95004, "switch")
    y = make_batch(2, 24, 95004, "switch")
    assert all(torch.equal(a, b) for a, b in zip(x, y))
