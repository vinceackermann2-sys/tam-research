from __future__ import annotations

import math

import pytest
import torch

from experiments.rlt.cpu_lastwrite_probe_ar import make_batch, make_models
from experiments.rlt.cpu_recurrence_ablation_au import (
    ABLATIONS, CANDIDATE_NAMES, apply_ablation, gate_statistics,
    probe_accuracy,
)
from experiments.rlt.model_gated_scan import GatedScanLightStateRLT


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_every_ablation_restores_exact_forward(name: str) -> None:
    torch.manual_seed(20301064)
    model = make_models()[name].eval()
    x, _, _ = make_batch(2, (1, 16, 63), 20291064)
    with torch.no_grad():
        reference = model(x)
        for ablation in ABLATIONS:
            with apply_ablation(model, ablation):
                logits = model(x)
                assert logits.shape == reference.shape
                assert bool(torch.isfinite(logits).all())
            torch.testing.assert_close(model(x), reference, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_encoder_and_recurrent_ablation_semantics(name: str) -> None:
    torch.manual_seed(20301064)
    model = make_models()[name].eval()
    x, _, _ = make_batch(2, (4, 40), 20291065)
    with torch.no_grad():
        memory = model.encode(x)
        scan = GatedScanLightStateRLT.gated_states(model, memory)
        with apply_ablation(model, "encoder_only"):
            torch.testing.assert_close(model.gated_states(memory), memory, atol=0.0, rtol=0.0)
        with apply_ablation(model, "recurrent_only"):
            torch.testing.assert_close(model.gated_states(memory), scan, atol=0.0, rtol=0.0)
        with apply_ablation(model, "static_gain"):
            torch.testing.assert_close(model.gated_states(memory), memory + scan, atol=0.0, rtol=0.0)


def test_adaptive_static_gain_matches_residual_same_weights() -> None:
    torch.manual_seed(20301064)
    models = make_models()
    adaptive = models["adaptive_scan"].eval()
    residual = models["residual_scan"].eval()
    residual.load_state_dict(adaptive.state_dict(), strict=True)
    x, _, _ = make_batch(2, (1, 40), 20291066)
    with torch.no_grad():
        with apply_ablation(adaptive, "static_gain"):
            torch.testing.assert_close(adaptive(x), residual(x), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_untrained_gate_stats_are_finite_and_bounded(name: str) -> None:
    torch.manual_seed(20301064)
    model = make_models()[name].eval()
    stats = gate_statistics(model)
    assert all(math.isfinite(value) for value in stats.values())
    assert 0 <= stats["gain_min"] <= stats["gain_max"] <= 2
    assert 0 <= stats["gain_near_zero_frac"] <= 1
    assert 0 <= stats["gain_near_two_frac"] <= 1
    assert 0 <= stats["retention_mean"] <= 1


def test_unknown_ablation_rejected_before_model_change() -> None:
    model = make_models()["residual_scan"]
    original = model.gated_states
    with pytest.raises(ValueError, match="unknown ablation"):
        with apply_ablation(model, "not-valid"):
            pass
    assert model.gated_states == original
