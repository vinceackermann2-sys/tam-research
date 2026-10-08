from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, parameter_count
from experiments.rlt.model_gated_scan import GatedScanLightStateRLT
from experiments.rlt.model_gated_scan_adaptive import AdaptiveResidualGatedScanRLT
from experiments.rlt.model_gated_scan_residual import ResidualGatedScanRLT


def tiny_cfg() -> RLTConfig:
    return RLTConfig(
        vocab_size=127, d_model=32, n_heads=4, n_stages=2,
        max_seq_len=32, ff_mult=2, swa_window=8,
    )


def full_cfg() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )


def test_exact_parameter_neutrality_15m() -> None:
    torch.manual_seed(151)
    cfg = full_cfg()
    counts = [
        parameter_count(GatedScanLightStateRLT(cfg)),
        parameter_count(ResidualGatedScanRLT(cfg)),
        parameter_count(AdaptiveResidualGatedScanRLT(cfg)),
    ]
    assert counts == [15_129_344] * 3


@pytest.mark.parametrize("length", [1, 4, 9, 16])
def test_zero_gate_projection_matches_static_residual_exactly(length: int) -> None:
    torch.manual_seed(152 + length)
    cfg = tiny_cfg()
    adaptive = AdaptiveResidualGatedScanRLT(cfg).eval()
    with torch.no_grad():
        adaptive.merge.weight[:, cfg.d_model:].zero_()
    static = ResidualGatedScanRLT(cfg).eval()
    static.load_state_dict(adaptive.state_dict(), strict=True)
    tokens = torch.randint(cfg.vocab_size, (2, length))
    with torch.no_grad():
        memory = adaptive.encode(tokens)
        actual = adaptive.gated_states(memory)
        expected = static.gated_states(memory)
        adaptive_logits = adaptive(tokens)
        static_logits = static(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(adaptive_logits, static_logits, atol=0.0, rtol=0.0)


def test_gate_gain_is_bounded_and_token_dependent() -> None:
    torch.manual_seed(159)
    model = AdaptiveResidualGatedScanRLT(tiny_cfg()).eval()
    tokens = torch.randint(127, (2, 11))
    with torch.no_grad():
        memory = model.encode(tokens)
        width = memory.shape[-1]
        logits = F.linear(memory, model.merge.weight[:, width:])
        gain = 2.0 * torch.sigmoid(logits.float())
    assert gain.shape == memory.shape
    assert bool((gain >= 0.0).all()) and bool((gain <= 2.0).all())
    assert float(gain.var()) > 0.0


def test_every_parameter_has_finite_gradient() -> None:
    torch.manual_seed(160)
    cfg = tiny_cfg()
    model = AdaptiveResidualGatedScanRLT(cfg)
    x = torch.randint(cfg.vocab_size, (2, 13))
    y = torch.randint(cfg.vocab_size, (2, 13))
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"nonfinite gradient for {name}"
    grad = model.merge.weight.grad
    assert torch.count_nonzero(grad[:, :cfg.d_model]) > 0
    assert torch.count_nonzero(grad[:, cfg.d_model:]) > 0
    assert model.start_state.grad.abs().sum() > 0


@pytest.mark.parametrize("prefix", [1, 7, 13])
def test_future_tokens_cannot_change_prefix_logits(prefix: int) -> None:
    torch.manual_seed(161 + prefix)
    model = AdaptiveResidualGatedScanRLT(tiny_cfg()).eval()
    a = torch.randint(127, (2, 17))
    b = a.clone()
    b[:, prefix:] = torch.randint(127, (2, 17 - prefix))
    with torch.no_grad():
        base = model(a)
        mutated = model(b)
        shortened = model(a[:, :prefix])
    torch.testing.assert_close(base[:, :prefix], mutated[:, :prefix], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(base[:, :prefix], shortened, atol=2e-5, rtol=2e-5)


def test_fullgraph_compile_eager_backend_cpu() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    torch.manual_seed(165)
    model = AdaptiveResidualGatedScanRLT(tiny_cfg())
    x = torch.randint(127, (2, 9))
    eager = model(x)
    compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
    actual = compiled(x)
    torch.testing.assert_close(actual, eager, atol=2e-5, rtol=2e-5)
