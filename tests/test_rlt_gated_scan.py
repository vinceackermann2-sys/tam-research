from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, parameter_count
from experiments.rlt.model_gated_scan import (
    GatedScanLightStateRLT,
    associative_affine_scan,
    sequential_affine_reference,
)
from experiments.rlt.model_light_state import LightStateRecurrentTransformer


@pytest.mark.parametrize("length", [1, 2, 3, 7, 16, 64])
def test_parallel_scan_matches_sequential_values_and_gradients(length: int) -> None:
    torch.manual_seed(1000 + length)
    a = (0.7 + 0.2 * torch.rand(2, length, 5, dtype=torch.float64)).requires_grad_()
    b = (torch.randn(2, length, 5, dtype=torch.float64) * 0.2).requires_grad_()
    initial = torch.randn(5, dtype=torch.float64, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    s_ref = initial.detach().clone().requires_grad_()

    got = associative_affine_scan(a, b, initial)
    expected = sequential_affine_reference(a_ref, b_ref, s_ref)
    torch.testing.assert_close(got, expected, atol=1e-11, rtol=1e-11)

    weights = torch.randn_like(got)
    got_grads = torch.autograd.grad((got * weights).sum(), (a, b, initial))
    ref_grads = torch.autograd.grad((expected * weights).sum(), (a_ref, b_ref, s_ref))
    for g, ref in zip(got_grads, ref_grads):
        torch.testing.assert_close(g, ref, atol=1e-10, rtol=1e-10)


def test_scan_batchwise_initial_state_and_prefix_causality() -> None:
    torch.manual_seed(11)
    a = torch.sigmoid(torch.randn(2, 13, 8))
    b = torch.randn(2, 13, 8) * 0.1
    initial = torch.randn(2, 8) * 0.1
    output = associative_affine_scan(a, b, initial)
    torch.testing.assert_close(
        output, sequential_affine_reference(a, b, initial), atol=2e-6, rtol=2e-6
    )
    altered_a = a.clone()
    altered_b = b.clone()
    altered_a[:, 8:] = 0.01
    altered_b[:, 8:] = 9.0
    altered = associative_affine_scan(altered_a, altered_b, initial)
    torch.testing.assert_close(output[:, :8], altered[:, :8], atol=1e-6, rtol=1e-6)


def test_15m_exact_param_count_and_all_merge_weights_trainable() -> None:
    torch.manual_seed(12)
    cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8,
        n_stages=2, max_seq_len=128, ff_mult=4, swa_window=32,
    )
    baseline = LightStateRecurrentTransformer(cfg)
    scan_model = GatedScanLightStateRLT(cfg)
    assert parameter_count(baseline) == 15_129_344
    assert parameter_count(scan_model) == parameter_count(baseline)

    small = RLTConfig(
        vocab_size=127, d_model=32, n_heads=4, n_stages=2,
        max_seq_len=32, ff_mult=2, swa_window=8,
    )
    scan_model = GatedScanLightStateRLT(small)
    x = torch.randint(0, small.vocab_size, (2, 8))
    y = torch.randint(0, small.vocab_size, (2, 8))
    logits = scan_model(x)
    assert logits.shape == (2, 8, small.vocab_size)
    loss = F.cross_entropy(logits.reshape(-1, small.vocab_size), y.reshape(-1))
    loss.backward()

    for name, parameter in scan_model.named_parameters():
        assert parameter.grad is not None, f"untrained parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}"
    width = small.d_model
    grad = scan_model.merge.weight.grad
    assert torch.count_nonzero(grad[:, :width]) > 0
    assert torch.count_nonzero(grad[:, width:]) > 0
    assert scan_model.start_state.grad.abs().sum() > 0


def test_future_tokens_do_not_change_past_logits() -> None:
    torch.manual_seed(13)
    cfg = RLTConfig(
        vocab_size=113, d_model=32, n_heads=4, n_stages=2,
        max_seq_len=32, ff_mult=2, swa_window=8,
    )
    model = GatedScanLightStateRLT(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (2, 12))
    b = a.clone()
    b[:, 6:] = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        full_a = model(a)
        full_b = model(b)
        shorter = model(a[:, :6])
    torch.testing.assert_close(full_a[:, :6], full_b[:, :6], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(full_a[:, :6], shorter, atol=2e-5, rtol=2e-5)


def test_fullgraph_torch_compile_eager_backend() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not supported")
    torch.manual_seed(14)
    cfg = RLTConfig(
        vocab_size=79, d_model=32, n_heads=4, n_stages=1,
        max_seq_len=16, ff_mult=2, swa_window=8,
    )
    model = GatedScanLightStateRLT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 7))
    y = model(x)
    compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
    result = compiled(x)
    torch.testing.assert_close(y, result, atol=1e-5, rtol=1e-5)
    assert math.isfinite(result.float().mean().item())
