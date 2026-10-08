from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, parameter_count
from experiments.rlt.model_gated_scan import GatedScanLightStateRLT
from experiments.rlt.model_gated_scan_residual import ResidualGatedScanRLT
from experiments.rlt.model_light_state import LightStateRecurrentTransformer


def small_cfg() -> RLTConfig:
    return RLTConfig(
        vocab_size=127,
        d_model=32,
        n_heads=4,
        n_stages=2,
        max_seq_len=32,
        ff_mult=2,
        swa_window=8,
    )


def large_cfg() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257,
        d_model=256,
        n_heads=8,
        n_stages=2,
        max_seq_len=128,
        ff_mult=4,
        swa_window=32,
    )


def test_parameter_neutrality_15m() -> None:
    torch.manual_seed(57)
    cfg = large_cfg()
    assert parameter_count(ResidualGatedScanRLT(cfg)) == 15_129_344
    assert parameter_count(GatedScanLightStateRLT(cfg)) == 15_129_344
    assert parameter_count(LightStateRecurrentTransformer(cfg)) == 15_129_344


@pytest.mark.parametrize("length", [1, 3, 8, 13])
def test_residual_state_exactly_equals_encoder_plus_scan(length: int) -> None:
    torch.manual_seed(58 + length)
    base = GatedScanLightStateRLT(small_cfg()).eval()
    residual = ResidualGatedScanRLT(small_cfg()).eval()
    residual.load_state_dict(base.state_dict(), strict=True)
    tokens = torch.randint(0, small_cfg().vocab_size, (2, length))
    with torch.no_grad():
        memory = base.encode(tokens)
        baseline_state = base.gated_states(memory)
        residual_state = residual.gated_states(memory)
    torch.testing.assert_close(
        residual_state, memory + baseline_state, atol=0, rtol=0
    )


def test_all_weights_and_both_gate_halves_receive_gradients() -> None:
    torch.manual_seed(71)
    model = ResidualGatedScanRLT(small_cfg())
    x = torch.randint(0, small_cfg().vocab_size, (2, 9))
    y = torch.randint(0, small_cfg().vocab_size, (2, 9))
    logits = model(x)
    assert logits.shape == (2, 9, small_cfg().vocab_size)
    loss = F.cross_entropy(logits.reshape(-1, small_cfg().vocab_size), y.reshape(-1))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"parameter did not train: {name}"
        assert bool(torch.isfinite(param.grad).all()), f"nonfinite gradient: {name}"
    grad = model.merge.weight.grad
    width = small_cfg().d_model
    assert torch.count_nonzero(grad[:, :width]) > 0
    assert torch.count_nonzero(grad[:, width:]) > 0
    assert model.start_state.grad.abs().sum() > 0


def test_causal_prefix_independence_and_sequence_truncation() -> None:
    torch.manual_seed(72)
    model = ResidualGatedScanRLT(small_cfg()).eval()
    a = torch.randint(0, small_cfg().vocab_size, (2, 15))
    b = a.clone()
    b[:, 7:] = torch.randint(0, small_cfg().vocab_size, (2, 8))
    with torch.no_grad():
        logits_a = model(a)
        logits_b = model(b)
        truncated = model(a[:, :7])
    torch.testing.assert_close(logits_a[:, :7], logits_b[:, :7], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(logits_a[:, :7], truncated, atol=2e-5, rtol=2e-5)


def test_fullgraph_compile_cpu_eager_backend() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    torch.manual_seed(73)
    model = ResidualGatedScanRLT(small_cfg())
    x = torch.randint(0, small_cfg().vocab_size, (2, 7))
    eager = model(x)
    compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
    actual = compiled(x)
    torch.testing.assert_close(actual, eager, atol=2e-5, rtol=2e-5)
