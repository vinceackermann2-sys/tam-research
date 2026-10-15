"""AX CPU-only validation of independently coded bounded-KV Transformer."""
from __future__ import annotations

import pytest
import torch

from experiments.rlt.model_sliding_window_ax import (
    SlidingWindowConfig, SlidingWindowTransformer, tiny_sliding_window_config,
)


def build():
    torch.manual_seed(202610109)
    return SlidingWindowTransformer(tiny_sliding_window_config())


def test_parameter_budget_and_receptive_field():
    model = build()
    assert sum(p.numel() for p in model.parameters()) == 27680
    assert model.receptive_distance == 14
    assert model.cfg.window == 8 and len(model.blocks) == 2
    assert model.out.weight is model.tok.weight


@pytest.mark.parametrize("partitions", [[1] * 17, [8, 8, 1], [3, 5, 9], [17]])
def test_step_cache_matches_independently_parallel_masked_attention(partitions):
    model = build().eval()
    x = torch.randint(16, (2, 17), generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        expected = model(x)
        state, outputs, cursor = model.init_stream(2), [], 0
        for length in partitions:
            logits, state = model.forward_chunk(x[:, cursor:cursor + length], state)
            outputs.append(logits)
            cursor += length
            for k, v in state.caches:
                assert 1 <= k.size(2) <= 8
                assert k.shape == v.shape == (2, 4, min(cursor, 8), 8)
        assert cursor == 17 and state.total_tokens == 17
        torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=2e-6, rtol=1e-6)


def test_causal_no_future_leakage():
    model = build().eval()
    x = torch.randint(16, (2, 32), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        baseline = model(x)
        for cut in (1, 7, 8, 15, 16, 30):
            modified = x.clone()
            modified[:, cut:] = (modified[:, cut:] + 5) % 16
            torch.testing.assert_close(model(modified)[:, :cut], baseline[:, :cut], atol=0, rtol=0)


def test_exact_total_receptive_field_for_two_layers_not_just_eight_tokens():
    model = build().eval()
    x = torch.randint(16, (2, 30), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = model(x)[:, -1]
        for distance in (15, 16, 20, 29):
            altered = x.clone()
            altered[:, -1 - distance] = (altered[:, -1 - distance] + 1) % 16
            torch.testing.assert_close(model(altered)[:, -1], expected, atol=0, rtol=0)


def test_two_layer_cache_can_carry_older_than_final_window_information():
    cfg = SlidingWindowConfig(vocab_size=16, width=8, heads=1, layers=2, window=8, ff_inner=16)
    model = SlidingWindowTransformer(cfg).eval()
    with torch.no_grad():
        model.pos.weight.zero_()
        model.tok.weight.zero_()
        model.tok.weight[1, 0] = 1.0
        model.tok.weight[2, 1] = 2.0
        for block in model.blocks:
            block.attn.qkv.weight.zero_()
            block.attn.qkv.weight[2*cfg.width:, :] = torch.eye(cfg.width)
            block.attn.proj.weight.copy_(torch.eye(cfg.width))
            for layer in block.ff:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_()
        x = torch.zeros((1, 15), dtype=torch.long)
        y = x.clone()
        x[0, 0] = 1
        y[0, 0] = 2
        last_x, last_y = model(x)[:, -1], model(y)[:, -1]
        assert (last_x - last_y).abs().max().item() > 1e-6
        z = torch.cat((x, torch.zeros((1, 1), dtype=torch.long)), dim=1)
        w = torch.cat((y, torch.zeros((1, 1), dtype=torch.long)), dim=1)
        torch.testing.assert_close(model(z)[:, -1], model(w)[:, -1], atol=0, rtol=0)


def test_long_distance_input_gradients_are_causally_zero():
    model = build()
    emb_out = []
    def save_embedding(_module, _inputs, output):
        output.retain_grad()
        emb_out.append(output)
    hook = model.tok.register_forward_hook(save_embedding)
    x = torch.randint(16, (2, 30), generator=torch.Generator().manual_seed(14))
    try:
        model(x)[:, -1, :].square().sum().backward()
    finally:
        hook.remove()
    grad = emb_out[0].grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[:, :-15]) == 0
    assert torch.count_nonzero(grad[:, -15:]) > 0
    assert any(p.grad is not None for p in model.parameters())


def test_input_validation():
    with pytest.raises(ValueError):
        SlidingWindowConfig(width=30, heads=4)
    with pytest.raises(ValueError):
        SlidingWindowConfig(window=0)
    model = build()
    with pytest.raises(ValueError):
        model(torch.empty(1, 0, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.ones(1, 8, dtype=torch.int))
    with pytest.raises(ValueError):
        model.init_stream(0)
