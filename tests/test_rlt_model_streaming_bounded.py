from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, parameter_count
from experiments.rlt.model_streaming_bounded import (
    BoundedChunkGatedRLT, BoundedStreamState,
    BoundedStreamingConfig, tiny_streaming_config,
)


def paired_models():
    cfg = tiny_streaming_config()
    torch.manual_seed(20261010)
    carry = BoundedChunkGatedRLT(cfg, carry_between_chunks=True)
    reset = BoundedChunkGatedRLT(cfg, carry_between_chunks=False)
    reset.load_state_dict(carry.state_dict(), strict=True)
    return carry, reset


def test_exact_parameter_match_and_shared_weights() -> None:
    carry, reset = paired_models()
    assert parameter_count(carry) == parameter_count(reset)
    assert carry.readout_parameter_count() == parameter_count(carry)
    assert set(carry.state_dict()) == set(reset.state_dict())
    for k, v in carry.state_dict().items():
        torch.testing.assert_close(v, reset.state_dict()[k], atol=0, rtol=0)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 32])
@pytest.mark.parametrize("keep_state", [True, False])
def test_chunked_stream_api_equals_full_fixed_framing(
    length: int, keep_state: bool,
) -> None:
    torch.manual_seed(172)
    cfg = tiny_streaming_config()
    model = BoundedChunkGatedRLT(cfg, carry_between_chunks=keep_state).eval()
    x = torch.randint(16, (2, length))
    with torch.no_grad():
        direct = model(x)
        s = model.init_stream(2)
        parts = []
        for start in range(0, length, cfg.chunk_size):
            logits, s = model.forward_chunk(x[:, start:start + cfg.chunk_size], s)
            parts.append(logits)
        streamed = torch.cat(parts, dim=1)
    assert s.total_tokens == length
    assert s.recurrent.shape == (2, 32)
    torch.testing.assert_close(direct, streamed, atol=0, rtol=0)


@pytest.mark.parametrize("carry_state", [True, False])
def test_future_token_modifications_do_not_change_past_logits(carry_state: bool) -> None:
    torch.manual_seed(173)
    cfg = tiny_streaming_config()
    model = BoundedChunkGatedRLT(cfg, carry_between_chunks=carry_state).eval()
    x = torch.randint(16, (2, 24))
    with torch.no_grad():
        base = model(x)
        for cut in (1, 5, 8, 9, 12, 16, 20):
            changed = x.clone()
            changed[:, cut:] = torch.randint(16, (2, x.size(1) - cut))
            other = model(changed)
            torch.testing.assert_close(
                base[:, :cut], other[:, :cut], atol=3e-5, rtol=3e-5
            )


def test_no_carry_model_has_no_access_to_previous_chunk() -> None:
    carry, reset = paired_models()
    carry.eval(); reset.eval()
    torch.manual_seed(174)
    x = torch.randint(16, (2, 16))
    y = x.clone()
    y[:, :8] = torch.randint(16, (2, 8))
    with torch.no_grad():
        reset_x = reset(x)
        reset_y = reset(y)
    torch.testing.assert_close(reset_x[:, 8:], reset_y[:, 8:], atol=0, rtol=0)

    # Strong model-level test: externally inject a distinct fixed-width state
    # at the same chunk boundary. Only carry architecture may observe it.
    state = carry.init_stream(2)
    altered = BoundedStreamState(state.recurrent + 3.0, state.total_tokens)
    with torch.no_grad():
        carry_a, _ = carry.forward_chunk(x[:, :8], state)
        carry_b, _ = carry.forward_chunk(x[:, :8], altered)
        reset_a, _ = reset.forward_chunk(x[:, :8], state)
        reset_b, _ = reset.forward_chunk(x[:, :8], altered)
    assert float((carry_a - carry_b).abs().max()) > 1e-5
    torch.testing.assert_close(reset_a, reset_b, atol=0, rtol=0)


def test_gradient_flows_from_next_chunk_to_earlier_state_only_with_carry() -> None:
    torch.manual_seed(175)
    x = torch.randint(16, (2, 16))
    for keep_state in (True, False):
        cfg = tiny_streaming_config()
        model = BoundedChunkGatedRLT(cfg, carry_between_chunks=keep_state)
        state = model.init_stream(2)
        _, state = model.forward_chunk(x[:, :8], state)
        state.recurrent.retain_grad()
        logits, _ = model.forward_chunk(x[:, 8:], state)
        loss = F.cross_entropy(logits[:, -1], torch.randint(16, (2,)))
        loss.backward()
        if keep_state:
            assert state.recurrent.grad is not None
            assert torch.isfinite(state.recurrent.grad).all()
            assert float(state.recurrent.grad.abs().sum()) > 0
        else:
            assert state.recurrent.grad is None


def test_fixed_size_state_and_attention_window_are_independent_of_history() -> None:
    cfg = tiny_streaming_config()
    model = BoundedChunkGatedRLT(cfg)
    for n in (8, 64, 128):
        x = torch.randint(16, (1, n))
        with torch.no_grad():
            s = model.init_stream(1)
            for i in range(0, n, 8):
                _, s = model.forward_chunk(x[:, i:i + 8], s)
                assert tuple(s.recurrent.shape) == (1, 32)
            assert s.total_tokens == n


def test_validate_chunk_shape_alignment_and_config_errors() -> None:
    with pytest.raises(ValueError):
        BoundedStreamingConfig(RLTConfig(max_seq_len=4, swa_window=4), chunk_size=8)
    with pytest.raises(ValueError):
        BoundedStreamingConfig(RLTConfig(max_seq_len=16, swa_window=4), chunk_size=8)
    carry, _ = paired_models()
    state = carry.init_stream(1)
    with pytest.raises(ValueError):
        carry.forward_chunk(torch.ones((1, 9), dtype=torch.long), state)
    with pytest.raises(ValueError):
        carry.forward_chunk(torch.ones((1, 8), dtype=torch.int32), state)
    with pytest.raises(ValueError):
        carry.forward_chunk(torch.ones((2, 8), dtype=torch.long), state)
    _, partial = carry.forward_chunk(torch.ones((1, 3), dtype=torch.long), state)
    with pytest.raises(ValueError):
        carry.forward_chunk(torch.ones((1, 8), dtype=torch.long), partial)
