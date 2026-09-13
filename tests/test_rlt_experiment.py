import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer


def _cfg(**overrides):
    values = {
        "vocab_size": 257,
        "d_model": 64,
        "n_heads": 4,
        "n_stages": 2,
        "max_seq_len": 32,
        "swa_window": 4,
    }
    values.update(overrides)
    return RLTConfig(**values)


def test_future_tokens_do_not_change_prefix_logits():
    torch.manual_seed(7)
    model = RecurrentLoopedTransformer(_cfg()).eval()
    a = torch.randint(0, 257, (2, 12))
    b = a.clone()
    b[:, 6:] = torch.randint(0, 257, (2, 6))
    with torch.no_grad():
        logits_a = model(a)[:, :6]
        logits_b = model(b)[:, :6]
    torch.testing.assert_close(logits_a, logits_b, atol=2e-6, rtol=2e-5)


def test_decoder_swa_cache_is_window_bounded():
    torch.manual_seed(8)
    model = RecurrentLoopedTransformer(_cfg(swa_window=3)).eval()
    tokens = torch.randint(0, 257, (2, 12))
    with torch.no_grad():
        model(tokens)
    assert model.last_cache_lengths
    assert all(length <= 3 for length in model.last_cache_lengths)


def test_late_loss_backprops_through_recurrent_start_state():
    torch.manual_seed(9)
    model = RecurrentLoopedTransformer(_cfg())
    tokens = torch.randint(0, 257, (2, 12))
    logits = model(tokens)
    loss = F.cross_entropy(logits[:, -2, :], tokens[:, -1])
    loss.backward()
    grad = model.start_state.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
