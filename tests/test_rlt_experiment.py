from dataclasses import replace

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig

from experiments.rlt.model import RecurrentLoopedLM, parameter_count


def _cfg(**overrides):
    cfg = ModelConfig(
        vocab_size=257,
        d_model=64,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        architecture="transformer",
    )
    return replace(cfg, **overrides)


def test_loop_count_does_not_change_parameter_count():
    assert parameter_count(RecurrentLoopedLM(_cfg(n_layers=2))) == parameter_count(
        RecurrentLoopedLM(_cfg(n_layers=11))
    )


def test_future_tokens_do_not_change_prefix_logits():
    torch.manual_seed(7)
    model = RecurrentLoopedLM(_cfg()).eval()
    a = torch.randint(0, 257, (2, 16))
    b = a.clone()
    b[:, 8:] = torch.randint(0, 257, (2, 8))
    with torch.no_grad():
        logits_a = model(a)[:, :8]
        logits_b = model(b)[:, :8]
    torch.testing.assert_close(logits_a, logits_b, atol=2e-6, rtol=2e-5)


def test_shared_block_receives_gradients():
    torch.manual_seed(8)
    model = RecurrentLoopedLM(_cfg())
    tokens = torch.randint(0, 257, (2, 16))
    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 257), tokens[:, 1:].reshape(-1)
    )
    loss.backward()
    grads = [p.grad for p in model.shared_block.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
