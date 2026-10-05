from __future__ import annotations

import torch

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v2_sequence.model import (
    CPWV2SequenceLM,
    sequence_parameter_count,
)
from tam_research.cpw_v2_sequence.protocol import SEQUENCE_PARAMETERS


def test_exact_parameter_count_and_physical_simplicity() -> None:
    model = CPWV2SequenceLM()
    assert sequence_parameter_count() == SEQUENCE_PARAMETERS
    for block in model.blocks:
        mixer = block.mixer
        assert mixer.sequence is not None
        assert mixer.world is None
        assert mixer.memory is None
    names = [name.lower() for name, _ in model.named_modules()]
    assert not any("attention" in name for name in names)
    assert not any("router" in name for name in names)


def test_finite_forward_backward_and_mixer_gradients() -> None:
    torch.manual_seed(1254)
    cfg = CPWV1Config(n_layers=2, max_seq_len=32)
    model = CPWV2SequenceLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    logits.float().square().mean().backward()
    grads = [
        p.grad for name, p in model.named_parameters()
        if ".mixer.sequence." in name
    ]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_strict_future_token_causality() -> None:
    torch.manual_seed(254)
    cfg = CPWV1Config(n_layers=3, max_seq_len=32)
    model = CPWV2SequenceLM(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 24))
    b = a.clone()
    b[:, 16:] = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    torch.testing.assert_close(la[:, :16], lb[:, :16], rtol=0, atol=2e-5)
