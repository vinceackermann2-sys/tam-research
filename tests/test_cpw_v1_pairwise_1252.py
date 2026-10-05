from __future__ import annotations

import torch

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v1_pairwise.model import (
    PAIRWISE_ARMS,
    PairwiseCPWResearchLM,
    pairwise_parameter_count,
)
from tam_research.cpw_v1_pairwise.protocol import EXPECTED_PARAMETERS


def test_pairwise_parameter_counts_and_physical_removal() -> None:
    for arm in sorted(PAIRWISE_ARMS):
        assert pairwise_parameter_count(arm) == EXPECTED_PARAMETERS[arm]
        model = PairwiseCPWResearchLM(arm)
        first = model.blocks[0].mixer
        assert (first.world is not None) == (arm != "sequence_memory")
        assert (first.sequence is not None) == (arm != "world_memory")
        assert (first.memory is not None) == (arm != "world_sequence")


def test_pairwise_forward_backward_and_retained_gradients() -> None:
    for arm in sorted(PAIRWISE_ARMS):
        torch.manual_seed(1252)
        cfg = CPWV1Config(n_layers=2, max_seq_len=32)
        model = PairwiseCPWResearchLM(arm, cfg)
        tokens = torch.randint(0, cfg.vocab_size, (1, 16))
        logits = model(tokens)
        assert logits.shape == (1, 16, cfg.vocab_size)
        assert torch.isfinite(logits).all()
        logits.float().square().mean().backward()
        grads = {
            name: p.grad
            for name, p in model.named_parameters()
            if ".mixer." in name
        }
        assert grads
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values())


def test_pairwise_future_token_causality() -> None:
    for arm in sorted(PAIRWISE_ARMS):
        torch.manual_seed(77)
        cfg = CPWV1Config(n_layers=2, max_seq_len=32)
        model = PairwiseCPWResearchLM(arm, cfg).eval()
        a = torch.randint(0, cfg.vocab_size, (1, 24))
        b = a.clone()
        b[:, 16:] = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            la = model(a)
            lb = model(b)
        torch.testing.assert_close(la[:, :16], lb[:, :16], rtol=0, atol=2e-5)
