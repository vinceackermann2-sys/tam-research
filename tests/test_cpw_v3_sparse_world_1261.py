from __future__ import annotations

import torch

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v2_sequence.model import CPWV2SequenceLM
from tam_research.cpw_v3_sparse_world.model import (
    ARM_WORLD_LAYERS,
    SparseWorldCPWResearchLM,
    sparse_world_parameter_count,
)
from tam_research.cpw_v3_sparse_world.protocol import CPW_V3_PARAMETERS


def test_exact_parameter_count_is_constant_across_sparse_world_arms() -> None:
    for arm in ARM_WORLD_LAYERS:
        assert sparse_world_parameter_count(arm) == CPW_V3_PARAMETERS


def test_physical_world_layer_layouts_are_exact() -> None:
    for arm, expected in ARM_WORLD_LAYERS.items():
        model = SparseWorldCPWResearchLM(arm)
        actual = tuple(
            i for i, block in enumerate(model.blocks)
            if block.mixer.world is not None
        )
        assert actual == expected
        for i, block in enumerate(model.blocks):
            if i in expected:
                assert block.mixer.world is not None
                assert block.mixer.sequence is None
            else:
                assert block.mixer.world is None
                assert block.mixer.sequence is not None


def test_sequence_only_is_exactly_load_compatible_with_cpw_v2() -> None:
    torch.manual_seed(1261)
    cfg = CPWV1Config(n_layers=3, max_seq_len=32)
    prior = CPWV2SequenceLM(cfg)
    current = SparseWorldCPWResearchLM("sequence_only", cfg)
    current.load_state_dict(prior.state_dict(), strict=True)
    tokens = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        a = prior(tokens)
        b = current(tokens)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_forward_backward_and_all_retained_predictors_receive_gradients() -> None:
    torch.manual_seed(61)
    cfg = CPWV1Config(max_seq_len=32)
    for arm in ARM_WORLD_LAYERS:
        model = SparseWorldCPWResearchLM(arm, cfg)
        tokens = torch.randint(0, cfg.vocab_size, (1, 17))
        logits = model(tokens)
        loss = logits.float().square().mean()
        loss.backward()
        assert torch.isfinite(logits).all()
        for block in model.blocks:
            params = (
                block.mixer.world.parameters()
                if block.mixer.world is not None
                else block.mixer.sequence.parameters()
            )
            grads = [p.grad for p in params]
            assert grads
            assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(62)
    cfg = CPWV1Config(max_seq_len=32)
    for arm in ARM_WORLD_LAYERS:
        model = SparseWorldCPWResearchLM(arm, cfg).eval()
        a = torch.randint(0, cfg.vocab_size, (1, 24))
        b = a.clone()
        b[:, 17:] = torch.randint(0, cfg.vocab_size, (1, 7))
        with torch.no_grad():
            la = model(a)
            lb = model(b)
        torch.testing.assert_close(la[:, :17], lb[:, :17], rtol=0, atol=2e-5)


def test_sparse_world_breaks_sequence_only_hard_15_token_horizon() -> None:
    torch.manual_seed(63)
    cfg = CPWV1Config(max_seq_len=32)
    a = torch.randint(0, cfg.vocab_size, (1, 17))
    b = a.clone()
    b[:, 0] = (b[:, 0] + 1) % cfg.vocab_size

    sequence = SparseWorldCPWResearchLM("sequence_only", cfg).eval()
    with torch.no_grad():
        sa = sequence(a)[:, -1]
        sb = sequence(b)[:, -1]
    torch.testing.assert_close(sa, sb, rtol=0, atol=0)

    for arm in ("world_last1", "world_2", "world_3"):
        torch.manual_seed(63)
        model = SparseWorldCPWResearchLM(arm, cfg).eval()
        with torch.no_grad():
            wa = model(a)[:, -1]
            wb = model(b)[:, -1]
        assert float((wa - wb).abs().max()) > 0.0
