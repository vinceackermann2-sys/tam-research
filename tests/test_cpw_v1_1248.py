from __future__ import annotations

import math

import torch

from tam_research.cpw_v1.model import (
    CPWV1Config,
    CPWV1ResearchLM,
    cpwv1_parameter_count,
)
from tam_research.cpw_v1.protocol import (
    EXPECTED_CPW_V1_PARAMETERS,
    TRANSFORMER_PARAMETERS,
)


def test_exact_parameter_count_and_reduction() -> None:
    count = cpwv1_parameter_count()
    assert count == EXPECTED_CPW_V1_PARAMETERS
    assert count < TRANSFORMER_PARAMETERS
    reduction = 1.0 - count / TRANSFORMER_PARAMETERS
    assert reduction > 0.07


def test_no_attention_router_or_auxiliary_objective() -> None:
    model = CPWV1ResearchLM()
    names = [name.lower() for name, _ in model.named_modules()]
    assert not any("attention" in name for name in names)
    assert not any("router" in name for name in names)
    assert not hasattr(model, "auxiliary_loss")


def test_forward_backward_and_predictor_gradients() -> None:
    torch.manual_seed(1248)
    model = CPWV1ResearchLM(CPWV1Config(n_layers=2))
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 24))
    logits = model(tokens)
    assert logits.shape == (2, 24, model.cfg.vocab_size)
    assert torch.isfinite(logits).all()
    logits.float().square().mean().backward()

    params = dict(model.named_parameters())
    for name in (
        "blocks.0.mixer.world.candidate.weight",
        "blocks.0.mixer.sequence.down.weight",
        "blocks.0.mixer.memory.down.weight",
    ):
        grad = params[name].grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0

    stats = model.router_stats()
    assert stats is not None
    mean = stats["mean"]
    assert math.isclose(float(mean["active_predictors"]), 3.0)
    assert float(mean["attention_present"]) == 0.0
    assert float(mean["router_present"]) == 0.0
    assert float(mean["world_state_norm"]) >= 0.0


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(1248001)
    model = CPWV1ResearchLM(CPWV1Config(n_layers=2)).eval()
    a = torch.randint(0, model.cfg.vocab_size, (1, 32))
    b = a.clone()
    b[:, 20:] = torch.randint(0, model.cfg.vocab_size, (1, 12))
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    torch.testing.assert_close(la[:, :20], lb[:, :20], rtol=0, atol=2e-5)
