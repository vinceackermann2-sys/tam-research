from __future__ import annotations

import math

import torch

from tam_research.cpw_v0.model import (
    CPWV0Config,
    CPWV0ResearchLM,
    cpwv0_parameter_count,
)
from tam_research.cpw_v0.protocol import (
    MAX_PARAMETER_MISMATCH_FRACTION,
    ROUTER_TOP_K,
    TRANSFORMER_PARAMETERS,
)


def test_parameter_match_is_within_frozen_tolerance() -> None:
    cpw = cpwv0_parameter_count()
    mismatch = abs(cpw - TRANSFORMER_PARAMETERS) / TRANSFORMER_PARAMETERS
    assert mismatch <= MAX_PARAMETER_MISMATCH_FRACTION
    assert cpw > TRANSFORMER_PARAMETERS


def test_forward_backward_auxiliary_and_router_contracts() -> None:
    torch.manual_seed(1245)
    model = CPWV0ResearchLM()
    tokens = torch.randint(0, 50_257, (1, 16))
    logits = model(tokens)
    assert logits.shape == (1, 16, 50_257)
    assert torch.isfinite(logits).all()

    aux = model.auxiliary_loss()
    assert torch.isfinite(aux)
    (logits.float().square().mean() + 0.05 * aux).backward()

    for name in (
        "blocks.0.mixer.world.candidate.weight",
        "blocks.0.mixer.sequence.down.weight",
        "blocks.0.mixer.memory.down.weight",
        "blocks.0.mixer.router.weight",
    ):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    stats = model.router_stats()
    assert stats is not None
    mean = stats["mean"]
    assert math.isclose(
        float(mean["active_fraction"]),
        ROUTER_TOP_K / 4,
        abs_tol=1e-6,
    )
    usage_sum = sum(
        float(mean[key])
        for key in (
            "attention_usage",
            "world_usage",
            "sequence_usage",
            "memory_usage",
        )
    )
    assert math.isclose(usage_sum, float(ROUTER_TOP_K), abs_tol=1e-5)
    assert float(mean["router_entropy"]) > 0.0


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(99)
    cfg = CPWV0Config(n_layers=2)
    model = CPWV0ResearchLM(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 24))
    b = a.clone()
    b[:, 16:] = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    torch.testing.assert_close(la[:, :16], lb[:, :16], rtol=0, atol=2e-5)


def test_no_persistent_drive_or_reward_module() -> None:
    names = [name.lower() for name, _ in CPWV0ResearchLM().named_modules()]
    forbidden = ("reward", "pleasure", "emotion", "hunger", "survival", "desire")
    assert not any(any(word in name for word in forbidden) for name in names)
