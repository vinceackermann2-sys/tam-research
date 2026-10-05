from __future__ import annotations

import torch

from tam_research.cpw_v0.model import CPWV0Config, CPWV0ResearchLM
from tam_research.cpw_v0_mechanism.model import CPWV0AblationLM
from tam_research.cpw_v0_mechanism.protocol import ARM_ORDER, CPW_PARAMETERS


CPW_ARMS = ARM_ORDER[1:]


def _grads_for(prefix: str, model: torch.nn.Module) -> list[torch.Tensor | None]:
    return [
        p.grad
        for name, p in model.named_parameters()
        if name.startswith(prefix)
    ]


def _all_zero_or_none(grads: list[torch.Tensor | None]) -> bool:
    return all(g is None or torch.count_nonzero(g).item() == 0 for g in grads)


def test_all_cpw_arms_have_identical_parameter_names_shapes_and_count() -> None:
    torch.manual_seed(1247)
    reference = CPWV0AblationLM("full")
    ref = [(n, tuple(p.shape)) for n, p in reference.named_parameters()]
    assert sum(p.numel() for p in reference.parameters()) == CPW_PARAMETERS

    for arm in CPW_ARMS:
        torch.manual_seed(1247)
        model = CPWV0AblationLM(arm)
        assert [(n, tuple(p.shape)) for n, p in model.named_parameters()] == ref
        assert sum(p.numel() for p in model.parameters()) == CPW_PARAMETERS


def test_full_arm_is_equivalent_to_frozen_cpw_v0() -> None:
    torch.manual_seed(99)
    cfg = CPWV0Config(n_layers=2)
    base = CPWV0ResearchLM(cfg).eval()
    panel = CPWV0AblationLM("full", cfg).eval()
    panel.load_state_dict(base.state_dict(), strict=True)
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a = base(tokens)
        aux_a = base.auxiliary_loss()
        b = panel(tokens)
        aux_b = panel.auxiliary_loss()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(aux_a, aux_b, rtol=0, atol=0)


def test_every_arm_is_causal_and_finite() -> None:
    for arm in CPW_ARMS:
        torch.manual_seed(7)
        cfg = CPWV0Config(n_layers=2)
        model = CPWV0AblationLM(arm, cfg).eval()
        a = torch.randint(0, cfg.vocab_size, (1, 24))
        b = a.clone()
        b[:, 16:] = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            la = model(a)
            lb = model(b)
        assert torch.isfinite(la).all()
        torch.testing.assert_close(la[:, :16], lb[:, :16], rtol=0, atol=2e-5)


def test_disabled_branches_and_uniform_router_have_no_task_gradient() -> None:
    cases = {
        "no_memory": "blocks.0.mixer.memory.",
        "no_sequence": "blocks.0.mixer.sequence.",
        "no_world": "blocks.0.mixer.world.",
        "no_attention": "blocks.0.mixer.attention.",
        "uniform_all": "blocks.0.mixer.router.",
    }
    for arm, prefix in cases.items():
        torch.manual_seed(11)
        cfg = CPWV0Config(n_layers=1)
        model = CPWV0AblationLM(arm, cfg)
        tokens = torch.randint(0, cfg.vocab_size, (1, 16))
        logits = model(tokens)
        loss = logits.float().square().mean() + 0.05 * model.auxiliary_loss()
        loss.backward()
        assert _all_zero_or_none(_grads_for(prefix, model)), (arm, prefix)


def test_no_aux_is_exactly_zero() -> None:
    torch.manual_seed(22)
    cfg = CPWV0Config(n_layers=2)
    model = CPWV0AblationLM("no_aux", cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    model(tokens)
    assert float(model.auxiliary_loss()) == 0.0


def test_uniform_all_reports_all_four_branches_active() -> None:
    torch.manual_seed(33)
    cfg = CPWV0Config(n_layers=2)
    model = CPWV0AblationLM("uniform_all", cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    model(tokens)
    stats = model.router_stats()
    assert stats is not None
    mean = stats["mean"]
    assert float(mean["active_fraction"]) == 1.0
    for key in (
        "attention_usage",
        "world_usage",
        "sequence_usage",
        "memory_usage",
    ):
        assert float(mean[key]) == 1.0
