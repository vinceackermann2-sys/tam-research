from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.pgw_v1.model import PGWV1Config, PGWV1ResearchLM
from tam_research.pgw_v2.model import PGWV2Config, PGWV2ResearchLM
from tam_research.pgw_v2_mechanism.model import (
    PGWControlConfig,
    PGWControlResearchLM,
    RoutingControlWorkspace,
    control_parameter_count,
)
from tam_research.pgw_v2_mechanism.protocol import (
    ARMS,
    EXPECTED_PARAMETERS,
    FIXED_RANDOM_POSITIONS,
    RECENCY_POSITIONS,
    REPLICATION_SEEDS,
    ROUTING_CONTROL_SEED,
    SMOKE_SEED,
)


def _control_small(mode: str) -> PGWControlConfig:
    return PGWControlConfig(
        vocab_size=127,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=128,
        local_attn_inner=24,
        chunk_size=64,
        workspace_layers=(0, 2),
        workspace_dim=16,
        workspace_slots=2,
        workspace_heads=4,
        predictor_rank=4,
        workspace_ff_rank=5,
        selected_events=4,
        control_mode=mode,
    )


def _signature(model: torch.nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(p.shape)) for name, p in model.named_parameters()]


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_panel_identity_and_frozen_routing_tables() -> None:
    assert ARMS == (
        "transformer",
        "mean_read_predictive",
        "token_read_predictive",
        "token_read_fixed_random",
        "token_read_recency",
        "no_workspace",
    )
    assert SMOKE_SEED == 1_180_001
    assert REPLICATION_SEEDS == (1_180_101, 1_180_102, 1_180_103)
    assert ROUTING_CONTROL_SEED == 1_180_999
    assert FIXED_RANDOM_POSITIONS == {
        4: (29, 45, 55, 60),
        9: (9, 40, 62, 63),
        14: (23, 26, 30, 56),
    }
    assert RECENCY_POSITIONS == (60, 61, 62, 63)


def test_all_full_size_arms_have_exact_parameter_count() -> None:
    transformer = ResearchLM(
        ModelConfig(
            architecture="transformer",
            d_model=256,
            n_layers=15,
            n_heads=8,
            max_seq_len=1024,
        )
    )
    assert parameter_count(transformer) == EXPECTED_PARAMETERS
    assert parameter_count(PGWV1ResearchLM(PGWV1Config())) == EXPECTED_PARAMETERS
    assert parameter_count(PGWV2ResearchLM(PGWV2Config())) == EXPECTED_PARAMETERS
    for mode in ("fixed_random", "recency", "no_workspace"):
        assert control_parameter_count(mode) == EXPECTED_PARAMETERS


def test_pgw_parameter_names_shapes_and_initial_values_are_identical() -> None:
    builders = [
        lambda: PGWV1ResearchLM(PGWV1Config()),
        lambda: PGWV2ResearchLM(PGWV2Config()),
        lambda: PGWControlResearchLM(PGWControlConfig(control_mode="fixed_random")),
        lambda: PGWControlResearchLM(PGWControlConfig(control_mode="recency")),
        lambda: PGWControlResearchLM(PGWControlConfig(control_mode="no_workspace")),
    ]
    models = []
    for build in builders:
        torch.manual_seed(1181)
        models.append(build())

    reference_signature = _signature(models[0])
    reference_values = _snapshot(models[0])
    for model in models[1:]:
        assert _signature(model) == reference_signature
        values = _snapshot(model)
        assert values.keys() == reference_values.keys()
        for name in reference_values:
            assert torch.equal(values[name], reference_values[name]), name


def test_fixed_random_positions_are_exact_and_content_independent() -> None:
    cfg = PGWControlConfig(control_mode="fixed_random")
    for layer, expected in FIXED_RANDOM_POSITIONS.items():
        module = RoutingControlWorkspace(
            cfg,
            layer_index=layer,
            control_mode="fixed_random",
        )
        indices = module._control_indices(batch=3, device=torch.device("cpu"))
        expected_tensor = torch.tensor(expected, dtype=torch.long)
        assert torch.equal(indices[0], expected_tensor)
        assert torch.equal(indices[1], expected_tensor)
        assert torch.equal(indices[2], expected_tensor)


def test_recency_positions_are_exact() -> None:
    cfg = PGWControlConfig(control_mode="recency")
    module = RoutingControlWorkspace(
        cfg,
        layer_index=4,
        control_mode="recency",
    )
    indices = module._control_indices(batch=2, device=torch.device("cpu"))
    expected = torch.tensor(RECENCY_POSITIONS, dtype=torch.long)
    assert torch.equal(indices[0], expected)
    assert torch.equal(indices[1], expected)


def test_no_workspace_has_zero_routing_and_no_workspace_state_influence() -> None:
    torch.manual_seed(1182)
    cfg = _control_small("no_workspace")
    model = PGWControlResearchLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 128))

    with torch.no_grad():
        before = model(tokens)
        for block in model.blocks:
            ws = block.mixer.workspace
            if ws is not None:
                ws.workspace_init.add_(100.0)
                ws.workspace_q.weight.add_(100.0)
                ws.workspace_k.weight.add_(100.0)
                ws.workspace_v.weight.add_(100.0)
                ws.workspace_out.weight.add_(100.0)
                ws.event_in.weight.add_(100.0)
                ws.broadcast.weight.add_(100.0)
                ws.workspace_ff_down.weight.add_(100.0)
                ws.workspace_ff_up.weight.add_(100.0)
        after = model(tokens)

    assert torch.equal(before, after)
    stats = model.router_stats()
    assert stats is not None
    assert float(stats["mean"]["selected_fraction"]) == 0.0


def test_no_workspace_backward_uses_predictor_but_not_workspace_path() -> None:
    torch.manual_seed(1183)
    cfg = _control_small("no_workspace")
    model = PGWControlResearchLM(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 128))
    labels = torch.randint(0, cfg.vocab_size, (2, 128))
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    predictor_grad = 0.0
    for block in model.blocks:
        ws = block.mixer.workspace
        if ws is None:
            continue
        assert ws.predict_down.weight.grad is not None
        assert ws.predict_up.weight.grad is not None
        predictor_grad += float(ws.predict_down.weight.grad.abs().sum())
        predictor_grad += float(ws.predict_up.weight.grad.abs().sum())
        for parameter in (
            ws.workspace_init,
            ws.event_in.weight,
            ws.workspace_q.weight,
            ws.workspace_k.weight,
            ws.workspace_v.weight,
            ws.workspace_out.weight,
            ws.broadcast.weight,
            ws.workspace_ff_down.weight,
            ws.workspace_ff_up.weight,
        ):
            assert parameter.grad is None
    assert predictor_grad > 0.0


def test_control_causality_for_all_custom_modes() -> None:
    for offset, mode in enumerate(("fixed_random", "recency", "no_workspace")):
        torch.manual_seed(1184 + offset)
        cfg = _control_small(mode)
        # Small model uses workspace layers 0/2, while fixed-random tables are
        # frozen for full layers 4/9/14. Exercise fixed-random causality with
        # a one-block full-index workspace directly below.
        if mode == "fixed_random":
            module = RoutingControlWorkspace(
                PGWControlConfig(control_mode="fixed_random"),
                layer_index=4,
                control_mode="fixed_random",
            ).eval()
            x = torch.randn(1, 128, 256)
            local = torch.randn_like(x)
            changed_x = x.clone()
            changed_local = local.clone()
            changed_x[:, 100, :] += 9.0
            changed_local[:, 100, :] -= 7.0
            with torch.no_grad():
                left = module(x, local)
                right = module(changed_x, changed_local)
            assert torch.equal(left[:, :100], right[:, :100])
            continue

        model = PGWControlResearchLM(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (1, 128))
        changed = tokens.clone()
        changed[0, 100] = (changed[0, 100] + 7) % cfg.vocab_size
        with torch.no_grad():
            left = model(tokens)
            right = model(changed)
        assert torch.equal(left[:, :100], right[:, :100])


def test_recency_selected_fraction_and_finite_backward() -> None:
    torch.manual_seed(1188)
    cfg = _control_small("recency")
    module = RoutingControlWorkspace(
        cfg,
        layer_index=0,
        control_mode="recency",
    )
    x = torch.randn(1, 128, cfg.d_model, requires_grad=True)
    local = torch.randn_like(x, requires_grad=True)
    out = module(x, local)
    loss = out.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert module.last_stats is not None
    assert math.isclose(
        float(module.last_stats["selected_fraction"]),
        0.0625,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
