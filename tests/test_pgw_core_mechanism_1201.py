from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.pgw_v2_mechanism.model import (
    PGWControlConfig,
    PGWControlResearchLM,
    RoutingControlWorkspace,
)
from tam_research.pgw_core_mechanism.model import (
    ChunkLocal256Config,
    ChunkLocal256ResearchLM,
    CoreControlWorkspace,
    PGWCoreConfig,
    PGWCoreResearchLM,
    chunk_local_256_parameter_count,
    core_control_parameter_count,
)
from tam_research.pgw_core_mechanism.protocol import (
    ARMS,
    EXPECTED_PARAMETERS,
    REPLICATION_SEEDS,
    SMOKE_SEED,
)


def _carry_small() -> PGWControlConfig:
    return PGWControlConfig(
        vocab_size=127,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        local_attn_inner=24,
        chunk_size=8,
        workspace_layers=(0, 2),
        workspace_dim=16,
        workspace_slots=2,
        workspace_heads=4,
        predictor_rank=4,
        workspace_ff_rank=5,
        selected_events=2,
        control_mode="no_workspace",
    )


def _core_small(mode: str) -> PGWCoreConfig:
    return PGWCoreConfig(
        vocab_size=127,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        local_attn_inner=24,
        chunk_size=8,
        workspace_layers=(0, 2),
        workspace_dim=16,
        workspace_slots=2,
        workspace_heads=4,
        predictor_rank=4,
        workspace_ff_rank=5,
        selected_events=2,
        core_mode=mode,
    )


def _local256_small() -> ChunkLocal256Config:
    return ChunkLocal256Config(
        vocab_size=127,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        local_attn_inner=32,
        chunk_size=8,
    )


def _signature(model: torch.nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(p.shape)) for name, p in model.named_parameters()]


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_frozen_core_panel_identity() -> None:
    assert ARMS == (
        "transformer",
        "chunk_local_256",
        "local224_predictor_carry",
        "local224_predictor_reset",
        "local224_only",
    )
    assert SMOKE_SEED == 1_200_001
    assert REPLICATION_SEEDS == (1_200_101, 1_200_102, 1_200_103)


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
    carry = PGWControlResearchLM(PGWControlConfig(control_mode="no_workspace"))
    assert parameter_count(transformer) == EXPECTED_PARAMETERS
    assert chunk_local_256_parameter_count() == EXPECTED_PARAMETERS
    assert parameter_count(carry) == EXPECTED_PARAMETERS
    assert core_control_parameter_count("predictor_reset") == EXPECTED_PARAMETERS
    assert core_control_parameter_count("local_only") == EXPECTED_PARAMETERS


def test_carry_reset_local_only_parameter_signatures_and_values_match() -> None:
    torch.manual_seed(1201)
    carry = PGWControlResearchLM(_carry_small())
    carry_values = _snapshot(carry)
    carry_signature = _signature(carry)

    torch.manual_seed(1201)
    reset = PGWCoreResearchLM(_core_small("predictor_reset"))
    torch.manual_seed(1201)
    local_only = PGWCoreResearchLM(_core_small("local_only"))

    for model in (reset, local_only):
        assert _signature(model) == carry_signature
        values = _snapshot(model)
        assert values.keys() == carry_values.keys()
        for name in carry_values:
            assert torch.equal(values[name], carry_values[name]), name


def test_chunk_local_256_cannot_cross_chunk_boundary() -> None:
    torch.manual_seed(1202)
    cfg = _local256_small()
    model = ChunkLocal256ResearchLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    changed = tokens.clone()
    changed[:, :8] = (changed[:, :8] + 17) % cfg.vocab_size

    with torch.no_grad():
        left = model(tokens)
        right = model(changed)

    assert torch.equal(left[:, 8:], right[:, 8:])


def test_carry_transmits_completed_chunk_context_to_next_chunk() -> None:
    torch.manual_seed(1203)
    cfg = _carry_small()
    module = RoutingControlWorkspace(
        cfg,
        layer_index=0,
        control_mode="no_workspace",
    ).eval()
    torch.nn.init.normal_(module.workspace_init, mean=0.0, std=0.02)
    x = torch.randn(1, 16, cfg.d_model)
    local = torch.randn_like(x)
    changed_x = x.clone()
    changed_local = local.clone()
    changed_x[:, 7, :] += 13.0
    changed_local[:, 7, :] -= 11.0

    with torch.no_grad():
        left = module(x, local)
        right = module(changed_x, changed_local)

    assert torch.equal(left[:, :7], right[:, :7])
    assert not torch.equal(left[:, 8, :], right[:, 8, :])


def test_reset_cannot_transmit_previous_chunk_context() -> None:
    torch.manual_seed(1204)
    cfg = _core_small("predictor_reset")
    module = CoreControlWorkspace(
        cfg,
        layer_index=0,
        core_mode="predictor_reset",
    ).eval()
    torch.nn.init.normal_(module.workspace_init, mean=0.0, std=0.02)
    x = torch.randn(1, 16, cfg.d_model)
    local = torch.randn_like(x)
    changed_x = x.clone()
    changed_local = local.clone()
    changed_x[:, 7, :] += 13.0
    changed_local[:, 7, :] -= 11.0

    with torch.no_grad():
        left = module(x, local)
        right = module(changed_x, changed_local)

    assert torch.equal(left[:, 8:], right[:, 8:])


def test_future_token_cannot_change_earlier_logits_for_core_controls() -> None:
    for offset, mode in enumerate(("predictor_reset", "local_only")):
        torch.manual_seed(1205 + offset)
        cfg = _core_small(mode)
        model = PGWCoreResearchLM(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (1, 24))
        changed = tokens.clone()
        changed[0, 19] = (changed[0, 19] + 9) % cfg.vocab_size
        with torch.no_grad():
            left = model(tokens)
            right = model(changed)
        assert torch.equal(left[:, :19], right[:, :19])


def test_predictor_receives_ce_gradient_in_carry_and_reset() -> None:
    builders = (
        lambda: PGWControlResearchLM(_carry_small()),
        lambda: PGWCoreResearchLM(_core_small("predictor_reset")),
    )
    for offset, build in enumerate(builders):
        torch.manual_seed(1207 + offset)
        model = build().train()
        tokens = torch.randint(0, 127, (2, 24))
        labels = torch.randint(0, 127, (2, 24))
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        assert torch.isfinite(loss)

        grad_sum = 0.0
        for block in model.blocks:
            ws = block.mixer.workspace
            if ws is None:
                continue
            assert ws.predict_down.weight.grad is not None
            assert ws.predict_up.weight.grad is not None
            grad_sum += float(ws.predict_down.weight.grad.abs().sum())
            grad_sum += float(ws.predict_up.weight.grad.abs().sum())
        assert grad_sum > 0.0


def test_local_only_disables_predictor_and_workspace_gradients() -> None:
    torch.manual_seed(1209)
    cfg = _core_small("local_only")
    model = PGWCoreResearchLM(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    labels = torch.randint(0, cfg.vocab_size, (2, 24))
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)

    for block in model.blocks:
        ws = block.mixer.workspace
        if ws is None:
            continue
        for parameter in ws.parameters():
            assert parameter.grad is None

    stats = model.router_stats()
    assert stats is not None
    assert math.isclose(
        float(stats["mean"]["selected_fraction"]),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
