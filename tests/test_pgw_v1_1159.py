from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.pgw_v1.model import (
    ChunkLocalAttention,
    PGWV1Config,
    PGWV1ResearchLM,
    PredictiveEventWorkspace,
    pgwv1_local_attention_parameter_count,
    pgwv1_parameter_count,
    pgwv1_workspace_parameter_count,
)
from tam_research.pgw_v1.protocol import (
    CHUNK_SIZE,
    EXPECTED_PARAMETERS,
    EXPECTED_SELECTED_FRACTION,
    LOCAL_ATTN_INNER,
    REPLICATION_SEEDS,
    SELECTED_EVENTS,
    SMOKE_SEED,
    WORKSPACE_LAYERS,
)


def _tiny_config() -> PGWV1Config:
    return PGWV1Config(
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
    )


def test_frozen_v1_identity() -> None:
    assert SMOKE_SEED == 1_160_001
    assert REPLICATION_SEEDS == (1_160_101, 1_160_102, 1_160_103)
    assert CHUNK_SIZE == 64
    assert SELECTED_EVENTS == 4
    assert EXPECTED_SELECTED_FRACTION == 0.0625
    assert LOCAL_ATTN_INNER == 224
    assert WORKSPACE_LAYERS == (4, 9, 14)


def test_exact_default_parameter_budget_matches_transformer() -> None:
    transformer = ResearchLM(
        ModelConfig(
            architecture="transformer",
            d_model=256,
            n_layers=15,
            n_heads=8,
            max_seq_len=1024,
        )
    )
    pgw = PGWV1ResearchLM(PGWV1Config())
    assert parameter_count(transformer) == EXPECTED_PARAMETERS
    assert parameter_count(pgw) == EXPECTED_PARAMETERS
    assert pgwv1_parameter_count() == EXPECTED_PARAMETERS
    assert pgwv1_local_attention_parameter_count() == 229_376
    assert pgwv1_workspace_parameter_count() == 163_840


def test_chunk_local_attention_cannot_cross_chunk_boundary() -> None:
    torch.manual_seed(1159)
    cfg = _tiny_config()
    local = ChunkLocalAttention(cfg).eval()
    x = torch.randn(1, 16, cfg.d_model)
    changed = x.clone()
    changed[:, :8, :] = changed[:, :8, :] + 17.0

    with torch.no_grad():
        left = local(x)
        right = local(changed)

    assert torch.equal(left[:, 8:, :], right[:, 8:, :])


def test_future_token_change_cannot_change_earlier_logits() -> None:
    torch.manual_seed(1160)
    cfg = _tiny_config()
    model = PGWV1ResearchLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 24))
    changed = tokens.clone()
    changed[0, 19] = (changed[0, 19] + 11) % cfg.vocab_size

    with torch.no_grad():
        left = model(tokens)
        right = model(changed)

    assert torch.equal(left[:, :19], right[:, :19])


def test_workspace_update_is_delayed_until_next_chunk() -> None:
    torch.manual_seed(1161)
    cfg = _tiny_config()
    workspace = PredictiveEventWorkspace(cfg).eval()
    x = torch.randn(1, 16, cfg.d_model)
    local = torch.randn_like(x)
    changed_x = x.clone()
    changed_local = local.clone()
    changed_x[:, 7, :] = torch.linspace(-20.0, 20.0, cfg.d_model)
    changed_local[:, 7, :] = torch.linspace(10.0, -10.0, cfg.d_model)

    with torch.no_grad():
        left = workspace(x, local)
        right = workspace(changed_x, changed_local)

    assert torch.equal(left[:, :7], right[:, :7])
    assert not torch.equal(left[:, 8:], right[:, 8:])


def test_router_stats_report_exact_sparse_fraction() -> None:
    torch.manual_seed(1162)
    cfg = _tiny_config()
    model = PGWV1ResearchLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        logits = model(tokens)
    assert torch.isfinite(logits).all()

    stats = model.router_stats()
    assert stats is not None
    expected = cfg.selected_events / cfg.chunk_size
    assert math.isclose(
        float(stats["mean"]["selected_fraction"]),
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert len(stats["per_layer"]) == len(cfg.workspace_layers)


def test_predictor_receives_ordinary_ce_gradient_and_backward_is_finite() -> None:
    torch.manual_seed(1163)
    cfg = _tiny_config()
    model = PGWV1ResearchLM(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    labels = torch.randint(0, cfg.vocab_size, (2, 24))

    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    predictor_grad_norm = 0.0
    for block in model.blocks:
        workspace = block.mixer.workspace
        if workspace is None:
            continue
        for parameter in (
            workspace.predict_down.weight,
            workspace.predict_up.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            predictor_grad_norm += float(parameter.grad.abs().sum())

    assert predictor_grad_norm > 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_salience_values_are_detached_from_autograd_graph() -> None:
    torch.manual_seed(1164)
    cfg = _tiny_config()
    module = PredictiveEventWorkspace(cfg).train()
    x = torch.randn(1, 16, cfg.d_model, requires_grad=True)
    local = torch.randn_like(x, requires_grad=True)
    output = module(x, local)

    assert output.requires_grad
    stats = module.last_stats
    assert stats is not None
    assert not stats["surprise_mean"].requires_grad
    assert not stats["surprise_std"].requires_grad


def test_rejects_non_chunk_aligned_sequence() -> None:
    cfg = _tiny_config()
    model = PGWV1ResearchLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 15))
    try:
        model(tokens)
    except ValueError as exc:
        assert "divisible by chunk_size" in str(exc)
    else:
        raise AssertionError("expected non-aligned sequence to be rejected")
