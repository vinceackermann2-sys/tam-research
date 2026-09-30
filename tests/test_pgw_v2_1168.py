from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.pgw_v1.model import PGWV1Config, PGWV1ResearchLM
from tam_research.pgw_v2.model import (
    PGWV2Config,
    PGWV2ResearchLM,
    TokenConditionedPredictiveEventWorkspace,
    pgwv2_parameter_count,
    pgwv2_workspace_parameter_count,
)
from tam_research.pgw_v2.protocol import (
    EXPECTED_PARAMETERS,
    EXPECTED_SELECTED_FRACTION,
    REPLICATION_SEEDS,
    SMOKE_SEED,
)


def _tiny_v2() -> PGWV2Config:
    return PGWV2Config(
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


def _tiny_v1() -> PGWV1Config:
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


def test_v2_frozen_identity() -> None:
    assert SMOKE_SEED == 1_170_001
    assert REPLICATION_SEEDS == (1_170_101, 1_170_102, 1_170_103)
    assert EXPECTED_SELECTED_FRACTION == 0.0625


def test_default_parameter_count_exactly_matches_transformer_and_v1() -> None:
    transformer = ResearchLM(
        ModelConfig(
            architecture="transformer",
            d_model=256,
            n_layers=15,
            n_heads=8,
            max_seq_len=1024,
        )
    )
    v1 = PGWV1ResearchLM(PGWV1Config())
    v2 = PGWV2ResearchLM(PGWV2Config())
    assert parameter_count(transformer) == EXPECTED_PARAMETERS
    assert parameter_count(v1) == EXPECTED_PARAMETERS
    assert parameter_count(v2) == EXPECTED_PARAMETERS
    assert pgwv2_parameter_count() == EXPECTED_PARAMETERS


def test_v2_adds_no_trainable_read_parameters_relative_to_v1() -> None:
    v1 = PGWV1ResearchLM(_tiny_v1())
    v2 = PGWV2ResearchLM(_tiny_v2())
    v1_signature = [(name, tuple(p.shape)) for name, p in v1.named_parameters()]
    v2_signature = [(name, tuple(p.shape)) for name, p in v2.named_parameters()]
    assert v2_signature == v1_signature
    v1_workspace = v1.blocks[0].mixer.workspace
    assert v1_workspace is not None
    assert pgwv2_workspace_parameter_count() > 0
    assert parameter_count(
        TokenConditionedPredictiveEventWorkspace(PGWV2Config())
    ) == parameter_count(v1_workspace.__class__(PGWV1Config()))


def test_token_conditioned_read_can_differ_by_token() -> None:
    torch.manual_seed(1168)
    cfg = _tiny_v2()
    module = TokenConditionedPredictiveEventWorkspace(cfg).eval()
    workspace = torch.randn(1, cfg.workspace_slots, cfg.workspace_dim)
    query = torch.randn(1, cfg.chunk_size, cfg.d_model)
    with torch.no_grad():
        read = module._read_workspace(workspace, query)
    assert read.shape == query.shape
    assert not torch.equal(read[:, 0, :], read[:, 1, :])


def test_first_chunk_read_is_zero_but_workspace_can_affect_next_chunk() -> None:
    torch.manual_seed(1169)
    cfg = _tiny_v2()
    module = TokenConditionedPredictiveEventWorkspace(cfg).eval()
    x = torch.randn(1, 16, cfg.d_model)
    local = torch.randn_like(x)

    with torch.no_grad():
        before = module(x, local)
        module.workspace_init.add_(25.0)
        after = module(x, local)

    assert torch.equal(before[:, :8], after[:, :8])
    assert not torch.equal(before[:, 8:], after[:, 8:])


def test_future_token_cannot_change_earlier_logits() -> None:
    torch.manual_seed(1170)
    cfg = _tiny_v2()
    model = PGWV2ResearchLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 24))
    changed = tokens.clone()
    changed[0, 19] = (changed[0, 19] + 13) % cfg.vocab_size

    with torch.no_grad():
        left = model(tokens)
        right = model(changed)

    assert torch.equal(left[:, :19], right[:, :19])


def test_completed_chunk_change_affects_only_later_chunk_via_workspace() -> None:
    torch.manual_seed(1171)
    cfg = _tiny_v2()
    module = TokenConditionedPredictiveEventWorkspace(cfg).eval()
    x = torch.randn(1, 16, cfg.d_model)
    local = torch.randn_like(x)
    changed_x = x.clone()
    changed_local = local.clone()
    changed_x[:, 7, :] = torch.linspace(-30.0, 30.0, cfg.d_model)
    changed_local[:, 7, :] = torch.linspace(15.0, -15.0, cfg.d_model)

    with torch.no_grad():
        left = module(x, local)
        right = module(changed_x, changed_local)

    assert torch.equal(left[:, :7], right[:, :7])
    assert not torch.equal(left[:, 8:], right[:, 8:])


def test_selected_fraction_is_exact() -> None:
    torch.manual_seed(1172)
    cfg = _tiny_v2()
    model = PGWV2ResearchLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        logits = model(tokens)
    assert torch.isfinite(logits).all()
    stats = model.router_stats()
    assert stats is not None
    assert math.isclose(
        float(stats["mean"]["selected_fraction"]),
        cfg.selected_events / cfg.chunk_size,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_predictor_gets_ce_gradient_and_salience_stats_are_detached() -> None:
    torch.manual_seed(1173)
    cfg = _tiny_v2()
    model = PGWV2ResearchLM(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    labels = torch.randint(0, cfg.vocab_size, (2, 24))
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    grad_sum = 0.0
    for block in model.blocks:
        workspace = block.mixer.workspace
        if workspace is None:
            continue
        for p in (workspace.predict_down.weight, workspace.predict_up.weight):
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
            grad_sum += float(p.grad.abs().sum())
        assert workspace.last_stats is not None
        assert not workspace.last_stats["surprise_mean"].requires_grad
        assert not workspace.last_stats["surprise_std"].requires_grad
    assert grad_sum > 0.0
