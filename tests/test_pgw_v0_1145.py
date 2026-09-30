from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.pgw_v0.model import (
    PGWConfig,
    PGWMixer,
    PGWResearchLM,
    pgw_mixer_parameter_count,
)
from tam_research.pgw_v0.protocol import (
    EXPECTED_SELECTED_FRACTION,
    PARAMETER_MISMATCH_LIMIT,
    REPLICATION_SEEDS,
    SELECTED_EVENTS,
    SEGMENT_SIZE,
    SMOKE_SEED,
)


def _tiny_config() -> PGWConfig:
    return PGWConfig(
        vocab_size=127,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        state_size=8,
        workspace_dim=20,
        workspace_slots=2,
        workspace_heads=4,
        segment_size=8,
        selected_events=2,
        auto_rank=4,
    )


def test_frozen_identity_and_route_fraction() -> None:
    assert SMOKE_SEED == 1_145_001
    assert REPLICATION_SEEDS == (1_145_101, 1_145_102, 1_145_103)
    assert SEGMENT_SIZE == 16
    assert SELECTED_EVENTS == 4
    assert EXPECTED_SELECTED_FRACTION == 0.25


def test_pgw_mixer_parameter_match_is_preregistered() -> None:
    assert pgw_mixer_parameter_count() == 261_904
    transformer_mixer = 4 * 256 * 256
    assert transformer_mixer == 262_144
    assert pgw_mixer_parameter_count() - transformer_mixer == -240


def test_full_25m_parameter_mismatch_is_below_point_one_percent() -> None:
    transformer = ResearchLM(
        ModelConfig(
            architecture="transformer",
            d_model=256,
            n_layers=15,
            n_heads=8,
            max_seq_len=1024,
        )
    )
    pgw = PGWResearchLM(PGWConfig())
    t = parameter_count(transformer)
    p = parameter_count(pgw)
    mismatch = abs(p - t) / t
    assert mismatch <= PARAMETER_MISMATCH_LIMIT
    assert p - t == -3_600


def test_future_token_change_cannot_change_earlier_logits() -> None:
    torch.manual_seed(1145)
    model = PGWResearchLM(_tiny_config()).eval()
    tokens = torch.randint(0, 127, (1, 24))
    changed = tokens.clone()
    changed[0, 17] = (changed[0, 17] + 11) % 127

    with torch.no_grad():
        left = model(tokens)
        right = model(changed)

    assert torch.equal(left[:, :17], right[:, :17])


def test_segment_workspace_update_is_delayed_until_later_segment() -> None:
    torch.manual_seed(1146)
    mixer = PGWMixer(_tiny_config()).eval()
    x = torch.randn(1, 16, 32)
    changed = x.clone()
    changed[:, 7, :] = torch.linspace(-20.0, 20.0, 32)

    with torch.no_grad():
        left = mixer(x)
        right = mixer(changed)

    assert torch.equal(left[:, :7], right[:, :7])
    assert not torch.equal(left[:, 8:], right[:, 8:])


def test_router_stats_report_exact_fixed_sparse_fraction() -> None:
    torch.manual_seed(1147)
    model = PGWResearchLM(_tiny_config()).eval()
    tokens = torch.randint(0, 127, (2, 24))
    with torch.no_grad():
        logits = model(tokens)
    assert torch.isfinite(logits).all()
    stats = model.router_stats()
    assert stats is not None
    assert math.isclose(
        float(stats["mean"]["selected_fraction"]),
        0.25,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_forward_backward_is_finite_and_auto_refinement_starts_identity() -> None:
    torch.manual_seed(1148)
    model = PGWResearchLM(_tiny_config()).train()
    for block in model.blocks:
        assert torch.count_nonzero(block.mixer.auto_up.weight).item() == 0

    tokens = torch.randint(0, 127, (2, 24))
    labels = torch.randint(0, 127, (2, 24))
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
