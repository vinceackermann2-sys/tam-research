from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.scales import SCALE_SPECS
from tam_research.attention_width_factorial.model import (
    WidthFactorialConfig,
    WidthFactorialResearchLM,
)
from tam_research.attention_width_50m.model import (
    baseline_50m_config,
    baseline_50m_parameter_count,
    build_baseline_50m,
    build_reduced_50m,
    reduced_50m_config,
    reduced_50m_parameter_count,
)
from tam_research.attention_width_50m.protocol import (
    ARMS,
    BASE_ATTN_INNER,
    BASE_FF_HIDDEN,
    D_MODEL,
    EFFECT_THRESHOLD_NLL,
    EXPECTED_PARAMETERS,
    MIN_THROUGHPUT_RATIO,
    N_HEADS,
    N_LAYERS,
    REDUCED_ATTN_INNER,
    REDUCED_FF_HIDDEN,
    REPLICATION_SEEDS,
    SMOKE_SEED,
)


def _baseline_small() -> ResearchLM:
    return ResearchLM(
        ModelConfig(
            architecture="transformer",
            vocab_size=127,
            d_model=96,
            n_layers=2,
            n_heads=3,
            max_seq_len=32,
            ff_mult=4,
        )
    )


def _reduced_small() -> WidthFactorialResearchLM:
    return WidthFactorialResearchLM(
        WidthFactorialConfig(
            vocab_size=127,
            d_model=96,
            n_layers=2,
            n_heads=3,
            max_seq_len=32,
            attention_inner=84,
            ff_hidden=408,
            chunk_size=8,
            locality="global",
        )
    )


def test_frozen_50m_replication_identity() -> None:
    assert ARMS == ("global384_ff1536", "global336_ff1632")
    assert SMOKE_SEED == 1_230_001
    assert REPLICATION_SEEDS == (1_230_101, 1_230_102, 1_230_103)
    assert EFFECT_THRESHOLD_NLL == 0.005
    assert MIN_THROUGHPUT_RATIO == 0.80


def test_geometry_matches_native_50m_scale_and_self_similar_rule() -> None:
    scale = SCALE_SPECS["50m"]
    assert (scale.d_model, scale.n_layers, scale.n_heads) == (
        D_MODEL,
        N_LAYERS,
        N_HEADS,
    )
    assert BASE_ATTN_INNER == D_MODEL == 384
    assert BASE_FF_HIDDEN == 4 * D_MODEL == 1536
    assert REDUCED_ATTN_INNER == 7 * D_MODEL // 8 == 336
    assert REDUCED_FF_HIDDEN * 4 == 17 * D_MODEL
    assert REDUCED_FF_HIDDEN == 1632
    assert REDUCED_ATTN_INNER % N_HEADS == 0
    assert REDUCED_ATTN_INNER // N_HEADS == 28


def test_exact_per_layer_parameter_reallocation() -> None:
    baseline_attention = 4 * D_MODEL * BASE_ATTN_INNER
    reduced_attention = 4 * D_MODEL * REDUCED_ATTN_INNER
    baseline_ff = 2 * D_MODEL * BASE_FF_HIDDEN
    reduced_ff = 2 * D_MODEL * REDUCED_FF_HIDDEN
    assert baseline_attention - reduced_attention == 73_728
    assert reduced_ff - baseline_ff == 73_728
    assert baseline_attention + baseline_ff == reduced_attention + reduced_ff


def test_both_full_size_arms_have_exact_50m_parameter_count() -> None:
    assert baseline_50m_parameter_count() == EXPECTED_PARAMETERS
    assert reduced_50m_parameter_count() == EXPECTED_PARAMETERS
    assert parameter_count(build_baseline_50m()) == EXPECTED_PARAMETERS
    assert parameter_count(build_reduced_50m()) == EXPECTED_PARAMETERS


def test_baseline_config_is_native_transformer_geometry() -> None:
    cfg = baseline_50m_config()
    assert cfg.architecture == "transformer"
    assert cfg.d_model == 384
    assert cfg.n_layers == 17
    assert cfg.n_heads == 12
    assert cfg.ff_mult == 4


def test_reduced_config_is_exact_frozen_geometry() -> None:
    cfg = reduced_50m_config()
    assert cfg.locality == "global"
    assert cfg.d_model == 384
    assert cfg.n_layers == 17
    assert cfg.n_heads == 12
    assert cfg.attention_inner == 336
    assert cfg.ff_hidden == 1632


def test_both_small_arms_are_global_and_future_causal() -> None:
    for offset, builder in enumerate((_baseline_small, _reduced_small)):
        torch.manual_seed(1230 + offset)
        model = builder().eval()
        tokens = torch.randint(0, 127, (1, 24))

        changed_past = tokens.clone()
        changed_past[:, 0] = (changed_past[:, 0] + 11) % 127
        changed_future = tokens.clone()
        changed_future[:, 19] = (changed_future[:, 19] + 13) % 127

        with torch.no_grad():
            base = model(tokens)
            past = model(changed_past)
            future = model(changed_future)

        assert not torch.equal(base[:, 8:], past[:, 8:])
        assert torch.equal(base[:, :19], future[:, :19])


def test_every_trainable_tensor_receives_finite_ce_gradient() -> None:
    for offset, builder in enumerate((_baseline_small, _reduced_small)):
        torch.manual_seed(1232 + offset)
        model = builder().train()
        tokens = torch.randint(0, 127, (2, 24))
        labels = torch.randint(0, 127, (2, 24))
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        assert torch.isfinite(loss)

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
