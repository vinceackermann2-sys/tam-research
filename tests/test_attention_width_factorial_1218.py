from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.pgw_core_mechanism.model import (
    ChunkLocal256Config,
    ChunkLocal256ResearchLM,
)
from tam_research.attention_width_factorial.model import (
    WidthFactorialConfig,
    WidthFactorialResearchLM,
    reduced_width_parameter_count,
)
from tam_research.attention_width_factorial.protocol import (
    ARMS,
    EFFECT_THRESHOLD_NLL,
    EXPECTED_PARAMETERS,
    REPLICATION_SEEDS,
    SMOKE_SEED,
)


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _signature(model: torch.nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(p.shape)) for name, p in model.named_parameters()]


def _baseline_small(local: bool) -> torch.nn.Module:
    if not local:
        return ResearchLM(
            ModelConfig(
                vocab_size=127,
                d_model=32,
                n_layers=3,
                n_heads=4,
                max_seq_len=32,
                ff_mult=4,
                architecture="transformer",
            )
        )
    return ChunkLocal256ResearchLM(
        ChunkLocal256Config(
            vocab_size=127,
            d_model=32,
            n_layers=3,
            n_heads=4,
            max_seq_len=32,
            ff_mult=4,
            local_attn_inner=32,
            chunk_size=8,
            workspace_layers=(),
        )
    )


def _reduced_small(locality: str) -> WidthFactorialResearchLM:
    return WidthFactorialResearchLM(
        WidthFactorialConfig(
            vocab_size=127,
            d_model=32,
            n_layers=3,
            n_heads=4,
            max_seq_len=32,
            attention_inner=28,
            ff_hidden=136,
            chunk_size=8,
            locality=locality,
        )
    )


def test_frozen_factorial_identity() -> None:
    assert ARMS == (
        "global256_ff1024",
        "local256_ff1024",
        "global224_ff1088",
        "local224_ff1088",
    )
    assert SMOKE_SEED == 1_220_001
    assert REPLICATION_SEEDS == (1_220_101, 1_220_102, 1_220_103)
    assert EFFECT_THRESHOLD_NLL == 0.005


def test_all_four_full_size_arms_have_exact_parameter_count() -> None:
    a = ResearchLM(
        ModelConfig(
            architecture="transformer",
            d_model=256,
            n_layers=15,
            n_heads=8,
            max_seq_len=1024,
        )
    )
    b = ChunkLocal256ResearchLM(ChunkLocal256Config())
    assert parameter_count(a) == EXPECTED_PARAMETERS
    assert parameter_count(b) == EXPECTED_PARAMETERS
    assert reduced_width_parameter_count("global") == EXPECTED_PARAMETERS
    assert reduced_width_parameter_count("local") == EXPECTED_PARAMETERS


def test_parameter_reallocation_is_exact_per_layer() -> None:
    d = 256
    attn_256 = 4 * d * 256
    attn_224 = 4 * d * 224
    ff_1024 = 2 * d * 1024
    ff_1088 = 2 * d * 1088
    assert attn_256 - attn_224 == 32_768
    assert ff_1088 - ff_1024 == 32_768
    assert attn_256 + ff_1024 == attn_224 + ff_1088


def test_baseline_global_local_parameter_signatures_and_seeded_values_match() -> None:
    torch.manual_seed(1218)
    global_model = _baseline_small(local=False)
    global_values = _snapshot(global_model)

    torch.manual_seed(1218)
    local_model = _baseline_small(local=True)
    local_values = _snapshot(local_model)

    assert _signature(global_model) == _signature(local_model)
    assert global_values.keys() == local_values.keys()
    for name in global_values:
        assert torch.equal(global_values[name], local_values[name]), name


def test_reduced_global_local_parameter_signatures_and_seeded_values_match() -> None:
    torch.manual_seed(1219)
    global_model = _reduced_small("global")
    global_values = _snapshot(global_model)

    torch.manual_seed(1219)
    local_model = _reduced_small("local")
    local_values = _snapshot(local_model)

    assert _signature(global_model) == _signature(local_model)
    assert global_values.keys() == local_values.keys()
    for name in global_values:
        assert torch.equal(global_values[name], local_values[name]), name


def test_global_can_cross_chunk_boundary_while_local_cannot() -> None:
    torch.manual_seed(1220)
    global_model = _reduced_small("global").eval()
    torch.manual_seed(1220)
    local_model = _reduced_small("local").eval()

    tokens = torch.randint(0, 127, (1, 16))
    changed = tokens.clone()
    changed[:, :8] = (changed[:, :8] + 19) % 127

    with torch.no_grad():
        g0 = global_model(tokens)
        g1 = global_model(changed)
        l0 = local_model(tokens)
        l1 = local_model(changed)

    assert not torch.equal(g0[:, 8:], g1[:, 8:])
    assert torch.equal(l0[:, 8:], l1[:, 8:])


def test_reduced_arms_are_future_causal() -> None:
    for offset, locality in enumerate(("global", "local")):
        torch.manual_seed(1221 + offset)
        model = _reduced_small(locality).eval()
        tokens = torch.randint(0, 127, (1, 24))
        changed = tokens.clone()
        changed[0, 19] = (changed[0, 19] + 7) % 127

        with torch.no_grad():
            left = model(tokens)
            right = model(changed)

        assert torch.equal(left[:, :19], right[:, :19])


def test_all_trainable_parameters_receive_ordinary_ce_gradients() -> None:
    builders = (
        lambda: _baseline_small(local=False),
        lambda: _baseline_small(local=True),
        lambda: _reduced_small("global"),
        lambda: _reduced_small("local"),
    )
    for offset, build in enumerate(builders):
        torch.manual_seed(1223 + offset)
        model = build().train()
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
