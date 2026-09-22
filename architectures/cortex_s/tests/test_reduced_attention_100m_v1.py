from __future__ import annotations

import math

import torch

from architectures.cortex_s.reduced_attention_100m_v1 import (
    ATTENTION_EVERY,
    D_MODEL,
    EXPECTED_REDUCED_ATTENTION_PARAMETERS,
    EXPECTED_TRANSFORMER_PARAMETERS,
    FF_HIDDEN,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    ReducedAttentionBlock,
    ReducedAttentionDenseLM,
    architecture_contract,
    attention_layer_indices,
    expected_parameter_count,
    parameter_contract,
)
from architectures.cortex_s.reduced_attention_100m_v1_protocol import (
    ENGINEERING_CALIBRATION_SEED,
    FORBIDDEN_SEEDS,
    SCIENTIFIC_PAIR_SEEDS,
    validate_protocol,
)


def test_attention_pattern_and_architecture_contract_are_frozen() -> None:
    assert D_MODEL == 512
    assert N_LAYERS == 24
    assert N_HEADS == 16
    assert MAX_SEQ_LEN == 1_024
    assert ATTENTION_EVERY == 6
    assert FF_HIDDEN == 2_902
    assert attention_layer_indices() == (5, 11, 17, 23)

    contract = architecture_contract()
    assert contract["name"] == "reduced_attention_dense_v1"
    assert contract["attention_layer_numbers_one_based"] == [6, 12, 18, 24]
    assert contract["world_state"] is False
    assert contract["moe"] is False
    assert contract["router"] is False
    assert contract["training_authorized"] is False


def test_parameter_formula_and_actual_meta_model_match_exactly() -> None:
    assert EXPECTED_TRANSFORMER_PARAMETERS == 101_803_520
    assert EXPECTED_REDUCED_ATTENTION_PARAMETERS == 101_799_424
    assert expected_parameter_count() == EXPECTED_REDUCED_ATTENTION_PARAMETERS

    contract = parameter_contract()
    assert contract["absolute_gap"] == 4_096
    assert math.isclose(
        float(contract["gap_fraction"]),
        4_096 / 101_803_520,
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert float(contract["gap_fraction"]) < 0.00005

    with torch.device("meta"):
        model = ReducedAttentionDenseLM()
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert actual == EXPECTED_REDUCED_ATTENTION_PARAMETERS


def test_attention_and_nonattention_blocks_have_expected_components() -> None:
    no_attention = ReducedAttentionBlock(0)
    assert no_attention.has_attention is False
    assert no_attention.attention is None
    assert no_attention.norm_attention is None

    with_attention = ReducedAttentionBlock(5)
    assert with_attention.has_attention is True
    assert with_attention.attention is not None
    assert with_attention.norm_attention is not None

    assert not hasattr(no_attention, "world")
    assert not hasattr(no_attention, "router")
    assert not hasattr(no_attention, "experts")


def test_small_block_forward_backward_is_finite() -> None:
    torch.manual_seed(1234)
    block = ReducedAttentionBlock(5)
    x = torch.randn(1, 8, D_MODEL, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    loss = y.float().square().mean()
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_protocol_matches_frozen_100m_2b_geometry_and_reserves_fresh_seeds() -> None:
    protocol = validate_protocol()
    assert protocol["train_tokens"] == 2_000_000_000
    assert protocol["val_tokens"] == 5_000_000
    assert protocol["seq_len"] == 512
    assert protocol["global_batch"] == 128
    assert protocol["tokens_per_optimizer_step"] == 65_536
    assert protocol["total_optimizer_steps"] == 30_518
    assert protocol["full_batch_token_exposures"] == 2_000_027_648
    assert protocol["adamw_betas"] == [0.9, 0.95]
    assert protocol["final_eval_batches"] == 50
    assert protocol["pair1_screen_max_nll_delta"] == 0.015
    assert protocol["three_pair_max_mean_nll_delta"] == 0.005
    assert protocol["three_pair_max_single_nll_delta"] == 0.020
    assert protocol["training_authorized"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_seeds_consumed"] is False
    assert protocol["250m_5b_authorized"] is False

    assert SCIENTIFIC_PAIR_SEEDS == (58_231, 58_232, 58_233)
    assert ENGINEERING_CALIBRATION_SEED == 2_026_092_001
    assert not (set(SCIENTIFIC_PAIR_SEEDS) & set(FORBIDDEN_SEEDS))
    assert ENGINEERING_CALIBRATION_SEED not in {
        *SCIENTIFIC_PAIR_SEEDS,
        *FORBIDDEN_SEEDS,
    }
