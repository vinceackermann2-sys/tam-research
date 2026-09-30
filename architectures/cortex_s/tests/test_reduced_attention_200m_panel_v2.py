from pathlib import Path

import torch

from architectures.cortex_s.reduced_attention_100m_v1 import (
    ReducedAttentionDenseLM as V1ReducedAttentionDenseLM,
)
from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
    ATTENTION_8,
    EARLY_4,
    EXPECTED_TRANSFORMER_PARAMETERS,
    PANEL_VARIANTS,
    V1_REPLICATE,
    ScheduledAttentionDenseLM,
    architecture_contract,
    build_fresh_transformer,
    expected_parameter_count,
)
from architectures.cortex_s.reduced_attention_200m_panel_v2_protocol import (
    ENGINEERING_SEED,
    EVAL_EXPOSURES_TOKENS,
    FINAL_EVAL_BATCHES,
    FULL_BATCH_TOKEN_EXPOSURES,
    MAX_200M_NLL_DELTA,
    MIN_TRAINING_TPS_RATIO,
    PAIR1_V1_200M_NLL_DELTA_REFERENCE,
    TOTAL_OPTIMIZER_STEPS,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "architectures" / "cortex_s" / "reduced_attention_200m_panel_v2.py"
PROTOCOL_PATH = (
    ROOT
    / "architectures"
    / "cortex_s"
    / "reduced_attention_200m_panel_v2_protocol.py"
)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_panel_variant_contract_is_exact_and_parameter_matched() -> None:
    assert V1_REPLICATE.attention_layers_one_based == (6, 12, 18, 24)
    assert EARLY_4.attention_layers_one_based == (1, 8, 16, 24)
    assert ATTENTION_8.attention_layers_one_based == (
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
    )
    assert V1_REPLICATE.ff_hidden == 2_902
    assert EARLY_4.ff_hidden == 2_902
    assert ATTENTION_8.ff_hidden == 2_731

    assert expected_parameter_count(V1_REPLICATE) == 101_799_424
    assert expected_parameter_count(EARLY_4) == 101_799_424
    assert expected_parameter_count(ATTENTION_8) == 101_795_328

    for variant in PANEL_VARIANTS:
        gap = abs(
            expected_parameter_count(variant)
            - EXPECTED_TRANSFORMER_PARAMETERS
        )
        assert gap / EXPECTED_TRANSFORMER_PARAMETERS < 0.0001


def test_panel_meta_models_match_frozen_parameter_counts_and_schedules() -> None:
    with torch.device("meta"):
        transformer = build_fresh_transformer()
        candidates = {
            variant.name: ScheduledAttentionDenseLM(variant)
            for variant in PANEL_VARIANTS
        }

    assert _count_parameters(transformer) == EXPECTED_TRANSFORMER_PARAMETERS
    for variant in PANEL_VARIANTS:
        model = candidates[variant.name]
        assert _count_parameters(model) == variant.expected_parameters
        actual_layers = tuple(
            block.layer_number
            for block in model.blocks
            if block.has_attention
        )
        assert actual_layers == variant.attention_layers_one_based


def test_v1_replicate_is_structurally_identical_to_frozen_v1_model() -> None:
    with torch.device("meta"):
        frozen = V1ReducedAttentionDenseLM()
        replicate = ScheduledAttentionDenseLM(V1_REPLICATE)

    frozen_shapes = {
        key: tuple(value.shape)
        for key, value in frozen.state_dict().items()
    }
    replicate_shapes = {
        key: tuple(value.shape)
        for key, value in replicate.state_dict().items()
    }
    assert replicate_shapes == frozen_shapes
    assert _count_parameters(replicate) == _count_parameters(frozen)


def test_panel_protocol_freezes_200m_geometry_and_progression_gate() -> None:
    protocol = validate_protocol()

    assert ENGINEERING_SEED == 2_026_092_901
    assert TOTAL_OPTIMIZER_STEPS == 3_052
    assert FULL_BATCH_TOKEN_EXPOSURES == 200_015_872
    assert EVAL_EXPOSURES_TOKENS == (
        50_003_968,
        100_007_936,
        150_011_904,
        200_015_872,
    )
    assert FINAL_EVAL_BATCHES == 50
    assert MAX_200M_NLL_DELTA == 0.025
    assert MIN_TRAINING_TPS_RATIO == 1.05
    assert abs(
        PAIR1_V1_200M_NLL_DELTA_REFERENCE - 0.1224118590354917
    ) < 1e-15

    assert protocol["model_order"] == [
        "fresh_transformer",
        "v1_replicate",
        "early_4",
        "attention_8",
    ]
    gate = protocol["progression_gate"]
    assert (
        gate["candidate_minus_fresh_transformer_final_nll_max"]
        == 0.025
    )
    assert (
        gate["candidate_over_fresh_transformer_training_tps_min"]
        == 1.05
    )
    assert gate["both_complete_and_finite"] is True
    assert gate["same_seed_data_order_and_training_geometry"] is True
    assert (
        gate["v1_replicate_pass_means_seed_sensitivity_diagnostic_only"]
        is True
    )


def test_panel_preparation_contains_no_gpu_or_scientific_authority() -> None:
    protocol = validate_protocol()

    assert protocol["engineering_panel_training_authorized"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["scientific_seeds_authorized"] is False
    assert protocol["replication_seeds_authorized"] is False
    assert protocol["250m_5b_authorized"] is False
    assert protocol["breakthrough_claim_allowed"] is False

    model_source = MODEL_PATH.read_text(encoding="utf-8")
    protocol_source = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "import modal" not in model_source
    assert "import modal" not in protocol_source
    assert "modal.App" not in model_source
    assert "modal.App" not in protocol_source
    assert 'gpu="H100!"' not in model_source
    assert 'gpu="H100!"' not in protocol_source


def test_panel_architecture_contract_remains_simple_dense_reduced_attention() -> None:
    contract = architecture_contract()
    assert contract["world_state"] is False
    assert contract["moe"] is False
    assert contract["router"] is False
    assert contract["gpu_authorized"] is False
    assert contract["training_authorized"] is False
