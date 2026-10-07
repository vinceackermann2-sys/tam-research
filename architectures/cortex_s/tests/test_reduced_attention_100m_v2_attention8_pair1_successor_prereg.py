from pathlib import Path

import torch

from architectures.cortex_s.reduced_attention_100m_v2_attention8 import (
    ATTENTION_LAYERS_ONE_BASED,
    CANDIDATE_NAME,
    EXPECTED_PARAMETERS,
    ReducedAttentionDenseV2Attention8LM,
    candidate_contract,
)
from architectures.cortex_s.reduced_attention_100m_v2_attention8_successor_protocol import (
    ENGINEERING_PANEL_ATTENTION8_NLL_DELTA,
    ENGINEERING_PANEL_ATTENTION8_TPS_RATIO,
    FORBIDDEN_PRIOR_SEEDS,
    FULL_BATCH_TOKEN_EXPOSURES,
    MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION,
    PAIR_SCREEN_MAX_NLL_DELTA,
    PAIR_SEED,
    PRIOR_PAIR_IS_SCIENTIFIC_EVIDENCE,
    PRIOR_PAIR_SEED,
    PRIOR_PAIR_TERMINAL_CLASSIFICATION,
    REQUIRED_MODAL_ACCOUNT,
    TOTAL_OPTIMIZER_STEPS,
    validate_protocol,
)
from tam_research.models import ModelConfig, ResearchLM


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = (
    ROOT
    / "architectures"
    / "cortex_s"
    / "reduced_attention_100m_v2_attention8_successor_protocol.py"
)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_attention8_v2_successor_candidate_is_the_same_frozen_architecture() -> None:
    assert CANDIDATE_NAME == "reduced_attention_dense_v2_attention8"
    assert ATTENTION_LAYERS_ONE_BASED == (3, 6, 9, 12, 15, 18, 21, 24)
    assert EXPECTED_PARAMETERS == 101_795_328

    with torch.device("meta"):
        candidate = ReducedAttentionDenseV2Attention8LM()
        transformer = ResearchLM(
            ModelConfig(
                vocab_size=50_257,
                d_model=512,
                n_layers=24,
                n_heads=16,
                max_seq_len=1_024,
                ff_mult=4,
                architecture="transformer",
            )
        )

    assert _count_parameters(candidate) == 101_795_328
    assert _count_parameters(transformer) == 101_803_520
    gap = abs(_count_parameters(candidate) - _count_parameters(transformer))
    assert gap / _count_parameters(transformer) < 0.0001

    contract = candidate_contract()
    assert contract["world_state"] is False
    assert contract["moe"] is False
    assert contract["router"] is False


def test_attention8_v2_successor_prereg_uses_fresh_seed_and_freezes_prior_seed() -> None:
    protocol = validate_protocol()

    assert PAIR_SEED == 60_232
    assert PRIOR_PAIR_SEED == 60_231
    assert PRIOR_PAIR_SEED in FORBIDDEN_PRIOR_SEEDS
    assert PAIR_SEED not in FORBIDDEN_PRIOR_SEEDS
    assert protocol["pair_seed_reserved_not_consumed"] == 60_232
    assert protocol["prior_pair_seed_consumed"] == 60_231
    assert protocol["scientific_seed_consumed"] is False


def test_attention8_v2_successor_records_prior_infra_failure_as_non_evidence() -> None:
    protocol = validate_protocol()

    assert (
        PRIOR_PAIR_TERMINAL_CLASSIFICATION
        == "SCIENTIFIC_ATTENTION8_V2_PAIR1_INFRA_FAILURE_NO_RETRY"
    )
    assert PRIOR_PAIR_IS_SCIENTIFIC_EVIDENCE is False
    assert protocol["prior_pair_not_scientific_evidence"] is True
    assert (
        protocol["prior_pair_terminal_classification"]
        == PRIOR_PAIR_TERMINAL_CLASSIFICATION
    )


def test_attention8_v2_successor_is_primary_only() -> None:
    protocol = validate_protocol()

    assert REQUIRED_MODAL_ACCOUNT == "primary"
    assert protocol["required_modal_account"] == "primary"


def test_attention8_v2_successor_freezes_same_2b_geometry_and_gates() -> None:
    protocol = validate_protocol()

    assert TOTAL_OPTIMIZER_STEPS == 30_518
    assert FULL_BATCH_TOKEN_EXPOSURES == 2_000_027_648
    assert protocol["train_tokens"] == 2_000_000_000
    assert protocol["seq_len"] == 512
    assert protocol["micro_batch_size"] == 64
    assert protocol["grad_accum_steps"] == 2
    assert protocol["tokens_per_optimizer_step"] == 65_536
    assert protocol["final_eval_batches"] == 50

    gate = protocol["pair1_gate"]
    assert PAIR_SCREEN_MAX_NLL_DELTA == 0.015
    assert MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION == 1.03
    assert gate["candidate_minus_transformer_final_nll_max"] == 0.015
    assert gate["candidate_over_transformer_training_tps_min_for_progression"] == 1.03
    assert gate["both_complete_exact_geometry_and_finite"] is True


def test_attention8_v2_successor_keeps_engineering_evidence_non_scientific() -> None:
    protocol = validate_protocol()
    evidence = protocol["engineering_evidence"]

    assert abs(ENGINEERING_PANEL_ATTENTION8_NLL_DELTA - (-0.01449833869934114)) < 1e-15
    assert abs(ENGINEERING_PANEL_ATTENTION8_TPS_RATIO - 1.0651255526571075) < 1e-15
    assert evidence["classification"] == "ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS"
    assert evidence["not_scientific_evidence"] is True
    assert evidence["seed"] == 2_026_092_901


def test_attention8_v2_successor_prereg_grants_no_execution_or_scale_authority() -> None:
    protocol = validate_protocol()

    assert protocol["scientific_training_authorized"] is False
    assert protocol["scientific_seed_consumed"] is False
    assert protocol["replication_authorized"] is False
    assert protocol["replication_seeds_authorized"] is False
    assert protocol["250m_5b_authorized"] is False
    assert protocol["breakthrough_claim_allowed"] is False

    source = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "import modal" not in source
    assert 'gpu="H100!"' not in source
    assert "modal run " not in source
