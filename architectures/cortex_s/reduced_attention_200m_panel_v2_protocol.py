from __future__ import annotations

from typing import Any, Final

from architectures.cortex_s import reduced_attention_100m_v1_protocol as v1
from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
    ATTENTION_8,
    EARLY_4,
    EXPECTED_TRANSFORMER_PARAMETERS,
    V1_REPLICATE,
    architecture_contract,
    expected_parameter_count,
)


EXPERIMENT_ID: Final = "reduced-attention-v2-100m-200m-engineering-panel"
ENGINEERING_SEED: Final = 2_026_092_901

TRAIN_TOKENS_TARGET: Final = 200_000_000
VAL_TOKENS: Final = 5_000_000
SEQ_LEN: Final = 512
MICRO_BATCH_SIZE: Final = 64
GRAD_ACCUM_STEPS: Final = 2
GLOBAL_BATCH: Final = 128
TOKENS_PER_OPTIMIZER_STEP: Final = 65_536
TOTAL_OPTIMIZER_STEPS: Final = 3_052
FULL_BATCH_TOKEN_EXPOSURES: Final = 200_015_872

LEARNING_RATE: Final = 3e-4
ADAMW_BETAS: Final = (0.9, 0.95)
WEIGHT_DECAY: Final = 0.1
WARMUP_RATIO: Final = 0.02
GRAD_CLIP: Final = 1.0

EVAL_THRESHOLDS_TOKENS: Final = (
    50_000_000,
    100_000_000,
    150_000_000,
    200_000_000,
)
EVAL_EXPOSURES_TOKENS: Final = (
    50_003_968,
    100_007_936,
    150_011_904,
    200_015_872,
)
PERIODIC_EVAL_BATCHES: Final = 20
FINAL_EVAL_BATCHES: Final = 50

DATA_DIR: Final = "/vol/data/tam100m-2b-curated-v1"
TRAIN_SHA256: Final = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256: Final = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256: Final = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

MAX_200M_NLL_DELTA: Final = 0.025
MIN_TRAINING_TPS_RATIO: Final = 1.05
PAIR1_V1_200M_NLL_DELTA_REFERENCE: Final = 0.1224118590354917

FORBIDDEN_SCIENTIFIC_OR_DATA_SEEDS: Final = (
    8_100,
    48_131,
    48_132,
    48_133,
    58_231,
    58_232,
    58_233,
    59_231,
)


def validate_protocol() -> dict[str, Any]:
    if ENGINEERING_SEED in FORBIDDEN_SCIENTIFIC_OR_DATA_SEEDS:
        raise RuntimeError("engineering seed collides with scientific/data seed")

    frozen_pairs = {
        "val_tokens": (VAL_TOKENS, v1.VAL_TOKENS),
        "seq_len": (SEQ_LEN, v1.SEQ_LEN),
        "micro_batch_size": (MICRO_BATCH_SIZE, v1.MICRO_BATCH_SIZE),
        "grad_accum_steps": (GRAD_ACCUM_STEPS, v1.GRAD_ACCUM_STEPS),
        "global_batch": (GLOBAL_BATCH, v1.GLOBAL_BATCH),
        "tokens_per_optimizer_step": (
            TOKENS_PER_OPTIMIZER_STEP,
            v1.TOKENS_PER_OPTIMIZER_STEP,
        ),
        "learning_rate": (LEARNING_RATE, v1.LEARNING_RATE),
        "adamw_betas": (ADAMW_BETAS, v1.ADAMW_BETAS),
        "weight_decay": (WEIGHT_DECAY, v1.WEIGHT_DECAY),
        "warmup_ratio": (WARMUP_RATIO, v1.WARMUP_RATIO),
        "grad_clip": (GRAD_CLIP, v1.GRAD_CLIP),
        "train_sha256": (TRAIN_SHA256, v1.TRAIN_SHA256),
        "val_sha256": (VAL_SHA256, v1.VAL_SHA256),
        "meta_sha256": (META_SHA256, v1.META_SHA256),
    }
    mismatches = {
        key: values
        for key, values in frozen_pairs.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(f"v1 matched-training contract drifted: {mismatches}")

    if TOTAL_OPTIMIZER_STEPS * TOKENS_PER_OPTIMIZER_STEP != FULL_BATCH_TOKEN_EXPOSURES:
        raise RuntimeError("panel full-batch exposure formula drifted")
    if FULL_BATCH_TOKEN_EXPOSURES < TRAIN_TOKENS_TARGET:
        raise RuntimeError("panel does not reach the 200M target")
    if FULL_BATCH_TOKEN_EXPOSURES - TRAIN_TOKENS_TARGET >= TOKENS_PER_OPTIMIZER_STEP:
        raise RuntimeError("panel exposure overshoot exceeds one optimizer step")

    expected_exposures = tuple(
        (
            (threshold + TOKENS_PER_OPTIMIZER_STEP - 1)
            // TOKENS_PER_OPTIMIZER_STEP
        )
        * TOKENS_PER_OPTIMIZER_STEP
        for threshold in EVAL_THRESHOLDS_TOKENS
    )
    if expected_exposures != EVAL_EXPOSURES_TOKENS:
        raise RuntimeError("panel evaluation exposure schedule drifted")

    parameter_counts = {
        "fresh_transformer": EXPECTED_TRANSFORMER_PARAMETERS,
        "v1_replicate": expected_parameter_count(V1_REPLICATE),
        "early_4": expected_parameter_count(EARLY_4),
        "attention_8": expected_parameter_count(ATTENTION_8),
    }
    for name, count in parameter_counts.items():
        gap = abs(count - EXPECTED_TRANSFORMER_PARAMETERS)
        if gap / EXPECTED_TRANSFORMER_PARAMETERS > 0.005:
            raise RuntimeError(f"{name} exceeds the 0.5% parameter tolerance")

    if MAX_200M_NLL_DELTA <= 0.0:
        raise RuntimeError("quality screen must be positive")
    if MIN_TRAINING_TPS_RATIO <= 1.0:
        raise RuntimeError("throughput screen must require a real speed advantage")
    recovered_fraction = 1.0 - (
        MAX_200M_NLL_DELTA / PAIR1_V1_200M_NLL_DELTA_REFERENCE
    )
    if recovered_fraction < 0.79:
        raise RuntimeError("quality gate no longer requires ~80% v1 deficit recovery")

    return {
        "experiment_id": EXPERIMENT_ID,
        "classification": "ENGINEERING_REDUCED_ATTENTION_V2_200M_PANEL_PREREG_ONLY",
        "engineering_seed_reserved_not_consumed": ENGINEERING_SEED,
        "models": architecture_contract(),
        "model_order": [
            "fresh_transformer",
            "v1_replicate",
            "early_4",
            "attention_8",
        ],
        "parameter_counts": parameter_counts,
        "train_tokens_target": TRAIN_TOKENS_TARGET,
        "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "val_tokens": VAL_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "global_batch": GLOBAL_BATCH,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "learning_rate": LEARNING_RATE,
        "adamw_betas": list(ADAMW_BETAS),
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "grad_clip": GRAD_CLIP,
        "eval_thresholds_tokens": list(EVAL_THRESHOLDS_TOKENS),
        "eval_exposures_tokens": list(EVAL_EXPOSURES_TOKENS),
        "periodic_eval_batches": PERIODIC_EVAL_BATCHES,
        "final_eval_batches": FINAL_EVAL_BATCHES,
        "data_dir": DATA_DIR,
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
        "progression_gate": {
            "candidate_minus_fresh_transformer_final_nll_max": (
                MAX_200M_NLL_DELTA
            ),
            "candidate_over_fresh_transformer_training_tps_min": (
                MIN_TRAINING_TPS_RATIO
            ),
            "both_complete_and_finite": True,
            "same_seed_data_order_and_training_geometry": True,
            "v1_replicate_pass_means_seed_sensitivity_diagnostic_only": True,
            "pair1_v1_200m_nll_delta_reference": (
                PAIR1_V1_200M_NLL_DELTA_REFERENCE
            ),
        },
        "engineering_panel_training_authorized": False,
        "gpu_authorized": False,
        "scientific_training_authorized": False,
        "scientific_seeds_authorized": False,
        "replication_seeds_authorized": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }
