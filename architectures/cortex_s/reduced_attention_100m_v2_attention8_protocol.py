from __future__ import annotations

from typing import Any, Final

from architectures.cortex_s import reduced_attention_100m_v1_protocol as v1
from architectures.cortex_s.reduced_attention_100m_v2_attention8 import (
    CANDIDATE_NAME,
    EXPECTED_PARAMETERS,
    candidate_contract,
)


EXPERIMENT_ID: Final = "reduced-attention-v2-attention8-pair1-100m-2b"
PAIR_SEED: Final = 60_231

TRAIN_TOKENS: Final = 2_000_000_000
VAL_TOKENS: Final = 5_000_000
SEQ_LEN: Final = 512
MICRO_BATCH_SIZE: Final = 64
GRAD_ACCUM_STEPS: Final = 2
GLOBAL_BATCH: Final = 128
TOKENS_PER_OPTIMIZER_STEP: Final = 65_536
TOTAL_OPTIMIZER_STEPS: Final = 30_518
FULL_BATCH_TOKEN_EXPOSURES: Final = 2_000_027_648

LEARNING_RATE: Final = 3e-4
ADAMW_BETAS: Final = (0.9, 0.95)
WEIGHT_DECAY: Final = 0.1
WARMUP_RATIO: Final = 0.02
GRAD_CLIP: Final = 1.0
FINAL_EVAL_BATCHES: Final = 50
CHECKPOINT_EVERY_TOKENS: Final = 200_000_000

TRAIN_SHA256: Final = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256: Final = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256: Final = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

PAIR_SCREEN_MAX_NLL_DELTA: Final = 0.015
MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION: Final = 1.03

ENGINEERING_PANEL_SEED: Final = 2_026_092_901
ENGINEERING_PANEL_ATTENTION8_FINAL_NLL: Final = 3.656175718307495
ENGINEERING_PANEL_TRANSFORMER_FINAL_NLL: Final = 3.670674057006836
ENGINEERING_PANEL_ATTENTION8_NLL_DELTA: Final = -0.01449833869934114
ENGINEERING_PANEL_ATTENTION8_TPS_RATIO: Final = 1.0651255526571075
ENGINEERING_PANEL_CLASSIFICATION: Final = "ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS"

FORBIDDEN_PRIOR_SEEDS: Final = (
    8_100,
    48_131,
    48_132,
    48_133,
    58_231,
    58_232,
    58_233,
    59_231,
    ENGINEERING_PANEL_SEED,
)


def validate_protocol() -> dict[str, Any]:
    if PAIR_SEED in FORBIDDEN_PRIOR_SEEDS:
        raise RuntimeError("v2 scientific pair seed collides with prior seed")

    frozen_pairs = {
        "train_tokens": (TRAIN_TOKENS, v1.TRAIN_TOKENS),
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
        raise RuntimeError(f"matched Pair-1 training contract drifted: {mismatches}")

    if TOTAL_OPTIMIZER_STEPS * TOKENS_PER_OPTIMIZER_STEP != FULL_BATCH_TOKEN_EXPOSURES:
        raise RuntimeError("2B full-batch exposure formula drifted")
    if FULL_BATCH_TOKEN_EXPOSURES < TRAIN_TOKENS:
        raise RuntimeError("2B pair does not reach target tokens")
    if FULL_BATCH_TOKEN_EXPOSURES - TRAIN_TOKENS >= TOKENS_PER_OPTIMIZER_STEP:
        raise RuntimeError("2B exposure overshoot exceeds one optimizer step")

    candidate = candidate_contract()
    if candidate["candidate_name"] != CANDIDATE_NAME:
        raise RuntimeError("candidate identity drift")
    if candidate["expected_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("candidate parameter drift")

    if ENGINEERING_PANEL_CLASSIFICATION != "ENGINEERING_PANEL_CANDIDATE_SCREEN_PASS":
        raise RuntimeError("engineering prerequisite classification drift")
    if ENGINEERING_PANEL_ATTENTION8_NLL_DELTA > 0.025:
        raise RuntimeError("engineering quality prerequisite no longer passes")
    if ENGINEERING_PANEL_ATTENTION8_TPS_RATIO < 1.05:
        raise RuntimeError("engineering throughput prerequisite no longer passes")

    return {
        "experiment_id": EXPERIMENT_ID,
        "classification": "SCIENTIFIC_PAIR1_V2_ATTENTION8_PREREG_ONLY",
        "candidate": candidate,
        "pair_seed_reserved_not_consumed": PAIR_SEED,
        "matched_models": ["fresh_transformer", CANDIDATE_NAME],
        "train_tokens": TRAIN_TOKENS,
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
        "checkpoint_every_tokens": CHECKPOINT_EVERY_TOKENS,
        "final_eval_batches": FINAL_EVAL_BATCHES,
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
        "pair1_gate": {
            "candidate_minus_transformer_final_nll_max": PAIR_SCREEN_MAX_NLL_DELTA,
            "candidate_over_transformer_training_tps_min_for_progression": (
                MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION
            ),
            "both_complete_exact_geometry_and_finite": True,
        },
        "engineering_evidence": {
            "seed": ENGINEERING_PANEL_SEED,
            "classification": ENGINEERING_PANEL_CLASSIFICATION,
            "attention8_final_nll": ENGINEERING_PANEL_ATTENTION8_FINAL_NLL,
            "transformer_final_nll": ENGINEERING_PANEL_TRANSFORMER_FINAL_NLL,
            "attention8_nll_delta": ENGINEERING_PANEL_ATTENTION8_NLL_DELTA,
            "attention8_training_tps_ratio": ENGINEERING_PANEL_ATTENTION8_TPS_RATIO,
            "not_scientific_evidence": True,
        },
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
        "replication_authorized": False,
        "replication_seeds_authorized": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }
