from __future__ import annotations

from typing import Any, Final

from architectures.cortex_s import reduced_attention_100m_v2_attention8_protocol as prior
from architectures.cortex_s.reduced_attention_100m_v2_attention8 import (
    CANDIDATE_NAME,
    EXPECTED_PARAMETERS,
    candidate_contract,
)


EXPERIMENT_ID: Final = "reduced-attention-v2-attention8-pair1-successor-100m-2b"
PAIR_SEED: Final = 60_232
PRIOR_PAIR_SEED: Final = 60_231
PRIOR_PAIR_TERMINAL_CLASSIFICATION: Final = (
    "SCIENTIFIC_ATTENTION8_V2_PAIR1_INFRA_FAILURE_NO_RETRY"
)
PRIOR_PAIR_IS_SCIENTIFIC_EVIDENCE: Final = False
REQUIRED_MODAL_ACCOUNT: Final = "primary"

TRAIN_TOKENS: Final = prior.TRAIN_TOKENS
VAL_TOKENS: Final = prior.VAL_TOKENS
SEQ_LEN: Final = prior.SEQ_LEN
MICRO_BATCH_SIZE: Final = prior.MICRO_BATCH_SIZE
GRAD_ACCUM_STEPS: Final = prior.GRAD_ACCUM_STEPS
GLOBAL_BATCH: Final = prior.GLOBAL_BATCH
TOKENS_PER_OPTIMIZER_STEP: Final = prior.TOKENS_PER_OPTIMIZER_STEP
TOTAL_OPTIMIZER_STEPS: Final = prior.TOTAL_OPTIMIZER_STEPS
FULL_BATCH_TOKEN_EXPOSURES: Final = prior.FULL_BATCH_TOKEN_EXPOSURES

LEARNING_RATE: Final = prior.LEARNING_RATE
ADAMW_BETAS: Final = prior.ADAMW_BETAS
WEIGHT_DECAY: Final = prior.WEIGHT_DECAY
WARMUP_RATIO: Final = prior.WARMUP_RATIO
GRAD_CLIP: Final = prior.GRAD_CLIP
FINAL_EVAL_BATCHES: Final = prior.FINAL_EVAL_BATCHES
CHECKPOINT_EVERY_TOKENS: Final = prior.CHECKPOINT_EVERY_TOKENS

TRAIN_SHA256: Final = prior.TRAIN_SHA256
VAL_SHA256: Final = prior.VAL_SHA256
META_SHA256: Final = prior.META_SHA256

PAIR_SCREEN_MAX_NLL_DELTA: Final = prior.PAIR_SCREEN_MAX_NLL_DELTA
MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION: Final = (
    prior.MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION
)

ENGINEERING_PANEL_SEED: Final = prior.ENGINEERING_PANEL_SEED
ENGINEERING_PANEL_ATTENTION8_FINAL_NLL: Final = (
    prior.ENGINEERING_PANEL_ATTENTION8_FINAL_NLL
)
ENGINEERING_PANEL_TRANSFORMER_FINAL_NLL: Final = (
    prior.ENGINEERING_PANEL_TRANSFORMER_FINAL_NLL
)
ENGINEERING_PANEL_ATTENTION8_NLL_DELTA: Final = (
    prior.ENGINEERING_PANEL_ATTENTION8_NLL_DELTA
)
ENGINEERING_PANEL_ATTENTION8_TPS_RATIO: Final = (
    prior.ENGINEERING_PANEL_ATTENTION8_TPS_RATIO
)
ENGINEERING_PANEL_CLASSIFICATION: Final = prior.ENGINEERING_PANEL_CLASSIFICATION

FORBIDDEN_PRIOR_SEEDS: Final = tuple(prior.FORBIDDEN_PRIOR_SEEDS) + (PRIOR_PAIR_SEED,)


def validate_protocol() -> dict[str, Any]:
    if PAIR_SEED in FORBIDDEN_PRIOR_SEEDS:
        raise RuntimeError("successor scientific pair seed collides with prior seed")
    if PRIOR_PAIR_SEED not in FORBIDDEN_PRIOR_SEEDS:
        raise RuntimeError("consumed prior Pair-1 seed must remain forbidden")
    if PAIR_SEED == PRIOR_PAIR_SEED:
        raise RuntimeError("successor seed must differ from consumed prior Pair-1 seed")
    if REQUIRED_MODAL_ACCOUNT != "primary":
        raise RuntimeError("successor scientific Pair-1 must be primary-only")
    if PRIOR_PAIR_IS_SCIENTIFIC_EVIDENCE is not False:
        raise RuntimeError("prior infrastructure failure cannot become scientific evidence")

    prior_protocol = prior.validate_protocol()
    frozen_pairs = {
        "train_tokens": (TRAIN_TOKENS, prior.TRAIN_TOKENS),
        "val_tokens": (VAL_TOKENS, prior.VAL_TOKENS),
        "seq_len": (SEQ_LEN, prior.SEQ_LEN),
        "micro_batch_size": (MICRO_BATCH_SIZE, prior.MICRO_BATCH_SIZE),
        "grad_accum_steps": (GRAD_ACCUM_STEPS, prior.GRAD_ACCUM_STEPS),
        "global_batch": (GLOBAL_BATCH, prior.GLOBAL_BATCH),
        "tokens_per_optimizer_step": (
            TOKENS_PER_OPTIMIZER_STEP,
            prior.TOKENS_PER_OPTIMIZER_STEP,
        ),
        "total_optimizer_steps": (
            TOTAL_OPTIMIZER_STEPS,
            prior.TOTAL_OPTIMIZER_STEPS,
        ),
        "full_batch_token_exposures": (
            FULL_BATCH_TOKEN_EXPOSURES,
            prior.FULL_BATCH_TOKEN_EXPOSURES,
        ),
        "learning_rate": (LEARNING_RATE, prior.LEARNING_RATE),
        "adamw_betas": (ADAMW_BETAS, prior.ADAMW_BETAS),
        "weight_decay": (WEIGHT_DECAY, prior.WEIGHT_DECAY),
        "warmup_ratio": (WARMUP_RATIO, prior.WARMUP_RATIO),
        "grad_clip": (GRAD_CLIP, prior.GRAD_CLIP),
        "final_eval_batches": (FINAL_EVAL_BATCHES, prior.FINAL_EVAL_BATCHES),
        "train_sha256": (TRAIN_SHA256, prior.TRAIN_SHA256),
        "val_sha256": (VAL_SHA256, prior.VAL_SHA256),
        "meta_sha256": (META_SHA256, prior.META_SHA256),
        "quality_gate": (PAIR_SCREEN_MAX_NLL_DELTA, prior.PAIR_SCREEN_MAX_NLL_DELTA),
        "systems_gate": (
            MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION,
            prior.MIN_TRAINING_TPS_RATIO_FOR_PROGRESSION,
        ),
    }
    mismatches = {
        key: values
        for key, values in frozen_pairs.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(f"successor Pair-1 contract drifted: {mismatches}")

    if TOTAL_OPTIMIZER_STEPS * TOKENS_PER_OPTIMIZER_STEP != FULL_BATCH_TOKEN_EXPOSURES:
        raise RuntimeError("2B full-batch exposure formula drifted")
    if FULL_BATCH_TOKEN_EXPOSURES < TRAIN_TOKENS:
        raise RuntimeError("2B successor pair does not reach target tokens")
    if FULL_BATCH_TOKEN_EXPOSURES - TRAIN_TOKENS >= TOKENS_PER_OPTIMIZER_STEP:
        raise RuntimeError("2B successor exposure overshoot exceeds one optimizer step")

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
        "classification": "SCIENTIFIC_PAIR1_V2_ATTENTION8_SUCCESSOR_PREREG_ONLY",
        "candidate": candidate,
        "pair_seed_reserved_not_consumed": PAIR_SEED,
        "prior_pair_seed_consumed": PRIOR_PAIR_SEED,
        "prior_pair_terminal_classification": PRIOR_PAIR_TERMINAL_CLASSIFICATION,
        "prior_pair_not_scientific_evidence": not PRIOR_PAIR_IS_SCIENTIFIC_EVIDENCE,
        "required_modal_account": REQUIRED_MODAL_ACCOUNT,
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
        "prior_protocol_classification": prior_protocol["classification"],
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
        "replication_authorized": False,
        "replication_seeds_authorized": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }
