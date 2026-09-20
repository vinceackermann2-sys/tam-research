from __future__ import annotations

from typing import Any

from architectures.cortex_s.experiments.scale100m_2b import protocol as base
from architectures.cortex_s.reduced_attention_100m_v1 import (
    EXPECTED_REDUCED_ATTENTION_PARAMETERS,
    EXPECTED_TRANSFORMER_PARAMETERS,
    architecture_contract,
    parameter_contract,
)


EXPERIMENT_ID = "reduced-attention-dense-v1-100m-2b-matched-quality"
PREREG_ISSUE = 1013

SCIENTIFIC_PAIR_SEEDS = (58_231, 58_232, 58_233)
ENGINEERING_CALIBRATION_SEED = 2_026_092_001
FORBIDDEN_SEEDS = (8_100, 48_131, 48_132, 48_133)

PAIR1_SCREEN_MAX_NLL_DELTA = 0.015
THREE_PAIR_MAX_MEAN_NLL_DELTA = 0.005
THREE_PAIR_MAX_SINGLE_NLL_DELTA = 0.020

TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
GLOBAL_BATCH = 128
TOKENS_PER_OPTIMIZER_STEP = 65_536
TOTAL_OPTIMIZER_STEPS = 30_518
FULL_BATCH_TOKEN_EXPOSURES = 2_000_027_648
LEARNING_RATE = 3e-4
ADAMW_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.02
GRAD_CLIP = 1.0
EVAL_EVERY_TOKENS = 200_000_000
CHECKPOINT_EVERY_TOKENS = 200_000_000
FINAL_EVAL_BATCHES = 50

DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"


def validate_protocol() -> dict[str, Any]:
    params = parameter_contract()

    frozen_pairs = {
        "train_tokens": (TRAIN_TOKENS, base.TRAIN_TOKENS),
        "val_tokens": (VAL_TOKENS, base.VAL_TOKENS),
        "seq_len": (SEQ_LEN, base.SEQ_LEN),
        "micro_batch_size": (MICRO_BATCH_SIZE, base.MICRO_BATCH_SIZE),
        "grad_accum_steps": (GRAD_ACCUM_STEPS, base.GRAD_ACCUM_STEPS),
        "global_batch": (GLOBAL_BATCH, base.GLOBAL_BATCH),
        "tokens_per_optimizer_step": (
            TOKENS_PER_OPTIMIZER_STEP,
            base.TOKENS_PER_OPTIMIZER_STEP,
        ),
        "total_optimizer_steps": (
            TOTAL_OPTIMIZER_STEPS,
            base.TOTAL_OPTIMIZER_STEPS,
        ),
        "full_batch_token_exposures": (
            FULL_BATCH_TOKEN_EXPOSURES,
            base.FULL_BATCH_TOKEN_EXPOSURES,
        ),
        "learning_rate": (LEARNING_RATE, base.LEARNING_RATE),
        "weight_decay": (WEIGHT_DECAY, base.WEIGHT_DECAY),
        "warmup_ratio": (WARMUP_RATIO, base.WARMUP_RATIO),
        "eval_every_tokens": (EVAL_EVERY_TOKENS, base.EVAL_EVERY_TOKENS),
        "checkpoint_every_tokens": (
            CHECKPOINT_EVERY_TOKENS,
            base.CHECKPOINT_EVERY_TOKENS,
        ),
        "train_sha256": (TRAIN_SHA256, base.TRAIN_SHA256),
        "val_sha256": (VAL_SHA256, base.VAL_SHA256),
        "meta_sha256": (META_SHA256, base.META_SHA256),
    }
    mismatches = {
        key: values
        for key, values in frozen_pairs.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(f"base 100M/2B protocol drifted: {mismatches}")

    if ADAMW_BETAS != (0.9, 0.95):
        raise RuntimeError("AdamW betas drifted")
    if GRAD_CLIP != 1.0:
        raise RuntimeError("grad clip drifted")
    if FINAL_EVAL_BATCHES != 50:
        raise RuntimeError("final eval batch count drifted")
    if set(SCIENTIFIC_PAIR_SEEDS) & set(FORBIDDEN_SEEDS):
        raise RuntimeError("scientific pair seed collides with forbidden seed")
    if ENGINEERING_CALIBRATION_SEED in {
        *SCIENTIFIC_PAIR_SEEDS,
        *FORBIDDEN_SEEDS,
    }:
        raise RuntimeError("engineering calibration seed collides with reserved seed")
    if len(set(SCIENTIFIC_PAIR_SEEDS)) != 3:
        raise RuntimeError("scientific pair seeds must be unique")
    if EXPECTED_TRANSFORMER_PARAMETERS != base.EXPECTED_TRANSFORMER_PARAMS:
        raise RuntimeError("Transformer parameter target drifted")
    if EXPECTED_REDUCED_ATTENTION_PARAMETERS != 101_799_424:
        raise RuntimeError("candidate parameter target drifted")

    return {
        "experiment_id": EXPERIMENT_ID,
        "prereg_issue": PREREG_ISSUE,
        "architecture": architecture_contract(),
        "scientific_pair_seeds_reserved_not_consumed": list(
            SCIENTIFIC_PAIR_SEEDS
        ),
        "engineering_calibration_seed_reserved_not_consumed": (
            ENGINEERING_CALIBRATION_SEED
        ),
        "forbidden_seeds": list(FORBIDDEN_SEEDS),
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "global_batch": GLOBAL_BATCH,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
        "learning_rate": LEARNING_RATE,
        "adamw_betas": list(ADAMW_BETAS),
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "grad_clip": GRAD_CLIP,
        "eval_every_tokens": EVAL_EVERY_TOKENS,
        "checkpoint_every_tokens": CHECKPOINT_EVERY_TOKENS,
        "final_eval_batches": FINAL_EVAL_BATCHES,
        "data_dir": DATA_DIR,
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "meta_sha256": META_SHA256,
        "pair1_screen_max_nll_delta": PAIR1_SCREEN_MAX_NLL_DELTA,
        "three_pair_max_mean_nll_delta": THREE_PAIR_MAX_MEAN_NLL_DELTA,
        "three_pair_max_single_nll_delta": THREE_PAIR_MAX_SINGLE_NLL_DELTA,
        "training_authorized": False,
        "gpu_authorized": False,
        "scientific_seeds_consumed": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }
