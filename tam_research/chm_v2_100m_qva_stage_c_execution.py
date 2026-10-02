from __future__ import annotations

"""Pure execution contract for CHM-v2 QVA Stage-C run-control #1182.

This module freezes identities, deterministic sample plans, and cost/resource
guards.  It deliberately contains no optimizer, training loop, Modal import,
GPU allocation, checkpoint writer, or scientific-launch authority.
"""

from typing import Any

import torch

from .chm_v1_100m_stage_c_execution import (
    build_start_plan,
    build_validation_start_plan as _build_validation_start_plan,
    start_plan_sha256,
)
from .chm_v2_100m_qva_stage_c import (
    CASES_PER_FAMILY,
    GENERATOR_VERSION,
    OPTIMIZER_STEPS_PER_MODEL,
    PROBE_SEED,
    SCIENTIFIC_SEED,
    TOKENS_PER_OPTIMIZER_STEP,
    TRAINING_TOKENS_PER_MODEL,
    VALIDATION_SEED,
    VALIDATION_TOKENS,
    WARMUP_STEPS,
    validate_protocol_manifest,
)
from .chm_v1_small_lm_protocol import GRAD_ACCUM, MICRO_BATCH, SESSION_LEN

CONTROL_ISSUE = 1182
PREREG_ISSUE = 1176
HYPOTHESIS_ISSUE = 1147
SYSTEMS_ISSUE = 1155

RESULT_ROOT = "/vol/chm-v2/100m-qva-stage-c/issue-1182/seed-2011761-v1"
TRIGGER_TITLE = "[modal-chm-v2-100m-qva-stage-c-1182-seed-2011761-v1]"
AUDIT_TITLE = "[modal-chm-v2-100m-qva-stage-c-1182-authority-audit-v1]"

TRAIN_STREAM_GENERATOR_SEED = 2_021_761
GPU_CLASS = "L4"
CPU_CORES = 4
RAM_GIB = 16
MAX_GPU_SECONDS = 10_800
MAX_BILLED_COMPUTE_USD = 3.00

QVA_BLOB = "a34a8dffc2c2c1702a909782c3aba2593e90947c"
STAGE_C_PROTOCOL_BLOB = "05e905937ff3a39276462a47fe93e8a499682d1a"
MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
SMALL_PROTOCOL_BLOB = "d5c2e405b5306f556e7fbe70aacd552867e69d3e"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"
DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"
RUNTIME_PROBE_BLOB = "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc"

TRIGGER_AUTHORIZED_BY_MODULE = False
GPU_ALLOCATION_AUTHORIZED_BY_MODULE = False
SCIENTIFIC_SEED_CONSUMED_BY_MODULE = False
REPLICATION_AUTHORIZED_BY_MODULE = False
STAGE_D_AUTHORIZED_BY_MODULE = False


def validate_execution_contract() -> dict[str, Any]:
    protocol = validate_protocol_manifest()
    if SCIENTIFIC_SEED != 2_011_761:
        raise RuntimeError("#1182 scientific seed drift")
    if TRAIN_STREAM_GENERATOR_SEED != SCIENTIFIC_SEED + 10_000:
        raise RuntimeError("#1182 training-plan seed derivation drift")
    if TRAINING_TOKENS_PER_MODEL != 33_554_432:
        raise RuntimeError("#1182 training token budget drift")
    if OPTIMIZER_STEPS_PER_MODEL != 2_048:
        raise RuntimeError("#1182 optimizer-step count drift")
    if WARMUP_STEPS != 40 or TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#1182 optimizer schedule drift")
    if (MICRO_BATCH, GRAD_ACCUM, SESSION_LEN) != (4, 4, 1024):
        raise RuntimeError("#1182 batching/session geometry drift")
    if PROBE_SEED != 977_301 or VALIDATION_SEED != 977_302:
        raise RuntimeError("#1182 inherited evaluator seed drift")
    if VALIDATION_TOKENS != 1_048_576 or CASES_PER_FAMILY != 128:
        raise RuntimeError("#1182 evaluator envelope drift")
    if GENERATOR_VERSION != "chm-v1-100m-heldout-aligned-v4":
        raise RuntimeError("#1182 aligned-v4 generator drift")
    if (GPU_CLASS, CPU_CORES, RAM_GIB) != ("L4", 4, 16):
        raise RuntimeError("#1182 resource geometry drift")
    if MAX_GPU_SECONDS != 10_800 or MAX_BILLED_COMPUTE_USD != 3.00:
        raise RuntimeError("#1182 runtime/cost envelope drift")
    if any(
        (
            TRIGGER_AUTHORIZED_BY_MODULE,
            GPU_ALLOCATION_AUTHORIZED_BY_MODULE,
            SCIENTIFIC_SEED_CONSUMED_BY_MODULE,
            REPLICATION_AUTHORIZED_BY_MODULE,
            STAGE_D_AUTHORIZED_BY_MODULE,
        )
    ):
        raise RuntimeError("#1182 pure execution contract must never self-authorize")
    return {
        "classification": "CHM_V2_100M_QVA_STAGE_C_RUN_CONTROL_TRIGGER_WITHHELD",
        "control_issue": CONTROL_ISSUE,
        "preregistration_issue": PREREG_ISSUE,
        "hypothesis_issue": HYPOTHESIS_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "scientific_seed_reserved": SCIENTIFIC_SEED,
        "train_stream_generator_seed": TRAIN_STREAM_GENERATOR_SEED,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "audit_title": AUDIT_TITLE,
        "training_tokens_per_model": TRAINING_TOKENS_PER_MODEL,
        "optimizer_steps_per_model": OPTIMIZER_STEPS_PER_MODEL,
        "warmup_steps": WARMUP_STEPS,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "gpu_class": GPU_CLASS,
        "cpu_cores": CPU_CORES,
        "ram_gib": RAM_GIB,
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "max_billed_compute_usd": MAX_BILLED_COMPUTE_USD,
        "protocol": protocol,
        "trigger_authorized_by_module": False,
        "gpu_allocation_authorized_by_module": False,
        "scientific_seed_consumed_by_module": False,
        "replication_authorized_by_module": False,
        "stage_d_authorized_by_module": False,
    }


def build_training_start_plan(train_tokens: int) -> torch.Tensor:
    plan = build_start_plan(
        shard_tokens=int(train_tokens),
        seq_len=SESSION_LEN,
        steps=OPTIMIZER_STEPS_PER_MODEL,
        batches_per_step=GRAD_ACCUM,
        batch_size=MICRO_BATCH,
        seed=TRAIN_STREAM_GENERATOR_SEED,
    )
    expected = (OPTIMIZER_STEPS_PER_MODEL, GRAD_ACCUM, MICRO_BATCH)
    if tuple(plan.shape) != expected or plan.numel() != 32_768:
        raise RuntimeError(f"#1182 training-plan shape drift: {tuple(plan.shape)}")
    return plan


def build_validation_start_plan(val_tokens: int) -> torch.Tensor:
    return _build_validation_start_plan(int(val_tokens))


def validate_live_rate_cap(hourly_resource_usd: float) -> dict[str, float]:
    hourly = float(hourly_resource_usd)
    if not (hourly > 0.0):
        raise RuntimeError("#1182 live resource rate must be positive")
    worst = hourly * (MAX_GPU_SECONDS / 3600.0)
    if worst > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError(
            f"#1182 live 3h worst-case ${worst:.6f} exceeds "
            f"${MAX_BILLED_COMPUTE_USD:.2f} cap"
        )
    return {
        "hourly_resource_usd": hourly,
        "worst_case_usd": worst,
        "max_gpu_seconds": float(MAX_GPU_SECONDS),
        "max_billed_compute_usd": float(MAX_BILLED_COMPUTE_USD),
    }


__all__ = [
    "AUDIT_TITLE",
    "CONTROL_ISSUE",
    "DUAL_ACCOUNT_BLOB",
    "DUAL_ACCOUNT_CLI_BLOB",
    "EVALUATOR_BLOB",
    "GPU_CLASS",
    "MAX_BILLED_COMPUTE_USD",
    "MAX_GPU_SECONDS",
    "MODEL_BLOB",
    "QVA_BLOB",
    "RESULT_ROOT",
    "RUNTIME_PROBE_BLOB",
    "SCIENTIFIC_SEED",
    "SMALL_PROTOCOL_BLOB",
    "STAGE_C_PROTOCOL_BLOB",
    "TRAIN_STREAM_GENERATOR_SEED",
    "TRIGGER_TITLE",
    "build_training_start_plan",
    "build_validation_start_plan",
    "start_plan_sha256",
    "validate_execution_contract",
    "validate_live_rate_cap",
]
