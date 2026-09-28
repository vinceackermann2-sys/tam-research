from __future__ import annotations

"""CHM-v2 ~100M value-projected EIEM run-control contract (#1115).

This module freezes the one-seed three-way LOCAL/RAW/VP development execution
envelope. It exposes deterministic CPU helpers and fail-closed governance
checks only; it does not create triggers, allocate GPUs, or authorize seed use.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .chm_v1_100m_scale import EXPECTED_EIEM_PARAMETERS, EXPECTED_LOCAL_PARAMETERS
from .chm_v1_100m_stage_c_eval import (
    CASES_PER_FAMILY,
    GENERATOR_VERSION,
    LONG_RANGE_FAMILIES,
    PROBE_SEED,
    TOTAL_PROBES,
    VALIDATION_BATCHES,
    VALIDATION_BATCH_SIZE,
    VALIDATION_SEED,
    VALIDATION_SESSION_LEN,
    VALIDATION_TOKENS,
)
from .chm_v1_100m_stage_c_execution import (
    build_training_start_plan,
    build_validation_start_plan,
    start_plan_sha256,
)
from .chm_v1_100m_stage_c_run_control_prep import (
    BETAS,
    CHECKPOINT_STEPS,
    GRAD_ACCUM,
    GRAD_CLIP,
    MICRO_BATCH,
    PEAK_LR,
    SESSION_LEN,
    TOKENS_PER_OPTIMIZER_STEP,
    TRAINING_TOKENS_PER_MODEL,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from .chm_v1_small_lm import LOCAL_WINDOW, RETRIEVAL_HOPS
from .chm_v1_small_lm_protocol import FLAT_TRAIN_TEMPERATURE
from .chm_v2_100m_value_projected_eiem import (
    EXPECTED_VP_DELTA_VS_LOCAL,
    EXPECTED_VP_EIEM_PARAMETERS,
    RESERVED_DEVELOPMENT_SEED,
    VALUE_RANK,
    validate_architecture_contract,
)

CONTROL_ISSUE = 1115
PREREG_ISSUE = 1112
IMPLEMENTATION_PR = 1113

ARCHITECTURE_BASE_SHA = "fb8afc3ba1cc155201c260287be88c194dd97cd3"
ARCHITECTURE_BASE_TREE = "4d26fa23a092af7cca1370773041ad06bc3e4bc7"

CHM_V2_MODULE_BLOB = "d0c2186231cead2a7c851d776d21e6ab6f73ec43"
MODEL_CONTRACT_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
STAGE_C_EXECUTION_BLOB = "26668b37a6062c641275e177b622948b36d0f227"
STAGE_C_PREP_BLOB = "bc4ed60885aaf991a2d6b9fe8f634ff973f3f722"
DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"
DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"
RUNTIME_ADMISSION_PROBE_BLOB = "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc"

SCIENTIFIC_SEED = RESERVED_DEVELOPMENT_SEED
PERMANENTLY_CONSUMED_CHM_V1_SEED = 977_001
TRAIN_STREAM_GENERATOR_SEED = 987_001

PHASE = "chm-v2-100m-value-projected-eiem-1115-seed-2011121-v2"
RESULT_ROOT = "/vol/chm-v2/100m-value-projected-eiem/issue-1115/seed-2011121-v2"
TRIGGER_TITLE = "[modal-chm-v2-100m-value-projected-eiem-1115-seed-2011121-v2]"
AUDIT_TITLE = "[modal-chm-v2-100m-value-projected-eiem-1115-authority-audit-v3]"

DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
VOLUME_NAME = "tam-research-data"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_GIB = 16
MAX_GPU_SECONDS = 14_400
MAX_COMPUTE_USD = 6.00
RETRIES = 0

CLASSIFICATION = "CHM_V2_100M_VALUE_PROJECTED_EIEM_RUN_CONTROL_IMPLEMENTATION_TRIGGER_WITHHELD"


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "control_issue": CONTROL_ISSUE,
        "prereg_issue": PREREG_ISSUE,
        "implementation_pr": IMPLEMENTATION_PR,
        "architecture_base_sha": ARCHITECTURE_BASE_SHA,
        "architecture_base_tree": ARCHITECTURE_BASE_TREE,
        "scientific_seed": SCIENTIFIC_SEED,
        "scientific_seed_authorized": False,
        "permanently_consumed_chm_v1_seed": PERMANENTLY_CONSUMED_CHM_V1_SEED,
        "phase": PHASE,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "audit_title": AUDIT_TITLE,
        "data_dir": DATA_DIR,
        "volume_name": VOLUME_NAME,
        "resource_envelope": {
            "gpu": GPU_CLASS,
            "cpu_cores": CPU_CORES,
            "ram_gib": RAM_GIB,
            "max_gpu_seconds": MAX_GPU_SECONDS,
            "max_compute_usd": MAX_COMPUTE_USD,
            "retries": RETRIES,
        },
        "training": {
            "tokens_per_model": TRAINING_TOKENS_PER_MODEL,
            "optimizer_steps": TRAINING_TOKENS_PER_MODEL // TOKENS_PER_OPTIMIZER_STEP,
            "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
            "warmup_steps": WARMUP_STEPS,
            "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM,
            "session_len": SESSION_LEN,
            "local_window": LOCAL_WINDOW,
            "retrieval_hops": RETRIEVAL_HOPS,
            "soft_temperature": FLAT_TRAIN_TEMPERATURE,
            "betas": tuple(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "peak_lr": PEAK_LR,
            "grad_clip": GRAD_CLIP,
            "training_stream_seed": TRAIN_STREAM_GENERATOR_SEED,
            "checkpoint_steps": tuple(CHECKPOINT_STEPS),
        },
        "evaluation": {
            "generator_version": GENERATOR_VERSION,
            "probe_seed": PROBE_SEED,
            "cases_per_family": CASES_PER_FAMILY,
            "total_probes": TOTAL_PROBES,
            "validation_seed": VALIDATION_SEED,
            "validation_tokens": VALIDATION_TOKENS,
            "validation_batches": VALIDATION_BATCHES,
            "validation_batch_size": VALIDATION_BATCH_SIZE,
            "validation_session_len": VALIDATION_SESSION_LEN,
        },
        "parameters": {
            "local": EXPECTED_LOCAL_PARAMETERS,
            "raw_eiem": EXPECTED_EIEM_PARAMETERS,
            "vp_eiem": EXPECTED_VP_EIEM_PARAMETERS,
            "vp_delta_fraction_vs_local": EXPECTED_VP_DELTA_VS_LOCAL,
            "value_rank": VALUE_RANK,
        },
        "trigger_authorized_by_module": False,
        "gpu_allocation_authorized_by_module": False,
        "scientific_seed_consumed_by_module": False,
        "stage_d_authorized": False,
        "scale_up_authorized": False,
        "multi_seed_replication_authorized": False,
    }


def validate_contract() -> dict[str, Any]:
    architecture = validate_architecture_contract()
    manifest = protocol_manifest()
    if CONTROL_ISSUE != 1115 or PREREG_ISSUE != 1112:
        raise RuntimeError("#1115 issue binding drift")
    if SCIENTIFIC_SEED != 2_011_121:
        raise RuntimeError("#1115 reserved development seed drift")
    if SCIENTIFIC_SEED == PERMANENTLY_CONSUMED_CHM_V1_SEED:
        raise RuntimeError("#1115 seed aliases consumed CHM-v1 seed")
    if TRAINING_TOKENS_PER_MODEL != 33_554_432:
        raise RuntimeError("#1115 token budget drift")
    if TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#1115 tokens/optimizer-step drift")
    if TRAINING_TOKENS_PER_MODEL // TOKENS_PER_OPTIMIZER_STEP != 2_048:
        raise RuntimeError("#1115 optimizer-step count drift")
    if WARMUP_STEPS != 40:
        raise RuntimeError("#1115 warmup drift")
    if tuple(CHECKPOINT_STEPS) != (512, 1024, 1536, 2048):
        raise RuntimeError("#1115 checkpoint schedule drift")
    if (MICRO_BATCH, GRAD_ACCUM, SESSION_LEN) != (4, 4, 1024):
        raise RuntimeError("#1115 batching/session drift")
    if (LOCAL_WINDOW, RETRIEVAL_HOPS, VALUE_RANK) != (512, 2, 32):
        raise RuntimeError("#1115 memory geometry drift")
    if FLAT_TRAIN_TEMPERATURE != 0.10:
        raise RuntimeError("#1115 soft retrieval temperature drift")
    if tuple(BETAS) != (0.9, 0.95):
        raise RuntimeError("#1115 AdamW beta drift")
    if (WEIGHT_DECAY, PEAK_LR, GRAD_CLIP) != (0.1, 3e-4, 1.0):
        raise RuntimeError("#1115 optimizer hyperparameter drift")
    if TRAIN_STREAM_GENERATOR_SEED != 987_001:
        raise RuntimeError("#1115 paired training-stream seed drift")
    if GENERATOR_VERSION != "chm-v1-100m-heldout-aligned-v4":
        raise RuntimeError("#1115 evaluator generator drift")
    if (PROBE_SEED, CASES_PER_FAMILY, TOTAL_PROBES) != (977_301, 128, 512):
        raise RuntimeError("#1115 probe envelope drift")
    if (
        VALIDATION_SEED,
        VALIDATION_TOKENS,
        VALIDATION_BATCHES,
        VALIDATION_BATCH_SIZE,
        VALIDATION_SESSION_LEN,
    ) != (977_302, 1_048_576, 128, 8, 1024):
        raise RuntimeError("#1115 validation envelope drift")
    if EXPECTED_VP_EIEM_PARAMETERS != 101_870_112:
        raise RuntimeError("#1115 VP parameter count drift")
    if abs(float(EXPECTED_VP_DELTA_VS_LOCAL)) > 0.001:
        raise RuntimeError("#1115 VP parameter fairness drift")
    if (GPU_CLASS, CPU_CORES, RAM_GIB, MAX_GPU_SECONDS, RETRIES) != ("L4", 4, 16, 14_400, 0):
        raise RuntimeError("#1115 resource envelope drift")
    if MAX_COMPUTE_USD != 6.00:
        raise RuntimeError("#1115 compute cap drift")
    return {**manifest, "architecture_contract": architecture}


def validate_seed_for_preparation(seed: int, *, request_execution: bool = False) -> int:
    value = int(seed)
    if request_execution:
        raise RuntimeError("#1115 implementation grants no scientific execution authority")
    if value == PERMANENTLY_CONSUMED_CHM_V1_SEED:
        raise RuntimeError("#1115 permanently consumed CHM-v1 seed refused")
    if value != SCIENTIFIC_SEED:
        raise RuntimeError(f"#1115 recognizes only reserved seed {SCIENTIFIC_SEED}")
    return value


def assert_no_execution_authority(
    *,
    gpu: bool = False,
    training: bool = False,
    trigger_creation: bool = False,
    scientific_seed_consumption: bool = False,
    stage_d: bool = False,
) -> None:
    requested = {
        "gpu": bool(gpu),
        "training": bool(training),
        "trigger_creation": bool(trigger_creation),
        "scientific_seed_consumption": bool(scientific_seed_consumption),
        "stage_d": bool(stage_d),
    }
    active = [name for name, enabled in requested.items() if enabled]
    if active:
        raise RuntimeError(f"#1115 implementation refuses execution authority: {active}")


def projected_worst_case_compute_usd(live_hourly_resource_usd: float) -> float:
    hourly = float(live_hourly_resource_usd)
    if hourly <= 0.0:
        raise ValueError("live hourly resource rate must be positive")
    return hourly * (MAX_GPU_SECONDS / 3600.0)


def validate_live_rate_cap(live_hourly_resource_usd: float) -> dict[str, float | bool]:
    projected = projected_worst_case_compute_usd(live_hourly_resource_usd)
    if projected > MAX_COMPUTE_USD:
        raise RuntimeError(
            f"#1115 execution must fail closed: 4h worst-case USD {projected:.6f} "
            f"> USD {MAX_COMPUTE_USD:.2f}"
        )
    return {
        "live_hourly_resource_usd": float(live_hourly_resource_usd),
        "worst_case_compute_usd": projected,
        "within_cap": True,
    }


def training_plan_for_tokens(train_tokens: int):
    plan = build_training_start_plan(int(train_tokens))
    if tuple(plan.shape) != (2_048, 4, 4):
        raise RuntimeError("#1115 training plan shape drift")
    return plan


def validation_plan_for_tokens(val_tokens: int):
    plan = build_validation_start_plan(int(val_tokens))
    if tuple(plan.shape) != (128, 1, 8):
        raise RuntimeError("#1115 validation plan shape drift")
    return plan


def plan_digest(plan) -> str:
    return start_plan_sha256(plan)


def summarize_three_way_probe_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != TOTAL_PROBES:
        raise RuntimeError(f"#1115 expected {TOTAL_PROBES} probe rows, got {len(rows)}")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        family = str(row["family"])
        grouped.setdefault(family, []).append(row)
    required = ("rare_fact", "overwrite", "two_hop", "local_negative")
    if tuple(sorted(grouped)) != tuple(sorted(required)):
        raise RuntimeError(f"#1115 probe family drift: {tuple(sorted(grouped))}")
    if any(len(grouped[name]) != CASES_PER_FAMILY for name in required):
        raise RuntimeError("#1115 probe family count drift")

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        count = float(len(items))
        local_acc = sum(float(bool(x["local_correct"])) for x in items) / count
        raw_acc = sum(float(bool(x["raw_correct"])) for x in items) / count
        vp_acc = sum(float(bool(x["vp_correct"])) for x in items) / count
        local_nll = sum(float(x["local_candidate_nll"]) for x in items) / count
        raw_nll = sum(float(x["raw_candidate_nll"]) for x in items) / count
        vp_nll = sum(float(x["vp_candidate_nll"]) for x in items) / count
        local_stale = sum(float(bool(x["local_stale_choice"])) for x in items) / count
        raw_stale = sum(float(bool(x["raw_stale_choice"])) for x in items) / count
        vp_stale = sum(float(bool(x["vp_stale_choice"])) for x in items) / count
        return {
            "local_accuracy": local_acc,
            "raw_accuracy": raw_acc,
            "vp_accuracy": vp_acc,
            "local_candidate_nll": local_nll,
            "raw_candidate_nll": raw_nll,
            "vp_candidate_nll": vp_nll,
            "local_stale_rate": local_stale,
            "raw_stale_rate": raw_stale,
            "vp_stale_rate": vp_stale,
            "vp_vs_local_accuracy_gain": vp_acc - local_acc,
            "vp_vs_local_nll_benefit": local_nll - vp_nll,
            "vp_vs_raw_accuracy_gain": vp_acc - raw_acc,
            "vp_vs_raw_nll_benefit": raw_nll - vp_nll,
        }

    per_family = {name: summarize(grouped[name]) for name in required}
    long_rows = [
        row
        for name in LONG_RANGE_FAMILIES
        for row in grouped[name]
    ]
    aggregate = summarize(long_rows)
    return {
        "aggregate_long_range": aggregate,
        "per_family": per_family,
        "local_negative": per_family["local_negative"],
        "overwrite": per_family["overwrite"],
    }
