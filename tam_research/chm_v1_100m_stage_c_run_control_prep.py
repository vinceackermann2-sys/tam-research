from __future__ import annotations

"""CHM-v1 ~100M Stage-C run-control preparation contract (#988).

CPU/CI-only preparation. This module freezes the future one-shot scientific
execution envelope and provenance requirements without providing a launcher,
trigger, GPU path, paid-compute path, or scientific execution authority.
"""

from typing import Any

from .chm_v1_100m_scale import (
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    FIRST_SCREEN_TOKEN_BUDGET,
    FUTURE_SCREEN_SEED,
)
from .chm_v1_100m_stage_c_eval import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CASES_PER_FAMILY,
    GENERATOR_VERSION,
    LONG_RANGE_PROBES,
    PROBE_SEED,
    TOTAL_PROBES,
    VALIDATION_SEED,
    VALIDATION_TOKENS,
)
from .chm_v1_small_lm_protocol import (
    BETAS,
    COMPILE_ENABLED,
    GRAD_ACCUM,
    GRAD_CLIP,
    MICRO_BATCH,
    PEAK_LR,
    SESSION_LEN,
    WEIGHT_DECAY,
)

PARENT_RESEARCH_ISSUE = 977
EVALUATOR_ISSUE = 986
RUN_CONTROL_PREP_ISSUE = 988

STARTING_MAIN_SHA = "72e432f3a0bbf1fb6a27da928f0078dab9491a73"
STARTING_MAIN_TREE = "ae017fff0156fd9409025a672dd314d776285c7a"
MODEL_CONTRACT_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"

SCIENTIFIC_SEED = 977_001
CONSUMED_STAGE_B_SEED = 977_201
STAGE_A_SMOKE_SEED = 977_099
EVALUATOR_ONLY_SEEDS = (PROBE_SEED, VALIDATION_SEED, BOOTSTRAP_SEED)
BLOCKED_NONSCIENTIFIC_SEEDS = (
    CONSUMED_STAGE_B_SEED,
    STAGE_A_SMOKE_SEED,
) + EVALUATOR_ONLY_SEEDS

TOKENS_PER_OPTIMIZER_STEP = MICRO_BATCH * SESSION_LEN * GRAD_ACCUM
TRAINING_TOKENS_PER_MODEL = FIRST_SCREEN_TOKEN_BUDGET
PAIR_TRAINING_TOKENS = 2 * TRAINING_TOKENS_PER_MODEL
OPTIMIZER_STEPS_PER_MODEL = TRAINING_TOKENS_PER_MODEL // TOKENS_PER_OPTIMIZER_STEP
WARMUP_STEPS = max(1, int(OPTIMIZER_STEPS_PER_MODEL * 0.02))
CHECKPOINT_STEPS = (512, 1024, 1536, 2048)
FINAL_EVALUATION_STEP = 2048
TRAIN_STREAM_GENERATOR_SEED = SCIENTIFIC_SEED + 10_000

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_GIB = 16
MAX_GPU_SECONDS = 43_200
MAX_STAGE_C_COMPUTE_USD = 25.00
RETRIES = 0

RESULT_ROOT = "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"
FUTURE_TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-988-seed-977001-v1]"
PHASE = "chm-v1-100m-stage-c-988-seed-977001-v1"

CLASSIFICATION = "CHM_V1_100M_STAGE_C_RUN_CONTROL_PREPARATION_NO_EXECUTION_AUTHORITY"


def validate_seed_for_preparation(seed: int, *, request_execution: bool = False) -> int:
    seed = int(seed)
    if request_execution:
        raise RuntimeError("#988 preparation grants no scientific execution authority")
    if seed in BLOCKED_NONSCIENTIFIC_SEEDS:
        raise RuntimeError(f"non-scientific/consumed seed refused by #988: {seed}")
    if seed != SCIENTIFIC_SEED:
        raise RuntimeError(f"#988 recognizes only future scientific seed {SCIENTIFIC_SEED}")
    if seed != FUTURE_SCREEN_SEED:
        raise RuntimeError("#988 scientific seed drifted from frozen #977 Stage-A contract")
    return seed


def checkpoint_role(step: int) -> str:
    step = int(step)
    if step not in CHECKPOINT_STEPS:
        raise ValueError(f"step {step} is outside the frozen #988 checkpoint schedule")
    return "final-scientific-evaluation" if step == FINAL_EVALUATION_STEP else "audit-only"


def projected_worst_case_compute_usd(live_hourly_resource_usd: float) -> float:
    hourly = float(live_hourly_resource_usd)
    if hourly <= 0:
        raise ValueError("live hourly resource rate must be positive")
    return hourly * (MAX_GPU_SECONDS / 3600.0)


def validate_live_rate_cap(live_hourly_resource_usd: float) -> dict[str, float | bool]:
    projected = projected_worst_case_compute_usd(live_hourly_resource_usd)
    if projected > MAX_STAGE_C_COMPUTE_USD:
        raise RuntimeError(
            f"#988 future execution must fail closed: 12h worst-case ${projected:.6f} "
            f"> ${MAX_STAGE_C_COMPUTE_USD:.2f}"
        )
    return {
        "live_hourly_resource_usd": float(live_hourly_resource_usd),
        "worst_case_compute_usd": projected,
        "within_cap": True,
    }


def validate_source_bindings(
    *,
    main_sha: str,
    main_tree: str,
    model_contract_blob: str,
    evaluator_blob: str,
) -> None:
    expected = {
        "main_sha": STARTING_MAIN_SHA,
        "main_tree": STARTING_MAIN_TREE,
        "model_contract_blob": MODEL_CONTRACT_BLOB,
        "evaluator_blob": EVALUATOR_BLOB,
    }
    observed = {
        "main_sha": str(main_sha),
        "main_tree": str(main_tree),
        "model_contract_blob": str(model_contract_blob),
        "evaluator_blob": str(evaluator_blob),
    }
    if observed != expected:
        raise RuntimeError(f"#988 frozen source binding mismatch: {observed} != {expected}")


def assert_no_execution_authority(
    *,
    gpu: bool = False,
    modal: bool = False,
    paid_compute: bool = False,
    training: bool = False,
    trigger_creation: bool = False,
    scientific_seed_consumption: bool = False,
) -> None:
    requested = {
        "gpu": bool(gpu),
        "modal": bool(modal),
        "paid_compute": bool(paid_compute),
        "training": bool(training),
        "trigger_creation": bool(trigger_creation),
        "scientific_seed_consumption": bool(scientific_seed_consumption),
    }
    active = [name for name, enabled in requested.items() if enabled]
    if active:
        raise RuntimeError(f"#988 preparation refuses execution authority: {active}")


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "parent_research_issue": PARENT_RESEARCH_ISSUE,
        "evaluator_issue": EVALUATOR_ISSUE,
        "run_control_prep_issue": RUN_CONTROL_PREP_ISSUE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "starting_main_tree": STARTING_MAIN_TREE,
        "source_bindings": {
            "model_contract_blob": MODEL_CONTRACT_BLOB,
            "evaluator_blob": EVALUATOR_BLOB,
        },
        "models": {
            "local_trainable_parameters": EXPECTED_LOCAL_PARAMETERS,
            "eiem_flat_trainable_parameters": EXPECTED_EIEM_PARAMETERS,
        },
        "scientific_seed_future_eligible_not_authorized": SCIENTIFIC_SEED,
        "consumed_stage_b_seed_never_reuse": CONSUMED_STAGE_B_SEED,
        "training": {
            "tokens_per_model": TRAINING_TOKENS_PER_MODEL,
            "pair_tokens_total": PAIR_TRAINING_TOKENS,
            "session_len": SESSION_LEN,
            "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM,
            "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
            "optimizer_steps_per_model": OPTIMIZER_STEPS_PER_MODEL,
            "warmup_steps": WARMUP_STEPS,
            "optimizer": "AdamW",
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "peak_lr": PEAK_LR,
            "grad_clip": GRAD_CLIP,
            "compile_enabled": COMPILE_ENABLED,
            "bf16_autocast_future_execution": True,
            "train_stream_generator_seed": TRAIN_STREAM_GENERATOR_SEED,
            "byte_identical_pair_stream_required": True,
        },
        "checkpoints": {
            "steps": list(CHECKPOINT_STEPS),
            "final_evaluation_step": FINAL_EVALUATION_STEP,
            "intermediate_role": "audit-only",
            "resume_from_checkpoint_authorized": False,
            "model_selection_authorized": False,
        },
        "evaluator": {
            "generator_version": GENERATOR_VERSION,
            "probe_seed": PROBE_SEED,
            "validation_seed": VALIDATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cases_per_family": CASES_PER_FAMILY,
            "total_probes": TOTAL_PROBES,
            "long_range_probes": LONG_RANGE_PROBES,
            "validation_tokens": VALIDATION_TOKENS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "future_resource_envelope_not_authorized": {
            "provider": "Modal",
            "gpu": "1x NVIDIA L4",
            "cpu_cores": CPU_CORES,
            "ram_gib": RAM_GIB,
            "max_gpu_seconds": MAX_GPU_SECONDS,
            "max_compute_usd": MAX_STAGE_C_COMPUTE_USD,
            "retries": RETRIES,
            "live_rate_check_required_before_allocation": True,
        },
        "future_result_root_reserved_not_dispatched": RESULT_ROOT,
        "future_trigger_title_reserved_not_created": FUTURE_TRIGGER_TITLE,
        "phase": PHASE,
        "one_shot_guards_required": {
            "actions_run_attempt_must_equal_one": True,
            "durable_dispatch_marker_before_allocation": True,
            "duplicate_trigger_refused": True,
            "duplicate_result_root_refused": True,
            "automatic_retry_refused": True,
            "resume_after_consumption_refused": True,
            "gpu_entry_consumes_scientific_seed": True,
        },
        "gpu_authorized": False,
        "modal_authorized": False,
        "paid_compute_authorized": False,
        "training_authorized": False,
        "trigger_creation_authorized": False,
        "scientific_execution_authorized": False,
        "scientific_seed_consumed": False,
        "stage_d_automatically_authorized": False,
    }


def validate_contract() -> dict[str, Any]:
    validate_seed_for_preparation(SCIENTIFIC_SEED)
    assert_no_execution_authority()
    validate_source_bindings(
        main_sha=STARTING_MAIN_SHA,
        main_tree=STARTING_MAIN_TREE,
        model_contract_blob=MODEL_CONTRACT_BLOB,
        evaluator_blob=EVALUATOR_BLOB,
    )
    if SCIENTIFIC_SEED != FUTURE_SCREEN_SEED:
        raise RuntimeError("#988 seed mismatch with #977")
    if TRAINING_TOKENS_PER_MODEL != 33_554_432:
        raise RuntimeError("#988 per-model scientific token budget drift")
    if PAIR_TRAINING_TOKENS != 67_108_864:
        raise RuntimeError("#988 pair token accounting drift")
    if TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#988 tokens/optimizer-step drift")
    if TRAINING_TOKENS_PER_MODEL % TOKENS_PER_OPTIMIZER_STEP:
        raise RuntimeError("#988 training token budget does not divide optimizer geometry")
    if OPTIMIZER_STEPS_PER_MODEL != 2_048 or WARMUP_STEPS != 40:
        raise RuntimeError("#988 optimizer/warmup step schedule drift")
    if tuple(CHECKPOINT_STEPS) != (512, 1024, 1536, 2048):
        raise RuntimeError("#988 checkpoint schedule drift")
    if FINAL_EVALUATION_STEP != OPTIMIZER_STEPS_PER_MODEL:
        raise RuntimeError("#988 final scientific evaluation must use only the completed training state")
    if SESSION_LEN != 1_024 or MICRO_BATCH != 4 or GRAD_ACCUM != 4:
        raise RuntimeError("#988 inherited matched training geometry drift")
    if tuple(BETAS) != (0.9, 0.95):
        raise RuntimeError("#988 AdamW beta drift")
    if WEIGHT_DECAY != 0.1 or PEAK_LR != 3e-4 or GRAD_CLIP != 1.0:
        raise RuntimeError("#988 optimizer hyperparameter drift")
    if COMPILE_ENABLED:
        raise RuntimeError("#988 requires torch.compile disabled")
    if EXPECTED_LOCAL_PARAMETERS != 101_803_520 or EXPECTED_EIEM_PARAMETERS != 101_836_800:
        raise RuntimeError("#988 Stage-A parameter binding drift")
    if GENERATOR_VERSION != "chm-v1-100m-heldout-aligned-v4":
        raise RuntimeError("#988 evaluator version drift")
    if (PROBE_SEED, VALIDATION_SEED, BOOTSTRAP_SEED) != (977_301, 977_302, 977_303):
        raise RuntimeError("#988 evaluator seed drift")
    if (CASES_PER_FAMILY, TOTAL_PROBES, LONG_RANGE_PROBES) != (128, 512, 384):
        raise RuntimeError("#988 evaluator probe envelope drift")
    if VALIDATION_TOKENS != 1_048_576 or BOOTSTRAP_RESAMPLES != 10_000:
        raise RuntimeError("#988 evaluator validation/bootstrap envelope drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_GIB != 16:
        raise RuntimeError("#988 future hardware envelope drift")
    if MAX_GPU_SECONDS != 43_200 or MAX_STAGE_C_COMPUTE_USD != 25.00 or RETRIES != 0:
        raise RuntimeError("#988 future paid envelope drift")
    if RESULT_ROOT != "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1":
        raise RuntimeError("#988 reserved result-root drift")
    if FUTURE_TRIGGER_TITLE != "[modal-chm-v1-100m-stage-c-988-seed-977001-v1]":
        raise RuntimeError("#988 reserved future trigger-title drift")

    manifest = protocol_manifest()
    for key in (
        "gpu_authorized",
        "modal_authorized",
        "paid_compute_authorized",
        "training_authorized",
        "trigger_creation_authorized",
        "scientific_execution_authorized",
        "scientific_seed_consumed",
        "stage_d_automatically_authorized",
    ):
        if manifest[key] is not False:
            raise RuntimeError(f"#988 authority flag unexpectedly enabled: {key}")
    return manifest
