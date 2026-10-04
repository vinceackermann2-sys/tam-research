from __future__ import annotations

"""CHM-v3 ~100M DAEC Stage-B systems contract (#1237).

Pure systems contract only: deterministic engineering stream, exact resource
geometry, projection arithmetic, and the predeclared PASS/STOP classifier.
This module allocates no GPU resources and grants no scientific execution.
"""

import hashlib
from typing import Any

import numpy as np
import torch

from .chm_v1_100m_scale import (
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    FIRST_SCREEN_TOKEN_BUDGET,
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
from .chm_v3_100m_daec import EXPECTED_DAEC_PARAMETERS

PARENT_HYPOTHESIS_ISSUE = 1234
SYSTEMS_ISSUE = 1237
STARTING_MAIN_SHA = "ee01b5a5d123c381dee861a3e202e93e5eeb49d0"
STARTING_MAIN_TREE = "af2fde0a72c1b32154345395ab0c1a85135ef2a9"

ENGINEERING_SEED = 1_234_201
STREAM_SEED = 1_244_201

HISTORICAL_CONSUMED_SCIENTIFIC_SEEDS = (
    977_001,
    2_011_121,
    2_011_371,
    2_011_431,
    2_011_761,
)
STAGE_A_ENGINEERING_SEED = 1_234_001

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_GIB = 16
MAX_GPU_SECONDS = 1_200
MAX_STAGE_B_COMPUTE_USD = 0.50

WARMUP_STEPS = 2
MEASURED_STEPS = 8
TOTAL_BENCHMARK_STEPS = WARMUP_STEPS + MEASURED_STEPS
TOKENS_PER_OPTIMIZER_STEP = MICRO_BATCH * SESSION_LEN * GRAD_ACCUM
MEASURED_TOKENS_PER_MODEL = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP
TRAIN_TOKENS = 2_000_000_000

MAX_PEAK_VRAM_GIB = 22.0
MIN_LOCAL_TOKENS_PER_SECOND = 8_000.0
MIN_RAW_TOKENS_PER_SECOND = 8_000.0
MIN_DAEC_TOKENS_PER_SECOND = 8_000.0
MIN_DAEC_RAW_THROUGHPUT_RATIO = 0.60
MAX_PROJECTED_THREE_MODEL_SECONDS = 9_600.0
MAX_PROJECTED_THREE_MODEL_COMPUTE_USD = 3.00

RESULT_ROOT = "/vol/chm-v3/100m-daec-stage-b/issue-1237/seed-1234201-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRIGGER_TITLE = "[modal-chm-v3-100m-daec-stage-b-1237-seed-1234201-v1]"
AUDIT_TITLE = "[modal-chm-v3-100m-daec-stage-b-1237-authority-audit-v1]"
PHASE = "chm-v3-100m-daec-stage-b-1237-v1"


def validate_engineering_seed(seed: int) -> int:
    value = int(seed)
    if value in HISTORICAL_CONSUMED_SCIENTIFIC_SEEDS:
        raise RuntimeError(f"#1237 refuses historical scientific seed {value}")
    if value == STAGE_A_ENGINEERING_SEED:
        raise RuntimeError(f"#1237 refuses Stage-A engineering seed {value}")
    if value != ENGINEERING_SEED:
        raise RuntimeError(f"#1237 accepts only systems engineering seed {ENGINEERING_SEED}")
    return value


def build_start_plan(train_tokens: int = TRAIN_TOKENS) -> torch.Tensor:
    train_tokens = int(train_tokens)
    hi = train_tokens - SESSION_LEN - 1
    if hi <= 0:
        raise ValueError("train token stream is too short for frozen sessions")
    generator = torch.Generator(device="cpu").manual_seed(STREAM_SEED)
    return torch.randint(
        0,
        hi,
        (TOTAL_BENCHMARK_STEPS, GRAD_ACCUM, MICRO_BATCH),
        generator=generator,
        dtype=torch.int64,
    )


def start_plan_sha256(plan: torch.Tensor) -> str:
    expected_shape = (TOTAL_BENCHMARK_STEPS, GRAD_ACCUM, MICRO_BATCH)
    if tuple(plan.shape) != expected_shape or plan.dtype != torch.int64:
        raise ValueError(f"#1237 start plan must be int64 {expected_shape}")
    array = plan.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def protocol_manifest() -> dict[str, Any]:
    plan = build_start_plan()
    return {
        "classification": "CHM_V3_100M_DAEC_STAGE_B_L4_SYSTEMS_PREFLIGHT_TRIGGER_WITHHELD",
        "parent_hypothesis_issue": PARENT_HYPOTHESIS_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "starting_main_tree": STARTING_MAIN_TREE,
        "engineering_seed": ENGINEERING_SEED,
        "stream_seed": STREAM_SEED,
        "historical_scientific_seeds_not_reusable": list(HISTORICAL_CONSUMED_SCIENTIFIC_SEEDS),
        "stage_a_engineering_seed_not_reusable": STAGE_A_ENGINEERING_SEED,
        "resource_plan": {
            "provider": "Modal",
            "gpu": "1x NVIDIA L4",
            "cpu_cores": CPU_CORES,
            "ram_gib": RAM_GIB,
            "max_gpu_seconds": MAX_GPU_SECONDS,
            "max_stage_b_compute_usd": MAX_STAGE_B_COMPUTE_USD,
            "retries": 0,
        },
        "training_geometry": {
            "session_len": SESSION_LEN,
            "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM,
            "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "measured_tokens_per_model": MEASURED_TOKENS_PER_MODEL,
            "bf16_autocast": True,
            "compile_enabled": COMPILE_ENABLED,
            "optimizer": "AdamW",
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "peak_lr": PEAK_LR,
            "grad_clip": GRAD_CLIP,
        },
        "models": {
            "local_parameters": EXPECTED_LOCAL_PARAMETERS,
            "raw_eiem_parameters": EXPECTED_EIEM_PARAMETERS,
            "daec_parameters": EXPECTED_DAEC_PARAMETERS,
        },
        "stream_plan": {
            "shape": list(plan.shape),
            "starts": int(plan.numel()),
            "sha256": start_plan_sha256(plan),
        },
        "future_screen_tokens_per_model_not_authorized": FIRST_SCREEN_TOKEN_BUDGET,
        "data_dir": DATA_DIR,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "audit_title": AUDIT_TITLE,
        "gpu_authorized_only_after_final_launcher_authority": True,
        "scientific_seed_authorized": False,
        "scientific_execution_authorized": False,
        "stage_c_authorized": False,
    }


def validate_contract() -> dict[str, Any]:
    validate_engineering_seed(ENGINEERING_SEED)
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_GIB != 16:
        raise RuntimeError("#1237 hardware envelope drift")
    if MAX_GPU_SECONDS != 1_200 or MAX_STAGE_B_COMPUTE_USD != 0.50:
        raise RuntimeError("#1237 paid envelope drift")
    if SESSION_LEN != 1_024 or MICRO_BATCH != 4 or GRAD_ACCUM != 4:
        raise RuntimeError("#1237 inherited training geometry drift")
    if TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#1237 tokens/optimizer-step drift")
    if WARMUP_STEPS != 2 or MEASURED_STEPS != 8:
        raise RuntimeError("#1237 measured workload drift")
    if MEASURED_TOKENS_PER_MODEL != 131_072:
        raise RuntimeError("#1237 measured token count drift")
    if build_start_plan().numel() != 160:
        raise RuntimeError("#1237 exact session-start count drift")
    if COMPILE_ENABLED:
        raise RuntimeError("#1237 requires torch.compile disabled")
    if tuple(BETAS) != (0.9, 0.95):
        raise RuntimeError("#1237 AdamW betas drift")
    if WEIGHT_DECAY != 0.1 or PEAK_LR != 3e-4 or GRAD_CLIP != 1.0:
        raise RuntimeError("#1237 optimizer geometry drift")
    if EXPECTED_DAEC_PARAMETERS != 101_853_697:
        raise RuntimeError("#1237 DAEC parameter count drift")
    return protocol_manifest()


def project_from_throughput(
    *,
    local_tokens_per_second: float,
    raw_tokens_per_second: float,
    daec_tokens_per_second: float,
    live_hourly_resource_usd: float,
) -> dict[str, float]:
    local_tps = float(local_tokens_per_second)
    raw_tps = float(raw_tokens_per_second)
    daec_tps = float(daec_tokens_per_second)
    hourly = float(live_hourly_resource_usd)
    if min(local_tps, raw_tps, daec_tps, hourly) <= 0.0:
        raise ValueError("throughputs and live hourly resource rate must be positive")
    local_seconds = FIRST_SCREEN_TOKEN_BUDGET / local_tps
    raw_seconds = FIRST_SCREEN_TOKEN_BUDGET / raw_tps
    daec_seconds = FIRST_SCREEN_TOKEN_BUDGET / daec_tps
    total_seconds = local_seconds + raw_seconds + daec_seconds
    return {
        "local_projected_seconds": local_seconds,
        "raw_projected_seconds": raw_seconds,
        "daec_projected_seconds": daec_seconds,
        "three_model_projected_seconds": total_seconds,
        "daec_raw_throughput_ratio": daec_tps / raw_tps,
        "three_model_projected_compute_usd": (total_seconds / 3600.0) * hourly,
        "live_hourly_resource_usd": hourly,
    }


def classify_stage_b(
    *,
    local: dict[str, Any],
    raw: dict[str, Any],
    daec: dict[str, Any],
    live_hourly_resource_usd: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    expected_stream = start_plan_sha256(build_start_plan())

    rows = (
        ("local", local, EXPECTED_LOCAL_PARAMETERS, MIN_LOCAL_TOKENS_PER_SECOND),
        ("raw", raw, EXPECTED_EIEM_PARAMETERS, MIN_RAW_TOKENS_PER_SECOND),
        ("daec", daec, EXPECTED_DAEC_PARAMETERS, MIN_DAEC_TOKENS_PER_SECOND),
    )
    for name, row, expected_params, min_tps in rows:
        if int(row.get("trainable_parameters", -1)) != expected_params:
            reasons.append(f"{name}_parameter_count_mismatch")
        if int(row.get("measured_tokens", -1)) != MEASURED_TOKENS_PER_MODEL:
            reasons.append(f"{name}_measured_token_count_mismatch")
        if not bool(row.get("finite_loss", False)):
            reasons.append(f"{name}_nonfinite_loss")
        if not bool(row.get("finite_parameters", False)):
            reasons.append(f"{name}_nonfinite_parameters")
        if str(row.get("start_plan_sha256", "")) != expected_stream:
            reasons.append(f"{name}_start_plan_digest_mismatch")
        tps = float(row.get("tokens_per_second", 0.0))
        if tps < min_tps:
            reasons.append(f"{name}_throughput_below_gate")
        peak_gib = float(row.get("peak_vram_bytes", float("inf"))) / (1024.0**3)
        if peak_gib > MAX_PEAK_VRAM_GIB:
            reasons.append(f"{name}_peak_vram_above_gate")

    digests = {
        str(local.get("start_plan_sha256", "")),
        str(raw.get("start_plan_sha256", "")),
        str(daec.get("start_plan_sha256", "")),
    }
    if digests != {expected_stream}:
        reasons.append("paired_start_plan_digest_mismatch")

    projection: dict[str, float] | None = None
    try:
        projection = project_from_throughput(
            local_tokens_per_second=float(local.get("tokens_per_second", 0.0)),
            raw_tokens_per_second=float(raw.get("tokens_per_second", 0.0)),
            daec_tokens_per_second=float(daec.get("tokens_per_second", 0.0)),
            live_hourly_resource_usd=live_hourly_resource_usd,
        )
    except ValueError:
        reasons.append("invalid_projection_inputs")

    if projection is not None:
        if projection["daec_raw_throughput_ratio"] < MIN_DAEC_RAW_THROUGHPUT_RATIO:
            reasons.append("daec_raw_throughput_ratio_below_gate")
        if projection["three_model_projected_seconds"] > MAX_PROJECTED_THREE_MODEL_SECONDS:
            reasons.append("projected_three_model_wall_time_above_gate")
        if projection["three_model_projected_compute_usd"] > MAX_PROJECTED_THREE_MODEL_COMPUTE_USD:
            reasons.append("projected_three_model_compute_cost_above_gate")

    passed = not reasons
    return {
        "classification": (
            "CHM_V3_100M_DAEC_STAGE_B_SYSTEMS_PASS"
            if passed
            else "CHM_V3_100M_DAEC_STAGE_B_SYSTEMS_STOP"
        ),
        "passed": passed,
        "stop_reasons": reasons,
        "projection": projection,
        "thresholds": {
            "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
            "min_local_tokens_per_second": MIN_LOCAL_TOKENS_PER_SECOND,
            "min_raw_tokens_per_second": MIN_RAW_TOKENS_PER_SECOND,
            "min_daec_tokens_per_second": MIN_DAEC_TOKENS_PER_SECOND,
            "min_daec_raw_throughput_ratio": MIN_DAEC_RAW_THROUGHPUT_RATIO,
            "max_projected_three_model_seconds": MAX_PROJECTED_THREE_MODEL_SECONDS,
            "max_projected_three_model_compute_usd": MAX_PROJECTED_THREE_MODEL_COMPUTE_USD,
        },
        "scientific_seed_authorized": False,
        "scientific_execution": False,
        "stage_c_authorized_automatically": False,
        "interpretation_ceiling": (
            "Systems feasibility only. PASS may justify preparing a separate Stage-C preregistration "
            "but does not establish modeling quality or authorize science."
        ),
    }
