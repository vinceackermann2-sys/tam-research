from __future__ import annotations

"""Frozen CHM-v1 ~100M Stage-B systems preflight contract (#983).

This module contains only deterministic protocol constants, seed/resource guards,
projection arithmetic, and the predeclared PASS/STOP classifier.  It grants no
scientific execution authority and does not itself allocate GPU resources.
"""

from typing import Any

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

PARENT_RESEARCH_ISSUE = 977
SYSTEMS_ISSUE = 983
STARTING_MAIN_SHA = "118a743f7eb1c224af71bd97dac0f53b2e565d73"
STARTING_MAIN_TREE = "a69e7698f754fb943c9f189d0623e2a6b02daeb6"

ENGINEERING_SEED = 977_201
RESERVED_SCIENTIFIC_SEED = 977_001
STAGE_A_SMOKE_SEED = 977_099
BLOCKED_PRIOR_SEEDS = (
    19_591,
    19_592,
    19_593,
    971_001,
    971_002,
    973_001,
    973_002,
    RESERVED_SCIENTIFIC_SEED,
    STAGE_A_SMOKE_SEED,
)

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_GIB = 16
MAX_GPU_SECONDS = 1_200
MAX_STAGE_B_COMPUTE_USD = 0.50

WARMUP_STEPS = 2
MEASURED_STEPS = 8
TOKENS_PER_OPTIMIZER_STEP = MICRO_BATCH * SESSION_LEN * GRAD_ACCUM
MEASURED_TOKENS_PER_MODEL = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP

MAX_PEAK_VRAM_GIB = 22.0
MIN_LOCAL_TOKENS_PER_SECOND = 3_500.0
MIN_EIEM_TOKENS_PER_SECOND = 2_500.0
MIN_EIEM_LOCAL_THROUGHPUT_RATIO = 0.60
MAX_PROJECTED_PAIR_SECONDS = 21_600.0
MAX_PROJECTED_PAIR_COMPUTE_USD = 6.00

RESULT_ROOT = "/vol/chm-v1/100m-stage-b/issue-983/seed-977201-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRIGGER_TITLE = "[modal-chm-v1-100m-stage-b-983-v1]"
PHASE = "chm-v1-100m-stage-b-983-v1"


def validate_engineering_seed(seed: int) -> int:
    seed = int(seed)
    if seed in BLOCKED_PRIOR_SEEDS:
        raise RuntimeError(f"blocked/scientific CHM seed refused by #983: {seed}")
    if seed != ENGINEERING_SEED:
        raise RuntimeError(f"#983 accepts only engineering seed {ENGINEERING_SEED}")
    return seed


def validate_contract() -> dict[str, Any]:
    validate_engineering_seed(ENGINEERING_SEED)
    if RESERVED_SCIENTIFIC_SEED == ENGINEERING_SEED:
        raise RuntimeError("Stage-B engineering seed aliases reserved scientific seed")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_GIB != 16:
        raise RuntimeError("#983 hardware envelope drift")
    if MAX_GPU_SECONDS != 1_200 or MAX_STAGE_B_COMPUTE_USD != 0.50:
        raise RuntimeError("#983 paid envelope drift")
    if SESSION_LEN != 1_024 or MICRO_BATCH != 4 or GRAD_ACCUM != 4:
        raise RuntimeError("#983 inherited training geometry drift")
    if TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#983 tokens/optimizer-step drift")
    if WARMUP_STEPS != 2 or MEASURED_STEPS != 8:
        raise RuntimeError("#983 measured workload drift")
    if MEASURED_TOKENS_PER_MODEL != 131_072:
        raise RuntimeError("#983 measured token count drift")
    if COMPILE_ENABLED:
        raise RuntimeError("#983 requires torch.compile disabled")
    if tuple(BETAS) != (0.9, 0.95):
        raise RuntimeError("#983 AdamW betas drift")
    if WEIGHT_DECAY != 0.1 or PEAK_LR != 3e-4 or GRAD_CLIP != 1.0:
        raise RuntimeError("#983 optimizer geometry drift")
    return protocol_manifest()


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": "CHM_V1_100M_STAGE_B_L4_SYSTEMS_PREFLIGHT_ONLY",
        "parent_research_issue": PARENT_RESEARCH_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "starting_main_tree": STARTING_MAIN_TREE,
        "engineering_seed": ENGINEERING_SEED,
        "reserved_scientific_seed_not_authorized": RESERVED_SCIENTIFIC_SEED,
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
            "eiem_flat_parameters": EXPECTED_EIEM_PARAMETERS,
        },
        "future_screen_tokens_per_model_not_authorized": FIRST_SCREEN_TOKEN_BUDGET,
        "data_dir": DATA_DIR,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "gpu_authorized_only_after_final_launcher_authority": True,
        "scientific_execution_authorized": False,
    }


def project_from_throughput(
    *,
    local_tokens_per_second: float,
    eiem_tokens_per_second: float,
    live_hourly_resource_usd: float,
) -> dict[str, float]:
    local_tps = float(local_tokens_per_second)
    eiem_tps = float(eiem_tokens_per_second)
    hourly = float(live_hourly_resource_usd)
    if local_tps <= 0 or eiem_tps <= 0 or hourly <= 0:
        raise ValueError("throughputs and live hourly resource rate must be positive")
    local_seconds = FIRST_SCREEN_TOKEN_BUDGET / local_tps
    eiem_seconds = FIRST_SCREEN_TOKEN_BUDGET / eiem_tps
    pair_seconds = local_seconds + eiem_seconds
    return {
        "local_projected_seconds": local_seconds,
        "eiem_projected_seconds": eiem_seconds,
        "pair_projected_seconds": pair_seconds,
        "eiem_local_throughput_ratio": eiem_tps / local_tps,
        "pair_projected_compute_usd": (pair_seconds / 3600.0) * hourly,
        "live_hourly_resource_usd": hourly,
    }


def classify_stage_b(
    *,
    local: dict[str, Any],
    eiem: dict[str, Any],
    live_hourly_resource_usd: float,
) -> dict[str, Any]:
    """Apply only the numerical/systems gate frozen in #983."""

    reasons: list[str] = []
    for name, row, expected_params, min_tps in (
        ("local", local, EXPECTED_LOCAL_PARAMETERS, MIN_LOCAL_TOKENS_PER_SECOND),
        ("eiem", eiem, EXPECTED_EIEM_PARAMETERS, MIN_EIEM_TOKENS_PER_SECOND),
    ):
        if int(row.get("trainable_parameters", -1)) != expected_params:
            reasons.append(f"{name}_parameter_count_mismatch")
        if int(row.get("measured_tokens", -1)) != MEASURED_TOKENS_PER_MODEL:
            reasons.append(f"{name}_measured_token_count_mismatch")
        if not bool(row.get("finite_loss", False)):
            reasons.append(f"{name}_nonfinite_loss")
        if not bool(row.get("finite_parameters", False)):
            reasons.append(f"{name}_nonfinite_parameters")
        tps = float(row.get("tokens_per_second", 0.0))
        if tps < min_tps:
            reasons.append(f"{name}_throughput_below_gate")
        peak_gib = float(row.get("peak_vram_bytes", float("inf"))) / (1024.0**3)
        if peak_gib > MAX_PEAK_VRAM_GIB:
            reasons.append(f"{name}_peak_vram_above_gate")

    projection = project_from_throughput(
        local_tokens_per_second=float(local.get("tokens_per_second", 0.0)),
        eiem_tokens_per_second=float(eiem.get("tokens_per_second", 0.0)),
        live_hourly_resource_usd=live_hourly_resource_usd,
    )
    if projection["eiem_local_throughput_ratio"] < MIN_EIEM_LOCAL_THROUGHPUT_RATIO:
        reasons.append("eiem_local_throughput_ratio_below_gate")
    if projection["pair_projected_seconds"] > MAX_PROJECTED_PAIR_SECONDS:
        reasons.append("projected_pair_wall_time_above_gate")
    if projection["pair_projected_compute_usd"] > MAX_PROJECTED_PAIR_COMPUTE_USD:
        reasons.append("projected_pair_compute_cost_above_gate")

    passed = not reasons
    return {
        "classification": (
            "CHM_V1_100M_STAGE_B_SYSTEMS_PASS"
            if passed
            else "CHM_V1_100M_STAGE_B_SYSTEMS_STOP"
        ),
        "passed": passed,
        "stop_reasons": reasons,
        "projection": projection,
        "thresholds": {
            "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
            "min_local_tokens_per_second": MIN_LOCAL_TOKENS_PER_SECOND,
            "min_eiem_tokens_per_second": MIN_EIEM_TOKENS_PER_SECOND,
            "min_eiem_local_throughput_ratio": MIN_EIEM_LOCAL_THROUGHPUT_RATIO,
            "max_projected_pair_seconds": MAX_PROJECTED_PAIR_SECONDS,
            "max_projected_pair_compute_usd": MAX_PROJECTED_PAIR_COMPUTE_USD,
        },
        "interpretation_ceiling": (
            "Systems feasibility only. PASS does not authorize or constitute a scientific result."
        ),
    }
