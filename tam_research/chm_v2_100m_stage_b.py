from __future__ import annotations

"""CHM-v2 ~100M QVA Stage-B systems contract (#1155).

Pure contract only: deterministic engineering stream, exact resource/workload
constants, projection arithmetic, and the predeclared PASS/STOP classifier.
This module does not allocate GPU resources and grants no scientific execution.
"""

import hashlib
from typing import Any

import numpy as np
import torch

from .chm_v1_100m_scale import EXPECTED_EIEM_PARAMETERS, FIRST_SCREEN_TOKEN_BUDGET
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
from .chm_v2_100m_qva import EXPECTED_QVA_PARAMETERS

PARENT_HYPOTHESIS_ISSUE = 1147
SYSTEMS_ISSUE = 1155
STARTING_MAIN_SHA = "53bcb541e4e564cec035e3b23dee772948892ae4"
STARTING_MAIN_TREE = "1cc5d365280fa07aecbc544d5e33c5f019687908"

ENGINEERING_SEED = 1_147_201
STREAM_SEED = ENGINEERING_SEED + 10_000
HISTORICAL_CONSUMED_CHM_V1_SEED = 977_001
STAGE_A_ENGINEERING_SEED = 1_147_099

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
MIN_V1_TOKENS_PER_SECOND = 10_000.0
MIN_QVA_TOKENS_PER_SECOND = 10_000.0
MIN_QVA_V1_THROUGHPUT_RATIO = 0.85
MAX_PROJECTED_PAIR_SECONDS = 7_200.0
MAX_PROJECTED_PAIR_COMPUTE_USD = 3.00

RESULT_ROOT = "/vol/chm-v2/100m-qva-stage-b/parent-1147/seed-1147201-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRIGGER_TITLE = "[modal-chm-v2-100m-qva-stage-b-parent-1147-v1]"
PHASE = "chm-v2-100m-qva-stage-b-1155-v1"


def validate_engineering_seed(seed: int) -> int:
    seed = int(seed)
    if seed in {HISTORICAL_CONSUMED_CHM_V1_SEED, STAGE_A_ENGINEERING_SEED}:
        raise RuntimeError(f"#1155 refuses historical/Stage-A seed {seed}")
    if seed != ENGINEERING_SEED:
        raise RuntimeError(f"#1155 accepts only systems engineering seed {ENGINEERING_SEED}")
    return seed


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
        raise ValueError(f"#1155 start plan must be int64 {expected_shape}")
    array = plan.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def protocol_manifest() -> dict[str, Any]:
    plan = build_start_plan()
    return {
        "classification": "CHM_V2_100M_QVA_STAGE_B_L4_SYSTEMS_PREFLIGHT_TRIGGER_WITHHELD",
        "parent_hypothesis_issue": PARENT_HYPOTHESIS_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "starting_main_tree": STARTING_MAIN_TREE,
        "engineering_seed": ENGINEERING_SEED,
        "stream_seed": STREAM_SEED,
        "historical_consumed_chm_v1_seed_not_reusable": HISTORICAL_CONSUMED_CHM_V1_SEED,
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
            "chm_v1_eiem_parameters": EXPECTED_EIEM_PARAMETERS,
            "chm_v2_qva_parameters": EXPECTED_QVA_PARAMETERS,
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
        "gpu_authorized_only_after_final_launcher_authority": True,
        "scientific_seed_authorized": False,
        "scientific_execution_authorized": False,
    }


def validate_contract() -> dict[str, Any]:
    validate_engineering_seed(ENGINEERING_SEED)
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_GIB != 16:
        raise RuntimeError("#1155 hardware envelope drift")
    if MAX_GPU_SECONDS != 1_200 or MAX_STAGE_B_COMPUTE_USD != 0.50:
        raise RuntimeError("#1155 paid envelope drift")
    if SESSION_LEN != 1_024 or MICRO_BATCH != 4 or GRAD_ACCUM != 4:
        raise RuntimeError("#1155 inherited training geometry drift")
    if TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#1155 tokens/optimizer-step drift")
    if WARMUP_STEPS != 2 or MEASURED_STEPS != 8:
        raise RuntimeError("#1155 measured workload drift")
    if MEASURED_TOKENS_PER_MODEL != 131_072:
        raise RuntimeError("#1155 measured token count drift")
    if build_start_plan().numel() != 160:
        raise RuntimeError("#1155 exact session-start count drift")
    if COMPILE_ENABLED:
        raise RuntimeError("#1155 requires torch.compile disabled")
    if tuple(BETAS) != (0.9, 0.95):
        raise RuntimeError("#1155 AdamW betas drift")
    if WEIGHT_DECAY != 0.1 or PEAK_LR != 3e-4 or GRAD_CLIP != 1.0:
        raise RuntimeError("#1155 optimizer geometry drift")
    return protocol_manifest()


def project_from_throughput(
    *,
    v1_tokens_per_second: float,
    qva_tokens_per_second: float,
    live_hourly_resource_usd: float,
) -> dict[str, float]:
    v1_tps = float(v1_tokens_per_second)
    qva_tps = float(qva_tokens_per_second)
    hourly = float(live_hourly_resource_usd)
    if v1_tps <= 0 or qva_tps <= 0 or hourly <= 0:
        raise ValueError("throughputs and live hourly resource rate must be positive")
    v1_seconds = FIRST_SCREEN_TOKEN_BUDGET / v1_tps
    qva_seconds = FIRST_SCREEN_TOKEN_BUDGET / qva_tps
    pair_seconds = v1_seconds + qva_seconds
    return {
        "v1_projected_seconds": v1_seconds,
        "qva_projected_seconds": qva_seconds,
        "pair_projected_seconds": pair_seconds,
        "qva_v1_throughput_ratio": qva_tps / v1_tps,
        "pair_projected_compute_usd": (pair_seconds / 3600.0) * hourly,
        "live_hourly_resource_usd": hourly,
    }


def classify_stage_b(
    *,
    v1: dict[str, Any],
    qva: dict[str, Any],
    live_hourly_resource_usd: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    expected_stream = start_plan_sha256(build_start_plan())

    for name, row, expected_params, min_tps in (
        ("v1", v1, EXPECTED_EIEM_PARAMETERS, MIN_V1_TOKENS_PER_SECOND),
        ("qva", qva, EXPECTED_QVA_PARAMETERS, MIN_QVA_TOKENS_PER_SECOND),
    ):
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

    if str(v1.get("start_plan_sha256", "")) != str(qva.get("start_plan_sha256", "")):
        reasons.append("paired_start_plan_digest_mismatch")

    projection = project_from_throughput(
        v1_tokens_per_second=float(v1.get("tokens_per_second", 0.0)),
        qva_tokens_per_second=float(qva.get("tokens_per_second", 0.0)),
        live_hourly_resource_usd=live_hourly_resource_usd,
    )
    if projection["qva_v1_throughput_ratio"] < MIN_QVA_V1_THROUGHPUT_RATIO:
        reasons.append("qva_v1_throughput_ratio_below_gate")
    if projection["pair_projected_seconds"] > MAX_PROJECTED_PAIR_SECONDS:
        reasons.append("projected_pair_wall_time_above_gate")
    if projection["pair_projected_compute_usd"] > MAX_PROJECTED_PAIR_COMPUTE_USD:
        reasons.append("projected_pair_compute_cost_above_gate")

    passed = not reasons
    return {
        "classification": (
            "CHM_V2_100M_QVA_STAGE_B_SYSTEMS_PASS"
            if passed
            else "CHM_V2_100M_QVA_STAGE_B_SYSTEMS_STOP"
        ),
        "passed": passed,
        "stop_reasons": reasons,
        "projection": projection,
        "thresholds": {
            "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
            "min_v1_tokens_per_second": MIN_V1_TOKENS_PER_SECOND,
            "min_qva_tokens_per_second": MIN_QVA_TOKENS_PER_SECOND,
            "min_qva_v1_throughput_ratio": MIN_QVA_V1_THROUGHPUT_RATIO,
            "max_projected_pair_seconds": MAX_PROJECTED_PAIR_SECONDS,
            "max_projected_pair_compute_usd": MAX_PROJECTED_PAIR_COMPUTE_USD,
        },
        "scientific_seed_authorized": False,
        "scientific_execution": False,
        "interpretation_ceiling": (
            "Systems feasibility only. PASS does not establish modeling quality or authorize science."
        ),
    }
