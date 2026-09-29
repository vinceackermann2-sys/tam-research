from __future__ import annotations

"""CHM-v2 100M host-staged L4 systems preflight contract (#1133).

Infrastructure-only. This module adds a CPU-resident/microbatch-only transport
path and pure systems classification. It does not launch Modal, train a
scientific result, reserve a scientific seed, or authorize a CHM-v2 rerun.
"""

import hashlib
from typing import Any, Sequence

import numpy as np
import torch

from .chm_v1_100m_scale import EXPECTED_EIEM_PARAMETERS
from .chm_v1_small_lm_protocol import (
    BETAS,
    GRAD_ACCUM,
    GRAD_CLIP,
    MICRO_BATCH,
    PEAK_LR,
    SESSION_LEN,
    WEIGHT_DECAY,
)
from .chm_v2_100m_value_projected_eiem import (
    EXPECTED_VP_EIEM_PARAMETERS,
    TRAINING_TOKENS_PER_MODEL,
)

CONTROL_ISSUE = 1133
ENGINEERING_SEED = 1_133_201
CONSUMED_SCIENTIFIC_SEED = 2_011_121

WARMUP_STEPS = 2
MEASURED_STEPS = 8
TOTAL_ENGINEERING_STEPS = WARMUP_STEPS + MEASURED_STEPS
TOKENS_PER_OPTIMIZER_STEP = MICRO_BATCH * SESSION_LEN * GRAD_ACCUM
MEASURED_TOKENS_PER_MODEL = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP

HISTORICAL_LOCAL_TOKENS_PER_SECOND = 28_534.744
MAX_PEAK_ALLOCATED_GIB = 18.0
MAX_PEAK_RESERVED_GIB = 19.0
MIN_RAW_TOKENS_PER_SECOND = 2_500.0
MIN_VP_TOKENS_PER_SECOND = 2_000.0
MAX_PROJECTED_THREE_MODEL_SECONDS = 4 * 60 * 60
MAX_PROJECTED_THREE_MODEL_COMPUTE_USD = 6.00

PASS_CLASSIFICATION = "CHM_V2_100M_HOST_STAGED_CORPUS_L4_SYSTEMS_PASS"
STOP_CLASSIFICATION = "CHM_V2_100M_HOST_STAGED_CORPUS_L4_SYSTEMS_STOP"

RESULT_ROOT = "/vol/chm-v2/100m-host-staged-l4-preflight/issue-1133/attempt-1133201-v1"
TRIGGER_TITLE = "[modal-chm-v2-100m-host-staged-l4-preflight-1133-v1]"


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": "CHM_V2_100M_HOST_STAGED_CORPUS_L4_MEMORY_PREFLIGHT_TRIGGER_WITHHELD",
        "control_issue": CONTROL_ISSUE,
        "engineering_seed": ENGINEERING_SEED,
        "engineering_seed_is_scientific": False,
        "consumed_scientific_seed": CONSUMED_SCIENTIFIC_SEED,
        "consumed_scientific_seed_reusable": False,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "models": {
            "raw_eiem_parameters": EXPECTED_EIEM_PARAMETERS,
            "vp_eiem_parameters": EXPECTED_VP_EIEM_PARAMETERS,
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
            "optimizer": "AdamW-fused",
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "peak_lr": PEAK_LR,
            "grad_clip": GRAD_CLIP,
            "soft_retrieval_temperature": 0.10,
            "torch_compile": False,
        },
        "transport": {
            "source_residency": "cpu_memmap",
            "transfer_scope": "current_microbatch_only",
            "token_transfer_dtype": "int64",
            "full_source_cuda_cache_authorized": False,
        },
        "scientific_execution_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "stage_d_authorized": False,
    }


def validate_contract() -> dict[str, Any]:
    if (CONTROL_ISSUE, ENGINEERING_SEED, CONSUMED_SCIENTIFIC_SEED) != (
        1133,
        1_133_201,
        2_011_121,
    ):
        raise RuntimeError("#1133 identity drift")
    if (SESSION_LEN, MICRO_BATCH, GRAD_ACCUM) != (1024, 4, 4):
        raise RuntimeError("#1133 frozen geometry drift")
    if TOKENS_PER_OPTIMIZER_STEP != 16_384:
        raise RuntimeError("#1133 tokens/optimizer-step drift")
    if (WARMUP_STEPS, MEASURED_STEPS, MEASURED_TOKENS_PER_MODEL) != (
        2,
        8,
        131_072,
    ):
        raise RuntimeError("#1133 engineering measurement envelope drift")
    if tuple(BETAS) != (0.9, 0.95):
        raise RuntimeError("#1133 AdamW beta drift")
    if WEIGHT_DECAY != 0.1 or PEAK_LR != 3e-4 or GRAD_CLIP != 1.0:
        raise RuntimeError("#1133 optimizer hyperparameter drift")
    if EXPECTED_EIEM_PARAMETERS != 101_836_800:
        raise RuntimeError("#1133 RAW-EIEM parameter binding drift")
    if EXPECTED_VP_EIEM_PARAMETERS <= EXPECTED_EIEM_PARAMETERS:
        raise RuntimeError("#1133 VP parameter accounting drift")
    if TRAINING_TOKENS_PER_MODEL != 33_554_432:
        raise RuntimeError("#1133 full-screen token budget drift")
    return protocol_manifest()


def engineering_start_plan(token_count: int) -> torch.Tensor:
    """Explicit deterministic start offsets shared by RAW and VP."""
    token_count = int(token_count)
    high = token_count - SESSION_LEN - 1
    if high <= 0:
        raise ValueError("token source is too short for the frozen session length")
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED)
    return torch.randint(
        0,
        high,
        (TOTAL_ENGINEERING_STEPS, GRAD_ACCUM, MICRO_BATCH),
        generator=generator,
        dtype=torch.int64,
    )


def start_plan_sha256(plan: torch.Tensor) -> str:
    plan = plan.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(plan.numpy().tobytes(order="C")).hexdigest()


def host_staged_gather(
    source: np.ndarray,
    starts_cpu: torch.Tensor | Sequence[int],
    *,
    seq_len: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Slice on CPU and transfer only one selected microbatch.

    source may be a NumPy memmap. No full-source torch tensor is created and
    no TokenBin CUDA cache is touched.
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    starts = torch.as_tensor(starts_cpu, dtype=torch.int64, device="cpu").reshape(-1)
    if starts.numel() == 0:
        raise ValueError("starts_cpu must contain at least one start offset")

    starts_list = [int(x) for x in starts.tolist()]
    n_tokens = int(len(source))
    if min(starts_list) < 0 or max(starts_list) + seq_len >= n_tokens:
        raise ValueError("sample start is outside token source")

    rows = [
        np.asarray(source[start : start + seq_len + 1], dtype=np.int64)
        for start in starts_list
    ]
    host = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.int64)
    transferred_bytes = int(host.nbytes)
    chunks = torch.from_numpy(host).to(device=torch.device(device), dtype=torch.long)
    return chunks[:, :-1], chunks[:, 1:], transferred_bytes


def projected_three_model_run(
    *,
    raw_tokens_per_second: float,
    vp_tokens_per_second: float,
    live_hourly_resource_usd: float,
) -> dict[str, float]:
    raw_tps = float(raw_tokens_per_second)
    vp_tps = float(vp_tokens_per_second)
    hourly = float(live_hourly_resource_usd)
    if raw_tps <= 0 or vp_tps <= 0 or hourly <= 0:
        raise ValueError("throughputs and live hourly rate must be positive")
    local_seconds = TRAINING_TOKENS_PER_MODEL / HISTORICAL_LOCAL_TOKENS_PER_SECOND
    raw_seconds = TRAINING_TOKENS_PER_MODEL / raw_tps
    vp_seconds = TRAINING_TOKENS_PER_MODEL / vp_tps
    total_seconds = local_seconds + raw_seconds + vp_seconds
    return {
        "historical_local_tokens_per_second": HISTORICAL_LOCAL_TOKENS_PER_SECOND,
        "local_projected_seconds": local_seconds,
        "raw_projected_seconds": raw_seconds,
        "vp_projected_seconds": vp_seconds,
        "three_model_projected_seconds": total_seconds,
        "live_hourly_resource_usd": hourly,
        "three_model_projected_compute_usd": total_seconds / 3600.0 * hourly,
    }


def classify_systems_preflight(
    *,
    raw: dict[str, Any],
    vp: dict[str, Any],
    live_hourly_resource_usd: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    expected = (
        ("raw_eiem", raw, EXPECTED_EIEM_PARAMETERS, MIN_RAW_TOKENS_PER_SECOND),
        ("vp_eiem", vp, EXPECTED_VP_EIEM_PARAMETERS, MIN_VP_TOKENS_PER_SECOND),
    )
    for name, row, expected_parameters, min_tps in expected:
        if int(row.get("trainable_parameters", -1)) != int(expected_parameters):
            reasons.append(f"{name}_parameter_count_mismatch")
        if int(row.get("warmup_steps", -1)) != WARMUP_STEPS:
            reasons.append(f"{name}_warmup_step_count_mismatch")
        if int(row.get("measured_steps", -1)) != MEASURED_STEPS:
            reasons.append(f"{name}_measured_step_count_mismatch")
        if int(row.get("measured_tokens", -1)) != MEASURED_TOKENS_PER_MODEL:
            reasons.append(f"{name}_measured_token_count_mismatch")
        if not bool(row.get("finite_loss", False)):
            reasons.append(f"{name}_nonfinite_loss")
        if not bool(row.get("finite_gradients", False)):
            reasons.append(f"{name}_nonfinite_gradients")
        if not bool(row.get("finite_parameters", False)):
            reasons.append(f"{name}_nonfinite_parameters")
        if not bool(row.get("host_staged_transport", False)):
            reasons.append(f"{name}_host_staged_transport_not_verified")
        if bool(row.get("full_source_cuda_cache_present", True)):
            reasons.append(f"{name}_full_source_cuda_cache_present")
        if float(row.get("peak_allocated_gib", float("inf"))) > MAX_PEAK_ALLOCATED_GIB:
            reasons.append(f"{name}_peak_allocated_above_gate")
        if float(row.get("peak_reserved_gib", float("inf"))) > MAX_PEAK_RESERVED_GIB:
            reasons.append(f"{name}_peak_reserved_above_gate")
        if float(row.get("tokens_per_second", 0.0)) < min_tps:
            reasons.append(f"{name}_throughput_below_gate")

    projection = projected_three_model_run(
        raw_tokens_per_second=float(raw.get("tokens_per_second", 0.0)),
        vp_tokens_per_second=float(vp.get("tokens_per_second", 0.0)),
        live_hourly_resource_usd=live_hourly_resource_usd,
    )
    if projection["three_model_projected_seconds"] > MAX_PROJECTED_THREE_MODEL_SECONDS:
        reasons.append("projected_three_model_wall_time_above_gate")
    if projection["three_model_projected_compute_usd"] > MAX_PROJECTED_THREE_MODEL_COMPUTE_USD:
        reasons.append("projected_three_model_compute_cost_above_gate")

    passed = not reasons
    return {
        "classification": PASS_CLASSIFICATION if passed else STOP_CLASSIFICATION,
        "passed": passed,
        "stop_reasons": reasons,
        "projection": projection,
        "thresholds": {
            "max_peak_allocated_gib": MAX_PEAK_ALLOCATED_GIB,
            "max_peak_reserved_gib": MAX_PEAK_RESERVED_GIB,
            "min_raw_tokens_per_second": MIN_RAW_TOKENS_PER_SECOND,
            "min_vp_tokens_per_second": MIN_VP_TOKENS_PER_SECOND,
            "max_projected_three_model_seconds": MAX_PROJECTED_THREE_MODEL_SECONDS,
            "max_projected_three_model_compute_usd": MAX_PROJECTED_THREE_MODEL_COMPUTE_USD,
        },
        "interpretation_ceiling": (
            "Systems feasibility only. PASS does not authorize a scientific seed, "
            "scientific rerun, Stage D, scale-up, or replication."
        ),
    }
