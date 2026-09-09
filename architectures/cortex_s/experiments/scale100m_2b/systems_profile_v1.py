from __future__ import annotations

import json
import math
import time
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from architectures.cortex_s.experiments.scale100m_2b.protocol import (
    CORTEX_100M_CONFIG,
    DATA_DIR,
    EXPECTED_CORTEX_PARAMS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    RESERVED_FRESH_SEEDS,
    SEQ_LEN,
    TOKENS_PER_OPTIMIZER_STEP,
)
from architectures.cortex_s.experiments.scale100m_2b.train import (
    _compile_model,
    _make_optimizer,
    _one_optimizer_step,
    seed_all,
)
from architectures.cortex_s.grouped_moe import (
    PhysicalPaddedGroupedSparseMoE,
    build_production_grouped_cortex_100m,
)
from architectures.cortex_s.language_model import parameter_count
from tam_research.data import TokenBin


CLASSIFICATION = "ENGINEERING_SYSTEMS_PROFILE_ONLY"
ENGINEERING_SEED = 2_026_090_906
CONSUMED_ENGINEERING_SEEDS = (
    910_001,
    2_026_090_901,
    2_026_090_902,
    2_026_090_903,
    2_026_090_904,
    2_026_090_905,
)
FORBIDDEN_SCIENTIFIC_SEEDS = (8_100, *RESERVED_FRESH_SEEDS)
TRIGGER_TITLE = "[modal-cortex-s-100m-systems-profile-v1]"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-profile-v1"

# Parent production preflight is consumed. This profile may diagnose it, never
# reinterpret it or change its frozen 8,500-second decision.
PARENT_V3_SOURCE_SHA = "3a9538228d100a40818030a64b5b9eaf7311f94a"
PARENT_V3_ISSUE = 802
PARENT_V3_RUN_ID = 34_345_899_535
PARENT_V3_JOB_ID = 102_447_296_828
PARENT_V3_CALIBRATION_SEED = 2_026_090_905
PARENT_V3_TPS = 241_834.85880920192
PARENT_V3_PROJECTED_FULL_SECONDS = 9_463.844871814677
PARENT_V3_FULL_GATE_SECONDS = 8_500.0
HISTORICAL_TRANSFORMER_TPS = 321_151.5755348581

# This is a bounded diagnostic, not another calibration or candidate benchmark.
COMPILE_TRIGGER_STEPS = 1
ADDITIONAL_WARMUP_STEPS = 3
MEASURED_STEPS = 10
PROFILE_STEPS = 2
PROFILE_TOP_K = 30
WEIGHT_MATERIALIZATION_REPEATS = 20
MAX_PEAK_VRAM_GIB = 70.0


def validate_profile_protocol() -> dict[str, Any]:
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("profile-v1 engineering seed is consumed or scientifically forbidden")
    if PARENT_V3_CALIBRATION_SEED not in CONSUMED_ENGINEERING_SEEDS:
        raise RuntimeError("parent v3 engineering seed must remain recorded as consumed")
    if MICRO_BATCH_SIZE != 64 or SEQ_LEN != 512 or GRAD_ACCUM_STEPS != 2:
        raise RuntimeError("profile-v1 must preserve the exact production batch shape")
    if TOKENS_PER_OPTIMIZER_STEP != 65_536:
        raise RuntimeError("profile-v1 production token-step contract drift")
    if CORTEX_100M_CONFIG.expert_hidden != 338:
        raise RuntimeError("profile-v1 logical expert width drift")
    if PROFILE_STEPS <= 0 or MEASURED_STEPS <= 0 or WEIGHT_MATERIALIZATION_REPEATS <= 0:
        raise RuntimeError("profile-v1 measurement counts must be positive")
    return {
        "classification": CLASSIFICATION,
        "engineering_seed": ENGINEERING_SEED,
        "consumed_engineering_seeds": list(CONSUMED_ENGINEERING_SEEDS),
        "forbidden_scientific_seeds": list(FORBIDDEN_SCIENTIFIC_SEEDS),
        "trigger_title": TRIGGER_TITLE,
        "result_root": RESULT_ROOT,
        "parent_v3_source_sha": PARENT_V3_SOURCE_SHA,
        "parent_v3_issue": PARENT_V3_ISSUE,
        "parent_v3_run_id": PARENT_V3_RUN_ID,
        "parent_v3_job_id": PARENT_V3_JOB_ID,
        "parent_v3_tps": PARENT_V3_TPS,
        "parent_v3_projected_full_seconds": PARENT_V3_PROJECTED_FULL_SECONDS,
        "parent_v3_full_gate_seconds": PARENT_V3_FULL_GATE_SECONDS,
        "historical_transformer_tps": HISTORICAL_TRANSFORMER_TPS,
        "compile_trigger_steps": COMPILE_TRIGGER_STEPS,
        "additional_warmup_steps": ADDITIONAL_WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "profile_steps": PROFILE_STEPS,
        "profile_top_k": PROFILE_TOP_K,
        "weight_materialization_repeats": WEIGHT_MATERIALIZATION_REPEATS,
        "full_training_authorized": False,
        "next_stage_authorized": False,
        "scientific_evidence": False,
    }


def zero_gpu_profile_contract_probe() -> dict[str, Any]:
    protocol = validate_profile_protocol()
    model = build_production_grouped_cortex_100m()
    actual = parameter_count(model)
    grouped_layers = sum(
        isinstance(block.moe, PhysicalPaddedGroupedSparseMoE) for block in model.blocks
    )
    if actual != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"profile-v1 parameter drift: {actual} != {EXPECTED_CORTEX_PARAMS}")
    if grouped_layers != CORTEX_100M_CONFIG.n_layers:
        raise RuntimeError("profile-v1 does not use grouped MoE in every production block")
    if any(block.moe.physical_hidden != 344 for block in model.blocks):
        raise RuntimeError("profile-v1 physical grouped width drift")
    return {
        "status": "PASS",
        "parameter_count": actual,
        "grouped_moe_layers": grouped_layers,
        "logical_expert_hidden": CORTEX_100M_CONFIG.expert_hidden,
        "physical_expert_hidden": 344,
        "production_moe_backend": "physical_padded_grouped_bf16",
        "protocol": protocol,
        "gpu_allocated": False,
        "full_training_authorized": False,
        "next_stage_authorized": False,
    }


def _event_device_time_us(event: Any) -> float:
    # PyTorch has renamed CUDA-specific profiler fields toward generic device
    # fields. Read either spelling so this diagnostic survives minor 2.10 builds.
    for name in (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _profile_top_events(profiler: Any) -> list[dict[str, Any]]:
    rows = []
    for event in profiler.key_averages():
        device_us = _event_device_time_us(event)
        cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        count = int(getattr(event, "count", 0) or 0)
        key = str(getattr(event, "key", getattr(event, "name", "unknown")))
        rows.append(
            {
                "key": key,
                "count": count,
                "self_device_time_us": device_us,
                "self_cpu_time_us": cpu_us,
            }
        )
    rows.sort(key=lambda row: row["self_device_time_us"], reverse=True)
    return rows[:PROFILE_TOP_K]


def _measure_weight_materialization(model: torch.nn.Module, device: torch.device) -> dict[str, float]:
    grouped = [block.moe for block in model.blocks]
    if len(grouped) != CORTEX_100M_CONFIG.n_layers:
        raise RuntimeError("weight materialization probe expected all grouped MoE blocks")

    # The current production grouped path physically pads FP32 logical parameters
    # and then casts the physical matrices to BF16 inside grouped_mm. Measure this
    # exact pad+cast work for every MoE block, without claiming it equals removable
    # end-to-end time. The checksum keeps the operations live until synchronization.
    with torch.no_grad():
        for _ in range(2):
            checksum = torch.zeros((), device=device, dtype=torch.float32)
            for moe in grouped:
                w1, w2 = moe._physical_matrices()
                w1_bf16 = w1.to(dtype=torch.bfloat16)
                w2_bf16 = w2.to(dtype=torch.bfloat16)
                checksum = checksum + w1_bf16[0, 0, 0].float() + w2_bf16[0, 0, 0].float()
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        checksum = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(WEIGHT_MATERIALIZATION_REPEATS):
            for moe in grouped:
                w1, w2 = moe._physical_matrices()
                w1_bf16 = w1.to(dtype=torch.bfloat16)
                w2_bf16 = w2.to(dtype=torch.bfloat16)
                checksum = checksum + w1_bf16[0, 0, 0].float() + w2_bf16[0, 0, 0].float()
        end.record()
        torch.cuda.synchronize(device)
        total_ms = float(start.elapsed_time(end))
        # Materialize checksum once so the probe cannot be optimized away.
        checksum_value = float(checksum)

    per_forward_ms = total_ms / WEIGHT_MATERIALIZATION_REPEATS
    return {
        "repeats": float(WEIGHT_MATERIALIZATION_REPEATS),
        "total_ms": total_ms,
        "ms_per_full_model_forward": per_forward_ms,
        "estimated_ms_per_optimizer_step_at_grad_accum_2": per_forward_ms * GRAD_ACCUM_STEPS,
        "checksum": checksum_value,
    }


def run_h100_systems_profile(*, source_sha: str, data_dir: str = DATA_DIR) -> dict[str, Any]:
    """Profile one exact compiled grouped production graph with an engineering seed.

    This consumes only ENGINEERING_SEED. It neither evaluates language quality nor
    authorizes any candidate implementation, scientific seed, or 2B run.
    """

    protocol = validate_profile_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("systems profile-v1 requires CUDA")

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    seed_all(ENGINEERING_SEED)
    model = build_production_grouped_cortex_100m().to(device)
    if parameter_count(model) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("profile-v1 exact model parameter count mismatch")
    optimizer = _make_optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 10_000)
    train_data = TokenBin(f"{data_dir}/train.bin")

    compile_started = time.perf_counter()
    runner = _compile_model(model)
    last_loss = _one_optimizer_step(
        model=model,
        runner=runner,
        optimizer=optimizer,
        train_data=train_data,
        generator=generator,
        device=device,
        lr=LEARNING_RATE,
    )
    torch.cuda.synchronize(device)
    compile_seconds = max(time.perf_counter() - compile_started, 0.0)

    for _ in range(ADDITIONAL_WARMUP_STEPS):
        last_loss = _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=LEARNING_RATE,
        )
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    measured_started = time.perf_counter()
    for _ in range(MEASURED_STEPS):
        last_loss = _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=LEARNING_RATE,
        )
    torch.cuda.synchronize(device)
    measured_seconds = max(time.perf_counter() - measured_started, 1e-9)
    measured_tokens = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP
    measured_tps = measured_tokens / measured_seconds
    measured_step_ms = 1000.0 * measured_seconds / MEASURED_STEPS
    measured_peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    # Profile only after clean throughput timing. Profiler overhead is diagnostic and
    # is never used as the production throughput estimate.
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as prof:
        for _ in range(PROFILE_STEPS):
            with record_function("cortex_s_profile_optimizer_step"):
                last_loss = _one_optimizer_step(
                    model=model,
                    runner=runner,
                    optimizer=optimizer,
                    train_data=train_data,
                    generator=generator,
                    device=device,
                    lr=LEARNING_RATE,
                )
        torch.cuda.synchronize(device)

    top_events = _profile_top_events(prof)
    profiler_table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=PROFILE_TOP_K,
    )

    materialization = _measure_weight_materialization(model, device)
    materialization["estimated_fraction_of_measured_optimizer_step"] = (
        materialization["estimated_ms_per_optimizer_step_at_grad_accum_2"]
        / max(measured_step_ms, 1e-9)
    )

    finite_loss = math.isfinite(float(last_loss))
    result = {
        "status": "PROFILE_COMPLETE" if finite_loss else "PROFILE_NONFINITE_ABORT",
        "scientific_status": CLASSIFICATION,
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "execution": "compiled",
        "compile_mode": "max-autotune-no-cudagraphs",
        "compile_seconds": compile_seconds,
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_seconds,
        "measured_optimizer_step_ms": measured_step_ms,
        "measured_training_tokens_per_second": measured_tps,
        "parent_v3_training_tokens_per_second": PARENT_V3_TPS,
        "throughput_ratio_profile_over_parent_v3": measured_tps / PARENT_V3_TPS,
        "historical_transformer_tokens_per_second": HISTORICAL_TRANSFORMER_TPS,
        "throughput_ratio_profile_over_transformer": measured_tps / HISTORICAL_TRANSFORMER_TPS,
        "peak_vram_gib": measured_peak_vram_gib,
        "last_train_loss": float(last_loss),
        "finite_loss": finite_loss,
        "profile_steps": PROFILE_STEPS,
        "top_device_events": top_events,
        "profiler_table": profiler_table,
        "physical_weight_pad_cast_probe": materialization,
        "protocol": protocol,
        "full_training_authorized": False,
        "next_stage_authorized": False,
        "scientific_evidence": False,
        "note": (
            "Diagnostic profile only. It may motivate a separately preregistered systems candidate, "
            "but cannot authorize, rerun, or reinterpret the consumed v3 production preflight."
        ),
    }
    if measured_peak_vram_gib > MAX_PEAK_VRAM_GIB:
        result["status"] = "PROFILE_VRAM_GUARD_FAIL"
    return json.loads(json.dumps(result))
