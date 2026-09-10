from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any

import torch

STEADY_WARMUP_STEPS = 5
STEADY_MEASURED_STEPS = 8
PROFILED_STEPS = 2
TOP_OPERATORS = 30

_CATEGORY_ORDER = (
    "grouped_moe",
    "recurrent_scan",
    "attention",
    "gemm_matmul",
    "routing_gather_scatter",
    "host_sync",
    "loss",
    "optimizer",
    "normalization_elementwise",
    "other",
)


def categorize_operator(name: str) -> str:
    key = name.lower()
    if any(token in key for token in ("grouped_mm", "grouped_moe", "groupedmoe")):
        return "grouped_moe"
    if any(token in key for token in ("affine_scan", "scan_triton", "cortex_affine", "associative_scan")):
        return "recurrent_scan"
    if any(token in key for token in ("scaled_dot_product", "flash_attention", "efficient_attention", "sdpa", "attention")):
        return "attention"
    if any(token in key for token in ("_local_scalar_dense", "cudaeventsynchronize", "cudadevicesynchronize", "cudastreamsynchronize", "synchronize")):
        return "host_sync"
    if any(token in key for token in ("cross_entropy", "nll_loss", "log_softmax")):
        return "loss"
    if any(token in key for token in ("adamw", "optimizer.step", "_foreach_", "clip_grad", "linalg_vector_norm")):
        return "optimizer"
    if any(token in key for token in ("index_select", "index_add", "scatter", "gather", "topk", "sort", "argsort", "bincount", "nonzero")):
        return "routing_gather_scatter"
    if any(token in key for token in ("gemm", "addmm", "bmm", "matmul", "aten::mm", "triton_mm")):
        return "gemm_matmul"
    if any(token in key for token in ("layer_norm", "native_layer_norm", "gelu", "sigmoid", "tanh", "softmax", "mul", "add", "sub", "div")):
        return "normalization_elementwise"
    return "other"


def _self_device_us(event: Any) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _self_cpu_us(event: Any) -> float:
    value = getattr(event, "self_cpu_time_total", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _event_record(event: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(event, "key", "")),
        "count": int(getattr(event, "count", 0)),
        "self_device_time_us": _self_device_us(event),
        "self_cpu_time_us": _self_cpu_us(event),
    }


def _aggregate(events: list[Any]) -> dict[str, Any]:
    device_by_category: dict[str, float] = defaultdict(float)
    cpu_by_category: dict[str, float] = defaultdict(float)
    calls_by_category: dict[str, int] = defaultdict(int)

    records = [_event_record(event) for event in events]
    for record in records:
        category = categorize_operator(record["name"])
        device_by_category[category] += float(record["self_device_time_us"])
        cpu_by_category[category] += float(record["self_cpu_time_us"])
        calls_by_category[category] += int(record["count"])

    total_device = sum(device_by_category.values())
    total_cpu = sum(cpu_by_category.values())

    categories: dict[str, Any] = {}
    for category in _CATEGORY_ORDER:
        device_us = float(device_by_category.get(category, 0.0))
        cpu_us = float(cpu_by_category.get(category, 0.0))
        categories[category] = {
            "self_device_time_us": device_us,
            "device_share": device_us / total_device if total_device > 0 else 0.0,
            "self_cpu_time_us": cpu_us,
            "cpu_share": cpu_us / total_cpu if total_cpu > 0 else 0.0,
            "calls": int(calls_by_category.get(category, 0)),
        }

    top_device = sorted(records, key=lambda item: item["self_device_time_us"], reverse=True)[:TOP_OPERATORS]
    top_cpu = sorted(records, key=lambda item: item["self_cpu_time_us"], reverse=True)[:TOP_OPERATORS]

    return {
        "total_self_device_time_us": total_device,
        "total_self_cpu_time_us": total_cpu,
        "categories": categories,
        "top_device_operators": top_device,
        "top_cpu_operators": top_cpu,
    }


def run_full_step_profile(
    *,
    training_module: Any,
    model: torch.nn.Module,
    runner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    generator: torch.Generator,
    device: torch.device,
    learning_rate: float,
) -> dict[str, Any]:
    """Profile the exact production optimizer-step function after compile/warmup.

    Throughput is measured only outside torch.profiler. Profiler timings are
    diagnostic attribution and must not be treated as a replacement systems gate.
    """

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-step profile requires CUDA")

    for _ in range(STEADY_WARMUP_STEPS):
        loss = training_module._one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=learning_rate,
        )
        if not math.isfinite(float(loss)):
            raise FloatingPointError("non-finite loss during profiler warmup")
    torch.cuda.synchronize(device)

    tokens_per_step = (
        int(training_module.MICRO_BATCH_SIZE)
        * int(training_module.SEQ_LEN)
        * int(training_module.GRAD_ACCUM_STEPS)
    )
    started = time.perf_counter()
    steady_losses: list[float] = []
    for _ in range(STEADY_MEASURED_STEPS):
        steady_losses.append(
            float(
                training_module._one_optimizer_step(
                    model=model,
                    runner=runner,
                    optimizer=optimizer,
                    train_data=train_data,
                    generator=generator,
                    device=device,
                    lr=learning_rate,
                )
            )
        )
    torch.cuda.synchronize(device)
    steady_seconds = max(time.perf_counter() - started, 1e-9)
    steady_tokens = STEADY_MEASURED_STEPS * tokens_per_step
    steady_tps = steady_tokens / steady_seconds

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        profiled_losses: list[float] = []
        for _ in range(PROFILED_STEPS):
            profiled_losses.append(
                float(
                    training_module._one_optimizer_step(
                        model=model,
                        runner=runner,
                        optimizer=optimizer,
                        train_data=train_data,
                        generator=generator,
                        device=device,
                        lr=learning_rate,
                    )
                )
            )
    torch.cuda.synchronize(device)

    if not all(math.isfinite(value) for value in steady_losses + profiled_losses):
        raise FloatingPointError("non-finite loss in full-step profile")

    averaged = prof.key_averages()
    aggregate = _aggregate(list(averaged))

    sync_keys = (
        "_local_scalar_dense",
        "cudaEventSynchronize",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
    )
    synchronization = []
    for event in averaged:
        name = str(getattr(event, "key", ""))
        if any(token.lower() in name.lower() for token in sync_keys):
            synchronization.append(_event_record(event))
    synchronization.sort(key=lambda item: item["self_cpu_time_us"], reverse=True)

    return {
        "classification": "ENGINEERING_FULL_STEP_PROFILE_V2_ONLY",
        "steady_state": {
            "warmup_steps": STEADY_WARMUP_STEPS,
            "measured_steps": STEADY_MEASURED_STEPS,
            "measured_tokens": steady_tokens,
            "measured_seconds": steady_seconds,
            "training_tokens_per_second": steady_tps,
            "last_loss": steady_losses[-1],
        },
        "profiler": {
            "profiled_steps": PROFILED_STEPS,
            "note": "Profiler self times are diagnostic attribution; CPU and device totals can overlap and must not be summed as wall time.",
            **aggregate,
            "synchronization_operators": synchronization,
        },
    }
