from __future__ import annotations

from collections import Counter, defaultdict
import math
import time
from typing import Any

import torch

STEADY_WARMUP_STEPS = 5
STEADY_MEASURED_STEPS = 4
PROFILED_STEPS = 2
EXPECTED_GROUPED_MM_CALLS_PER_OPTIMIZER_STEP = 96
GROUPED_MM_DOMINANT_MIN_PER_STEP = 80.0
OTHER_DOMINANT_MIN_SHARE = 0.50
ATTRIBUTION_COVERAGE_MIN_SHARE = 0.90
MAX_PARENT_DEPTH = 16
MAX_STACK_FRAMES = 12
MAX_RAW_SYNC_RECORDS = 120

_MEANINGFUL_CLASSES = (
    "grouped_mm",
    "training_scalar_sync",
    "routing",
    "recurrent_scan",
    "optimizer",
    "loss",
    "compiled_graph_other",
    "runtime_other",
    "unknown",
)


def _name(event: Any) -> str:
    return str(getattr(event, "key", getattr(event, "name", "")) or "")


def _cpu_us(event: Any) -> float:
    value = getattr(event, "self_cpu_time_total", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _device_us(event: Any) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _stack(event: Any) -> list[str]:
    value = getattr(event, "stack", None)
    if not value:
        return []
    frames = []
    for frame in list(value)[-MAX_STACK_FRAMES:]:
        frames.append(str(frame))
    return frames


def _parent_chain(event: Any) -> list[str]:
    chain: list[str] = []
    seen: set[int] = set()
    parent = getattr(event, "cpu_parent", None)
    while parent is not None and len(chain) < MAX_PARENT_DEPTH:
        identity = id(parent)
        if identity in seen:
            break
        seen.add(identity)
        chain.append(_name(parent))
        parent = getattr(parent, "cpu_parent", None)
    return chain


def classify_source(event: Any) -> tuple[str, list[str], list[str]]:
    parents = _parent_chain(event)
    stack = _stack(event)
    evidence = "\n".join([_name(event), *parents, *stack]).lower()

    # Order is intentional: preserve the most specific production path before
    # generic compiled-region/runtime labels.
    if any(token in evidence for token in ("grouped_mm", "_grouped_mm", "triton_grouped_mm")):
        source = "grouped_mm"
    elif any(
        token in evidence
        for token in (
            "_local_scalar_dense",
            "aten::item",
            "torch.isfinite",
            "isfinite",
            "float(loss",
            "clip_grad_norm",
        )
    ):
        source = "training_scalar_sync"
    elif any(
        token in evidence
        for token in ("aten::sort", "aten::topk", "aten::bincount", "index_add", "index_select", "scatter", "gather", "argsort")
    ):
        source = "routing"
    elif any(token in evidence for token in ("affine_scan", "scan_triton", "associative_scan", "cortex_affine")):
        source = "recurrent_scan"
    elif any(token in evidence for token in ("adamw", "optimizer", "_foreach_", "clip_grad")):
        source = "optimizer"
    elif any(token in evidence for token in ("cross_entropy", "nll_loss", "log_softmax")):
        source = "loss"
    elif any(token in evidence for token in ("compiledfxgraph", "compiledfunction", "torch-compiled", "compiled region")):
        source = "compiled_graph_other"
    elif parents:
        source = "runtime_other"
    else:
        source = "unknown"
    return source, parents, stack


def _sync_record(event: Any) -> dict[str, Any]:
    source, parents, stack = classify_source(event)
    return {
        "name": _name(event),
        "source": source,
        "self_cpu_time_us": _cpu_us(event),
        "self_device_time_us": _device_us(event),
        "parent_chain": parents,
        "stack": stack,
    }


def _is_stream_sync(event: Any) -> bool:
    return "cudastreamsynchronize" in _name(event).lower()


def _is_sync_event(event: Any) -> bool:
    lowered = _name(event).lower()
    return any(
        token in lowered
        for token in (
            "cudastreamsynchronize",
            "cudadevicesynchronize",
            "cudaeventsynchronize",
            "_local_scalar_dense",
        )
    )


def aggregate_sync_attribution(events: list[Any], profiled_steps: int = PROFILED_STEPS) -> dict[str, Any]:
    if profiled_steps <= 0:
        raise ValueError("profiled_steps must be positive")

    records = [_sync_record(event) for event in events if _is_sync_event(event)]
    stream_records = [record for record in records if record["name"].lower().find("cudastreamsynchronize") >= 0]
    counts = Counter(record["source"] for record in stream_records)
    cpu_by_source: dict[str, float] = defaultdict(float)
    for record in stream_records:
        cpu_by_source[record["source"]] += float(record["self_cpu_time_us"])

    total_stream = len(stream_records)
    per_step = {
        source: float(counts.get(source, 0)) / float(profiled_steps)
        for source in _MEANINGFUL_CLASSES
    }
    known_count = sum(
        count
        for source, count in counts.items()
        if source not in {"unknown", "compiled_graph_other"}
    )
    coverage = float(known_count) / float(total_stream) if total_stream else 0.0
    grouped_per_step = per_step["grouped_mm"]

    ranked = sorted(
        ((source, int(count), float(cpu_by_source.get(source, 0.0))) for source, count in counts.items()),
        key=lambda row: (row[1], row[2]),
        reverse=True,
    )
    largest_source = ranked[0][0] if ranked else "unknown"
    largest_count = ranked[0][1] if ranked else 0

    if total_stream and grouped_per_step >= GROUPED_MM_DOMINANT_MIN_PER_STEP and largest_source == "grouped_mm":
        classification = "GROUPED_MM_HOST_SYNC_DOMINANT"
    elif total_stream and largest_source not in {"grouped_mm", "unknown", "compiled_graph_other"} and largest_count / total_stream >= OTHER_DOMINANT_MIN_SHARE:
        classification = "OTHER_HOST_SYNC_SOURCE_DOMINANT"
    elif total_stream and coverage >= ATTRIBUTION_COVERAGE_MIN_SHARE and largest_count / total_stream < OTHER_DOMINANT_MIN_SHARE:
        classification = "MIXED_HOST_SYNC_SOURCES"
    else:
        classification = "ATTRIBUTION_INCONCLUSIVE"

    raw_top = sorted(records, key=lambda item: float(item["self_cpu_time_us"]), reverse=True)[:MAX_RAW_SYNC_RECORDS]
    return {
        "classification": classification,
        "profiled_steps": int(profiled_steps),
        "stream_sync_events": total_stream,
        "stream_sync_events_per_optimizer_step": float(total_stream) / float(profiled_steps),
        "expected_grouped_mm_calls_per_optimizer_step": EXPECTED_GROUPED_MM_CALLS_PER_OPTIMIZER_STEP,
        "stream_sync_counts_by_source": {source: int(counts.get(source, 0)) for source in _MEANINGFUL_CLASSES},
        "stream_sync_events_per_step_by_source": per_step,
        "stream_sync_cpu_self_time_us_by_source": {
            source: float(cpu_by_source.get(source, 0.0)) for source in _MEANINGFUL_CLASSES
        },
        "non_generic_attribution_coverage": coverage,
        "largest_source": largest_source,
        "largest_source_count": int(largest_count),
        "raw_sync_records_top_cpu": raw_top,
    }


def run_host_sync_profile(
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
    """Attribute synchronization in the exact production optimizer-step path.

    Unprofiled throughput is a sanity measurement only. Profiler self-times are
    diagnostic attribution and must never substitute for the frozen systems gate.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("host-sync profile requires CUDA")

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
            raise FloatingPointError("non-finite loss during attribution warmup")
    torch.cuda.synchronize(device)

    tokens_per_step = (
        int(training_module.MICRO_BATCH_SIZE)
        * int(training_module.SEQ_LEN)
        * int(training_module.GRAD_ACCUM_STEPS)
    )
    start = time.perf_counter()
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
    steady_seconds = max(time.perf_counter() - start, 1e-9)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
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
        raise FloatingPointError("non-finite loss in host-sync attribution profile")

    events = list(prof.events())
    attribution = aggregate_sync_attribution(events, PROFILED_STEPS)
    key_averages = list(prof.key_averages())
    top_cpu = sorted(
        (
            {
                "name": _name(event),
                "count": int(getattr(event, "count", 0)),
                "self_cpu_time_us": _cpu_us(event),
                "self_device_time_us": _device_us(event),
            }
            for event in key_averages
        ),
        key=lambda item: item["self_cpu_time_us"],
        reverse=True,
    )[:30]

    return {
        "classification": "ENGINEERING_HOST_SYNC_ATTRIBUTION_V1_ONLY",
        "steady_state_sanity": {
            "warmup_steps": STEADY_WARMUP_STEPS,
            "measured_steps": STEADY_MEASURED_STEPS,
            "measured_tokens": STEADY_MEASURED_STEPS * tokens_per_step,
            "measured_seconds": steady_seconds,
            "training_tokens_per_second": (STEADY_MEASURED_STEPS * tokens_per_step) / steady_seconds,
            "last_loss": steady_losses[-1],
        },
        "profiler": {
            "profiled_steps": PROFILED_STEPS,
            "with_stack": True,
            "event_count": len(events),
            "attribution": attribution,
            "top_cpu_operators": top_cpu,
            "note": (
                "Profiler timings are diagnostic only. CPU and device self times can overlap; "
                "this result does not replace the frozen 100M/2B systems gate."
            ),
        },
    }
