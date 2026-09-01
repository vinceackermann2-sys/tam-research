from __future__ import annotations

"""Issue #400 synthetic AERA-v26 coalesced-state movement probe.

This is a systems-only primitive benchmark. It uses synthetic production-shaped
state tensors and never loads a model, corpus, checkpoint, optimizer, or scientific
seed. The only GPU-authorized caller is the separately guarded issue #400 Modal
launcher/workflow.
"""

from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable

import torch

from .aera import AERAState
from .aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    _merge_epi_state,
    _select_epi_state,
)
from .aera_hardware_core_v26 import (
    merge_packed_epi_state,
    pack_ephemeral_epi_state,
    select_packed_epi_state,
    unpack_ephemeral_epi_state,
)


RESEARCH_ISSUE = 400
DESIGN_SEED = 398_400
D_MODEL = 200
MEMORY_DIM = 50
CAPACITY = 48
PACKED_FLOAT_WIDTH = D_MODEL + 2 * CAPACITY * MEMORY_DIM + CAPACITY
BATCH_SIZES: tuple[int, ...] = (8, 64)
SELECTED_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

WARMUP_CALLS = 10
TIMED_ROUNDS = 5
CALLS_PER_ROUND = 200

MAX_KERNEL_RATIO = 0.80
MAX_BATCH64_GEOMEAN_LATENCY_RATIO = 0.90
MAX_BATCH64_ROW_LATENCY_RATIO = 1.05
MAX_BATCH8_ROW_LATENCY_RATIO = 1.10


@dataclass(frozen=True)
class SyntheticMovementCase:
    batch_size: int
    selected_fraction: float
    run_idx: torch.Tensor
    base: AERAState
    update: AERAState


def issue400_protocol() -> dict[str, Any]:
    return {
        "version": "aera-v26-issue400-synthetic-state-movement",
        "research_issue": RESEARCH_ISSUE,
        "design_seed": DESIGN_SEED,
        "design_seed_is_scientific_seed": False,
        "d_model": D_MODEL,
        "memory_dim": MEMORY_DIM,
        "capacity": CAPACITY,
        "packed_float_width": PACKED_FLOAT_WIDTH,
        "batch_sizes": list(BATCH_SIZES),
        "selected_fractions": list(SELECTED_FRACTIONS),
        "warmup_calls": WARMUP_CALLS,
        "timed_rounds": TIMED_ROUNDS,
        "calls_per_round": CALLS_PER_ROUND,
        "timing_clock": "CUDA events",
        "timing_order": "legacy/v26 interleaved and rotated by round",
        "max_kernel_ratio_each_row": MAX_KERNEL_RATIO,
        "max_batch64_geomean_latency_ratio": MAX_BATCH64_GEOMEAN_LATENCY_RATIO,
        "max_batch64_row_latency_ratio": MAX_BATCH64_ROW_LATENCY_RATIO,
        "max_batch8_row_latency_ratio": MAX_BATCH8_ROW_LATENCY_RATIO,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def cpu_contract_preflight() -> dict[str, Any]:
    if PACKED_FLOAT_WIDTH != 5_048:
        raise RuntimeError("issue400 packed float width drifted")
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue400 batch sizes drifted")
    if SELECTED_FRACTIONS != (0.25, 0.50, 0.75, 1.00):
        raise RuntimeError("issue400 selected fractions drifted")
    if (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) != (10, 5, 200):
        raise RuntimeError("issue400 timing protocol drifted")
    if (
        MAX_KERNEL_RATIO != 0.80
        or MAX_BATCH64_GEOMEAN_LATENCY_RATIO != 0.90
        or MAX_BATCH64_ROW_LATENCY_RATIO != 1.05
        or MAX_BATCH8_ROW_LATENCY_RATIO != 1.10
    ):
        raise RuntimeError("issue400 PASS thresholds drifted")
    return {
        "protocol": issue400_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
    }


def _make_random_state(
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> AERAState:
    # Generate on CPU from the fixed design generator, then transfer. This keeps
    # synthetic case construction independent of GPU RNG implementation details.
    stream = torch.randn(batch_size, D_MODEL, generator=generator)
    keys = torch.randn(batch_size, CAPACITY, MEMORY_DIM, generator=generator)
    values = torch.randn(batch_size, CAPACITY, MEMORY_DIM, generator=generator)
    strengths = torch.rand(batch_size, CAPACITY, generator=generator)
    valid = torch.rand(batch_size, CAPACITY, generator=generator) > 0.37
    return AERAState(
        stream=stream.to(device),
        memory=ContextualEpisodicMemoryState(
            keys=keys.to(device),
            values=values.to(device),
            strengths=strengths.to(device),
            valid=valid.to(device),
        ),
    )


def make_case(
    *,
    batch_size: int,
    selected_fraction: float,
    generator: torch.Generator,
    device: torch.device,
) -> SyntheticMovementCase:
    if batch_size not in BATCH_SIZES:
        raise ValueError("issue400 unsupported batch size")
    if selected_fraction not in SELECTED_FRACTIONS:
        raise ValueError("issue400 unsupported selected fraction")
    run_count = max(1, int(round(batch_size * selected_fraction)))
    permutation = torch.randperm(batch_size, generator=generator)
    run_idx = permutation[:run_count].sort().values.to(device)
    base = _make_random_state(
        batch_size=batch_size, generator=generator, device=device
    )
    update = _make_random_state(
        batch_size=run_count, generator=generator, device=device
    )
    return SyntheticMovementCase(
        batch_size=batch_size,
        selected_fraction=selected_fraction,
        run_idx=run_idx,
        base=base,
        update=update,
    )


def legacy_componentwise_movement(
    case: SyntheticMovementCase,
) -> tuple[AERAState, AERAState]:
    selected = _select_epi_state(case.base, case.run_idx)
    merged = _merge_epi_state(case.base, case.update, case.run_idx)
    return selected, merged


def v26_coalesced_movement(
    case: SyntheticMovementCase,
) -> tuple[AERAState, AERAState]:
    base_packed = pack_ephemeral_epi_state(case.base)
    selected_packed = select_packed_epi_state(base_packed, case.run_idx)
    selected = unpack_ephemeral_epi_state(selected_packed)
    update_packed = pack_ephemeral_epi_state(case.update)
    merged_packed = merge_packed_epi_state(
        base_packed, update_packed, case.run_idx
    )
    merged = unpack_ephemeral_epi_state(merged_packed)
    return selected, merged


def _state_tensors(state: AERAState) -> tuple[torch.Tensor, ...]:
    memory = state.memory
    if not isinstance(memory, ContextualEpisodicMemoryState):
        raise TypeError("issue400 requires contextual episodic memory state")
    return (
        state.stream,
        memory.keys,
        memory.values,
        memory.strengths,
        memory.valid,
    )


def _state_bit_equal(left: AERAState, right: AERAState) -> bool:
    return all(
        torch.equal(a, b)
        for a, b in zip(_state_tensors(left), _state_tensors(right))
    )


def _state_finite(state: AERAState) -> bool:
    tensors = _state_tensors(state)
    return all(bool(torch.isfinite(t).all()) for t in tensors[:-1])


def _clone_state(state: AERAState) -> AERAState:
    memory = state.memory
    if not isinstance(memory, ContextualEpisodicMemoryState):
        raise TypeError("issue400 requires contextual episodic memory state")
    return AERAState(
        stream=state.stream.clone(),
        memory=ContextualEpisodicMemoryState(
            keys=memory.keys.clone(),
            values=memory.values.clone(),
            strengths=memory.strengths.clone(),
            valid=memory.valid.clone(),
        ),
    )


def correctness_row(case: SyntheticMovementCase) -> dict[str, Any]:
    base_before = _clone_state(case.base)
    update_before = _clone_state(case.update)
    legacy_selected, legacy_merged = legacy_componentwise_movement(case)
    v26_selected, v26_merged = v26_coalesced_movement(case)

    selected_exact = _state_bit_equal(legacy_selected, v26_selected)
    merged_exact = _state_bit_equal(legacy_merged, v26_merged)
    source_unchanged = _state_bit_equal(case.base, base_before) and _state_bit_equal(
        case.update, update_before
    )
    finite = all(
        _state_finite(state)
        for state in (legacy_selected, legacy_merged, v26_selected, v26_merged)
    )
    output_device_matches = all(
        tensor.device.type == case.run_idx.device.type
        for tensor in _state_tensors(v26_merged)
    )
    passed = bool(
        selected_exact
        and merged_exact
        and source_unchanged
        and finite
        and output_device_matches
    )
    return {
        "pass": passed,
        "selected_exact": selected_exact,
        "merged_exact": merged_exact,
        "source_unchanged": source_unchanged,
        "finite": finite,
        "output_device_matches_index_device": output_device_matches,
    }


def run_cpu_correctness_matrix() -> dict[str, dict[str, Any]]:
    generator = torch.Generator().manual_seed(DESIGN_SEED)
    rows: dict[str, dict[str, Any]] = {}
    for batch_size in BATCH_SIZES:
        for fraction in SELECTED_FRACTIONS:
            case = make_case(
                batch_size=batch_size,
                selected_fraction=fraction,
                generator=generator,
                device=torch.device("cpu"),
            )
            key = _row_key(batch_size, fraction)
            rows[key] = correctness_row(case)
            if not rows[key]["pass"]:
                raise RuntimeError(f"issue400 CPU correctness failed for {key}")
    return rows


def _row_key(batch_size: int, fraction: float) -> str:
    return f"batch{batch_size}_fraction{int(round(fraction * 100)):03d}"


def _timed_round_us(
    call: Callable[[], tuple[AERAState, AERAState]],
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    output = None
    for _ in range(CALLS_PER_ROUND):
        output = call()
    end.record()
    end.synchronize()
    if output is None:
        raise RuntimeError("issue400 timed loop produced no output")
    elapsed_ms = float(start.elapsed_time(end))
    return elapsed_ms * 1000.0 / CALLS_PER_ROUND


def _timed_summaries(
    calls: dict[str, Callable[[], tuple[AERAState, AERAState]]],
) -> dict[str, dict[str, Any]]:
    for call in calls.values():
        for _ in range(WARMUP_CALLS):
            call()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in calls}
    names = list(calls)
    for round_index in range(TIMED_ROUNDS):
        order = list(reversed(names)) if round_index % 2 else names
        for name in order:
            samples[name].append(_timed_round_us(calls[name]))

    result: dict[str, dict[str, Any]] = {}
    for name, values in samples.items():
        result[name] = {
            "round_us_per_call": values,
            "median_us_per_call": float(statistics.median(values)),
            "mean_us_per_call": float(statistics.fmean(values)),
        }
    return result


def _profile_call(
    call: Callable[[], tuple[AERAState, AERAState]],
) -> dict[str, Any]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        output = call()
    torch.cuda.synchronize()
    del output

    cuda_events = 0
    for event in profile.events():
        device_type = getattr(event, "device_type", None)
        if device_type == torch.autograd.DeviceType.CUDA or str(device_type).endswith(
            "CUDA"
        ):
            cuda_events += 1

    relevant_calls = {"cat": 0, "index_select": 0, "index_copy": 0}
    for item in profile.key_averages():
        key = str(item.key).lower()
        for token in relevant_calls:
            if token in key:
                relevant_calls[token] += int(item.count)
    return {
        "cuda_device_events": int(cuda_events),
        "relevant_operator_calls": relevant_calls,
    }


def _peak_vram(
    call: Callable[[], tuple[AERAState, AERAState]],
) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = call()
    torch.cuda.synchronize()
    result = {
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
    }
    del output
    return result


def run_state_movement_probe() -> dict[str, Any]:
    cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("issue400 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue400 requires NVIDIA L4, found {device_name}")

    generator = torch.Generator().manual_seed(DESIGN_SEED)
    rows: dict[str, dict[str, Any]] = {}
    for batch_size in BATCH_SIZES:
        for fraction in SELECTED_FRACTIONS:
            case = make_case(
                batch_size=batch_size,
                selected_fraction=fraction,
                generator=generator,
                device=device,
            )
            key = _row_key(batch_size, fraction)
            correctness = correctness_row(case)
            if not correctness["pass"]:
                raise RuntimeError(f"issue400 GPU correctness failed for {key}")

            calls = {
                "legacy": lambda case=case: legacy_componentwise_movement(case),
                "v26": lambda case=case: v26_coalesced_movement(case),
            }
            timing = _timed_summaries(calls)
            legacy_us = timing["legacy"]["median_us_per_call"]
            v26_us = timing["v26"]["median_us_per_call"]
            latency_ratio = v26_us / legacy_us

            profiles = {name: _profile_call(call) for name, call in calls.items()}
            legacy_kernels = profiles["legacy"]["cuda_device_events"]
            v26_kernels = profiles["v26"]["cuda_device_events"]
            if legacy_kernels <= 0 or v26_kernels <= 0:
                raise RuntimeError(
                    f"issue400 profiler found no CUDA device events for {key}"
                )
            kernel_ratio = v26_kernels / legacy_kernels

            rows[key] = {
                "batch_size": batch_size,
                "selected_fraction": fraction,
                "run_count": int(case.run_idx.numel()),
                "correctness": correctness,
                "timing": timing,
                "latency_ratio_v26_over_legacy": float(latency_ratio),
                "profiles": profiles,
                "kernel_event_ratio_v26_over_legacy": float(kernel_ratio),
                "vram": {name: _peak_vram(call) for name, call in calls.items()},
                "kernel_ratio_pass": bool(kernel_ratio <= MAX_KERNEL_RATIO),
                "row_latency_pass": bool(
                    latency_ratio
                    <= (
                        MAX_BATCH64_ROW_LATENCY_RATIO
                        if batch_size == 64
                        else MAX_BATCH8_ROW_LATENCY_RATIO
                    )
                ),
            }

    batch64_ratios = [
        row["latency_ratio_v26_over_legacy"]
        for row in rows.values()
        if row["batch_size"] == 64
    ]
    batch64_geomean = math.exp(
        sum(math.log(value) for value in batch64_ratios) / len(batch64_ratios)
    )
    correctness_pass = all(row["correctness"]["pass"] for row in rows.values())
    kernel_pass = all(row["kernel_ratio_pass"] for row in rows.values())
    row_latency_pass = all(row["row_latency_pass"] for row in rows.values())
    geomean_pass = batch64_geomean <= MAX_BATCH64_GEOMEAN_LATENCY_RATIO
    overall_pass = bool(
        correctness_pass and kernel_pass and row_latency_pass and geomean_pass
    )

    return {
        "protocol": issue400_protocol(),
        "device": device_name,
        "rows": rows,
        "batch64_geomean_latency_ratio_v26_over_legacy": float(batch64_geomean),
        "correctness_pass": correctness_pass,
        "kernel_launch_reduction_pass": kernel_pass,
        "row_latency_pass": row_latency_pass,
        "batch64_geomean_latency_pass": geomean_pass,
        "overall_pass": overall_pass,
        "decision": "PASS" if overall_pass else "FAIL",
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
