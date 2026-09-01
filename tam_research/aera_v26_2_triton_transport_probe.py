from __future__ import annotations

"""Issue #408 synthetic L4 gate for the AERA-v26.2 fused Triton transport.

This module is intentionally systems-only. It constructs production-shaped synthetic
state tensors and compares the merged v26.1 componentwise reference transport with
the concrete v26.2 two-kernel Triton backend. It never imports or builds an AERA
model, loads a checkpoint/corpus, trains, creates an optimizer, calls backward, or
consumes a scientific seed.
"""

from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable

import torch

from .aera import AERAState
from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v26_1 import TorchComponentwiseStateTransport
from .aera_hardware_core_v26_2_triton import (
    TritonFusedStateTransport,
    fused_triton_transport_v26_2_protocol,
)

RESEARCH_ISSUE = 408
DESIGN_SEED = 406_408
D_MODEL = 200
MEMORY_DIM = 50
CAPACITY = 48
BATCH_SIZES: tuple[int, ...] = (8, 64)
SELECTED_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
DTYPE_NAMES: tuple[str, ...] = ("float32", "bfloat16")

WARMUP_CALLS = 10
TIMED_ROUNDS = 5
CALLS_PER_ROUND = 200
MAX_KERNEL_RATIO = 0.25
MAX_BATCH64_GEOMEAN_LATENCY_RATIO = 0.90
MAX_BATCH64_ROW_LATENCY_RATIO = 1.05
MAX_BATCH8_ROW_LATENCY_RATIO = 1.10


@dataclass(frozen=True)
class SyntheticTransportCase:
    dtype_name: str
    batch_size: int
    selected_fraction: float
    run_idx: torch.Tensor
    base: AERAState
    update: AERAState


def issue408_protocol() -> dict[str, Any]:
    protocol = dict(fused_triton_transport_v26_2_protocol())
    protocol.update(
        {
            "probe_version": "aera-v26.2-issue408-synthetic-triton-transport-l4",
            "design_seed": DESIGN_SEED,
            "design_seed_is_scientific_seed": False,
            "d_model": D_MODEL,
            "memory_dim": MEMORY_DIM,
            "capacity": CAPACITY,
            "batch_sizes": list(BATCH_SIZES),
            "selected_fractions": list(SELECTED_FRACTIONS),
            "dtypes": list(DTYPE_NAMES),
            "warmup_calls": WARMUP_CALLS,
            "timed_rounds": TIMED_ROUNDS,
            "calls_per_round": CALLS_PER_ROUND,
            "timing_clock": "CUDA events",
            "timing_order": "reference/candidate interleaved and rotated by round",
            "max_kernel_ratio_each_row": MAX_KERNEL_RATIO,
            "max_batch64_geomean_latency_ratio_each_dtype": MAX_BATCH64_GEOMEAN_LATENCY_RATIO,
            "max_batch64_row_latency_ratio": MAX_BATCH64_ROW_LATENCY_RATIO,
            "max_batch8_row_latency_ratio": MAX_BATCH8_ROW_LATENCY_RATIO,
            "synthetic_only": True,
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
    )
    return protocol


def cpu_contract_preflight() -> dict[str, Any]:
    if (D_MODEL, MEMORY_DIM, CAPACITY) != (200, 50, 48):
        raise RuntimeError("issue408 production geometry drifted")
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue408 batch sizes drifted")
    if SELECTED_FRACTIONS != (0.25, 0.50, 0.75, 1.00):
        raise RuntimeError("issue408 selected fractions drifted")
    if DTYPE_NAMES != ("float32", "bfloat16"):
        raise RuntimeError("issue408 dtype set drifted")
    if (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) != (10, 5, 200):
        raise RuntimeError("issue408 timing protocol drifted")
    if (
        MAX_KERNEL_RATIO != 0.25
        or MAX_BATCH64_GEOMEAN_LATENCY_RATIO != 0.90
        or MAX_BATCH64_ROW_LATENCY_RATIO != 1.05
        or MAX_BATCH8_ROW_LATENCY_RATIO != 1.10
    ):
        raise RuntimeError("issue408 frozen PASS thresholds drifted")
    return {
        "protocol": issue408_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "scientific_seed_consumed": False,
    }


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported issue408 dtype: {name}")


def _make_random_state(
    *,
    batch_size: int,
    dtype: torch.dtype,
    generator: torch.Generator,
    device: torch.device,
) -> AERAState:
    # Construct all randomness on CPU from the design-only generator, then cast and
    # transfer. GPU RNG behavior therefore cannot alter the frozen synthetic cases.
    stream = torch.randn(batch_size, D_MODEL, generator=generator).to(dtype=dtype)
    keys = torch.randn(
        batch_size, CAPACITY, MEMORY_DIM, generator=generator
    ).to(dtype=dtype)
    values = torch.randn(
        batch_size, CAPACITY, MEMORY_DIM, generator=generator
    ).to(dtype=dtype)
    strengths = torch.rand(batch_size, CAPACITY, generator=generator).to(dtype=dtype)
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
    dtype_name: str,
    batch_size: int,
    selected_fraction: float,
    generator: torch.Generator,
    device: torch.device,
) -> SyntheticTransportCase:
    if dtype_name not in DTYPE_NAMES:
        raise ValueError("issue408 unsupported dtype")
    if batch_size not in BATCH_SIZES:
        raise ValueError("issue408 unsupported batch size")
    if selected_fraction not in SELECTED_FRACTIONS:
        raise ValueError("issue408 unsupported selected fraction")
    dtype = _dtype_from_name(dtype_name)
    run_count = max(1, int(round(batch_size * selected_fraction)))
    permutation = torch.randperm(batch_size, generator=generator)
    run_idx = permutation[:run_count].sort().values.to(device=device, dtype=torch.long)
    base = _make_random_state(
        batch_size=batch_size,
        dtype=dtype,
        generator=generator,
        device=device,
    )
    update = _make_random_state(
        batch_size=run_count,
        dtype=dtype,
        generator=generator,
        device=device,
    )
    return SyntheticTransportCase(
        dtype_name=dtype_name,
        batch_size=batch_size,
        selected_fraction=selected_fraction,
        run_idx=run_idx,
        base=base,
        update=update,
    )


def _state_tensors(state: AERAState) -> tuple[torch.Tensor, ...]:
    memory = state.memory
    if not isinstance(memory, ContextualEpisodicMemoryState):
        raise TypeError("issue408 requires contextual episodic memory state")
    return (
        state.stream,
        memory.keys,
        memory.values,
        memory.strengths,
        memory.valid,
    )


def _clone_state(state: AERAState) -> AERAState:
    memory = state.memory
    if not isinstance(memory, ContextualEpisodicMemoryState):
        raise TypeError("issue408 requires contextual episodic memory state")
    return AERAState(
        stream=state.stream.clone(),
        memory=ContextualEpisodicMemoryState(
            keys=memory.keys.clone(),
            values=memory.values.clone(),
            strengths=memory.strengths.clone(),
            valid=memory.valid.clone(),
        ),
    )


def _state_bit_equal(left: AERAState, right: AERAState) -> bool:
    return all(
        torch.equal(a, b)
        for a, b in zip(_state_tensors(left), _state_tensors(right))
    )


def _state_finite(state: AERAState) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in _state_tensors(state)[:-1])


def _state_meta_equal(left: AERAState, right: AERAState) -> bool:
    return all(
        a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        for a, b in zip(_state_tensors(left), _state_tensors(right))
    )


def _row_key(dtype_name: str, batch_size: int, fraction: float) -> str:
    return (
        f"{dtype_name}_batch{batch_size}_fraction"
        f"{int(round(fraction * 100)):03d}"
    )


def correctness_row(
    case: SyntheticTransportCase,
    *,
    reference: TorchComponentwiseStateTransport,
    candidate: TritonFusedStateTransport,
) -> dict[str, Any]:
    base_before = _clone_state(case.base)
    update_before = _clone_state(case.update)
    run_idx_before = case.run_idx.clone()

    reference_selected = reference.select(case.base, case.run_idx)
    reference_merged = reference.merge(case.base, case.update, case.run_idx)
    candidate_selected = candidate.select(case.base, case.run_idx)
    candidate_merged = candidate.merge(case.base, case.update, case.run_idx)
    torch.cuda.synchronize()

    selected_exact = _state_bit_equal(reference_selected, candidate_selected)
    merged_exact = _state_bit_equal(reference_merged, candidate_merged)
    meta_exact = _state_meta_equal(reference_selected, candidate_selected) and _state_meta_equal(
        reference_merged, candidate_merged
    )
    source_unchanged = (
        _state_bit_equal(case.base, base_before)
        and _state_bit_equal(case.update, update_before)
        and torch.equal(case.run_idx, run_idx_before)
    )
    finite = all(
        _state_finite(state)
        for state in (
            reference_selected,
            reference_merged,
            candidate_selected,
            candidate_merged,
        )
    )
    candidate_cuda_only = all(
        tensor.device.type == "cuda"
        for state in (candidate_selected, candidate_merged)
        for tensor in _state_tensors(state)
    )
    passed = bool(
        selected_exact
        and merged_exact
        and meta_exact
        and source_unchanged
        and finite
        and candidate_cuda_only
    )
    return {
        "pass": passed,
        "selected_exact": selected_exact,
        "merged_exact": merged_exact,
        "dtype_device_shape_exact": meta_exact,
        "source_and_index_unchanged": source_unchanged,
        "finite": finite,
        "candidate_cuda_only": candidate_cuda_only,
    }


def _transport_call(
    transport: Any,
    case: SyntheticTransportCase,
) -> tuple[AERAState, AERAState]:
    selected = transport.select(case.base, case.run_idx)
    merged = transport.merge(case.base, case.update, case.run_idx)
    return selected, merged


def _timed_round_us(call: Callable[[], tuple[AERAState, AERAState]]) -> float:
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
        raise RuntimeError("issue408 timed loop produced no output")
    return float(start.elapsed_time(end)) * 1000.0 / CALLS_PER_ROUND


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

    return {
        name: {
            "round_us_per_call": values,
            "median_us_per_call": float(statistics.median(values)),
            "mean_us_per_call": float(statistics.fmean(values)),
        }
        for name, values in samples.items()
    }


def _profile_call(call: Callable[[], tuple[AERAState, AERAState]]) -> dict[str, Any]:
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
    cuda_names: list[str] = []
    for event in profile.events():
        device_type = getattr(event, "device_type", None)
        if device_type == torch.autograd.DeviceType.CUDA or str(device_type).endswith(
            "CUDA"
        ):
            cuda_events += 1
            cuda_names.append(str(getattr(event, "name", "")))

    relevant_calls = {
        "cat": 0,
        "stack": 0,
        "index_select": 0,
        "index_copy": 0,
    }
    for item in profile.key_averages():
        key = str(item.key).lower()
        for token in relevant_calls:
            if token in key:
                relevant_calls[token] += int(item.count)

    triton_names = sorted(
        {
            name
            for name in cuda_names
            if "fused_select_kernel" in name or "fused_merge_kernel" in name
        }
    )
    return {
        "cuda_device_events": int(cuda_events),
        "cuda_event_names": cuda_names,
        "triton_transport_kernel_names": triton_names,
        "relevant_operator_calls": relevant_calls,
    }


def _peak_vram(call: Callable[[], tuple[AERAState, AERAState]]) -> dict[str, float]:
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


def run_triton_transport_probe() -> dict[str, Any]:
    cpu_contract_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("issue408 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue408 requires NVIDIA L4, found {device_name}")

    reference = TorchComponentwiseStateTransport()
    candidate = TritonFusedStateTransport(max_batch=max(BATCH_SIZES))
    generator = torch.Generator().manual_seed(DESIGN_SEED)
    rows: dict[str, dict[str, Any]] = {}

    for dtype_name in DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            for fraction in SELECTED_FRACTIONS:
                case = make_case(
                    dtype_name=dtype_name,
                    batch_size=batch_size,
                    selected_fraction=fraction,
                    generator=generator,
                    device=device,
                )
                key = _row_key(dtype_name, batch_size, fraction)
                correctness = correctness_row(
                    case, reference=reference, candidate=candidate
                )
                if not correctness["pass"]:
                    raise RuntimeError(f"issue408 GPU correctness failed for {key}")

                calls = {
                    "reference": lambda case=case: _transport_call(reference, case),
                    "candidate": lambda case=case: _transport_call(candidate, case),
                }
                timing = _timed_summaries(calls)
                reference_us = timing["reference"]["median_us_per_call"]
                candidate_us = timing["candidate"]["median_us_per_call"]
                latency_ratio = candidate_us / reference_us

                profiles = {name: _profile_call(call) for name, call in calls.items()}
                reference_events = profiles["reference"]["cuda_device_events"]
                candidate_events = profiles["candidate"]["cuda_device_events"]
                if reference_events <= 0 or candidate_events <= 0:
                    raise RuntimeError(
                        f"issue408 profiler found no CUDA device events for {key}"
                    )
                kernel_ratio = candidate_events / reference_events
                candidate_ops = profiles["candidate"]["relevant_operator_calls"]
                no_pack_ops = candidate_ops["cat"] == 0 and candidate_ops["stack"] == 0
                no_auxiliary_cuda_kernel = candidate_events <= 2

                rows[key] = {
                    "dtype": dtype_name,
                    "batch_size": batch_size,
                    "selected_fraction": fraction,
                    "run_count": int(case.run_idx.numel()),
                    "correctness": correctness,
                    "timing": timing,
                    "latency_ratio_candidate_over_reference": float(latency_ratio),
                    "profiles": profiles,
                    "kernel_event_ratio_candidate_over_reference": float(kernel_ratio),
                    "vram": {name: _peak_vram(call) for name, call in calls.items()},
                    "kernel_ratio_pass": bool(kernel_ratio <= MAX_KERNEL_RATIO),
                    "no_cat_or_stack_pass": bool(no_pack_ops),
                    "no_auxiliary_cuda_kernel_pass": bool(no_auxiliary_cuda_kernel),
                    "row_latency_pass": bool(
                        latency_ratio
                        <= (
                            MAX_BATCH64_ROW_LATENCY_RATIO
                            if batch_size == 64
                            else MAX_BATCH8_ROW_LATENCY_RATIO
                        )
                    ),
                }

    dtype_geomeans: dict[str, float] = {}
    dtype_geomean_pass: dict[str, bool] = {}
    for dtype_name in DTYPE_NAMES:
        ratios = [
            row["latency_ratio_candidate_over_reference"]
            for row in rows.values()
            if row["dtype"] == dtype_name and row["batch_size"] == 64
        ]
        geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        dtype_geomeans[dtype_name] = float(geomean)
        dtype_geomean_pass[dtype_name] = bool(
            geomean <= MAX_BATCH64_GEOMEAN_LATENCY_RATIO
        )

    correctness_pass = all(row["correctness"]["pass"] for row in rows.values())
    kernel_pass = all(row["kernel_ratio_pass"] for row in rows.values())
    row_latency_pass = all(row["row_latency_pass"] for row in rows.values())
    no_pack_ops_pass = all(row["no_cat_or_stack_pass"] for row in rows.values())
    no_auxiliary_kernel_pass = all(
        row["no_auxiliary_cuda_kernel_pass"] for row in rows.values()
    )
    geomean_pass = all(dtype_geomean_pass.values())
    overall_pass = bool(
        correctness_pass
        and kernel_pass
        and row_latency_pass
        and no_pack_ops_pass
        and no_auxiliary_kernel_pass
        and geomean_pass
    )

    return {
        "protocol": issue408_protocol(),
        "device": device_name,
        "rows": rows,
        "batch64_geomean_latency_ratio_by_dtype": dtype_geomeans,
        "batch64_geomean_latency_pass_by_dtype": dtype_geomean_pass,
        "correctness_pass": correctness_pass,
        "kernel_launch_reduction_pass": kernel_pass,
        "row_latency_pass": row_latency_pass,
        "no_cat_or_stack_pass": no_pack_ops_pass,
        "no_auxiliary_cuda_kernel_pass": no_auxiliary_kernel_pass,
        "batch64_geomean_latency_pass": geomean_pass,
        "overall_pass": overall_pass,
        "decision": "PASS" if overall_pass else "FAIL",
        "synthetic_only": True,
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
