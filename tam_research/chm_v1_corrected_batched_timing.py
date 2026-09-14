from __future__ import annotations

"""Frozen CPU-only systems timing protocol for CHM-v1 issue #971.

This module compares the generic frozen batched indexed path with the corrected
#970 clipped batched path.  It is deliberately synthetic and CPU-only: no
scientific corpus, training, CUDA/Modal execution, or scientific seed is
reachable here.  Timing is descriptive systems evidence; correctness is gated
before any timing sample is interpreted.
"""

from collections import defaultdict
import platform
import time
from typing import Any, Literal

import numpy as np
import torch

from .chm_v1_batched_eval import retrieve_many_exact
from .chm_v1_optimized_batched_eval import retrieve_many_optimized_exact
from .chm_v1_optimized_replication import OptimizedEpisodicState
from .chm_v1_post_materialization_profiler import (
    QueryGeometry,
    build_query_batch,
    synthetic_state_payload,
)

ISSUE = 971
DIAGNOSTIC_SEEDS = (971_001, 971_002)
SMOKE_SEED = 971_099
BLOCKED_SCIENTIFIC_SEEDS = (19_591, 19_592, 19_593, 8_611, 8_612, 8_613)
MEMORY_SIZES = (512, 1_024, 4_096)
GEOMETRIES: tuple[QueryGeometry, ...] = ("exact", "near", "random")
ADDRESS_WIDTH = 32
VALUE_WIDTH = 256
QUERY_COUNT = 32
NEAR_NOISE_STD = 0.05
REPEATS = 7
PathName = Literal["frozen", "corrected"]
RegimeName = Literal["warm", "cold"]


def validate_systems_seed(seed: int, *, protocol: bool = False) -> int:
    seed = int(seed)
    if seed in BLOCKED_SCIENTIFIC_SEEDS:
        raise RuntimeError(
            f"scientific seed refused by #971 CPU systems timing protocol: {seed}"
        )
    if protocol and seed not in DIAGNOSTIC_SEEDS:
        raise RuntimeError(f"unfrozen #971 protocol seed refused: {seed}")
    return seed


def _state(name: str, keys: torch.Tensor, values: torch.Tensor) -> OptimizedEpisodicState:
    if keys.device.type != "cpu" or values.device.type != "cpu":
        raise RuntimeError("#971 timing accepts CPU tensors only")
    state = OptimizedEpisodicState(name)
    state.write(keys, values)
    return state


def _query_seed(seed: int, geometry: QueryGeometry) -> int:
    offset = {"exact": 11, "near": 23, "random": 37}[geometry]
    return int(seed + offset)


def _correctness_gate(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    label: str,
) -> dict[str, Any]:
    frozen_state = _state(f"{label}-correctness-frozen", keys, values)
    corrected_state = _state(f"{label}-correctness-corrected", keys, values)

    frozen = retrieve_many_exact(
        frozen_state,
        queries,
        mode="indexed",
        verify_indexed_exactness=True,
    )
    corrected = retrieve_many_optimized_exact(
        corrected_state,
        queries,
        verify_indexed_exactness=True,
    )

    if corrected.results != frozen.results:
        mismatch = next(
            i
            for i, (left, right) in enumerate(zip(corrected.results, frozen.results))
            if left != right
        )
        raise AssertionError(
            f"corrected/frozen full SearchResult mismatch at {label} query {mismatch}: "
            f"corrected={corrected.results[mismatch]} frozen={frozen.results[mismatch]}"
        )
    if not torch.equal(corrected.values, frozen.values):
        raise AssertionError(f"corrected/frozen selected-value mismatch at {label}")

    query_matrix = queries.detach().float().cpu().numpy().astype(np.float32, copy=False)
    flat_answer_matches = 0
    for query, indexed in zip(query_matrix, corrected.results):
        flat = frozen_state.index().flat_search(query)
        match = indexed.item_id == flat.item_id and indexed.position == flat.position
        flat_answer_matches += int(match)
        if not match:
            raise AssertionError(
                f"corrected/flat answer mismatch at {label}: corrected={indexed} flat={flat}"
            )

    return {
        "query_count": int(len(corrected.results)),
        "corrected_frozen_full_result_parity": True,
        "corrected_frozen_value_parity": True,
        "flat_answer_match_rate": flat_answer_matches / max(len(corrected.results), 1),
        "address_vector_reads": int(
            sum(result.address_vector_reads for result in corrected.results)
        ),
        "directory_nodes_visited": int(
            sum(result.directory_nodes_visited for result in corrected.results)
        ),
    }


def _measure_call(
    path: PathName,
    state: OptimizedEpisodicState,
    queries: torch.Tensor,
) -> dict[str, int]:
    before_index_build = float(state.index_build_seconds_total)
    before_search = float(state.indexed_search_seconds_total)
    before_sidecar = float(state.optimized_sidecar_build_seconds_total)
    before_wrapper = float(state.optimized_wrapper_seconds_total)

    started = time.perf_counter_ns()
    if path == "frozen":
        retrieve_many_exact(
            state,
            queries,
            mode="indexed",
            verify_indexed_exactness=False,
        )
    elif path == "corrected":
        retrieve_many_optimized_exact(
            state,
            queries,
            verify_indexed_exactness=False,
        )
    else:
        raise ValueError(f"unknown path {path!r}")
    wall_ns = time.perf_counter_ns() - started

    index_build_ns = int(
        round((state.index_build_seconds_total - before_index_build) * 1e9)
    )
    search_ns = int(round((state.indexed_search_seconds_total - before_search) * 1e9))
    sidecar_ns = int(
        round((state.optimized_sidecar_build_seconds_total - before_sidecar) * 1e9)
    )
    optimized_wrapper_ns = int(
        round((state.optimized_wrapper_seconds_total - before_wrapper) * 1e9)
    )

    if path == "corrected":
        # The #970 wrapper counter includes sidecar construction on a cold call.
        residual_wrapper_ns = max(0, optimized_wrapper_ns - sidecar_ns)
    else:
        residual_wrapper_ns = max(0, wall_ns - index_build_ns - search_ns)

    return {
        "wall_ns": int(wall_ns),
        "index_build_ns": int(index_build_ns),
        "sidecar_build_ns": int(sidecar_ns),
        "search_ns": int(search_ns),
        "wrapper_ns": int(residual_wrapper_ns),
    }


def _median_components(samples: list[dict[str, int]]) -> dict[str, int]:
    if len(samples) != REPEATS:
        raise AssertionError(f"expected {REPEATS} samples, got {len(samples)}")
    keys = samples[0].keys()
    return {
        key: int(np.median(np.asarray([sample[key] for sample in samples], dtype=np.int64)))
        for key in keys
    }


def _timing_regime(
    *,
    regime: RegimeName,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    label: str,
) -> dict[str, Any]:
    samples: dict[PathName, list[dict[str, int]]] = {
        "frozen": [],
        "corrected": [],
    }

    warm_states: dict[PathName, OptimizedEpisodicState] | None = None
    if regime == "warm":
        warm_states = {
            "frozen": _state(f"{label}-warm-frozen", keys, values),
            "corrected": _state(f"{label}-warm-corrected", keys, values),
        }
        warm_states["frozen"].index()
        warm_states["corrected"].optimized_index()
    elif regime != "cold":
        raise ValueError(f"unknown regime {regime!r}")

    for repeat in range(REPEATS):
        order: tuple[PathName, PathName] = (
            ("frozen", "corrected") if repeat % 2 == 0 else ("corrected", "frozen")
        )
        for path in order:
            if regime == "warm":
                assert warm_states is not None
                state = warm_states[path]
            else:
                state = _state(f"{label}-cold-{path}-{repeat}", keys, values)
            samples[path].append(_measure_call(path, state, queries))

    frozen_median = _median_components(samples["frozen"])
    corrected_median = _median_components(samples["corrected"])
    ratio = corrected_median["wall_ns"] / max(frozen_median["wall_ns"], 1)
    return {
        "regime": regime,
        "repeats": REPEATS,
        "alternating_order": True,
        "frozen_samples": samples["frozen"],
        "corrected_samples": samples["corrected"],
        "frozen_median": frozen_median,
        "corrected_median": corrected_median,
        "corrected_over_frozen_wall_ratio": float(ratio),
    }


def benchmark_configuration(
    *,
    seed: int,
    memory_size: int,
    geometry: QueryGeometry,
) -> dict[str, Any]:
    """Run one frozen #971 configuration after an untimed correctness gate."""

    seed = validate_systems_seed(seed)
    if memory_size not in MEMORY_SIZES:
        raise ValueError(f"memory_size must be one of {MEMORY_SIZES}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}")

    keys, values = synthetic_state_payload(
        session_length=memory_size,
        key_width=ADDRESS_WIDTH,
        value_width=VALUE_WIDTH,
        seed=seed,
    )
    batch = build_query_batch(
        keys,
        query_count=QUERY_COUNT,
        geometry=geometry,
        noise_std=NEAR_NOISE_STD,
        seed=_query_seed(seed, geometry),
    )
    label = f"issue-{ISSUE}-seed-{seed}-n-{memory_size}-{geometry}"

    correctness = _correctness_gate(
        keys=keys,
        values=values,
        queries=batch.queries,
        label=label,
    )
    warm = _timing_regime(
        regime="warm",
        keys=keys,
        values=values,
        queries=batch.queries,
        label=label,
    )
    cold = _timing_regime(
        regime="cold",
        keys=keys,
        values=values,
        queries=batch.queries,
        label=label,
    )

    return {
        "seed": int(seed),
        "memory_size": int(memory_size),
        "geometry": geometry,
        "noise_std": float(batch.noise_std),
        "address_width": ADDRESS_WIDTH,
        "value_width": VALUE_WIDTH,
        "query_count": QUERY_COUNT,
        "correctness": correctness,
        "warm": warm,
        "cold": cold,
    }


def classify_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(DIAGNOSTIC_SEEDS) * len(MEMORY_SIZES) * len(GEOMETRIES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} protocol rows, got {len(rows)}")

    by_key = {
        (int(row["seed"]), int(row["memory_size"]), str(row["geometry"])): row
        for row in rows
    }
    primary_checks: list[dict[str, Any]] = []
    for seed in DIAGNOSTIC_SEEDS:
        for memory_size in MEMORY_SIZES:
            row = by_key[(seed, memory_size, "near")]
            ratio = float(row["warm"]["corrected_over_frozen_wall_ratio"])
            threshold = 1.05 if memory_size == 512 else 0.90
            primary_checks.append(
                {
                    "seed": seed,
                    "memory_size": memory_size,
                    "ratio": ratio,
                    "threshold": threshold,
                    "pass": ratio <= threshold,
                }
            )
    warm_pass = all(check["pass"] for check in primary_checks)

    cold_checks: list[dict[str, Any]] = []
    for seed in DIAGNOSTIC_SEEDS:
        for memory_size in (1_024, 4_096):
            row = by_key[(seed, memory_size, "near")]
            ratio = float(row["cold"]["corrected_over_frozen_wall_ratio"])
            cold_checks.append(
                {
                    "seed": seed,
                    "memory_size": memory_size,
                    "ratio": ratio,
                    "threshold": 1.10,
                    "pass": ratio <= 1.10,
                }
            )
    cold_pass = all(check["pass"] for check in cold_checks)

    if not warm_pass:
        classification = "CORRECTED_BATCHED_SYSTEMS_STOP"
    elif cold_pass:
        classification = "CORRECTED_BATCHED_SYSTEMS_PASS"
    else:
        classification = "CORRECTED_BATCHED_AMORTIZATION_REQUIRED"

    warnings: list[dict[str, Any]] = []
    for regime in ("warm", "cold"):
        for memory_size in MEMORY_SIZES:
            for geometry in ("exact", "random"):
                ratios = [
                    float(
                        by_key[(seed, memory_size, geometry)][regime][
                            "corrected_over_frozen_wall_ratio"
                        ]
                    )
                    for seed in DIAGNOSTIC_SEEDS
                ]
                if all(ratio > 1.20 for ratio in ratios):
                    warnings.append(
                        {
                            "regime": regime,
                            "memory_size": memory_size,
                            "geometry": geometry,
                            "ratios": ratios,
                            "classification": "DESCRIPTIVE_REGRESSION_WARNING",
                        }
                    )

    return {
        "classification": classification,
        "warm_primary_pass": warm_pass,
        "cold_secondary_pass": cold_pass,
        "primary_checks": primary_checks,
        "cold_checks": cold_checks,
        "descriptive_regression_warnings": warnings,
    }


def run_frozen_protocol() -> dict[str, Any]:
    """Execute the complete frozen #971 CPU timing protocol exactly once."""

    for seed in DIAGNOSTIC_SEEDS:
        validate_systems_seed(seed, protocol=True)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in DIAGNOSTIC_SEEDS:
        for memory_size in MEMORY_SIZES:
            for geometry in GEOMETRIES:
                rows.append(
                    benchmark_configuration(
                        seed=seed,
                        memory_size=memory_size,
                        geometry=geometry,
                    )
                )

    decision = classify_protocol(rows)
    return {
        "measurement_kind": "chm_v1_corrected_batched_systems_timing_cpu",
        "issue": ISSUE,
        "scientific_credit": False,
        "gpu_authorized": False,
        "paid_scientific_compute": False,
        "modal_trigger": False,
        "timing_is_ci_gate": False,
        "device": "cpu",
        "protocol": {
            "diagnostic_seeds": list(DIAGNOSTIC_SEEDS),
            "memory_sizes": list(MEMORY_SIZES),
            "geometries": list(GEOMETRIES),
            "address_width": ADDRESS_WIDTH,
            "value_width": VALUE_WIDTH,
            "query_count": QUERY_COUNT,
            "near_noise_std": NEAR_NOISE_STD,
            "repeats": REPEATS,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
        "decision": decision,
        "interpretation_ceiling": (
            "CPU systems timing only. Even a pass does not authorize scientific seeds, "
            "GPU/Modal execution, training, or claims of modeling/capability improvement."
        ),
    }
