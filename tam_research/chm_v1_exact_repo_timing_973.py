from __future__ import annotations

"""Exact-repository CPU systems replication for CHM-v1 issue #973.

This runner reuses the merged #971 benchmark implementation directly and changes
only the fresh diagnostic seed namespace and terminal classification labels. It
contains no CUDA/Modal path, training loop, optimizer, backward pass, corpus
loader, or scientific-seed execution authority.
"""

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .chm_v1_corrected_batched_timing import (
    ADDRESS_WIDTH,
    GEOMETRIES,
    MEMORY_SIZES,
    NEAR_NOISE_STD,
    QUERY_COUNT,
    REPEATS,
    VALUE_WIDTH,
    benchmark_configuration,
)

ISSUE = 973
DIAGNOSTIC_SEEDS = (973_001, 973_002)
SMOKE_SEED = 973_099
BLOCKED_SCIENTIFIC_SEEDS = (19_591, 19_592, 19_593, 8_611, 8_612, 8_613)
BLOCKED_CONSUMED_DIAGNOSTIC_SEEDS = (971_001, 971_002)


def validate_seed(seed: int, *, protocol: bool = False) -> int:
    seed = int(seed)
    if seed in BLOCKED_SCIENTIFIC_SEEDS:
        raise RuntimeError(f"scientific seed refused by #973 exact-repo CPU runner: {seed}")
    if seed in BLOCKED_CONSUMED_DIAGNOSTIC_SEEDS:
        raise RuntimeError(f"consumed #971 diagnostic seed refused by #973 runner: {seed}")
    if protocol and seed not in DIAGNOSTIC_SEEDS:
        raise RuntimeError(f"unfrozen #973 protocol seed refused: {seed}")
    return seed


def _expected_keys() -> set[tuple[int, int, str]]:
    return {
        (seed, memory_size, geometry)
        for seed in DIAGNOSTIC_SEEDS
        for memory_size in MEMORY_SIZES
        for geometry in GEOMETRIES
    }


def classify_exact_repo_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(DIAGNOSTIC_SEEDS) * len(MEMORY_SIZES) * len(GEOMETRIES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} #973 rows, got {len(rows)}")

    by_key = {
        (int(row["seed"]), int(row["memory_size"]), str(row["geometry"])): row
        for row in rows
    }
    if set(by_key) != _expected_keys():
        raise ValueError("rows do not exactly match the frozen #973 cases")

    correctness_failures: list[dict[str, Any]] = []
    for key, row in by_key.items():
        correctness = row["correctness"]
        passed = (
            bool(correctness["corrected_frozen_full_result_parity"])
            and bool(correctness["corrected_frozen_value_parity"])
            and float(correctness["flat_answer_match_rate"]) == 1.0
        )
        if not passed:
            correctness_failures.append({"case": key, "correctness": correctness})

    if correctness_failures:
        return {
            "classification": "CORRECTNESS_FAIL",
            "correctness_failures": correctness_failures,
            "warm_primary_pass": False,
            "cold_secondary_pass": False,
            "primary_checks": [],
            "cold_checks": [],
            "descriptive_regression_warnings": [],
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
        classification = "EXACT_REPO_CORRECTED_BATCHED_SYSTEMS_STOP"
    elif cold_pass:
        classification = "EXACT_REPO_CORRECTED_BATCHED_SYSTEMS_PASS"
    else:
        classification = "EXACT_REPO_CORRECTED_BATCHED_AMORTIZATION_REQUIRED"

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
        "correctness_failures": [],
        "warm_primary_pass": warm_pass,
        "cold_secondary_pass": cold_pass,
        "primary_checks": primary_checks,
        "cold_checks": cold_checks,
        "descriptive_regression_warnings": warnings,
    }


def validate_checkout_identity(expected_source_sha: str) -> str:
    expected = expected_source_sha.strip().lower()
    if len(expected) != 40 or any(ch not in "0123456789abcdef" for ch in expected):
        raise RuntimeError("expected source SHA must be exactly 40 lowercase hex characters")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path.cwd()
    ).strip().lower()
    if actual != expected:
        raise RuntimeError(f"checked-out source drift: expected {expected}, got {actual}")
    return actual


def run_exact_repo_protocol(*, source_sha: str) -> dict[str, Any]:
    source_sha = validate_checkout_identity(source_sha)
    if torch.cuda.is_available():
        raise RuntimeError("#973 is CPU-only; CUDA must be unavailable")
    for seed in DIAGNOSTIC_SEEDS:
        validate_seed(seed, protocol=True)

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

    decision = classify_exact_repo_protocol(rows)
    return {
        "measurement_kind": "chm_v1_exact_repo_corrected_batched_systems_timing_cpu",
        "issue": ISSUE,
        "source_sha": source_sha,
        "scientific_credit": False,
        "gpu_authorized": False,
        "paid_scientific_compute": False,
        "modal_trigger": False,
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
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
        "decision": decision,
        "interpretation_ceiling": (
            "Exact-repository CPU systems timing only. Even a PASS does not authorize "
            "GPU/Modal, scientific seeds, training, or modeling/capability claims."
        ),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_exact_repo_protocol(source_sha=args.source_sha)
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    _main()
