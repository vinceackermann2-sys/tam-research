"""CPU-only and evidence-neutral adjudication of future CPW binding metrics.

#1319 is a *preauthority* protocol. This module NEVER verifies that a run
was authorized, that metrics came from training, or that a result exists.
Positive return values are PROVISIONAL metrics screens, NOT science results.
No weights, accelerator, checkpoints, filesystem writes or launch APIs.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from tam_research.cpw_binding_science.harness import ARMS, PARAMETERS, LONG_DISTANCES
from tam_research.cpw_binding_v2.task import SCORED_DELAYS

CANDIDATES = ("afm_last1", "afm_first1", "r1_final")
REPLICATIONS = 3
STEPS = 1500
MICROBATCH = 32
ACCUM = 2
EXAMPLES_PER_DISTANCE = 512
COUNTERFACTUAL_SOURCE_GROUPS = 512
TRANSFORMER_LONG_MIN = 0.80
TRANSFORMER_D256_MIN = 0.70
SEQUENCE_LEAKAGE_MAX = 0.05
CANDIDATE_LONG_MIN = 0.80
CANDIDATE_D256_MIN = 0.70
CANDIDATE_DISTANCE_MARGIN = 0.40
CANDIDATE_FOUR_QUERY_MIN = 0.70
STRONG_TRANSFORMER_TOLERANCE = 0.05

# These immutable labels intentionally do not certify scientific provenance.
PROVISIONAL = "PROVISIONAL_METRICS_ONLY"


def _number(value: Any, label: str, *, lo: float = 0.0,
            hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a real number")
    x = float(value)
    if not math.isfinite(x) or x < lo or (hi is not None and x > hi):
        raise ValueError(f"{label} outside finite expected range")
    return x


def _count(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be exactly {expected}")


def _validate_result(arm: str, seed: int, record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("arm") != arm or record.get("seed") != seed:
        raise ValueError(f"{arm}: arm/seed identity mismatch")
    _count(record.get("steps"), STEPS, f"{arm} optimizer steps")
    _count(record.get("examples_seen"), STEPS * MICROBATCH * ACCUM,
           f"{arm} examples_seen")
    _count(record.get("parameters"), PARAMETERS[arm], f"{arm} parameter count")
    train_time = _number(record.get("training_seconds"), f"{arm} training seconds")
    total_time = _number(record.get("total_compute_seconds"), f"{arm} total seconds")
    throughput = _number(record.get("examples_per_second"), f"{arm} throughput")
    _number(record.get("peak_vram_gib"), f"{arm} peak vram")
    _number(record.get("last_train_query_loss"), f"{arm} last query loss")
    if train_time <= 0 or total_time < train_time or throughput <= 0:
        raise ValueError(f"{arm}: invalid compute telemetry")

    e = record.get("evaluation")
    if not isinstance(e, Mapping):
        raise ValueError(f"{arm}: evaluation missing")
    by = e.get("by_distance")
    if not isinstance(by, Mapping) or set(by) != {str(d) for d in SCORED_DELAYS}:
        raise ValueError(f"{arm}: exactly four scored delays required")
    per_distance: dict[str, dict[str, float]] = {}
    for d in SCORED_DELAYS:
        k = str(d)
        row = by[k]
        if not isinstance(row, Mapping):
            raise ValueError(f"{arm}: malformed delay {k}")
        _count(row.get("examples"), EXAMPLES_PER_DISTANCE, f"{arm} delay {k} examples")
        per_distance[k] = {
            "accuracy": _number(row.get("accuracy"), f"{arm} d{k} accuracy", hi=1.0),
            "nll": _number(row.get("nll"), f"{arm} d{k} NLL"),
        }
    long_acc = sum(per_distance[str(d)]["accuracy"] for d in LONG_DISTANCES) / 3
    long_nll = sum(per_distance[str(d)]["nll"] for d in LONG_DISTANCES) / 3
    if not math.isclose(_number(e.get("long_mean_accuracy"), f"{arm} mean accuracy",
                                hi=1.0), long_acc, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{arm}: inconsistent reported long mean accuracy")
    if not math.isclose(_number(e.get("long_mean_nll"), f"{arm} mean NLL"),
                        long_nll, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{arm}: inconsistent reported long mean NLL")
    _count(e.get("counterfactual_groups"), COUNTERFACTUAL_SOURCE_GROUPS,
           f"{arm} counterfactual groups")
    all_four = _number(e.get("counterfactual_all_four_accuracy"),
                       f"{arm} all-four accuracy", hi=1.0)
    individual = _number(e.get("counterfactual_individual_accuracy"),
                         f"{arm} per-query accuracy", hi=1.0)
    if all_four > individual + 1e-12:
        raise ValueError(f"{arm}: all-four accuracy exceeds individual accuracy")
    keyblind = _number(e.get("keyblind_fixed_query_accuracy"),
                       f"{arm} keyblind accuracy", hi=1.0)
    if not math.isclose(keyblind, 0.25, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{arm}: counterfactual key-blind baseline != .25")
    return {
        "by_distance": per_distance,
        "long_mean_accuracy": long_acc,
        "long_mean_nll": long_nll,
        "all_four_accuracy": all_four,
        "per_query_accuracy": individual,
        "training_seconds": train_time,
        "total_compute_seconds": total_time,
        "examples_per_second": throughput,
    }


def _check_rows(rows: Any) -> tuple[str, dict[int, dict[str, dict[str, Any]]]]:
    if not isinstance(rows, list) or len(rows) != REPLICATIONS:
        raise ValueError("exactly three paired replication rows required")
    normalized: dict[int, dict[str, dict[str, Any]]] = {}
    source_sha: str | None = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("replication row must be a mapping")
        seed = row.get("seed")
        if type(seed) is not int or seed < 0 or seed in normalized:
            raise ValueError("seed identifiers must be unique nonnegative integers")
        if row.get("phase") != "replication":
            raise ValueError("only frozen replication phase may be screened")
        sha = row.get("source_sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("source_sha must be a 40-character commit SHA")
        if source_sha is None:
            source_sha = sha
        if sha != source_sha:
            raise ValueError("different source SHAs cannot be paired")
        raw_arms = row.get("arms")
        if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(ARMS):
            raise ValueError("all five exact arms required on every seed")
        normalized[seed] = {}
        for arm in ARMS:
            if not isinstance(raw_arms[arm], Mapping):
                raise ValueError(f"{arm} result must be a mapping")
            normalized[seed][arm] = _validate_result(arm, seed, raw_arms[arm])
    assert source_sha is not None
    return source_sha, normalized


def screen_replications(rows: Any) -> dict[str, Any]:
    """Return provisional metric screening ONLY, never scientific authorization.

    The source_sha and seed fields here are *unverified claims* inside supplied
    records; callers must separately audit physical run evidence and provenance.
    """
    result: dict[str, Any] = {
        "evidence_level": PROVISIONAL,
        "verified_scientific_run": False,
        "scientific_gpu_authorized": False,
        "breakthrough_claim_allowed": False,
        "status": "INVALID_INCOMPLETE",
        "reason": None,
        "candidates": {},
    }
    try:
        sha, normalized = _check_rows(rows)
    except (ValueError, TypeError, KeyError) as exc:
        result["reason"] = str(exc)
        return result
    seeds = sorted(normalized)
    result["source_sha_unverified"] = sha
    result["paired_seed_ids_unverified"] = seeds
    transformer_bad = []
    local_bad = []
    for seed in seeds:
        control = normalized[seed]["transformer"]
        if (control["long_mean_accuracy"] < TRANSFORMER_LONG_MIN or
                control["by_distance"]["256"]["accuracy"] < TRANSFORMER_D256_MIN):
            transformer_bad.append(seed)
        local = normalized[seed]["sequence_only"]
        if local["long_mean_accuracy"] > SEQUENCE_LEAKAGE_MAX:
            local_bad.append(seed)
    result["transformer_invalid_seeds"] = transformer_bad
    result["negative_control_invalid_seeds"] = local_bad
    if transformer_bad:
        result["status"] = "INVALID_TRANSFORMER_CONTROL"
        result["reason"] = "positive control undertrained on at least one paired seed"
        return result
    if local_bad:
        result["status"] = "INVALID_NEGATIVE_CONTROL"
        result["reason"] = "sequence-only negative control exceeded leakage gate"
        return result

    result["status"] = "VALID_CONTROLS_PROVISIONAL"
    for arm in CANDIDATES:
        per_seed = {}
        for seed in seeds:
            rec = normalized[seed][arm]
            seq = normalized[seed]["sequence_only"]
            transformer = normalized[seed]["transformer"]
            tests = {
                "long_mean": rec["long_mean_accuracy"] >= CANDIDATE_LONG_MIN,
                "distance256": rec["by_distance"]["256"]["accuracy"] >= CANDIDATE_D256_MIN,
                "margin_each_long_distance": all(
                    rec["by_distance"][str(d)]["accuracy"]
                    - seq["by_distance"][str(d)]["accuracy"] >= CANDIDATE_DISTANCE_MARGIN
                    for d in LONG_DISTANCES
                ),
                "lower_nll_each_long_distance": all(
                    rec["by_distance"][str(d)]["nll"] < seq["by_distance"][str(d)]["nll"]
                    for d in LONG_DISTANCES
                ),
                "counterfactual_all_four": rec["all_four_accuracy"] >= CANDIDATE_FOUR_QUERY_MIN,
            }
            passed = all(tests.values())
            per_seed[str(seed)] = {
                "tests": tests,
                "passed": passed,
                "within_005_transformer": (
                    rec["long_mean_accuracy"]
                    >= transformer["long_mean_accuracy"] - STRONG_TRANSFORMER_TOLERANCE
                ),
            }
        count = sum(int(r["passed"]) for r in per_seed.values())
        strong = count == 3 and all(
            r["within_005_transformer"] for r in per_seed.values()
        )
        result["candidates"][arm] = {
            "passed_seeds": count,
            "seed_decisions": per_seed,
            "metric_classification": (
                "PROVISIONAL_STRONG_3_OF_3" if strong
                else "PROVISIONAL_SUPPORT_2_OF_3" if count >= 2
                else "PROVISIONAL_NO_SUPPORT"
            ),
            "mean_training_seconds": sum(
                normalized[s][arm]["training_seconds"] for s in seeds
            ) / 3,
            "mean_total_compute_seconds": sum(
                normalized[s][arm]["total_compute_seconds"] for s in seeds
            ) / 3,
            "mean_examples_per_second": sum(
                normalized[s][arm]["examples_per_second"] for s in seeds
            ) / 3,
        }
    return result
