from __future__ import annotations

"""Executable, zero-credit CHM-v1 / EIEM scientific gate for issue #854.

This module is orchestration/measurement plumbing only.  It does not launch a
GPU, authorize paid compute, train a model, alter any architecture, change the
frozen data/optimizer/probe protocol, or consume a reserved scientific seed.

A future authorized runner must emit one record for each reserved seed.  This
module then applies the already-preregistered pass/fail criteria mechanically
and refuses to classify incomplete evidence as a pass.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .chm_v1_long_memory_eval import DEFAULT_CASES_PER_FAMILY
from .chm_v1_long_memory_eval_v3 import (
    GENERATOR_VERSION as SCIENTIFIC_GENERATOR_VERSION,
    generate_probe_suite as generate_v3_probe_suite,
)
from .chm_v1_small_lm import SCIENTIFIC_SEEDS

EVALUATOR_SEED = 8_540_911
LONG_RANGE_FAMILIES = ("rare_fact", "overwrite", "two_hop")
ALL_FAMILIES = LONG_RANGE_FAMILIES + ("local_negative",)

# Required evidence keys are intentionally explicit.  A missing field means the
# scientific record is incomplete, never implicitly successful.
REQUIRED_MEASUREMENT_KEYS = frozenset(
    {
        "trainable_params",
        "train_tokens",
        "validation_version",
        "train_curve",
        "distance_memory_slices",
        "nodes_visited",
        "index_build_update_write_time",
        "batch1_and_throughput_wall_clock",
        "training_wall_clock_tokens_per_second_vram_compile",
        "state_bytes",
        "failures_and_consumed_seeds",
    }
)
REQUIRED_ABLATION_KEYS = frozenset(
    {
        "memory_disabled_reset",
        "wrong_session_shuffled_memory",
        "random_address_projections",
        "flat_exhaustive_same_encoder_state",
        "capacity_distance_slices",
    }
)
REQUIRED_SYSTEM_FLAGS = (
    "no_nan_inf",
    "no_cross_session_aliasing",
    "no_hidden_persistent_state",
    "no_base_parameter_mutation",
)
STOP_CONDITION_FLAGS = (
    "oracle_or_future_leakage",
    "benefit_disappears_out_of_template",
    "index_overhead_erases_practical_advantage",
    "simpler_control_reproduces_frontier",
    "numerical_or_fairness_violation",
)


def build_scientific_probe_suite(encode: Callable[[str], Sequence[int]]):
    """Construct exactly the cue-free v3 held-out scientific envelope."""
    suite = generate_v3_probe_suite(
        encode,
        seed=EVALUATOR_SEED,
        cases_per_family=DEFAULT_CASES_PER_FAMILY,
    )
    expected = DEFAULT_CASES_PER_FAMILY * len(ALL_FAMILIES)
    if len(suite) != expected:
        raise RuntimeError(f"scientific probe count drifted: {len(suite)} != {expected}")
    versions = {probe.generator_version for probe in suite}
    if versions != {SCIENTIFIC_GENERATOR_VERSION}:
        raise RuntimeError(f"scientific generator drifted: {sorted(versions)}")
    if any(probe.used_for_training for probe in suite):
        raise RuntimeError("held-out scientific probes were marked as training data")
    return suite


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_keys(mapping: Mapping[str, Any], required: frozenset[str], path: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")


def _validate_record(record: Mapping[str, Any]) -> int:
    seed = int(record.get("seed", -1))
    if seed not in SCIENTIFIC_SEEDS:
        raise ValueError(f"unexpected scientific seed {seed}")
    if record.get("generator_version") != SCIENTIFIC_GENERATOR_VERSION:
        raise ValueError(
            f"seed {seed} must use {SCIENTIFIC_GENERATOR_VERSION}, "
            f"got {record.get('generator_version')!r}"
        )

    language = _mapping(record.get("language"), f"seed {seed}.language")
    _number(language.get("local_nll"), f"seed {seed}.language.local_nll")
    _number(language.get("eiem_flat_nll"), f"seed {seed}.language.eiem_flat_nll")

    families = _mapping(record.get("families"), f"seed {seed}.families")
    for family in ALL_FAMILIES:
        row = _mapping(families.get(family), f"seed {seed}.families.{family}")
        _number(row.get("local_accuracy"), f"seed {seed}.{family}.local_accuracy")
        _number(row.get("eiem_accuracy"), f"seed {seed}.{family}.eiem_accuracy")
    overwrite = _mapping(families["overwrite"], f"seed {seed}.families.overwrite")
    _number(overwrite.get("local_stale_error"), f"seed {seed}.overwrite.local_stale_error")
    _number(overwrite.get("eiem_stale_error"), f"seed {seed}.overwrite.eiem_stale_error")

    indexed = _mapping(record.get("indexed"), f"seed {seed}.indexed")
    _number(indexed.get("exact_match_rate"), f"seed {seed}.indexed.exact_match_rate")
    _number(indexed.get("indexed_reads_1024"), f"seed {seed}.indexed.indexed_reads_1024")
    _number(indexed.get("flat_reads_1024"), f"seed {seed}.indexed.flat_reads_1024")

    systems = _mapping(record.get("systems"), f"seed {seed}.systems")
    for flag in REQUIRED_SYSTEM_FLAGS:
        if not isinstance(systems.get(flag), bool):
            raise ValueError(f"seed {seed}.systems.{flag} must be boolean")

    stop = _mapping(record.get("stop_conditions"), f"seed {seed}.stop_conditions")
    for flag in STOP_CONDITION_FLAGS:
        if not isinstance(stop.get(flag), bool):
            raise ValueError(f"seed {seed}.stop_conditions.{flag} must be boolean")

    measurements = _mapping(record.get("measurements"), f"seed {seed}.measurements")
    _require_keys(measurements, REQUIRED_MEASUREMENT_KEYS, f"seed {seed}.measurements")
    ablations = _mapping(record.get("ablations"), f"seed {seed}.ablations")
    _require_keys(ablations, REQUIRED_ABLATION_KEYS, f"seed {seed}.ablations")
    return seed


def evaluate_scientific_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply issue #854's frozen three-seed FULL PASS criteria.

    The function is deliberately conservative about missing evidence: malformed
    or incomplete records raise rather than being interpreted as a failed or
    successful scientific result.
    """
    if len(records) != len(SCIENTIFIC_SEEDS):
        raise ValueError(
            f"need exactly {len(SCIENTIFIC_SEEDS)} scientific seed records, got {len(records)}"
        )

    by_seed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        seed = _validate_record(record)
        if seed in by_seed:
            raise ValueError(f"duplicate scientific seed {seed}")
        by_seed[seed] = record
    expected = set(SCIENTIFIC_SEEDS)
    if set(by_seed) != expected:
        raise ValueError(f"scientific seed set drifted: {sorted(by_seed)} != {sorted(expected)}")

    ordered = [by_seed[seed] for seed in SCIENTIFIC_SEEDS]

    # Criterion 1: exact indexed lookup must reproduce flat lookup on every call.
    exact_rates = [
        _number(_mapping(r["indexed"], "indexed")["exact_match_rate"], "exact_match_rate")
        for r in ordered
    ]
    exactness_pass = all(rate == 1.0 for rate in exact_rates)

    # Criterion 2: >=+10 pp mean gain across the three long-range families,
    # plus a win in each family for >=2/3 paired seeds.
    gains: list[float] = []
    family_wins: dict[str, int] = {}
    for family in LONG_RANGE_FAMILIES:
        wins = 0
        for record in ordered:
            row = _mapping(_mapping(record["families"], "families")[family], family)
            local = _number(row["local_accuracy"], f"{family}.local_accuracy")
            eiem = _number(row["eiem_accuracy"], f"{family}.eiem_accuracy")
            gains.append(eiem - local)
            wins += int(eiem > local)
        family_wins[family] = wins
    mean_long_range_gain = sum(gains) / len(gains)
    long_range_pass = mean_long_range_gain >= 0.10 and all(
        wins >= 2 for wins in family_wins.values()
    )

    # Criterion 3: aggregate overwrite stale-value error across equal-size
    # preregistered per-seed probe sets, then apply the frozen exception rule.
    local_stale = sum(
        _number(_mapping(_mapping(r["families"], "families")["overwrite"], "overwrite")["local_stale_error"], "local_stale_error")
        for r in ordered
    ) / len(ordered)
    eiem_stale = sum(
        _number(_mapping(_mapping(r["families"], "families")["overwrite"], "overwrite")["eiem_stale_error"], "eiem_stale_error")
        for r in ordered
    ) / len(ordered)
    if local_stale < 0.01:
        stale_state_pass = eiem_stale <= local_stale + 0.005
        stale_rule = "eiem<=local+0.005_when_local<0.01"
    else:
        stale_state_pass = eiem_stale <= 0.5 * local_stale
        stale_rule = "eiem<=0.5*local"

    # Criterion 4: ordinary language regression ceiling.
    nll_deltas = []
    for record in ordered:
        language = _mapping(record["language"], "language")
        nll_deltas.append(
            _number(language["eiem_flat_nll"], "eiem_flat_nll")
            - _number(language["local_nll"], "local_nll")
        )
    mean_nll_delta = sum(nll_deltas) / len(nll_deltas)
    language_pass = mean_nll_delta <= 0.10 and all(delta <= 0.15 for delta in nll_deltas)

    # Criterion 5: local negative control may not regress by >2 pp on any seed.
    local_control_deltas = []
    for record in ordered:
        row = _mapping(_mapping(record["families"], "families")["local_negative"], "local_negative")
        local_control_deltas.append(
            _number(row["eiem_accuracy"], "local_negative.eiem_accuracy")
            - _number(row["local_accuracy"], "local_negative.local_accuracy")
        )
    local_control_pass = all(delta >= -0.02 for delta in local_control_deltas)

    # Criterion 6: <=25% of exhaustive address-vector reads at the exact
    # preregistered 1,024-item slice, aggregated as counts rather than averaging
    # per-seed ratios.
    indexed_reads = sum(
        _number(_mapping(r["indexed"], "indexed")["indexed_reads_1024"], "indexed_reads_1024")
        for r in ordered
    )
    flat_reads = sum(
        _number(_mapping(r["indexed"], "indexed")["flat_reads_1024"], "flat_reads_1024")
        for r in ordered
    )
    if flat_reads <= 0:
        raise ValueError("flat_reads_1024 must be positive")
    sparse_read_fraction = indexed_reads / flat_reads
    sparse_reads_pass = sparse_read_fraction <= 0.25

    # Criterion 7 plus frozen stop conditions.
    systems_pass = all(
        bool(_mapping(r["systems"], "systems")[flag])
        for r in ordered
        for flag in REQUIRED_SYSTEM_FLAGS
    )
    no_stop_condition = not any(
        bool(_mapping(r["stop_conditions"], "stop_conditions")[flag])
        for r in ordered
        for flag in STOP_CONDITION_FLAGS
    )

    criteria = {
        "exactness": exactness_pass,
        "long_range_capability": long_range_pass,
        "stale_state": stale_state_pass,
        "ordinary_language": language_pass,
        "local_negative_control": local_control_pass,
        "sparse_reads_1024": sparse_reads_pass,
        "systems_sanity": systems_pass,
        "no_stop_condition": no_stop_condition,
    }
    full_pass = all(criteria.values())

    return {
        "classification": "FULL_PASS_SMALL_LM_ONLY" if full_pass else "FAIL_SMALL_LM_GATE",
        "full_pass": full_pass,
        "generator_version": SCIENTIFIC_GENERATOR_VERSION,
        "scientific_seeds": list(SCIENTIFIC_SEEDS),
        "criteria": criteria,
        "metrics": {
            "mean_long_range_absolute_gain": mean_long_range_gain,
            "family_seed_wins": family_wins,
            "local_stale_error": local_stale,
            "eiem_stale_error": eiem_stale,
            "stale_gate_rule": stale_rule,
            "mean_nll_delta": mean_nll_delta,
            "per_seed_nll_deltas": nll_deltas,
            "per_seed_local_control_deltas": local_control_deltas,
            "indexed_flat_exact_match_rates": exact_rates,
            "indexed_reads_1024": indexed_reads,
            "flat_reads_1024": flat_reads,
            "sparse_read_fraction_1024": sparse_read_fraction,
        },
        "interpretation_ceiling": (
            "small-LM long-memory signal plus exact sparse inference lookup in this setup only; "
            "not novelty, SOTA, large-scale training efficiency, production speedup, AGI, or breakthrough"
        ),
    }
