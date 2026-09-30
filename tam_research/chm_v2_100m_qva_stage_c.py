from __future__ import annotations

"""CHM-v2 100M QVA Stage-C one-seed development protocol (#1176).

Pure protocol / scoring / classification layer only.  This module does not
train models, construct optimizers, launch Modal/GPU work, authorize the
reserved scientific seed, or create checkpoints.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .chm_v1_100m_scale import (
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    FIRST_SCREEN_TOKEN_BUDGET,
    PARAMETER_MISMATCH_LIMIT,
)
from .chm_v1_100m_stage_c_eval import (
    ALL_FAMILIES,
    CASES_PER_FAMILY,
    GENERATOR_VERSION,
    LONG_RANGE_FAMILIES,
    PROBE_SEED,
    TOTAL_PROBES,
    VALIDATION_SEED,
    VALIDATION_TOKENS,
    candidate_score,
    local_aligned_final_logits,
    eiem_flat_final_logits,
)
from .chm_v1_long_memory_eval import EncodedProbe
from .chm_v2_100m_qva import (
    CHMV2100MEIEMQVA,
    EXPECTED_QVA_PARAMETERS,
    QVA_GATE_INIT,
    QVA_RANK,
    qva_hard_flat_final_logits,
)

ISSUE = 1176
SCIENTIFIC_SEED = 2_011_761
SCIENTIFIC_SEED_AUTHORIZED_BY_MODULE = False

TRAINING_TOKENS_PER_MODEL = FIRST_SCREEN_TOKEN_BUDGET
OPTIMIZER_STEPS_PER_MODEL = 2_048
WARMUP_STEPS = 40
TOKENS_PER_OPTIMIZER_STEP = 16_384

MIN_AGGREGATE_ACCURACY_GAIN = 0.05
MIN_AGGREGATE_CANDIDATE_NLL_BENEFIT = 0.05
MIN_FAMILY_ACCURACY_GAIN = 0.03
MIN_ANY_FAMILY_ACCURACY_GAIN = -0.01
MIN_LONG_RANGE_FAMILIES_AT_GAIN = 2
MIN_LONG_RANGE_FAMILIES_POSITIVE_NLL = 2

MIN_QVA_VS_RAW_NLL_BENEFIT = 0.05
MIN_QVA_VS_RAW_FAMILY_NONWORSE_NLL = 2

MAX_LANGUAGE_NLL_REGRESSION = 0.03
MIN_LOCAL_NEGATIVE_ACCURACY_GAIN = -0.02
MAX_LOCAL_NEGATIVE_CANDIDATE_NLL_REGRESSION = 0.03

POSITIVE_CLASSIFICATION = "CHM_V2_100M_QVA_POSITIVE_DEVELOPMENT_SCREEN"
STOP_CLASSIFICATION = "CHM_V2_100M_QVA_STOP"

REQUIRED_INTEGRITY_FLAGS = (
    "paired_backbone_initialization_identical",
    "raw_qva_address_initialization_identical",
    "raw_qva_legacy_gate_initialization_identical",
    "qva_initialization_exact",
    "byte_identical_training_stream",
    "matched_optimizer_schedule",
    "finite_losses",
    "finite_parameters",
    "no_cross_session_state_aliasing",
    "no_future_self_leakage",
)


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": "CHM_V2_100M_QVA_STAGE_C_PREREGISTRATION_NO_EXECUTION_AUTHORITY",
        "issue": ISSUE,
        "scientific_seed_reserved": SCIENTIFIC_SEED,
        "scientific_seed_authorized": SCIENTIFIC_SEED_AUTHORIZED_BY_MODULE,
        "models": ("LOCAL", "RAW_EIEM", "QVA_EIEM"),
        "local_parameters": EXPECTED_LOCAL_PARAMETERS,
        "raw_eiem_parameters": EXPECTED_EIEM_PARAMETERS,
        "qva_parameters": EXPECTED_QVA_PARAMETERS,
        "qva_rank": QVA_RANK,
        "qva_gate_init": QVA_GATE_INIT,
        "training_tokens_per_model": TRAINING_TOKENS_PER_MODEL,
        "optimizer_steps_per_model": OPTIMIZER_STEPS_PER_MODEL,
        "warmup_steps": WARMUP_STEPS,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "generator_version": GENERATOR_VERSION,
        "probe_seed": PROBE_SEED,
        "cases_per_family": CASES_PER_FAMILY,
        "probe_count": TOTAL_PROBES,
        "validation_seed": VALIDATION_SEED,
        "validation_tokens": VALIDATION_TOKENS,
        "bootstrap_gate": False,
        "gpu_authorized": False,
        "modal_authorized": False,
        "training_launch_authorized": False,
        "stage_d_authorized": False,
        "multi_seed_replication_authorized": False,
    }


def validate_protocol_manifest() -> dict[str, Any]:
    manifest = protocol_manifest()
    if SCIENTIFIC_SEED != 2_000_000 + 10 * ISSUE + 1:
        raise RuntimeError("#1176 scientific-seed derivation drift")
    if TRAINING_TOKENS_PER_MODEL != 33_554_432:
        raise RuntimeError("#1176 token budget drift")
    if (
        OPTIMIZER_STEPS_PER_MODEL != 2_048
        or WARMUP_STEPS != 40
        or TOKENS_PER_OPTIMIZER_STEP != 16_384
    ):
        raise RuntimeError("#1176 optimizer schedule accounting drift")
    if QVA_RANK != 32 or float(QVA_GATE_INIT) != -4.0:
        raise RuntimeError("#1176 QVA architecture drift")
    mismatch = (EXPECTED_QVA_PARAMETERS - EXPECTED_LOCAL_PARAMETERS) / EXPECTED_LOCAL_PARAMETERS
    if abs(mismatch) > PARAMETER_MISMATCH_LIMIT:
        raise RuntimeError("#1176 QVA parameter mismatch exceeds frozen 0.1% limit")
    if SCIENTIFIC_SEED_AUTHORIZED_BY_MODULE:
        raise RuntimeError("#1176 protocol module must never self-authorize the reserved seed")
    return manifest


def triplet_probe_row(
    *,
    probe: EncodedProbe,
    local_logits: torch.Tensor,
    raw_logits: torch.Tensor,
    qva_logits: torch.Tensor,
) -> dict[str, Any]:
    if probe.generator_version != GENERATOR_VERSION:
        raise RuntimeError("#1176 accepts only the frozen aligned-v4 probe generator")

    def score(logits: torch.Tensor) -> dict[str, Any]:
        return candidate_score(
            logits,
            candidate_token_ids=probe.candidate_token_ids,
            answer_token_id=probe.answer_token_id,
            stale_token_ids=probe.stale_token_ids,
        )

    local = score(local_logits)
    raw = score(raw_logits)
    qva = score(qva_logits)
    return {
        "family": probe.family,
        "case_id": int(probe.case_id),
        "generator_version": probe.generator_version,
        "evidence_distance": int(probe.evidence_distance),
        "local_correct": bool(local["correct"]),
        "raw_correct": bool(raw["correct"]),
        "qva_correct": bool(qva["correct"]),
        "local_candidate_nll": float(local["candidate_nll"]),
        "raw_candidate_nll": float(raw["candidate_nll"]),
        "qva_candidate_nll": float(qva["candidate_nll"]),
        "local_stale_choice": bool(local["stale_choice"]),
        "raw_stale_choice": bool(raw["stale_choice"]),
        "qva_stale_choice": bool(qva["stale_choice"]),
    }


@torch.no_grad()
def score_probe_triplet(
    *,
    probe: EncodedProbe,
    local_model: torch.nn.Module,
    raw_model: torch.nn.Module,
    qva_model: CHMV2100MEIEMQVA,
) -> dict[str, Any]:
    return triplet_probe_row(
        probe=probe,
        local_logits=local_aligned_final_logits(local_model, probe),
        raw_logits=eiem_flat_final_logits(raw_model, probe.prompt_ids),
        qva_logits=qva_hard_flat_final_logits(qva_model, probe.prompt_ids),
    )


@torch.no_grad()
def qva_integration_diagnostics(
    model: CHMV2100MEIEMQVA,
    hidden: torch.Tensor,
    memory: torch.Tensor,
) -> dict[str, float]:
    """Report-only QVA gate/correction statistics; never used by the gate."""
    if hidden.shape != memory.shape:
        raise ValueError("QVA diagnostic hidden/memory shapes must match")
    qva = model.qva
    z = F.silu(
        qva.query_projection(qva.hidden_norm(hidden))
        + qva.value_projection(qva.memory_norm(memory))
    )
    delta = qva.output_projection(z)
    gate = torch.sigmoid(qva.gate_bias + qva.gate_projection(z))
    correction = gate.to(delta.dtype) * delta
    hidden_norm = hidden.float().norm(dim=-1).clamp_min(1e-12)
    ratio = correction.float().norm(dim=-1) / hidden_norm
    return {
        "gate_mean": float(gate.float().mean().item()),
        "gate_median": float(gate.float().median().item()),
        "gate_min": float(gate.float().min().item()),
        "gate_max": float(gate.float().max().item()),
        "correction_ratio_mean": float(ratio.mean().item()),
        "correction_ratio_median": float(ratio.median().item()),
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    if len(rows) != TOTAL_PROBES:
        raise ValueError(f"#1176 requires exactly {TOTAL_PROBES} scored probe rows")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        family = str(row.get("family"))
        if family not in ALL_FAMILIES:
            raise ValueError(f"#1176 unexpected family {family!r}")
        if row.get("generator_version") != GENERATOR_VERSION:
            raise ValueError("#1176 scored row has wrong evaluator version")
        case_id = int(row.get("case_id", -1))
        key = (family, case_id)
        if key in seen:
            raise ValueError(f"#1176 duplicate scored case {key}")
        seen.add(key)
        for model in ("local", "raw", "qva"):
            nll = float(row[f"{model}_candidate_nll"])
            if not np.isfinite(nll):
                raise ValueError(f"#1176 non-finite {model} candidate NLL")
            for suffix in ("correct", "stale_choice"):
                if not isinstance(row.get(f"{model}_{suffix}"), (bool, np.bool_)):
                    raise ValueError(f"#1176 {model}_{suffix} must be boolean")
        by_family[family].append(row)
    for family in ALL_FAMILIES:
        if len(by_family[family]) != CASES_PER_FAMILY:
            raise ValueError(f"#1176 {family} row-count drift")
    return by_family


def _pair_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
) -> dict[str, float]:
    return {
        "accuracy_gain": float(
            np.mean(
                [
                    float(bool(row[f"{candidate}_correct"]))
                    - float(bool(row[f"{baseline}_correct"]))
                    for row in rows
                ]
            )
        ),
        "candidate_nll_benefit": float(
            np.mean(
                [
                    float(row[f"{baseline}_candidate_nll"])
                    - float(row[f"{candidate}_candidate_nll"])
                    for row in rows
                ]
            )
        ),
    }


def _integrity_stop_reasons(integrity: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected_counts = {
        "local_trainable_parameters": EXPECTED_LOCAL_PARAMETERS,
        "raw_trainable_parameters": EXPECTED_EIEM_PARAMETERS,
        "qva_trainable_parameters": EXPECTED_QVA_PARAMETERS,
        "training_tokens_per_model": TRAINING_TOKENS_PER_MODEL,
        "optimizer_steps_per_model": OPTIMIZER_STEPS_PER_MODEL,
        "scientific_seed": SCIENTIFIC_SEED,
        "generator_version": GENERATOR_VERSION,
        "probe_seed": PROBE_SEED,
        "cases_per_family": CASES_PER_FAMILY,
        "probe_count": TOTAL_PROBES,
        "validation_seed": VALIDATION_SEED,
        "validation_tokens": VALIDATION_TOKENS,
    }
    for key, expected in expected_counts.items():
        if integrity.get(key) != expected:
            reasons.append(f"integrity_{key}_mismatch")

    local = int(integrity.get("local_trainable_parameters", -1))
    qva = int(integrity.get("qva_trainable_parameters", -1))
    if local > 0 and abs((qva - local) / local) > PARAMETER_MISMATCH_LIMIT:
        reasons.append("qva_parameter_mismatch_above_point_one_percent")

    for flag in REQUIRED_INTEGRITY_FLAGS:
        if integrity.get(flag) is not True:
            reasons.append(f"integrity_{flag}_failed")
    return reasons


def classify_stage_c(
    *,
    integrity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    local_language_nll: float,
    raw_language_nll: float,
    qva_language_nll: float,
) -> dict[str, Any]:
    """Apply the frozen #1176 development gate mechanically."""
    reasons = _integrity_stop_reasons(integrity)
    by_family = _validate_rows(rows)

    family_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for family in ALL_FAMILIES:
        family_rows = by_family[family]
        family_metrics[family] = {
            "qva_vs_local": _pair_metrics(family_rows, baseline="local", candidate="qva"),
            "qva_vs_raw": _pair_metrics(family_rows, baseline="raw", candidate="qva"),
            "raw_vs_local": _pair_metrics(family_rows, baseline="local", candidate="raw"),
        }

    long_rows = [row for family in LONG_RANGE_FAMILIES for row in by_family[family]]
    qva_vs_local = _pair_metrics(long_rows, baseline="local", candidate="qva")
    qva_vs_raw = _pair_metrics(long_rows, baseline="raw", candidate="qva")
    raw_vs_local = _pair_metrics(long_rows, baseline="local", candidate="raw")

    if qva_vs_local["accuracy_gain"] < MIN_AGGREGATE_ACCURACY_GAIN:
        reasons.append("qva_vs_local_aggregate_accuracy_gain_below_gate")
    if qva_vs_local["candidate_nll_benefit"] < MIN_AGGREGATE_CANDIDATE_NLL_BENEFIT:
        reasons.append("qva_vs_local_aggregate_candidate_nll_benefit_below_gate")

    local_family_acc = [
        family_metrics[family]["qva_vs_local"]["accuracy_gain"]
        for family in LONG_RANGE_FAMILIES
    ]
    if sum(x >= MIN_FAMILY_ACCURACY_GAIN for x in local_family_acc) < MIN_LONG_RANGE_FAMILIES_AT_GAIN:
        reasons.append("fewer_than_two_long_range_families_meet_qva_vs_local_accuracy_gain")
    if any(x < MIN_ANY_FAMILY_ACCURACY_GAIN for x in local_family_acc):
        reasons.append("qva_vs_local_long_range_family_accuracy_regression")

    local_family_nll = [
        family_metrics[family]["qva_vs_local"]["candidate_nll_benefit"]
        for family in LONG_RANGE_FAMILIES
    ]
    if sum(x > 0.0 for x in local_family_nll) < MIN_LONG_RANGE_FAMILIES_POSITIVE_NLL:
        reasons.append("fewer_than_two_long_range_families_have_positive_qva_vs_local_nll")

    if qva_vs_raw["candidate_nll_benefit"] < MIN_QVA_VS_RAW_NLL_BENEFIT:
        reasons.append("qva_vs_raw_aggregate_candidate_nll_benefit_below_gate")
    if qva_vs_raw["accuracy_gain"] <= 0.0:
        reasons.append("qva_aggregate_accuracy_not_above_raw")
    raw_family_nll = [
        family_metrics[family]["qva_vs_raw"]["candidate_nll_benefit"]
        for family in LONG_RANGE_FAMILIES
    ]
    if sum(x >= 0.0 for x in raw_family_nll) < MIN_QVA_VS_RAW_FAMILY_NONWORSE_NLL:
        reasons.append("fewer_than_two_long_range_families_qva_nll_nonworse_than_raw")

    local_language_nll = float(local_language_nll)
    raw_language_nll = float(raw_language_nll)
    qva_language_nll = float(qva_language_nll)
    if not all(np.isfinite(x) for x in (local_language_nll, raw_language_nll, qva_language_nll)):
        reasons.append("nonfinite_ordinary_language_nll")
    qva_language_delta = qva_language_nll - local_language_nll
    if qva_language_delta > MAX_LANGUAGE_NLL_REGRESSION:
        reasons.append("qva_ordinary_language_nll_regression_above_gate")

    local_negative = family_metrics["local_negative"]["qva_vs_local"]
    if local_negative["accuracy_gain"] < MIN_LOCAL_NEGATIVE_ACCURACY_GAIN:
        reasons.append("qva_local_negative_accuracy_regression_above_gate")
    local_negative_nll_regression = -local_negative["candidate_nll_benefit"]
    if local_negative_nll_regression > MAX_LOCAL_NEGATIVE_CANDIDATE_NLL_REGRESSION:
        reasons.append("qva_local_negative_candidate_nll_regression_above_gate")

    overwrite_rows = by_family["overwrite"]
    local_stale = float(
        np.mean([float(bool(row["local_stale_choice"])) for row in overwrite_rows])
    )
    raw_stale = float(
        np.mean([float(bool(row["raw_stale_choice"])) for row in overwrite_rows])
    )
    qva_stale = float(
        np.mean([float(bool(row["qva_stale_choice"])) for row in overwrite_rows])
    )
    if local_stale < 0.01:
        stale_pass = qva_stale <= local_stale + 0.01
        stale_rule = "qva<=local+0.01_when_local<0.01"
    else:
        stale_pass = qva_stale <= 0.80 * local_stale
        stale_rule = "qva<=0.80*local"
    if not stale_pass:
        reasons.append("qva_overwrite_stale_state_guard_failed")

    passed = not reasons
    return {
        "classification": POSITIVE_CLASSIFICATION if passed else STOP_CLASSIFICATION,
        "passed": passed,
        "stop_reasons": reasons,
        "metrics": {
            "aggregate_long_range": {
                "qva_vs_local": qva_vs_local,
                "qva_vs_raw": qva_vs_raw,
                "raw_vs_local": raw_vs_local,
            },
            "family_metrics": family_metrics,
            "ordinary_language": {
                "local_nll": local_language_nll,
                "raw_nll": raw_language_nll,
                "qva_nll": qva_language_nll,
                "qva_minus_local_nll": qva_language_delta,
                "raw_minus_local_nll": raw_language_nll - local_language_nll,
            },
            "local_negative_qva_candidate_nll_regression": local_negative_nll_regression,
            "overwrite": {
                "local_stale_choice_rate": local_stale,
                "raw_stale_choice_rate": raw_stale,
                "qva_stale_choice_rate": qva_stale,
                "stale_rule": stale_rule,
            },
        },
        "interpretation_ceiling": (
            "One-seed QVA development evidence only. A positive result may justify only "
            "separately preregistered multi-seed replication; no Stage D, scaling, "
            "SOTA, AGI, RSI, novelty, or breakthrough claim is authorized."
        ),
    }
