from __future__ import annotations

"""Checkpoint-only payload-vs-integration decomposition for CHM-v1 Stage-C (#1037).

This is a pure posthoc diagnostic layer.  It introduces no launcher, checkpoint
loader, optimizer, parameter update, scientific seed, or Stage-D authority.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .chm_v1_100m_stage_c_eval import ALL_FAMILIES, GENERATOR_VERSION, LONG_RANGE_FAMILIES, candidate_score
from .chm_v1_100m_stage_c_postmortem import (
    _fused_final_logits,
    _hidden_chunks,
    _prior_bank,
    no_memory_final_logits,
    oracle_evidence_final_logits,
)
from .chm_v1_long_memory_eval import EncodedProbe

ISSUE = 1037
CLASSIFICATION = (
    "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_DECOMPOSITION_"
    "PREREGISTRATION_NO_EXECUTION_AUTHORITY"
)

CONDITIONS = ("NO_MEMORY", "ORACLE_LEARNED_GATE", "ORACLE_UNIT_GATE")

GATE_SUPPRESSION_MAX_MEDIAN = 0.05
GATE_SUPPRESSION_MIN_UNIT_MINUS_LEARNED_NLL = 0.05
GATE_SUPPRESSION_MIN_UNIT_BENEFIT = -0.01

PAYLOAD_WEAK_MIN_USEFUL_NLL = 0.05
PAYLOAD_WEAK_MAX_ACCURACY_GAIN = 0.02

LEARNED_HARM_MAX_BENEFIT = -0.02
UNIT_DESTABILIZATION_MAX_BENEFIT = -0.10


def protocol_manifest() -> dict[str, Any]:
    return {
        "issue": ISSUE,
        "classification": CLASSIFICATION,
        "conditions": CONDITIONS,
        "generator_version": GENERATOR_VERSION,
        "checkpoint_only": True,
        "unit_gate_value": 1.0,
        "gate_sweep_authorized": False,
        "training_authorized": False,
        "parameter_updates_authorized": False,
        "optimizer_authorized": False,
        "checkpoint_mutation_authorized": False,
        "modal_authorized": False,
        "gpu_authorized": False,
        "new_scientific_seed_authorized": False,
        "stage_d_authorized": False,
        "prior_stage_c_result_changed": False,
    }


def validate_protocol_manifest() -> dict[str, Any]:
    manifest = protocol_manifest()
    if CONDITIONS != ("NO_MEMORY", "ORACLE_LEARNED_GATE", "ORACLE_UNIT_GATE"):
        raise RuntimeError("#1037 diagnostic condition drift")
    if manifest["unit_gate_value"] != 1.0:
        raise RuntimeError("#1037 unit gate must remain exactly 1.0")
    return manifest


def gate_statistics(model: torch.nn.Module) -> dict[str, float]:
    logits = getattr(model, "memory_gate_logit", None)
    if logits is None:
        raise RuntimeError("#1037 model has no memory_gate_logit")
    gate = torch.sigmoid(logits.detach().float()).cpu().reshape(-1)
    if gate.numel() < 1 or not torch.isfinite(gate).all():
        raise RuntimeError("#1037 memory gate must be finite and non-empty")

    def q(value: float) -> float:
        return float(torch.quantile(gate, torch.tensor(value, dtype=gate.dtype)).item())

    return {
        "count": float(gate.numel()),
        "min": float(gate.min().item()),
        "p05": q(0.05),
        "median": q(0.50),
        "mean": float(gate.mean().item()),
        "p95": q(0.95),
        "max": float(gate.max().item()),
        "fraction_lt_0p01": float((gate < 0.01).float().mean().item()),
        "fraction_lt_0p05": float((gate < 0.05).float().mean().item()),
        "fraction_gt_0p10": float((gate > 0.10).float().mean().item()),
        "l1": float(gate.abs().sum().item()),
        "l2": float(torch.linalg.vector_norm(gate).item()),
    }


def _oracle_targets(probe: EncodedProbe) -> tuple[int, ...]:
    if probe.family == "local_negative":
        return ()
    if probe.family == "two_hop":
        if probe.first_evidence_end_token is None:
            raise RuntimeError("#1037 two-hop probe is missing first evidence endpoint")
        return (int(probe.first_evidence_end_token), int(probe.evidence_end_token))
    return (int(probe.evidence_end_token),)


def _candidate_delta_trace(
    logits: torch.Tensor,
    no_memory_logits: torch.Tensor,
    probe: EncodedProbe,
) -> dict[str, float]:
    answer_id = int(probe.answer_token_id)
    wrong_ids = [int(x) for x in probe.candidate_token_ids if int(x) != answer_id]
    if not wrong_ids:
        raise RuntimeError("#1037 candidate set has no wrong alternatives")

    answer_delta = float((logits[answer_id] - no_memory_logits[answer_id]).float().item())
    wrong_deltas = [
        float((logits[token_id] - no_memory_logits[token_id]).float().item())
        for token_id in wrong_ids
    ]
    best_wrong_delta = max(wrong_deltas)
    return {
        "answer_logit_delta": answer_delta,
        "best_wrong_logit_delta": best_wrong_delta,
        "answer_minus_best_wrong_delta_contribution": answer_delta - best_wrong_delta,
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left32 = left.detach().float()
    right32 = right.detach().float()
    if float(torch.linalg.vector_norm(left32).item()) == 0.0:
        return 0.0
    if float(torch.linalg.vector_norm(right32).item()) == 0.0:
        return 0.0
    return float(F.cosine_similarity(left32, right32, dim=0).item())


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()).item())


@torch.no_grad()
def payload_integration_probe_row(
    model: torch.nn.Module,
    probe: EncodedProbe,
) -> dict[str, Any]:
    if probe.generator_version != GENERATOR_VERSION:
        raise RuntimeError("#1037 accepts only the frozen aligned-v4 generator")

    hidden_chunks = _hidden_chunks(model, probe.prompt_ids)
    final_hidden = hidden_chunks[-1][1]
    _, values, positions = _prior_bank(model, hidden_chunks)
    no_memory_logits = _fused_final_logits(model, final_hidden, final_hidden[0, -1])

    learned_state = final_hidden[0, -1]
    unit_state = final_hidden[0, -1]
    traces: list[dict[str, Any]] = []

    targets = _oracle_targets(probe)
    if targets:
        if values is None:
            raise RuntimeError("#1037 long-range oracle has no completed prior-chunk memory")
        position_to_index = {int(position): idx for idx, position in enumerate(positions)}
        gate = torch.sigmoid(model.memory_gate_logit).to(final_hidden.dtype)

        for hop, target in enumerate(targets):
            if target not in position_to_index:
                raise RuntimeError(f"#1037 oracle endpoint {target} is outside completed memory")
            memory = values[position_to_index[target]]

            learned_before = learned_state
            unit_before = unit_state
            learned_contribution = gate * memory

            learned_state = learned_before + learned_contribution
            unit_state = unit_before + memory

            learned_logits = _fused_final_logits(model, final_hidden, learned_state)
            unit_logits = _fused_final_logits(model, final_hidden, unit_state)

            learned_before_norm = _norm(learned_before)
            learned_contribution_norm = _norm(learned_contribution)
            traces.append(
                {
                    "hop": int(hop),
                    "target_position": int(target),
                    "raw_memory_norm": _norm(memory),
                    "learned_query_norm": learned_before_norm,
                    "unit_query_norm": _norm(unit_before),
                    "learned_gated_memory_norm": learned_contribution_norm,
                    "learned_contribution_norm_ratio": (
                        0.0
                        if learned_before_norm == 0.0
                        else learned_contribution_norm / learned_before_norm
                    ),
                    "learned_query_raw_memory_cosine": _cosine(learned_before, memory),
                    "learned_query_gated_memory_cosine": _cosine(
                        learned_before, learned_contribution
                    ),
                    "unit_query_raw_memory_cosine": _cosine(unit_before, memory),
                    "learned_gate_candidate_delta": _candidate_delta_trace(
                        learned_logits, no_memory_logits, probe
                    ),
                    "unit_gate_candidate_delta": _candidate_delta_trace(
                        unit_logits, no_memory_logits, probe
                    ),
                }
            )

    learned_final_logits = _fused_final_logits(model, final_hidden, learned_state)
    unit_final_logits = _fused_final_logits(model, final_hidden, unit_state)

    # Learned-gate oracle must remain bit-identical to the already-merged #1002
    # diagnostic so #1037 cannot redefine the first postmortem after results.
    frozen_learned = oracle_evidence_final_logits(model, probe)
    torch.testing.assert_close(
        learned_final_logits,
        frozen_learned,
        rtol=0.0,
        atol=0.0,
    )
    frozen_no_memory = no_memory_final_logits(model, probe.prompt_ids)
    torch.testing.assert_close(
        no_memory_logits,
        frozen_no_memory,
        rtol=0.0,
        atol=0.0,
    )

    def score(logits: torch.Tensor) -> dict[str, Any]:
        return candidate_score(
            logits,
            candidate_token_ids=probe.candidate_token_ids,
            answer_token_id=probe.answer_token_id,
            stale_token_ids=probe.stale_token_ids,
        )

    return {
        "family": probe.family,
        "case_id": int(probe.case_id),
        "generator_version": probe.generator_version,
        "no_memory": score(no_memory_logits),
        "oracle_learned_gate": score(learned_final_logits),
        "oracle_unit_gate": score(unit_final_logits),
        "hop_traces": traces,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("#1037 mean requires at least one value")
    return float(sum(float(value) for value in values) / len(values))


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float]:
    return {
        "accuracy": _mean([float(bool(row[field]["correct"])) for row in rows]),
        "candidate_nll": _mean([float(row[field]["candidate_nll"]) for row in rows]),
        "stale_choice_rate": _mean(
            [float(bool(row[field]["stale_choice"])) for row in rows]
        ),
    }


def _trace_values(
    rows: Sequence[Mapping[str, Any]],
    path: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for row in rows:
        for trace in row["hop_traces"]:
            current: Any = trace
            for key in path:
                current = current[key]
            values.append(float(current))
    return values


def summarize_payload_integration(
    rows: Sequence[Mapping[str, Any]],
    gate_stats: Mapping[str, float],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("#1037 requires at least one diagnostic row")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row["family"])
        if family not in ALL_FAMILIES:
            raise RuntimeError(f"#1037 unknown family {family!r}")
        grouped[family].append(row)

    per_family: dict[str, Any] = {}
    for family in ALL_FAMILIES:
        family_rows = grouped.get(family, [])
        if not family_rows:
            continue
        learned_margin = _trace_values(
            family_rows,
            ("learned_gate_candidate_delta", "answer_minus_best_wrong_delta_contribution"),
        )
        unit_margin = _trace_values(
            family_rows,
            ("unit_gate_candidate_delta", "answer_minus_best_wrong_delta_contribution"),
        )
        contribution_ratios = _trace_values(
            family_rows,
            ("learned_contribution_norm_ratio",),
        )
        no_memory = _condition_summary(family_rows, "no_memory")
        learned = _condition_summary(family_rows, "oracle_learned_gate")
        unit = _condition_summary(family_rows, "oracle_unit_gate")
        per_family[family] = {
            "conditions": {
                "no_memory": no_memory,
                "oracle_learned_gate": learned,
                "oracle_unit_gate": unit,
            },
            "learned_oracle_benefit": no_memory["candidate_nll"] - learned["candidate_nll"],
            "unit_oracle_benefit": no_memory["candidate_nll"] - unit["candidate_nll"],
            "unit_minus_learned_benefit": learned["candidate_nll"] - unit["candidate_nll"],
            "learned_accuracy_gain": learned["accuracy"] - no_memory["accuracy"],
            "unit_accuracy_gain": unit["accuracy"] - no_memory["accuracy"],
            "median_learned_contribution_norm_ratio": (
                None if not contribution_ratios else float(np.median(contribution_ratios))
            ),
            "mean_learned_answer_minus_best_wrong_delta": (
                None if not learned_margin else _mean(learned_margin)
            ),
            "mean_unit_answer_minus_best_wrong_delta": (
                None if not unit_margin else _mean(unit_margin)
            ),
        }

    long_rows = [row for row in rows if row["family"] in LONG_RANGE_FAMILIES]
    if not long_rows:
        raise RuntimeError("#1037 requires long-range rows")

    no_memory = _condition_summary(long_rows, "no_memory")
    learned = _condition_summary(long_rows, "oracle_learned_gate")
    unit = _condition_summary(long_rows, "oracle_unit_gate")
    learned_margin = _trace_values(
        long_rows,
        ("learned_gate_candidate_delta", "answer_minus_best_wrong_delta_contribution"),
    )
    unit_margin = _trace_values(
        long_rows,
        ("unit_gate_candidate_delta", "answer_minus_best_wrong_delta_contribution"),
    )
    contribution_ratios = _trace_values(
        long_rows,
        ("learned_contribution_norm_ratio",),
    )

    aggregate = {
        "learned_oracle_benefit": no_memory["candidate_nll"] - learned["candidate_nll"],
        "unit_oracle_benefit": no_memory["candidate_nll"] - unit["candidate_nll"],
        "unit_minus_learned_benefit": learned["candidate_nll"] - unit["candidate_nll"],
        "learned_accuracy_gain": learned["accuracy"] - no_memory["accuracy"],
        "unit_accuracy_gain": unit["accuracy"] - no_memory["accuracy"],
        "median_learned_contribution_norm_ratio": float(np.median(contribution_ratios)),
        "mean_learned_answer_minus_best_wrong_delta": _mean(learned_margin),
        "mean_unit_answer_minus_best_wrong_delta": _mean(unit_margin),
    }
    return {
        "row_count": len(rows),
        "gate_statistics": {str(key): float(value) for key, value in gate_stats.items()},
        "per_family": per_family,
        "aggregate": aggregate,
    }


def classify_payload_integration(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    gate = summary["gate_statistics"]
    agg = summary["aggregate"]
    tags: list[str] = []

    if (
        float(gate["median"]) <= GATE_SUPPRESSION_MAX_MEDIAN
        and float(agg["unit_minus_learned_benefit"])
        >= GATE_SUPPRESSION_MIN_UNIT_MINUS_LEARNED_NLL
        and float(agg["unit_oracle_benefit"]) >= GATE_SUPPRESSION_MIN_UNIT_BENEFIT
    ):
        tags.append("GATE_SUPPRESSION_SIGNAL")

    if (
        float(agg["unit_oracle_benefit"]) < PAYLOAD_WEAK_MIN_USEFUL_NLL
        and float(agg["unit_accuracy_gain"]) <= PAYLOAD_WEAK_MAX_ACCURACY_GAIN
        and float(agg["mean_unit_answer_minus_best_wrong_delta"]) <= 0.0
    ):
        tags.append("PAYLOAD_DIRECTION_WEAK_SIGNAL")

    if (
        float(agg["learned_oracle_benefit"]) <= LEARNED_HARM_MAX_BENEFIT
        and float(agg["mean_learned_answer_minus_best_wrong_delta"]) <= 0.0
    ):
        tags.append("LEARNED_GATE_HARM_SIGNAL")

    if float(agg["unit_oracle_benefit"]) <= UNIT_DESTABILIZATION_MAX_BENEFIT:
        tags.append("UNIT_GATE_DESTABILIZATION_SIGNAL")

    if not tags:
        tags.append("PAYLOAD_INTEGRATION_DECOMPOSITION_INCONCLUSIVE")

    return {
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_DECOMPOSITION_DIAGNOSTIC_ONLY",
        "tags": tags,
        "stage_c_result_changed": False,
        "first_postmortem_result_changed": False,
        "new_training_authorized": False,
        "new_scientific_seed_authorized": False,
        "stage_d_authorized": False,
    }
