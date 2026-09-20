from __future__ import annotations

"""Checkpoint-only CHM-v1 payload/gate bootstrap diagnostic (#1014).

This module is post-STOP localization only.  It contains no checkpoint loader,
launcher, optimizer, parameter update, or paid-compute authority.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

from .chm_v1_100m_stage_c_eval import (
    GENERATOR_VERSION,
    LONG_RANGE_FAMILIES,
    PROBE_SEED,
)
from .chm_v1_100m_stage_c_postmortem import (
    _fused_final_logits,
    _hidden_chunks,
    _prior_bank,
    no_memory_final_logits,
)
from .chm_v1_long_memory_eval import EncodedProbe
from .chm_v1_small_lm import LOCAL_WINDOW

ISSUE = 1014
CLASSIFICATION = (
    "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_DIAGNOSTIC_"
    "PREREGISTRATION_NO_EXECUTION_AUTHORITY"
)
SOURCE_POSTMORTEM_ISSUE = 1008
SOURCE_RESULT_COMMENT = 5750912351
SOURCE_CHECKPOINT_SHA256 = (
    "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831"
)
SOURCE_CHECKPOINT_BYTES = 407_424_818

INITIAL_GATE_LOGIT = -4.0
CONDITIONS = (
    "NO_MEMORY",
    "ANSWER_TOKEN_LEARNED_GATE",
    "ANSWER_TOKEN_FULL_RESIDUAL",
    "LABEL_DIRECTION_LEARNED_GATE",
    "LABEL_DIRECTION_FULL_RESIDUAL",
)

GATE_SUPPRESSION_FULL_MIN = 0.10
GATE_SUPPRESSION_LEARNED_MAX = 0.05
GATE_SUPPRESSION_MEDIAN_MAX = 0.05
RAW_PAYLOAD_FORMAT_LABEL_MIN = 0.10
RAW_PAYLOAD_FORMAT_RAW_MAX = 0.05
RAW_PAYLOAD_USABLE_MIN = 0.05
RESIDUAL_CHANNEL_WEAK_MAX = 0.10


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "issue": ISSUE,
        "source_postmortem_issue": SOURCE_POSTMORTEM_ISSUE,
        "source_result_comment": SOURCE_RESULT_COMMENT,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_checkpoint_bytes": SOURCE_CHECKPOINT_BYTES,
        "generator_version": GENERATOR_VERSION,
        "probe_seed": PROBE_SEED,
        "families": tuple(LONG_RANGE_FAMILIES),
        "conditions": CONDITIONS,
        "initial_gate_logit": INITIAL_GATE_LOGIT,
        "checkpoint_only": True,
        "production_checkpoint_loading_authorized": False,
        "parameter_updates_authorized": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "backward_authorized": False,
        "gpu_authorized": False,
        "modal_authorized": False,
        "new_scientific_seed_authorized": False,
        "stage_d_authorized": False,
    }


def validate_protocol_manifest() -> dict[str, Any]:
    if LOCAL_WINDOW != 512:
        raise RuntimeError("#1014 requires the frozen 512-token local window")
    if INITIAL_GATE_LOGIT != -4.0:
        raise RuntimeError("#1014 gate initialization drift")
    if SOURCE_CHECKPOINT_BYTES != 407_424_818:
        raise RuntimeError("#1014 checkpoint byte binding drift")
    if len(SOURCE_CHECKPOINT_SHA256) != 64:
        raise RuntimeError("#1014 checkpoint SHA-256 binding drift")
    return protocol_manifest()


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().float().flatten().cpu()
    if x.numel() == 0:
        raise ValueError("stats require at least one value")
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "p05": float(torch.quantile(x, 0.05).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


@torch.no_grad()
def gate_state_summary(model: torch.nn.Module) -> dict[str, Any]:
    if not hasattr(model, "memory_gate_logit"):
        raise RuntimeError("#1014 model has no memory_gate_logit")
    logits = model.memory_gate_logit.detach().float()
    if logits.ndim != 1:
        raise RuntimeError("#1014 memory_gate_logit must be a vector")
    gate = torch.sigmoid(logits)
    delta = (logits - INITIAL_GATE_LOGIT).abs()
    return {
        "dimensions": int(logits.numel()),
        "logit": _tensor_stats(logits),
        "sigmoid": _tensor_stats(gate),
        "sigmoid_fraction_ge_0p05": float((gate >= 0.05).float().mean().item()),
        "sigmoid_fraction_ge_0p10": float((gate >= 0.10).float().mean().item()),
        "sigmoid_fraction_ge_0p25": float((gate >= 0.25).float().mean().item()),
        "mean_abs_logit_delta_from_init": float(delta.mean().item()),
        "max_abs_logit_delta_from_init": float(delta.max().item()),
    }


def unique_answer_token_position(probe: EncodedProbe) -> int:
    if probe.generator_version != GENERATOR_VERSION:
        raise RuntimeError("#1014 accepts only the frozen aligned-v4 probe generator")
    if probe.family not in LONG_RANGE_FAMILIES:
        raise RuntimeError("#1014 primary payload diagnostic is long-range only")
    final_chunk_start = (int(probe.query_token) // LOCAL_WINDOW) * LOCAL_WINDOW
    if final_chunk_start <= 0:
        raise RuntimeError("#1014 long-range probe has no completed prior memory")
    matches = [
        index
        for index, token_id in enumerate(probe.prompt_ids[:final_chunk_start])
        if int(token_id) == int(probe.answer_token_id)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "#1014 expected exactly one answer-token occurrence in completed prior memory; "
            f"family={probe.family} case={probe.case_id} matches={matches}"
        )
    return int(matches[0])


def _condition_score(
    logits: torch.Tensor,
    probe: EncodedProbe,
) -> dict[str, Any]:
    candidate_ids = tuple(int(x) for x in probe.candidate_token_ids)
    answer_id = int(probe.answer_token_id)
    if answer_id not in candidate_ids:
        raise RuntimeError("#1014 answer token is missing from candidate set")
    index = torch.tensor(candidate_ids, dtype=torch.long, device=logits.device)
    candidate_logits = logits.index_select(0, index).float()
    answer_index = candidate_ids.index(answer_id)
    answer_logit = float(candidate_logits[answer_index].item())
    wrong = torch.cat(
        (candidate_logits[:answer_index], candidate_logits[answer_index + 1 :])
    )
    if wrong.numel() == 0:
        raise RuntimeError("#1014 requires at least one wrong candidate")
    best_wrong = float(wrong.max().item())
    log_probs = torch.log_softmax(candidate_logits, dim=0)
    predicted = int(candidate_ids[int(torch.argmax(candidate_logits).item())])
    return {
        "predicted_token_id": predicted,
        "correct": predicted == answer_id,
        "candidate_nll": float(-log_probs[answer_index].item()),
        "answer_logit": answer_logit,
        "best_wrong_logit": best_wrong,
        "answer_margin": answer_logit - best_wrong,
    }


@torch.no_grad()
def payload_gate_probe_row(
    model: torch.nn.Module,
    probe: EncodedProbe,
) -> dict[str, Any]:
    """Score the five frozen #1014 counterfactual conditions for one probe."""
    answer_position = unique_answer_token_position(probe)
    hidden_chunks = _hidden_chunks(model, probe.prompt_ids)
    final_hidden = hidden_chunks[-1][1]
    _, prior_values, prior_positions = _prior_bank(model, hidden_chunks)
    if prior_values is None:
        raise RuntimeError("#1014 answer-token oracle has no completed prior memory")
    position_to_index = {int(position): idx for idx, position in enumerate(prior_positions)}
    if answer_position not in position_to_index:
        raise RuntimeError("#1014 answer-token position is not in the prior memory bank")

    query_state = final_hidden[0, -1]
    raw_payload = prior_values[position_to_index[answer_position]]

    label_weight = model.backbone.lm_head.weight[int(probe.answer_token_id)]
    label_direction = F.normalize(label_weight.float(), dim=0)
    raw_norm = raw_payload.float().norm()
    label_payload = (label_direction * raw_norm).to(
        device=query_state.device,
        dtype=query_state.dtype,
    )

    learned_raw_state = model._integrate(query_state, raw_payload)
    full_raw_state = query_state + raw_payload
    learned_label_state = model._integrate(query_state, label_payload)
    full_label_state = query_state + label_payload

    logits = {
        "no_memory": no_memory_final_logits(model, probe.prompt_ids),
        "answer_token_learned_gate": _fused_final_logits(
            model, final_hidden, learned_raw_state
        ),
        "answer_token_full_residual": _fused_final_logits(
            model, final_hidden, full_raw_state
        ),
        "label_direction_learned_gate": _fused_final_logits(
            model, final_hidden, learned_label_state
        ),
        "label_direction_full_residual": _fused_final_logits(
            model, final_hidden, full_label_state
        ),
    }

    raw_float = raw_payload.float()
    query_float = query_state.float()
    learned_delta = (learned_raw_state - query_state).float()
    full_delta = (full_raw_state - query_state).float()
    raw_cosine = float(
        F.cosine_similarity(raw_float, label_direction, dim=0).item()
    )
    return {
        "family": probe.family,
        "case_id": int(probe.case_id),
        "generator_version": probe.generator_version,
        "answer_token_id": int(probe.answer_token_id),
        "answer_token_position": answer_position,
        "evidence_end_token": int(probe.evidence_end_token),
        "answer_position_minus_evidence_end": (
            answer_position - int(probe.evidence_end_token)
        ),
        "conditions": {
            name: _condition_score(value, probe) for name, value in logits.items()
        },
        "payload": {
            "raw_answer_hidden_norm": float(raw_norm.item()),
            "query_state_norm": float(query_float.norm().item()),
            "raw_answer_hidden_to_label_direction_cosine": raw_cosine,
            "learned_gate_residual_norm": float(learned_delta.norm().item()),
            "full_strength_residual_norm": float(full_delta.norm().item()),
        },
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return float(sum(float(value) for value in values) / len(values))


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    condition_names = (
        "no_memory",
        "answer_token_learned_gate",
        "answer_token_full_residual",
        "label_direction_learned_gate",
        "label_direction_full_residual",
    )
    conditions: dict[str, Any] = {}
    for condition in condition_names:
        conditions[condition] = {
            "accuracy": _mean(
                [float(bool(row["conditions"][condition]["correct"])) for row in rows]
            ),
            "candidate_nll": _mean(
                [float(row["conditions"][condition]["candidate_nll"]) for row in rows]
            ),
            "answer_margin": _mean(
                [float(row["conditions"][condition]["answer_margin"]) for row in rows]
            ),
        }
    payload = {
        "raw_answer_hidden_to_label_direction_cosine": _mean(
            [
                float(row["payload"]["raw_answer_hidden_to_label_direction_cosine"])
                for row in rows
            ]
        ),
        "raw_answer_hidden_norm": _mean(
            [float(row["payload"]["raw_answer_hidden_norm"]) for row in rows]
        ),
        "query_state_norm": _mean(
            [float(row["payload"]["query_state_norm"]) for row in rows]
        ),
        "learned_gate_residual_norm": _mean(
            [float(row["payload"]["learned_gate_residual_norm"]) for row in rows]
        ),
        "full_strength_residual_norm": _mean(
            [float(row["payload"]["full_strength_residual_norm"]) for row in rows]
        ),
        "mean_answer_position_minus_evidence_end": _mean(
            [float(row["answer_position_minus_evidence_end"]) for row in rows]
        ),
    }
    return {"conditions": conditions, "payload": payload}


def summarize_payload_gate_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    gate_state: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("#1014 requires diagnostic rows")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row["family"])
        if family not in LONG_RANGE_FAMILIES:
            raise RuntimeError(f"#1014 non-long-range family in diagnostic: {family!r}")
        grouped[family].append(row)
    missing = [family for family in LONG_RANGE_FAMILIES if not grouped.get(family)]
    if missing:
        raise RuntimeError(f"#1014 missing long-range families: {missing}")

    per_family = {
        family: _summarize_rows(grouped[family]) for family in LONG_RANGE_FAMILIES
    }
    aggregate = _summarize_rows(rows)
    return {
        "rows": len(rows),
        "gate_state": dict(gate_state),
        "aggregate": aggregate,
        "per_family": per_family,
    }


def classify_payload_gate_signals(summary: Mapping[str, Any]) -> dict[str, Any]:
    conditions = summary["aggregate"]["conditions"]
    no_memory_nll = float(conditions["no_memory"]["candidate_nll"])

    def benefit(name: str) -> float:
        return no_memory_nll - float(conditions[name]["candidate_nll"])

    benefits = {
        "answer_token_learned_gate": benefit("answer_token_learned_gate"),
        "answer_token_full_residual": benefit("answer_token_full_residual"),
        "label_direction_learned_gate": benefit("label_direction_learned_gate"),
        "label_direction_full_residual": benefit("label_direction_full_residual"),
    }
    gate_median = float(summary["gate_state"]["sigmoid"]["median"])
    tags: list[str] = []

    if (
        benefits["label_direction_full_residual"] >= GATE_SUPPRESSION_FULL_MIN
        and benefits["label_direction_learned_gate"] < GATE_SUPPRESSION_LEARNED_MAX
        and gate_median < GATE_SUPPRESSION_MEDIAN_MAX
    ):
        tags.append("GATE_SUPPRESSION_SIGNAL")

    if (
        benefits["label_direction_full_residual"] >= RAW_PAYLOAD_FORMAT_LABEL_MIN
        and benefits["answer_token_full_residual"] < RAW_PAYLOAD_FORMAT_RAW_MAX
    ):
        tags.append("RAW_PAYLOAD_FORMAT_SIGNAL")

    if benefits["answer_token_full_residual"] >= RAW_PAYLOAD_USABLE_MIN:
        tags.append("RAW_PAYLOAD_USABLE_SIGNAL")

    if benefits["label_direction_full_residual"] < RESIDUAL_CHANNEL_WEAK_MAX:
        tags.append("RESIDUAL_CHANNEL_WEAK_SIGNAL")

    if not tags:
        tags.append("PAYLOAD_GATE_DIAGNOSTIC_INCONCLUSIVE")

    return {
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_DIAGNOSTIC_ONLY",
        "tags": tags,
        "nll_benefits": benefits,
        "gate_sigmoid_median": gate_median,
        "stage_c_result_changed": False,
        "stage_d_authorized": False,
        "new_training_authorized": False,
        "new_scientific_seed_authorized": False,
    }
