from __future__ import annotations

"""Checkpoint-only CHM-v1 ~100M Stage-C STOP postmortem (#1002).

This module localizes the already-terminal #990 STOP result without training,
checkpoint mutation, paid-compute launchers, new scientific seeds, or Stage-D
authority.  All conditions operate on a frozen final EIEM checkpoint and the
unchanged aligned-v4 held-out probes.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .chm_v1_100m_scale import FIRST_SCREEN_TOKEN_BUDGET
from .chm_v1_100m_stage_c_eval import (
    ALL_FAMILIES,
    GENERATOR_VERSION,
    LONG_RANGE_FAMILIES,
    PROBE_SEED,
    candidate_score,
    eiem_flat_final_logits,
)
from .chm_v1_long_memory_eval import EncodedProbe
from .chm_v1_small_lm import EpisodicState, LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden

ISSUE = 1002
CLASSIFICATION = "CHM_V1_100M_STAGE_C_POSTMORTEM_PREREGISTRATION_NO_EXECUTION_AUTHORITY"
FINAL_STEP = 2048
FINAL_TOKENS = FIRST_SCREEN_TOKEN_BUDGET
SOFT_TEMPERATURE = 0.10
CONDITIONS = ("NO_MEMORY", "HARD_EXACT", "SOFT_TRAIN_T0P10", "ORACLE_EVIDENCE_ENDPOINT")

SOFT_HARD_MIN_NLL = 0.05
ADDRESS_MIN_NLL = 0.10
ADDRESS_MAX_HIT_RATE = 0.50
PAYLOAD_WEAK_MAX_NLL = 0.05
OVERWRITE_MIN_IMPROVEMENT = 0.05


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "issue": ISSUE,
        "source_result": "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH",
        "generator_version": GENERATOR_VERSION,
        "probe_seed": PROBE_SEED,
        "conditions": CONDITIONS,
        "soft_temperature": SOFT_TEMPERATURE,
        "final_step": FINAL_STEP,
        "final_tokens": FINAL_TOKENS,
        "checkpoint_only": True,
        "parameter_updates_authorized": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "gpu_authorized": False,
        "modal_authorized": False,
        "new_scientific_seed_authorized": False,
        "stage_d_authorized": False,
    }


def validate_protocol_manifest() -> dict[str, Any]:
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("#1002 requires the frozen two-hop CHM-v1 mechanism")
    if LOCAL_WINDOW != 512:
        raise RuntimeError("#1002 requires the frozen 512-token local window")
    if SOFT_TEMPERATURE != 0.10:
        raise RuntimeError("#1002 soft diagnostic temperature drift")
    if FINAL_STEP != 2048 or FINAL_TOKENS != 33_554_432:
        raise RuntimeError("#1002 final-checkpoint envelope drift")
    return protocol_manifest()


def validate_checkpoint_payload(payload: Mapping[str, Any], *, expected_kind: str) -> Mapping[str, Any]:
    """Fail closed unless a checkpoint is exactly the frozen final Stage-C artifact."""
    if expected_kind not in {"local", "eiem"}:
        raise ValueError("expected_kind must be local or eiem")
    if payload.get("kind") != expected_kind:
        raise RuntimeError(f"#1002 checkpoint kind mismatch: {payload.get('kind')!r}")
    if int(payload.get("step", -1)) != FINAL_STEP:
        raise RuntimeError("#1002 requires only the final step-2048 checkpoint")
    if int(payload.get("tokens_seen", -1)) != FINAL_TOKENS:
        raise RuntimeError("#1002 final checkpoint token count mismatch")
    if payload.get("resume_authorized") is not False:
        raise RuntimeError("#1002 checkpoint must remain non-resumable")
    if payload.get("scientific_evaluation_authorized") is not True:
        raise RuntimeError("#1002 checkpoint is not the frozen final scientific-evaluation artifact")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("#1002 checkpoint has no model_state_dict")
    return payload


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _chunks(ids: Sequence[int]) -> list[tuple[int, tuple[int, ...]]]:
    values = tuple(int(x) for x in ids)
    if not values:
        raise ValueError("prompt_ids must be non-empty")
    return [(start, values[start : start + LOCAL_WINDOW]) for start in range(0, len(values), LOCAL_WINDOW)]


@torch.no_grad()
def _hidden_chunks(model: torch.nn.Module, prompt_ids: Sequence[int]) -> list[tuple[int, torch.Tensor]]:
    device = _model_device(model)
    out: list[tuple[int, torch.Tensor]] = []
    for start, ids in _chunks(prompt_ids):
        tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out.append((start, _hidden(model.backbone, tokens)))
    return out


def _prior_bank(
    model: torch.nn.Module,
    hidden_chunks: Sequence[tuple[int, torch.Tensor]],
) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[int, ...]]:
    if len(hidden_chunks) <= 1:
        return None, None, ()
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    positions: list[int] = []
    for start, hidden in hidden_chunks[:-1]:
        keys.append(model.key_for(hidden)[0])
        values.append(hidden[0])
        positions.extend(range(start, start + hidden.shape[1]))
    return torch.cat(keys, dim=0), torch.cat(values, dim=0), tuple(positions)


def _fused_final_logits(model: torch.nn.Module, final_hidden: torch.Tensor, query_state: torch.Tensor) -> torch.Tensor:
    fused = final_hidden.clone()
    fused[0, -1] = query_state
    return model.backbone.lm_head(fused)[0, -1]


@torch.no_grad()
def no_memory_final_logits(model: torch.nn.Module, prompt_ids: Sequence[int]) -> torch.Tensor:
    hidden_chunks = _hidden_chunks(model, prompt_ids)
    final_hidden = hidden_chunks[-1][1]
    return _fused_final_logits(model, final_hidden, final_hidden[0, -1])


def _target_endpoint_for_hop(probe: EncodedProbe, hop: int) -> int | None:
    if probe.family == "local_negative":
        return None
    if probe.family == "two_hop":
        return probe.first_evidence_end_token if hop == 0 else probe.evidence_end_token
    return probe.evidence_end_token


def _rank_trace(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    positions: Sequence[int],
    target_position: int | None,
    selected_position: int,
) -> dict[str, Any]:
    q = query.detach().float().cpu().numpy().astype(np.float32, copy=False)
    k = keys.detach().float().cpu().numpy().astype(np.float32, copy=False)
    # Same ordering geometry as the exact index: normalized vectors make
    # maximum dot product equivalent to minimum squared Euclidean distance.
    delta = k.astype(np.float64, copy=False) - q.astype(np.float64, copy=False)[None, :]
    distances = np.einsum("nd,nd->n", delta, delta, dtype=np.float64)
    order = np.lexsort((np.asarray(positions, dtype=np.int64), distances))
    pos_to_index = {int(position): idx for idx, position in enumerate(positions)}
    selected_idx = pos_to_index[int(selected_position)]
    selected_score = float(np.dot(k[selected_idx].astype(np.float64), q.astype(np.float64)))
    trace: dict[str, Any] = {
        "selected_position": int(selected_position),
        "selected_score": selected_score,
        "target_position": None if target_position is None else int(target_position),
        "target_rank": None,
        "target_score": None,
        "selected_equals_target": None,
    }
    if target_position is not None and int(target_position) in pos_to_index:
        target_idx = pos_to_index[int(target_position)]
        target_order_index = int(np.where(order == target_idx)[0][0])
        target_score = float(np.dot(k[target_idx].astype(np.float64), q.astype(np.float64)))
        trace.update(
            {
                "target_rank": target_order_index + 1,
                "target_score": target_score,
                "selected_equals_target": int(selected_position) == int(target_position),
            }
        )
    return trace


@torch.no_grad()
def hard_exact_final_logits_with_trace(
    model: torch.nn.Module,
    probe: EncodedProbe,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Reproduce #986 HARD_EXACT logits while exposing non-causal address traces."""
    hidden_chunks = _hidden_chunks(model, probe.prompt_ids)
    final_hidden = hidden_chunks[-1][1]
    keys, values, positions = _prior_bank(model, hidden_chunks)
    query_state = final_hidden[0, -1]
    traces: list[dict[str, Any]] = []
    if keys is not None:
        assert values is not None
        state = EpisodicState("stage-c-1002-hard-trace")
        state.write(keys, values)
        for hop in range(RETRIEVAL_HOPS):
            query = model.query_for(query_state)
            value, result, _, _, _, _ = state.retrieve(
                query,
                mode="flat",
                verify_indexed_exactness=False,
            )
            selected_position = positions[result.position]
            traces.append(
                _rank_trace(
                    query=query,
                    keys=keys,
                    positions=positions,
                    target_position=_target_endpoint_for_hop(probe, hop),
                    selected_position=selected_position,
                )
            )
            query_state = model._integrate(query_state, value)
    logits = _fused_final_logits(model, final_hidden, query_state)
    # The trace path must never redefine the frozen scientific scorer.
    frozen = eiem_flat_final_logits(model, probe.prompt_ids)
    torch.testing.assert_close(logits, frozen, rtol=0.0, atol=0.0)
    return logits, traces


@torch.no_grad()
def soft_train_final_logits_with_trace(
    model: torch.nn.Module,
    probe: EncodedProbe,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    hidden_chunks = _hidden_chunks(model, probe.prompt_ids)
    final_hidden = hidden_chunks[-1][1]
    keys, values, positions = _prior_bank(model, hidden_chunks)
    query_state = final_hidden[0, -1]
    traces: list[dict[str, Any]] = []
    if keys is not None:
        assert values is not None
        position_to_index = {int(position): idx for idx, position in enumerate(positions)}
        for hop in range(RETRIEVAL_HOPS):
            query = model.query_for(query_state)
            scores = torch.einsum("d,sd->s", query, keys) / SOFT_TEMPERATURE
            weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
            memory = torch.matmul(weights, values)
            target = _target_endpoint_for_hop(probe, hop)
            target_mass = None
            if target is not None and int(target) in position_to_index:
                target_mass = float(weights[position_to_index[int(target)]].item())
            safe = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
            entropy = float((-(safe * safe.log()).sum()).item())
            top_index = int(torch.argmax(weights).item())
            traces.append(
                {
                    "top1_position": int(positions[top_index]),
                    "target_position": None if target is None else int(target),
                    "target_mass": target_mass,
                    "entropy": entropy,
                }
            )
            query_state = model._integrate(query_state, memory)
    return _fused_final_logits(model, final_hidden, query_state), traces


@torch.no_grad()
def oracle_evidence_final_logits(model: torch.nn.Module, probe: EncodedProbe) -> torch.Tensor:
    hidden_chunks = _hidden_chunks(model, probe.prompt_ids)
    final_hidden = hidden_chunks[-1][1]
    _, values, positions = _prior_bank(model, hidden_chunks)
    query_state = final_hidden[0, -1]

    if probe.family == "local_negative":
        # Its evidence is in the current chunk and is intentionally not episodic.
        return _fused_final_logits(model, final_hidden, query_state)

    if values is None:
        raise RuntimeError("#1002 long-range oracle has no completed prior-chunk memory")
    position_to_index = {int(position): idx for idx, position in enumerate(positions)}
    targets: tuple[int, ...]
    if probe.family == "two_hop":
        if probe.first_evidence_end_token is None:
            raise RuntimeError("#1002 two-hop oracle is missing first evidence endpoint")
        targets = (int(probe.first_evidence_end_token), int(probe.evidence_end_token))
    else:
        targets = (int(probe.evidence_end_token),)
    for target in targets:
        if target not in position_to_index:
            raise RuntimeError(f"#1002 oracle endpoint {target} is outside completed memory")
        query_state = model._integrate(query_state, values[position_to_index[target]])
    return _fused_final_logits(model, final_hidden, query_state)


def _score(logits: torch.Tensor, probe: EncodedProbe) -> dict[str, Any]:
    return candidate_score(
        logits,
        candidate_token_ids=probe.candidate_token_ids,
        answer_token_id=probe.answer_token_id,
        stale_token_ids=probe.stale_token_ids,
    )


@torch.no_grad()
def diagnostic_probe_row(model: torch.nn.Module, probe: EncodedProbe) -> dict[str, Any]:
    if probe.generator_version != GENERATOR_VERSION:
        raise RuntimeError("#1002 accepts only the frozen aligned-v4 probe generator")
    no_memory = no_memory_final_logits(model, probe.prompt_ids)
    hard, hard_trace = hard_exact_final_logits_with_trace(model, probe)
    soft, soft_trace = soft_train_final_logits_with_trace(model, probe)
    oracle = oracle_evidence_final_logits(model, probe)
    return {
        "family": probe.family,
        "case_id": int(probe.case_id),
        "generator_version": probe.generator_version,
        "no_memory": _score(no_memory, probe),
        "hard_exact": _score(hard, probe),
        "soft_train_t0p10": _score(soft, probe),
        "oracle_evidence_endpoint": _score(oracle, probe),
        "hard_trace": hard_trace,
        "soft_trace": soft_trace,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return float(sum(float(x) for x in values) / len(values))


def summarize_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("#1002 requires diagnostic rows")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row["family"])
        if family not in ALL_FAMILIES:
            raise RuntimeError(f"#1002 unknown family {family!r}")
        grouped[family].append(row)

    per_family: dict[str, Any] = {}
    for family in ALL_FAMILIES:
        family_rows = grouped.get(family, [])
        if not family_rows:
            continue
        condition_summary: dict[str, Any] = {}
        for field in ("no_memory", "hard_exact", "soft_train_t0p10", "oracle_evidence_endpoint"):
            condition_summary[field] = {
                "accuracy": _mean([float(bool(row[field]["correct"])) for row in family_rows]),
                "candidate_nll": _mean([float(row[field]["candidate_nll"]) for row in family_rows]),
                "stale_choice_rate": _mean([float(bool(row[field]["stale_choice"])) for row in family_rows]),
            }
        hard_ranks = [
            float(trace["target_rank"])
            for row in family_rows
            for trace in row["hard_trace"]
            if trace.get("target_rank") is not None
        ]
        hard_hits = [
            float(bool(trace["selected_equals_target"]))
            for row in family_rows
            for trace in row["hard_trace"]
            if trace.get("selected_equals_target") is not None
        ]
        soft_masses = [
            float(trace["target_mass"])
            for row in family_rows
            for trace in row["soft_trace"]
            if trace.get("target_mass") is not None
        ]
        soft_entropies = [float(trace["entropy"]) for row in family_rows for trace in row["soft_trace"]]
        per_family[family] = {
            "conditions": condition_summary,
            "hard_authoritative_endpoint_hit_rate": None if not hard_hits else _mean(hard_hits),
            "hard_authoritative_endpoint_median_rank": None if not hard_ranks else float(np.median(hard_ranks)),
            "soft_authoritative_endpoint_mean_mass": None if not soft_masses else _mean(soft_masses),
            "soft_mean_entropy": None if not soft_entropies else _mean(soft_entropies),
        }

    long_rows = [row for row in rows if row["family"] in LONG_RANGE_FAMILIES]
    if not long_rows:
        raise RuntimeError("#1002 requires long-range diagnostic rows")

    def nll(field: str) -> float:
        return _mean([float(row[field]["candidate_nll"]) for row in long_rows])

    hard_hits = [
        float(bool(trace["selected_equals_target"]))
        for row in long_rows
        for trace in row["hard_trace"]
        if trace.get("selected_equals_target") is not None
    ]
    aggregate = {
        "soft_minus_hard_nll_benefit": nll("hard_exact") - nll("soft_train_t0p10"),
        "oracle_minus_hard_nll_benefit": nll("hard_exact") - nll("oracle_evidence_endpoint"),
        "oracle_minus_no_memory_nll_benefit": nll("no_memory") - nll("oracle_evidence_endpoint"),
        "hard_authoritative_endpoint_hit_rate": None if not hard_hits else _mean(hard_hits),
    }
    return {"rows": len(rows), "per_family": per_family, "aggregate": aggregate}


def classify_diagnostic_signals(summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    per_family = summary["per_family"]
    tags: list[str] = []

    soft_accuracy_wins = 0
    for family in LONG_RANGE_FAMILIES:
        conditions = per_family[family]["conditions"]
        if conditions["soft_train_t0p10"]["accuracy"] > conditions["hard_exact"]["accuracy"]:
            soft_accuracy_wins += 1
    if (
        float(aggregate["soft_minus_hard_nll_benefit"]) >= SOFT_HARD_MIN_NLL
        and soft_accuracy_wins >= 2
    ):
        tags.append("SOFT_HARD_MISMATCH_SIGNAL")

    hit_rate = aggregate.get("hard_authoritative_endpoint_hit_rate")
    if (
        float(aggregate["oracle_minus_hard_nll_benefit"]) >= ADDRESS_MIN_NLL
        and hit_rate is not None
        and float(hit_rate) < ADDRESS_MAX_HIT_RATE
    ):
        tags.append("ADDRESS_SELECTION_SIGNAL")

    if float(aggregate["oracle_minus_no_memory_nll_benefit"]) < PAYLOAD_WEAK_MAX_NLL:
        tags.append("PAYLOAD_INTEGRATION_WEAK_SIGNAL")

    overwrite = per_family["overwrite"]["conditions"]
    if (
        float(overwrite["hard_exact"]["stale_choice_rate"])
        - float(overwrite["oracle_evidence_endpoint"]["stale_choice_rate"])
        >= OVERWRITE_MIN_IMPROVEMENT
        and float(overwrite["hard_exact"]["candidate_nll"])
        - float(overwrite["oracle_evidence_endpoint"]["candidate_nll"])
        >= OVERWRITE_MIN_IMPROVEMENT
    ):
        tags.append("OVERWRITE_RECENCY_SIGNAL")

    if not tags:
        tags.append("POSTMORTEM_INCONCLUSIVE")
    return {
        "classification": "CHM_V1_100M_STAGE_C_POSTMORTEM_DIAGNOSTIC_ONLY",
        "tags": tags,
        "stage_c_result_changed": False,
        "stage_d_authorized": False,
        "new_training_authorized": False,
    }
