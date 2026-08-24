from __future__ import annotations

"""CPU-only post-#313 audit of AERA-v21 write kinetics and interference.

This is a deterministic diagnostic reproduction of the already-observed #313
controlled seed. It is not independent evidence and changes no production model
mechanics. Same-checkpoint evaluation interventions only localize whether the
remaining integrated-memory failure comes from weak one-shot correction,
irrelevant writes, candidate interference, or downstream injection.
"""

from contextlib import ExitStack, contextmanager
import json
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import (
    CHUNK_SIZE,
    DISTRACT,
    EVAL_SEED,
    FULL_ACCURACY_MIN,
    FULL_OVER_STREAM_MIN,
    N_KEYS,
    N_OVERWRITES,
    OVERWRITE_ACCURACY_MIN,
    QUERY,
    SAME_CHECKPOINT_MEMORY_DROP_MIN,
    STALE_ERROR_MAX,
    VALUE_START,
    WRITE,
    _evaluate,
    _query_logits,
    deployment_safety_and_isolation,
    make_batch,
)
from aera_v21_conflict_free_memory_objective_cpu import (
    train_pair_with_conflict_free_objective,
)
from aera_v21_memory_code_objective_cpu import _decode_with_frozen_model_head
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21

MATERIAL_GAIN = 0.15
RAW_RECALL_STRONG = 0.95
DIAGNOSTIC_ONE_SHOT_LR = 1.0


@contextmanager
def _keep_candidate_one_preserve_gate(
    model: HardwareAwareAERATextLMV21,
) -> Iterator[None]:
    """Suppress all candidates except index 1 without changing its learned gate."""
    originals: list[tuple[torch.nn.Module, object]] = []
    for stage in model.stages:
        gate = stage.pair_write_gate
        original = gate.forward
        originals.append((gate, original))

        def forward_keep_one(
            this: torch.nn.Module,
            x: torch.Tensor,
            _orig=original,
        ) -> torch.Tensor:
            logits = _orig(x)
            masked = torch.full_like(logits, -30.0)
            if logits.size(1) > 1:
                masked[:, 1, :] = logits[:, 1, :]
            return masked

        gate.forward = MethodType(forward_keep_one, gate)
    try:
        yield
    finally:
        for gate, original in originals:
            gate.forward = original  # type: ignore[method-assign]


@contextmanager
def _temporary_memory_lr(
    model: HardwareAwareAERATextLMV21,
    lr: float | None,
) -> Iterator[None]:
    if lr is None:
        yield
        return
    originals: list[tuple[torch.nn.Module, float]] = []
    for stage in model.stages:
        memory = stage.memory
        originals.append((memory, float(memory.lr)))
        memory.lr = float(lr)
    try:
        yield
    finally:
        for memory, original in originals:
            memory.lr = original


def _prefix(chunk: torch.Tensor) -> int:
    prefix = chunk[:, 0]
    if not bool(prefix.eq(prefix[0]).all()):
        raise RuntimeError("audit expects synchronized chunk types across the batch")
    return int(prefix[0])


def _accuracy(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float((prediction[mask] == target[mask]).float().mean())


def _group_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    overwrite: torch.Tensor,
    stale: torch.Tensor,
) -> dict[str, Any]:
    non_overwrite = ~overwrite
    result: dict[str, Any] = {
        "overall_accuracy": float((prediction == target).float().mean()),
        "non_overwrite_accuracy": _accuracy(prediction, target, non_overwrite),
        "overwrite_current_value_accuracy": _accuracy(prediction, target, overwrite),
        "overwrite_stale_value_error": _accuracy(prediction, stale, overwrite),
        "query_ordinal_accuracy": [
            float((prediction[:, i] == target[:, i]).float().mean())
            for i in range(prediction.size(1))
        ],
    }
    return result


@torch.no_grad()
def _evaluate_mode(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
    *,
    write_chunks_only: bool = False,
    keep_candidate_one: bool = False,
    memory_lr: float | None = None,
) -> dict[str, Any]:
    """Chunkwise final evaluation plus pre-injection stage-0 raw memory decode."""
    model.eval()
    model.set_memory_pretraining_mode(False)
    state = None
    logits_parts: list[torch.Tensor] = []
    raw_predictions: list[torch.Tensor] = []
    raw_target_stale_margins: list[torch.Tensor] = []
    query_index = 0

    with ExitStack() as stack:
        if keep_candidate_one:
            stack.enter_context(_keep_candidate_one_preserve_gate(model))
        stack.enter_context(_temporary_memory_lr(model, memory_lr))

        for start in range(0, batch.tokens.size(1), CHUNK_SIZE):
            chunk = batch.tokens[:, start : start + CHUNK_SIZE]
            prefix = _prefix(chunk)

            if prefix == QUERY:
                if state is None:
                    raise RuntimeError("query reached before memory state exists")
                stage0 = model.stages[0]
                pos = torch.arange(chunk.size(1), device=chunk.device)
                events = model.token_emb(chunk) + model.local_pos(pos)[None, :, :]
                h = stage0.norm(events)
                query_h = h[:, 1:2]
                q = F.normalize(stage0.memory.q(query_h), dim=-1)
                recalled = torch.einsum(
                    "btd,bdm->btm",
                    q,
                    state.stages[0].memory.matrix,
                ).squeeze(1)
                payload_code = stage0.memory.out(recalled)
                raw_logits = _decode_with_frozen_model_head(model, payload_code)
                legal = raw_logits[:, VALUE_START : VALUE_START + 16]
                raw_prediction = legal.argmax(dim=-1) + VALUE_START
                raw_predictions.append(raw_prediction.cpu())

                target = batch.query_targets[:, query_index].to(raw_logits.device)
                stale = batch.stale_targets[:, query_index].to(raw_logits.device)
                overwrite = batch.overwrite_mask[:, query_index].to(raw_logits.device)
                safe_stale = torch.where(overwrite, stale, target)
                row = torch.arange(raw_logits.size(0), device=raw_logits.device)
                margin = raw_logits[row, target] - raw_logits[row, safe_stale]
                raw_target_stale_margins.append(margin.cpu())
                query_index += 1

            update = True
            if write_chunks_only:
                update = prefix == WRITE
            out = model(
                chunk,
                state=state,
                hard=True,
                route_mode="hard_sparse",
                update_memory=update,
                return_block_logits=False,
            )
            logits = out["logits"]
            if not isinstance(logits, torch.Tensor):
                raise TypeError("expected tensor logits")
            logits_parts.append(logits)
            state = out["state"]

    if query_index != batch.query_targets.size(1):
        raise RuntimeError("query count mismatch in chunkwise audit")

    logits = torch.cat(logits_parts, dim=1)
    selected = _query_logits(logits, batch.query_positions)
    target = batch.query_targets.to(selected.device)
    prediction = selected.argmax(dim=-1).cpu()
    final = _group_metrics(
        prediction,
        batch.query_targets,
        batch.overwrite_mask,
        batch.stale_targets,
    )
    final["query_nll"] = float(
        F.cross_entropy(selected.float().reshape(-1, selected.size(-1)), target.reshape(-1))
    )

    raw_prediction = torch.stack(raw_predictions, dim=1)
    raw = _group_metrics(
        raw_prediction,
        batch.query_targets,
        batch.overwrite_mask,
        batch.stale_targets,
    )
    margins = torch.stack(raw_target_stale_margins, dim=1)
    overwrite = batch.overwrite_mask
    raw["overwrite_target_minus_stale_logit_margin"] = (
        float(margins[overwrite].float().mean()) if bool(overwrite.any()) else float("nan")
    )
    return {"final": final, "raw_memory_decode": raw}


@torch.no_grad()
def trace_write_kinetics(
    model: HardwareAwareAERATextLMV21,
    batch: Any,
) -> dict[str, float]:
    """Trace learned gates and effective eta for semantically distinct writes."""
    model.eval()
    model.set_memory_pretraining_mode(False)
    state = None
    groups: dict[str, list[float]] = {
        "initial_target_pair_gate": [],
        "overwrite_target_pair_gate": [],
        "initial_target_strength": [],
        "overwrite_target_strength": [],
        "initial_target_eta": [],
        "overwrite_target_eta": [],
        "write_chunk_strength": [],
        "write_non_target_strength": [],
        "query_all_candidate_strength": [],
        "distract_all_candidate_strength": [],
    }

    for chunk_index, start in enumerate(range(0, batch.tokens.size(1), CHUNK_SIZE)):
        chunk = batch.tokens[:, start : start + CHUNK_SIZE]
        prefix = _prefix(chunk)
        out = model(
            chunk,
            state=state,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )
        state = out["state"]
        stage0 = model.stages[0]
        pair_gate = stage0.last_pair_gate
        pair_strength = stage0.last_pair_strength
        if pair_gate is None or pair_strength is None:
            raise RuntimeError("missing v21 pair-write trace")
        end = stage0.last_end_controls
        if end is None:
            raise RuntimeError("missing end controller trace")
        chunk_strength = (end["novelty"] * end["memory_write"]).clamp(0.0, 1.0)
        lr = float(stage0.memory.lr)

        if prefix == WRITE:
            groups["write_chunk_strength"].append(float(chunk_strength.float().mean()))
            target_gate = pair_gate[:, 1, 0]
            target_strength = pair_strength[:, 1, 0]
            mask = torch.ones(pair_strength.size(1), dtype=torch.bool, device=pair_strength.device)
            mask[1] = False
            groups["write_non_target_strength"].append(
                float(pair_strength[:, mask, :].float().mean())
            )
            if chunk_index < N_KEYS:
                prefix_name = "initial"
            elif chunk_index < N_KEYS + N_OVERWRITES:
                prefix_name = "overwrite"
            else:
                raise RuntimeError("unexpected WRITE chunk ordering")
            groups[f"{prefix_name}_target_pair_gate"].append(float(target_gate.float().mean()))
            groups[f"{prefix_name}_target_strength"].append(float(target_strength.float().mean()))
            groups[f"{prefix_name}_target_eta"].append(float((lr * target_strength).float().mean()))
        elif prefix == QUERY:
            groups["query_all_candidate_strength"].append(float(pair_strength.float().mean()))
        elif prefix == DISTRACT:
            groups["distract_all_candidate_strength"].append(float(pair_strength.float().mean()))

    def mean(name: str) -> float:
        xs = groups[name]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {name + "_mean": mean(name) for name in groups}


def _original_gate_pass_for_normal_lr1(
    normal_lr1: dict[str, Any],
    stream_eval: dict[str, Any],
    memory_off: dict[str, Any],
    safety: dict[str, Any],
) -> bool:
    final = normal_lr1["final"]
    return bool(
        final["overall_accuracy"] >= FULL_ACCURACY_MIN
        and final["overall_accuracy"] - stream_eval["query_accuracy"] >= FULL_OVER_STREAM_MIN
        and final["overall_accuracy"] - memory_off["query_accuracy"] >= SAME_CHECKPOINT_MEMORY_DROP_MIN
        and final["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN
        and final["overwrite_stale_value_error"] <= STALE_ERROR_MAX
        and safety["deployment_base_parameters_unchanged"]
        and safety["session_isolation_exact"]
    )


def _diagnose(
    modes: dict[str, dict[str, Any]],
    *,
    normal_lr1_original_gate_pass: bool,
) -> tuple[str, dict[str, bool]]:
    normal = modes["normal"]
    write_only = modes["write_chunks_only"]
    candidate = modes["write_chunks_only_candidate1"]
    candidate_lr1 = modes["write_chunks_only_candidate1_lr1"]

    downstream = bool(
        normal["raw_memory_decode"]["overall_accuracy"] >= RAW_RECALL_STRONG
        and normal["raw_memory_decode"]["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN
        and normal["final"]["overall_accuracy"] < FULL_ACCURACY_MIN
    )
    pollution = bool(
        write_only["final"]["overall_accuracy"] - normal["final"]["overall_accuracy"] >= MATERIAL_GAIN
        or write_only["raw_memory_decode"]["overall_accuracy"]
        - normal["raw_memory_decode"]["overall_accuracy"]
        >= MATERIAL_GAIN
    )
    candidate_interference = bool(
        candidate["final"]["overall_accuracy"] - write_only["final"]["overall_accuracy"] >= MATERIAL_GAIN
        or candidate["raw_memory_decode"]["overall_accuracy"]
        - write_only["raw_memory_decode"]["overall_accuracy"]
        >= MATERIAL_GAIN
    )
    under_correction = bool(
        candidate["final"]["overwrite_current_value_accuracy"] < OVERWRITE_ACCURACY_MIN
        and candidate_lr1["raw_memory_decode"]["overall_accuracy"] >= RAW_RECALL_STRONG
        and candidate_lr1["final"]["overwrite_current_value_accuracy"] >= OVERWRITE_ACCURACY_MIN
        and candidate_lr1["final"]["overwrite_stale_value_error"] <= STALE_ERROR_MAX
    )
    unresolved_even_oracle_lr1 = bool(
        candidate_lr1["raw_memory_decode"]["overall_accuracy"] < RAW_RECALL_STRONG
        or candidate_lr1["final"]["overwrite_current_value_accuracy"] < OVERWRITE_ACCURACY_MIN
        or candidate_lr1["final"]["overwrite_stale_value_error"] > STALE_ERROR_MAX
    )
    findings = {
        "downstream_read_or_injection_blocker": downstream,
        "query_or_distractor_write_pollution_material": pollution,
        "pair_candidate_selectivity_or_interference_material": candidate_interference,
        "one_shot_delta_under_correction_localized": under_correction,
        "normal_candidates_lr1_passes_original_end_to_end_gates": normal_lr1_original_gate_pass,
        "oracle_write_only_lr1_still_insufficient": unresolved_even_oracle_lr1,
    }

    if downstream:
        diagnosis = "downstream_read_or_injection_blocker"
    elif under_correction:
        diagnosis = "one_shot_delta_correction_write_kinetics_blocker"
    elif pollution:
        diagnosis = "query_or_distractor_write_pollution_blocker"
    elif candidate_interference:
        diagnosis = "pair_candidate_selectivity_or_interference_blocker"
    elif normal_lr1_original_gate_pass:
        diagnosis = "simple_delta_under_correction_dominates_with_normal_candidate_traffic"
    else:
        diagnosis = "matrix_interference_or_write_read_representation_mismatch_remains"
    return diagnosis, findings


def run_audit() -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    full, stream_only, training = train_pair_with_conflict_free_objective()
    eval_batch = make_batch(24, EVAL_SEED)

    reference = _evaluate(full, eval_batch, memory=True)
    modes = {
        "normal": _evaluate_mode(full, eval_batch),
        "write_chunks_only": _evaluate_mode(full, eval_batch, write_chunks_only=True),
        "write_chunks_only_candidate1": _evaluate_mode(
            full,
            eval_batch,
            write_chunks_only=True,
            keep_candidate_one=True,
        ),
        "normal_lr1": _evaluate_mode(full, eval_batch, memory_lr=DIAGNOSTIC_ONE_SHOT_LR),
        "write_chunks_only_candidate1_lr1": _evaluate_mode(
            full,
            eval_batch,
            write_chunks_only=True,
            keep_candidate_one=True,
            memory_lr=DIAGNOSTIC_ONE_SHOT_LR,
        ),
    }
    normal = modes["normal"]["final"]
    if abs(reference["query_accuracy"] - normal["overall_accuracy"]) > 1e-7:
        raise RuntimeError("chunkwise audit changed normal query accuracy")
    if abs(reference["query_nll"] - normal["query_nll"]) > 1e-6:
        raise RuntimeError("chunkwise audit changed normal query NLL")

    activity = trace_write_kinetics(full, eval_batch)
    memory_off = _evaluate(full, eval_batch, memory=False)
    stream_eval = _evaluate(stream_only, eval_batch, memory=False)
    safety = deployment_safety_and_isolation(full, eval_batch)
    lr1_gate_pass = _original_gate_pass_for_normal_lr1(
        modes["normal_lr1"], stream_eval, memory_off, safety
    )
    diagnosis, findings = _diagnose(
        modes,
        normal_lr1_original_gate_pass=lr1_gate_pass,
    )

    return {
        "scope": "aera_v21_post_313_write_kinetics_interference_audit_cpu",
        "diagnostic_reproduction_only": True,
        "independent_evidence": False,
        "training_protocol": {
            "source": "exact_313_conflict_free_training",
            "steps": 500,
            "lr": 4e-3,
            "final_local_code": training["final_local_code"],
        },
        "normal_reference": {
            "query_accuracy": reference["query_accuracy"],
            "query_nll": reference["query_nll"],
            "overwrite_current_value_accuracy": reference["overwrite_current_value_accuracy"],
            "stale_value_error_rate": reference["stale_value_error_rate"],
        },
        "write_kinetics": activity,
        "counterfactuals": modes,
        "controls": {
            "stream_only_query_accuracy": stream_eval["query_accuracy"],
            "memory_off_query_accuracy": memory_off["query_accuracy"],
            "deployment_safety": safety,
            "normal_lr1_original_end_to_end_gate_pass": lr1_gate_pass,
        },
        "thresholds": {
            "material_gain": MATERIAL_GAIN,
            "raw_recall_strong": RAW_RECALL_STRONG,
            "overwrite_accuracy_min": OVERWRITE_ACCURACY_MIN,
            "stale_error_max": STALE_ERROR_MAX,
            "diagnostic_one_shot_lr": DIAGNOSTIC_ONE_SHOT_LR,
        },
        "findings": findings,
        "diagnosis": diagnosis,
        "claims": {
            "production_change_authorized": False,
            "gpu_authorized": False,
            "real_language_run_authorized": False,
            "v22_authorized": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_audit()
    print("AERA_V21_WRITE_KINETICS_AUDIT_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
