from __future__ import annotations

"""Issue #578 first-divergence localizer for the consumed #571 systems FAIL.

This module is diagnostic only. It wraps the frozen v26 execution path, records
observations, returns every production output unchanged, and restores all instance
method wrappers in ``finally``. It performs no training and changes no threshold.
"""

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch

from . import aera_v25_post8471_triage as triage
from . import aera_v26_5_end_to_end_systems as base
from . import aera_v26_8_issue562_end_to_end_systems as issue562
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    DUPLICATE_SIMILARITY,
)
from .aera_hardware_core_v26 import CoalescedFICEMMemory, HardwareAwareAERATextLMV26
from .aera_hardware_core_v26_8_ficem_read_mixed_strength_precision import (
    StrengthPrecisionTritonFICEMReadWriteBackend,
)

RESEARCH_ISSUE = 578
SOURCE_MAIN = "bbf548edf47fd91948c54819f1cee47c4c567ed6"
SOURCE_TREE = "ed4020eec5a51819c5fed3cc02f2895c2c8d8821"
ISSUE571_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
ISSUE574_INSPECTOR_RUN = 33744802059
ISSUE574_INSPECTOR_JOB = 100614716963
ISSUE574_MERGE = SOURCE_MAIN
V26_8_CANDIDATE_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}
BATCH_SIZES = (8, 64)
INTEGRATED_ATOL = 1e-2
INTEGRATED_RTOL = 1e-2
DUPLICATE_THRESHOLD = 0.95
MAX_GPU_SECONDS = 300


def localization_protocol() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "issue571_result_sha256": ISSUE571_RESULT_SHA256,
        "issue574_inspector_run": ISSUE574_INSPECTOR_RUN,
        "issue574_inspector_job": ISSUE574_INSPECTOR_JOB,
        "checkpoint": base.CHECKPOINT_RELATIVE_DIR,
        "checkpoint_hashes": dict(CHECKPOINT_HASHES),
        "batch_sizes": list(BATCH_SIZES),
        "token_seed_rule": "138471 + 10000 + batch_size",
        "hard": True,
        "route_mode": "hard_sparse",
        "update_memory": True,
        "cuda_autocast": "bfloat16",
        "integrated_atol": INTEGRATED_ATOL,
        "integrated_rtol": INTEGRATED_RTOL,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "candidate_backend": StrengthPrecisionTritonFICEMReadWriteBackend.name,
        "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
        "v26_6_write_blob": V26_6_WRITE_BLOB,
        "instrumentation": "instance wrappers return original outputs unchanged and restore in finally",
        "candidate_backend_adjudication_internal_decisions_available": False,
        "reference_formula_replay_from_exact_update_inputs": True,
        "transformer_timing_performed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_written": False,
        "scientific_seed_consumed": False,
        "repair_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def cpu_contract_preflight_issue578() -> dict[str, Any]:
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue578 batch surface drifted")
    if triage.DIAGNOSTIC_SEED != 138_471:
        raise RuntimeError("issue578 diagnostic seed lineage drifted")
    if (INTEGRATED_ATOL, INTEGRATED_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue578 integrated tolerance drifted")
    if float(DUPLICATE_SIMILARITY) != DUPLICATE_THRESHOLD:
        raise RuntimeError("issue578 duplicate threshold drifted")
    if base.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue578 checkpoint path drifted")
    candidate_protocol = issue562.issue562_systems_protocol()
    if candidate_protocol["candidate_backend"] != StrengthPrecisionTritonFICEMReadWriteBackend.name:
        raise RuntimeError("issue578 candidate backend identity drifted")
    if candidate_protocol["v26_8_candidate_blob"] != V26_8_CANDIDATE_BLOB:
        raise RuntimeError("issue578 candidate blob drifted")
    if candidate_protocol["v26_6_write_blob"] != V26_6_WRITE_BLOB:
        raise RuntimeError("issue578 inherited WRITE blob drifted")
    return {
        "protocol": localization_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "localization_measurement_performed": False,
        "scientific_seed_consumed": False,
        "repair_authorized": False,
        "architecture_freeze_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().clone()


def _memory_snapshot(state: ContextualEpisodicMemoryState) -> dict[str, torch.Tensor]:
    return {
        "keys": _cpu(state.keys),
        "values": _cpu(state.values),
        "strengths": _cpu(state.strengths),
        "valid": _cpu(state.valid),
    }


def _stage_state_snapshot(state) -> dict[str, torch.Tensor] | None:
    memory = getattr(state, "memory", None)
    stream = getattr(state, "stream", None)
    if not isinstance(stream, torch.Tensor) or not isinstance(memory, ContextualEpisodicMemoryState):
        return None
    result = {"stream": _cpu(stream)}
    result.update(_memory_snapshot(memory))
    return result


def _controller_snapshot(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: _cpu(value) for name, value in output.items() if isinstance(value, torch.Tensor)}


def _route_snapshot(meta: dict[str, object]) -> dict[str, Any]:
    gate = meta.get("stage_route_gate")
    probability = meta.get("stage_route_probability")
    result: dict[str, Any] = {
        "gate": _cpu(gate) if isinstance(gate, torch.Tensor) else None,
        "probability": _cpu(probability) if isinstance(probability, torch.Tensor) else None,
        "executed_fraction": float(meta.get("executed_fraction", 0.0)),
    }
    if isinstance(gate, torch.Tensor):
        selected = (gate[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
        result["selected_population"] = _cpu(selected)
    else:
        result["selected_population"] = None
    return result


def _reference_formula_adjudication(
    projected_new_keys: torch.Tensor,
    normalized_old_keys: torch.Tensor,
    write_strength: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> dict[str, Any]:
    """Replay frozen reference adjudication from exact update inputs.

    This is explicitly diagnostic replay. Candidate Triton kernel-internal decision
    tensors are not exposed by the production backend and are never fabricated.
    """
    with torch.no_grad():
        new_strengths = write_strength[..., 0].clamp(0.0, 1.0)
        initial_new_valid = new_strengths > 0.0
        incoming_similarity = torch.einsum("bkd,bjd->bkj", projected_new_keys, projected_new_keys)
        k_count = projected_new_keys.size(1)
        position = torch.arange(k_count, device=projected_new_keys.device)
        later = position[None, :, None] < position[None, None, :]
        incoming_pair_ge = (
            incoming_similarity.ge(DUPLICATE_THRESHOLD)
            & initial_new_valid[:, :, None]
            & initial_new_valid[:, None, :]
            & later
        )
        shadowed_incoming = incoming_pair_ge.any(dim=2)
        final_new_valid = initial_new_valid & ~shadowed_incoming

        old_similarity = torch.einsum("bkd,bsd->bks", projected_new_keys, normalized_old_keys)
        old_pair_ge = (
            old_similarity.ge(DUPLICATE_THRESHOLD)
            & final_new_valid[:, :, None]
            & state.valid[:, None, :]
        )
        duplicate_old = old_pair_ge.any(dim=1)
        keep_old = state.valid & ~duplicate_old

        incoming_relevant = (
            initial_new_valid[:, :, None]
            & initial_new_valid[:, None, :]
            & later
        )
        old_relevant = final_new_valid[:, :, None] & state.valid[:, None, :]
        incoming_margin = (incoming_similarity - DUPLICATE_THRESHOLD).abs()
        old_margin = (old_similarity - DUPLICATE_THRESHOLD).abs()
        incoming_min_margin = (
            float(incoming_margin[incoming_relevant].min())
            if bool(incoming_relevant.any())
            else None
        )
        old_min_margin = (
            float(old_margin[old_relevant].min())
            if bool(old_relevant.any())
            else None
        )

        return {
            "kind": "torch_reference_formula_replay_from_exact_update_inputs",
            "actual_candidate_backend_internal_decisions_available": False,
            "incoming_min_abs_margin_to_0_95": incoming_min_margin,
            "old_min_abs_margin_to_0_95": old_min_margin,
            "initial_new_valid": _cpu(initial_new_valid),
            "incoming_similarity": _cpu(incoming_similarity),
            "incoming_pair_ge_threshold": _cpu(incoming_pair_ge),
            "shadowed_incoming": _cpu(shadowed_incoming),
            "final_new_valid": _cpu(final_new_valid),
            "old_similarity": _cpu(old_similarity),
            "old_pair_ge_threshold": _cpu(old_pair_ge),
            "duplicate_old": _cpu(duplicate_old),
            "keep_old": _cpu(keep_old),
        }


@contextmanager
def _capture_model(model: HardwareAwareAERATextLMV26, *, label: str) -> Iterator[dict[str, Any]]:
    """Capture production execution by instance wrapping and restore every wrapper."""
    records: dict[str, Any] = {}
    context: dict[str, Any] = {"route_call": 0, "current": None}
    restorations: list[tuple[object, str, object]] = []
    stage_count = len(model.stages)

    original_route = model._route_one_stage

    def wrapped_route(this, x, stage, stage_state, router, *, route_mode, update_memory):
        call = int(context["route_call"])
        context["route_call"] = call + 1
        chunk_index = call // stage_count
        stage_index = call % stage_count
        key = f"chunk{chunk_index}.stage{stage_index}"
        record: dict[str, Any] = {
            "label": label,
            "chunk": chunk_index,
            "stage": stage_index,
            "input": _cpu(x),
            "pre_state": _stage_state_snapshot(stage_state),
            "controller_calls": [],
            "reads": [],
            "updates": [],
        }
        records[key] = record
        context["current"] = key
        try:
            y, new_state, meta = original_route(
                x,
                stage,
                stage_state,
                router,
                route_mode=route_mode,
                update_memory=update_memory,
            )
            record["route"] = _route_snapshot(meta)
            record["output"] = _cpu(y)
            record["post_state"] = _stage_state_snapshot(new_state)
            executed = bool(record["controller_calls"])
            record["reasoner_steps"] = (
                _cpu(getattr(stage.reasoner, "last_steps", None)) if executed else None
            )
            record["selected_write_indices"] = (
                _cpu(getattr(stage, "last_selected_indices", None)) if executed else None
            )
            record["selected_write_strengths"] = (
                _cpu(getattr(stage, "last_pair_strength", None)) if executed else None
            )
            record["pair_gate"] = (
                _cpu(getattr(stage, "last_pair_gate", None)) if executed else None
            )
            record["candidate_count"] = int(getattr(stage, "last_candidate_count", 0)) if executed else 0
            record["selected_count"] = int(getattr(stage, "last_selected_count", 0)) if executed else 0
            if record["reads"] and record["controller_calls"]:
                recalled = record["reads"][-1].get("recalled")
                start = record["controller_calls"][0]["output"].get("memory_read")
                if isinstance(recalled, torch.Tensor) and isinstance(start, torch.Tensor):
                    record["applied_read"] = start[:, None, :] * recalled
                else:
                    record["applied_read"] = None
            else:
                record["applied_read"] = None
            return y, new_state, meta
        finally:
            context["current"] = None

    object.__setattr__(model, "_route_one_stage", MethodType(wrapped_route, model))
    restorations.append((model, "_route_one_stage", original_route))

    for router_index, router in enumerate(model.stage_routers):
        original_router = router.forward

        def make_router(original, expected_stage):
            def wrapped(this, first_event, stream, *, mode):
                gate, logits = original(first_event, stream, mode=mode)
                key = context.get("current")
                if key is not None and int(records[key]["stage"]) == expected_stage:
                    records[key]["router_call"] = {
                        "first_event": _cpu(first_event),
                        "stream": _cpu(stream),
                        "gate": _cpu(gate),
                        "logits": _cpu(logits),
                        "probability": _cpu(torch.sigmoid(logits)),
                    }
                return gate, logits
            return wrapped

        object.__setattr__(router, "forward", MethodType(make_router(original_router, router_index), router))
        restorations.append((router, "forward", original_router))

    for stage_index, stage in enumerate(model.stages):
        controller = stage.controller
        original_controller = controller.forward

        def make_controller(original, expected_stage):
            def wrapped(this, event, stream):
                output = original(event, stream)
                key = context.get("current")
                if key is not None and int(records[key]["stage"]) == expected_stage:
                    calls = records[key]["controller_calls"]
                    role = "start" if len(calls) == 0 else "end" if len(calls) == 1 else f"extra_{len(calls)}"
                    calls.append(
                        {
                            "role": role,
                            "event": _cpu(event),
                            "stream": _cpu(stream),
                            "output": _controller_snapshot(output),
                        }
                    )
                return output
            return wrapped

        object.__setattr__(controller, "forward", MethodType(make_controller(original_controller, stage_index), controller))
        restorations.append((controller, "forward", original_controller))

        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise TypeError("issue578 requires CoalescedFICEMMemory stages")

        original_read = memory.read_with_reuse

        def make_read(original, expected_stage):
            def wrapped(this, identity_source, context_source, state):
                recalled, projected_query, normalized_old_keys = original(
                    identity_source, context_source, state
                )
                key = context.get("current")
                if key is not None and int(records[key]["stage"]) == expected_stage:
                    records[key]["reads"].append(
                        {
                            "identity_source": _cpu(identity_source),
                            "context_source": _cpu(context_source),
                            "pre_memory": _memory_snapshot(state),
                            "recalled": _cpu(recalled),
                            "projected_query": _cpu(projected_query),
                            "normalized_old_keys": _cpu(normalized_old_keys),
                        }
                    )
                return recalled, projected_query, normalized_old_keys
            return wrapped

        object.__setattr__(memory, "read_with_reuse", MethodType(make_read(original_read, stage_index), memory))
        restorations.append((memory, "read_with_reuse", original_read))

        original_projected = memory.update_block_from_projected

        def make_projected(original, expected_stage):
            def wrapped(this, projected_new_keys, normalized_old_keys, payload_source, write_strength, state):
                result = original(
                    projected_new_keys,
                    normalized_old_keys,
                    payload_source,
                    write_strength,
                    state,
                )
                key = context.get("current")
                if key is not None and int(records[key]["stage"]) == expected_stage:
                    records[key]["updates"].append(
                        {
                            "kind": "projected",
                            "projected_new_keys": _cpu(projected_new_keys),
                            "normalized_old_keys": _cpu(normalized_old_keys),
                            "payload_source": _cpu(payload_source),
                            "write_strength": _cpu(write_strength),
                            "pre_memory": _memory_snapshot(state),
                            "adjudication_replay": _reference_formula_adjudication(
                                projected_new_keys,
                                normalized_old_keys,
                                write_strength,
                                state,
                            ),
                            "post_memory": _memory_snapshot(result),
                        }
                    )
                return result
            return wrapped

        object.__setattr__(memory, "update_block_from_projected", MethodType(make_projected(original_projected, stage_index), memory))
        restorations.append((memory, "update_block_from_projected", original_projected))

        original_direct = memory.update_block

        def make_direct(original, expected_stage):
            def wrapped(this, identity_source, context_source, payload_source, write_strength, state):
                result = original(
                    identity_source,
                    context_source,
                    payload_source,
                    write_strength,
                    state,
                )
                key = context.get("current")
                if key is not None and int(records[key]["stage"]) == expected_stage:
                    records[key]["updates"].append(
                        {
                            "kind": "direct_known_empty_or_nonreuse",
                            "identity_source": _cpu(identity_source),
                            "context_source": _cpu(context_source),
                            "payload_source": _cpu(payload_source),
                            "write_strength": _cpu(write_strength),
                            "pre_memory": _memory_snapshot(state),
                            "actual_backend_adjudication_decisions_available": False,
                            "adjudication_replay": None,
                            "post_memory": _memory_snapshot(result),
                        }
                    )
                return result
            return wrapped

        object.__setattr__(memory, "update_block", MethodType(make_direct(original_direct, stage_index), memory))
        restorations.append((memory, "update_block", original_direct))

    try:
        yield records
    finally:
        for obj, name, original in reversed(restorations):
            object.__setattr__(obj, name, original)


def _tensor_comparison(a: torch.Tensor | None, b: torch.Tensor | None, *, decision_bearing: bool = False) -> dict[str, Any]:
    if a is None or b is None:
        return {
            "available": False if a is None and b is None else True,
            "left_available": a is not None,
            "right_available": b is not None,
            "metadata_exact": a is None and b is None,
            "exact": a is None and b is None,
            "allclose": a is None and b is None,
            "decision_bearing": decision_bearing,
            "failure": not (a is None and b is None),
        }
    metadata_exact = a.dtype == b.dtype and a.shape == b.shape
    exact = bool(metadata_exact and torch.equal(a, b))
    is_continuous = a.is_floating_point() and b.is_floating_point()
    allclose = bool(
        metadata_exact
        and (
            torch.allclose(a.float(), b.float(), atol=INTEGRATED_ATOL, rtol=INTEGRATED_RTOL, equal_nan=True)
            if is_continuous
            else torch.equal(a, b)
        )
    )
    result: dict[str, Any] = {
        "available": True,
        "left_dtype": str(a.dtype),
        "right_dtype": str(b.dtype),
        "left_shape": list(a.shape),
        "right_shape": list(b.shape),
        "metadata_exact": bool(metadata_exact),
        "exact": exact,
        "allclose": allclose,
        "decision_bearing": decision_bearing,
        "failure": not allclose,
    }
    if metadata_exact and a.numel() > 0 and is_continuous:
        diff = (a.float() - b.float()).abs().reshape(-1)
        max_value, max_index = diff.max(dim=0)
        flat_a = a.float().reshape(-1)
        flat_b = b.float().reshape(-1)
        idx = int(max_index)
        result.update(
            {
                "max_abs": float(max_value),
                "max_abs_flat_index": idx,
                "left_at_max": float(flat_a[idx]),
                "right_at_max": float(flat_b[idx]),
            }
        )
    elif metadata_exact and a.numel() > 0:
        mismatch = (a != b).reshape(-1)
        result["mismatch_count"] = int(mismatch.sum())
        if bool(mismatch.any()):
            result["first_mismatch_flat_index"] = int(mismatch.nonzero(as_tuple=False)[0])
    return result


def _state_boundaries(prefix: str, left: dict[str, torch.Tensor] | None, right: dict[str, torch.Tensor] | None) -> list[dict[str, Any]]:
    fields = ("stream", "keys", "values", "strengths", "valid")
    rows: list[dict[str, Any]] = []
    for field in fields:
        a = left.get(field) if left else None
        b = right.get(field) if right else None
        rows.append(
            {
                "name": f"{prefix}.{field}",
                "comparison": _tensor_comparison(a, b, decision_bearing=(field == "valid")),
            }
        )
    return rows


def _first_item(rows: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    for row in rows:
        if predicate(row["comparison"]):
            return {
                "name": row["name"],
                "chunk": row.get("chunk"),
                "stage": row.get("stage"),
                "comparison": row["comparison"],
            }
    return None


def _compare_records(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    boundaries: list[dict[str, Any]] = []
    threshold_margin_diagnostics: list[dict[str, Any]] = []
    keys = sorted(
        set(reference) | set(candidate),
        key=lambda key: tuple(int(part.replace("chunk", "").replace("stage", "")) for part in key.split(".")),
    )
    by_chunk: dict[int, list[str]] = {}
    for key in keys:
        chunk = int(key.split(".")[0].replace("chunk", ""))
        by_chunk.setdefault(chunk, []).append(key)

    def add(name: str, chunk: int, stage: int | None, comparison: dict[str, Any]) -> None:
        boundaries.append({"name": name, "chunk": chunk, "stage": stage, "comparison": comparison})

    for chunk in sorted(by_chunk):
        for key in by_chunk[chunk]:
            left = reference.get(key)
            right = candidate.get(key)
            stage = int(key.split(".")[1].replace("stage", ""))
            if left is None or right is None:
                add(f"{key}.record_presence", chunk, stage, _tensor_comparison(None if left is None else torch.tensor([1]), None if right is None else torch.tensor([1]), decision_bearing=True))
                continue

            lroute, rroute = left.get("route", {}), right.get("route", {})
            add(f"{key}.route.gate", chunk, stage, _tensor_comparison(lroute.get("gate"), rroute.get("gate"), decision_bearing=True))
            add(f"{key}.route.probability", chunk, stage, _tensor_comparison(lroute.get("probability"), rroute.get("probability")))
            add(f"{key}.route.selected_population", chunk, stage, _tensor_comparison(lroute.get("selected_population"), rroute.get("selected_population"), decision_bearing=True))
            lrouter, rrouter = left.get("router_call", {}), right.get("router_call", {})
            add(f"{key}.route.logits", chunk, stage, _tensor_comparison(lrouter.get("logits"), rrouter.get("logits")))

            for row in _state_boundaries(f"{key}.pre_state", left.get("pre_state"), right.get("pre_state")):
                row.update({"chunk": chunk, "stage": stage})
                boundaries.append(row)

            lreads, rreads = left.get("reads", []), right.get("reads", [])
            lread = lreads[0] if lreads else None
            rread = rreads[0] if rreads else None
            for field in ("identity_source", "context_source", "recalled", "projected_query", "normalized_old_keys"):
                add(
                    f"{key}.read.{field}",
                    chunk,
                    stage,
                    _tensor_comparison(lread.get(field) if lread else None, rread.get(field) if rread else None),
                )
            add(f"{key}.read.applied", chunk, stage, _tensor_comparison(left.get("applied_read"), right.get("applied_read")))

            lcalls, rcalls = left.get("controller_calls", []), right.get("controller_calls", [])
            lend = next((call for call in lcalls if call.get("role") == "end"), None)
            rend = next((call for call in rcalls if call.get("role") == "end"), None)
            add(f"{key}.end_controller.event", chunk, stage, _tensor_comparison(lend.get("event") if lend else None, rend.get("event") if rend else None))
            for field in ("novelty", "memory_write", "depth_logits"):
                add(
                    f"{key}.end_controller.{field}",
                    chunk,
                    stage,
                    _tensor_comparison(
                        lend.get("output", {}).get(field) if lend else None,
                        rend.get("output", {}).get(field) if rend else None,
                    ),
                )
            add(f"{key}.reasoner_steps", chunk, stage, _tensor_comparison(left.get("reasoner_steps"), right.get("reasoner_steps"), decision_bearing=True))
            add(f"{key}.stage.output", chunk, stage, _tensor_comparison(left.get("output"), right.get("output")))

            lpost, rpost = left.get("post_state"), right.get("post_state")
            if lpost or rpost:
                add(f"{key}.stage.stream", chunk, stage, _tensor_comparison(lpost.get("stream") if lpost else None, rpost.get("stream") if rpost else None))

            add(f"{key}.write.selected_indices", chunk, stage, _tensor_comparison(left.get("selected_write_indices"), right.get("selected_write_indices"), decision_bearing=True))
            add(f"{key}.write.selected_strengths", chunk, stage, _tensor_comparison(left.get("selected_write_strengths"), right.get("selected_write_strengths")))
            add(f"{key}.write.pair_gate", chunk, stage, _tensor_comparison(left.get("pair_gate"), right.get("pair_gate")))

            lupdates, rupdates = left.get("updates", []), right.get("updates", [])
            lu = lupdates[0] if lupdates else None
            ru = rupdates[0] if rupdates else None
            for field in ("projected_new_keys", "payload_source", "write_strength"):
                add(
                    f"{key}.update.{field}",
                    chunk,
                    stage,
                    _tensor_comparison(lu.get(field) if lu else None, ru.get(field) if ru else None),
                )
            lrep = lu.get("adjudication_replay") if lu else None
            rrep = ru.get("adjudication_replay") if ru else None
            threshold_margin_diagnostics.append(
                {
                    "chunk": chunk,
                    "stage": stage,
                    "reference_incoming_min_abs_margin_to_0_95": lrep.get("incoming_min_abs_margin_to_0_95") if lrep else None,
                    "candidate_incoming_min_abs_margin_to_0_95": rrep.get("incoming_min_abs_margin_to_0_95") if rrep else None,
                    "reference_old_min_abs_margin_to_0_95": lrep.get("old_min_abs_margin_to_0_95") if lrep else None,
                    "candidate_old_min_abs_margin_to_0_95": rrep.get("old_min_abs_margin_to_0_95") if rrep else None,
                    "candidate_backend_internal_decisions_available": False,
                }
            )
            for field in ("initial_new_valid", "shadowed_incoming", "final_new_valid", "duplicate_old", "keep_old"):
                add(
                    f"{key}.adjudication_replay.{field}",
                    chunk,
                    stage,
                    _tensor_comparison(lrep.get(field) if lrep else None, rrep.get(field) if rrep else None, decision_bearing=True),
                )
            for field in ("incoming_similarity", "old_similarity"):
                add(
                    f"{key}.adjudication_replay.{field}",
                    chunk,
                    stage,
                    _tensor_comparison(lrep.get(field) if lrep else None, rrep.get(field) if rrep else None),
                )

            lpm = lu.get("post_memory") if lu else None
            rpm = ru.get("post_memory") if ru else None
            if lpm or rpm:
                for field in ("keys", "values", "strengths", "valid"):
                    add(
                        f"{key}.post_write.{field}",
                        chunk,
                        stage,
                        _tensor_comparison(lpm.get(field) if lpm else None, rpm.get(field) if rpm else None, decision_bearing=(field == "valid")),
                    )

        start = chunk * chunk_size
        end = min((chunk + 1) * chunk_size, reference_logits.size(1), candidate_logits.size(1))
        add(
            f"chunk{chunk}.output_logits",
            chunk,
            None,
            _tensor_comparison(reference_logits[:, start:end], candidate_logits[:, start:end]),
        )

    first_bitwise = _first_item(boundaries, lambda c: c.get("available") and not c.get("exact", True))
    first_tolerance_or_metadata = _first_item(boundaries, lambda c: bool(c.get("failure")))
    first_discrete = _first_item(
        boundaries,
        lambda c: bool(c.get("decision_bearing")) and bool(c.get("failure")),
    )
    failures = [
        {"name": row["name"], "chunk": row["chunk"], "stage": row["stage"], "comparison": row["comparison"]}
        for row in boundaries
        if row["comparison"].get("failure")
    ]
    return {
        "boundary_count": len(boundaries),
        "first_bitwise_difference": first_bitwise,
        "first_integrated_tolerance_or_metadata_failure": first_tolerance_or_metadata,
        "first_discrete_decision_difference": first_discrete,
        "failures": failures,
        "boundaries": boundaries,
        "threshold_margin_diagnostics": threshold_margin_diagnostics,
        "candidate_backend_internal_adjudication_decisions_available": False,
        "adjudication_replay_is_diagnostic_only": True,
    }


@torch.inference_mode()
def run_first_divergence_localization(
    *, run_dir: str = base.CHECKPOINT_RELATIVE_DIR
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue578 first-divergence localization requires one separately authorized NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    hashes_before = base.checkpoint_hashes(run_dir)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError(f"issue578 checkpoint hash drift before localization: {hashes_before}")
    reference, candidate, transformer, candidate_backend_names = issue562.load_models_v26_8(
        run_dir=run_dir,
        device=device,
    )
    del transformer
    torch.cuda.empty_cache()

    reference_versions = tuple(int(parameter._version) for parameter in reference.parameters())
    candidate_versions = tuple(int(parameter._version) for parameter in candidate.parameters())
    rows: dict[str, Any] = {}

    for batch_size in BATCH_SIZES:
        generator = torch.Generator(device="cpu").manual_seed(
            triage.DIAGNOSTIC_SEED + 10_000 + batch_size
        )
        tokens = torch.randint(
            0,
            triage.VOCAB_SIZE,
            (batch_size, triage.SEQ_LEN),
            generator=generator,
        ).to(device)

        with _capture_model(reference, label="reference") as reference_records:
            reference_output = triage._model_forward(reference, tokens, update_memory=True)
        with _capture_model(candidate, label="candidate") as candidate_records:
            candidate_output = triage._model_forward(candidate, tokens, update_memory=True)

        reference_logits = reference_output.get("logits")
        candidate_logits = candidate_output.get("logits")
        reference_state = reference_output.get("state")
        candidate_state = candidate_output.get("state")
        if not isinstance(reference_logits, torch.Tensor) or not isinstance(candidate_logits, torch.Tensor):
            raise RuntimeError("issue578 model output missing logits")
        if not isinstance(reference_state, HardwareAERAState) or not isinstance(candidate_state, HardwareAERAState):
            raise RuntimeError("issue578 model output missing state")

        comparison = _compare_records(
            reference_records,
            candidate_records,
            _cpu(reference_logits),
            _cpu(candidate_logits),
            chunk_size=reference.cfg.chunk_size,
        )
        rows[str(batch_size)] = {
            "comparison": comparison,
            "final_logit": _tensor_comparison(_cpu(reference_logits), _cpu(candidate_logits)),
            "final_state": [
                {
                    row["name"].split(".")[-1]: row["comparison"]
                    for row in _state_boundaries(
                        f"final.stage{stage_index}",
                        _stage_state_snapshot(reference_state.stages[stage_index]),
                        _stage_state_snapshot(candidate_state.stages[stage_index]),
                    )
                }
                for stage_index in range(len(reference_state.stages))
            ],
        }
        del reference_output, candidate_output, reference_logits, candidate_logits, tokens
        torch.cuda.empty_cache()

    versions_unchanged = bool(
        reference_versions == tuple(int(parameter._version) for parameter in reference.parameters())
        and candidate_versions == tuple(int(parameter._version) for parameter in candidate.parameters())
    )
    hashes_after = base.checkpoint_hashes(run_dir)
    hashes_unchanged = hashes_before == hashes_after == CHECKPOINT_HASHES

    return {
        "research_issue": RESEARCH_ISSUE,
        "scope": "first_divergence_localization_of_consumed_issue571_only",
        "device": torch.cuda.get_device_name(device),
        "protocol": localization_protocol(),
        "candidate_backend_names": list(candidate_backend_names),
        "rows": rows,
        "parameter_versions_unchanged": versions_unchanged,
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "checkpoint_hashes_unchanged": hashes_unchanged,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_written": False,
        "scientific_seed_consumed": False,
        "repair_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
