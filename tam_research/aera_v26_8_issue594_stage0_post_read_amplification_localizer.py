from __future__ import annotations

"""Issue #594 observational localization of stage0 post-READ amplification.

This diagnostic preserves the exact #588 checkpoint/token/autocast/hard-sparse
fixture and observes only production tensors already flowing through chunk1 stage0.
Hooks and instance wrappers return original outputs unchanged and are restored in
``finally``. No production module is invoked a second time and no threshold changes.
"""

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch

from . import aera_v25_post8471_triage as triage
from . import aera_v26_5_end_to_end_systems as base
from . import aera_v26_8_issue562_end_to_end_systems as issue562
from . import aera_v26_8_issue578_first_divergence_localizer as issue578
from .aera_hardware_core_v26 import HardwareAwareAERATextLMV26

RESEARCH_ISSUE = 594
SOURCE_MAIN = "9547bb0b4c340d793acb1e12c655dc3d22513234"
SOURCE_TREE = "6cd794ae3d893ab8f9f29e3d342c71aa081da324"
ISSUE588_RESULT_PATH = "/vol/aera-v26/issue588-first-divergence-guard-repair1/result.json"
ISSUE588_RESULT_SHA256 = "495c6f49210074580553aa4b55bf0970624a8abaee910f6d2bf7315e26d2a540"
ISSUE590_TRIGGER = 590
ISSUE590_RUN = 33753926605
ISSUE590_JOB = 100643674944
ISSUE593_TRIGGER = 593
ISSUE593_RUN = 33764045085
ISSUE593_JOB = 100677235816
ISSUE592_TESTED_HEAD = "45c007b52811c0b62f5da254fd2ae862d6ed81aa"
ISSUE592_CI_RUN = 33763382230
ISSUE592_CI_JOB = 100674987954
ISSUE592_MERGE = SOURCE_MAIN

CHECKPOINT_HASHES = dict(issue578.CHECKPOINT_HASHES)
BATCH_SIZES = issue578.BATCH_SIZES
INTEGRATED_ATOL = issue578.INTEGRATED_ATOL
INTEGRATED_RTOL = issue578.INTEGRATED_RTOL
TARGET_CHUNK = 1
TARGET_STAGE = 0
MAX_GPU_SECONDS = 300


def localization_protocol() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "issue588_result_path": ISSUE588_RESULT_PATH,
        "issue588_result_sha256": ISSUE588_RESULT_SHA256,
        "issue590_trigger": ISSUE590_TRIGGER,
        "issue590_run": ISSUE590_RUN,
        "issue590_job": ISSUE590_JOB,
        "issue593_trigger": ISSUE593_TRIGGER,
        "issue593_run": ISSUE593_RUN,
        "issue593_job": ISSUE593_JOB,
        "checkpoint": base.CHECKPOINT_RELATIVE_DIR,
        "checkpoint_hashes": dict(CHECKPOINT_HASHES),
        "batch_sizes": list(BATCH_SIZES),
        "token_seed_rule": "138471 + 10000 + batch_size",
        "target_chunk": TARGET_CHUNK,
        "target_stage": TARGET_STAGE,
        "hard": True,
        "route_mode": "hard_sparse",
        "update_memory": True,
        "cuda_autocast": "bfloat16",
        "integrated_atol": INTEGRATED_ATOL,
        "integrated_rtol": INTEGRATED_RTOL,
        "instrumentation": "observational hooks/wrappers; original outputs returned unchanged; restored in finally",
        "production_module_reinvocation": False,
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


def cpu_contract_preflight_issue594() -> dict[str, Any]:
    parent = issue578.cpu_contract_preflight_issue578()
    if BATCH_SIZES != (8, 64):
        raise RuntimeError("issue594 batch surface drifted")
    if (TARGET_CHUNK, TARGET_STAGE) != (1, 0):
        raise RuntimeError("issue594 target boundary drifted")
    if (INTEGRATED_ATOL, INTEGRATED_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue594 integrated tolerance drifted")
    if parent["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue578 CPU preflight unexpectedly authorizes GPU")
    return {
        "protocol": localization_protocol(),
        "gpu_authorized_by_cpu_preflight": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "localization_measurement_performed": False,
        "scientific_seed_consumed": False,
        "repair_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return issue578._cpu(value)


def _controller_snapshot(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: _cpu(value)
        for name, value in output.items()
        if isinstance(value, torch.Tensor)
    }


def _comparison(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    *,
    decision_bearing: bool = False,
) -> dict[str, Any]:
    """Use the exact frozen #578 tolerance contract, adding discrete values only."""
    result = issue578._tensor_comparison(
        left,
        right,
        decision_bearing=decision_bearing,
    )
    if (
        left is not None
        and right is not None
        and result.get("metadata_exact")
        and left.numel() > 0
        and not (left.is_floating_point() and right.is_floating_point())
        and result.get("mismatch_count", 0)
    ):
        idx = int(result["first_mismatch_flat_index"])
        result["left_at_first_mismatch"] = left.reshape(-1)[idx].item()
        result["right_at_first_mismatch"] = right.reshape(-1)[idx].item()
    return result


def _presence_comparison(left_present: bool, right_present: bool, *, decision_bearing: bool = False) -> dict[str, Any]:
    return _comparison(
        torch.tensor([1], dtype=torch.int8) if left_present else None,
        torch.tensor([1], dtype=torch.int8) if right_present else None,
        decision_bearing=decision_bearing,
    )


@contextmanager
def _capture_stage0_post_read(
    model: HardwareAwareAERATextLMV26,
    *,
    label: str,
) -> Iterator[dict[str, Any]]:
    """Observe chunk1/stage0 without changing any production return value."""
    record: dict[str, Any] = {
        "label": label,
        "norm_outputs": [],
        "tokenwise_context_calls": [],
        "attn_calls": [],
        "expert_calls": [],
        "expert_run_selected_calls": [],
        "controller_calls": [],
        "controller_proj_calls": [],
        "route_end_controls": None,
    }
    context: dict[str, Any] = {
        "route_call": 0,
        "active": False,
        "controller_index": None,
    }
    restorations: list[tuple[object, str, object]] = []
    handles: list[Any] = []
    stage_count = len(model.stages)
    if TARGET_STAGE >= stage_count:
        raise RuntimeError("issue594 target stage unavailable")
    stage = model.stages[TARGET_STAGE]

    original_route = model._route_one_stage

    def wrapped_route(this, x, stage_module, stage_state, router, *, route_mode, update_memory):
        call = int(context["route_call"])
        context["route_call"] = call + 1
        chunk_index = call // stage_count
        stage_index = call % stage_count
        active = chunk_index == TARGET_CHUNK and stage_index == TARGET_STAGE
        previous_active = bool(context["active"])
        context["active"] = active
        if active:
            record["route_input"] = _cpu(x)
            context["controller_index"] = None
        try:
            y, new_state, meta = original_route(
                x,
                stage_module,
                stage_state,
                router,
                route_mode=route_mode,
                update_memory=update_memory,
            )
            if active:
                end_controls = meta.get("end") if isinstance(meta, dict) else None
                record["route_end_controls"] = (
                    _controller_snapshot(end_controls)
                    if isinstance(end_controls, dict)
                    else None
                )
                record["route_output"] = _cpu(y)
            return y, new_state, meta
        finally:
            context["active"] = previous_active
            if active:
                context["controller_index"] = None

    object.__setattr__(model, "_route_one_stage", MethodType(wrapped_route, model))
    restorations.append((model, "_route_one_stage", original_route))

    def norm_hook(module, args, output):
        if context["active"]:
            record["norm_outputs"].append(_cpu(output))

    handles.append(stage.norm.register_forward_hook(norm_hook))

    original_context = stage._tokenwise_context

    def wrapped_context(this, h, state, start_control):
        output = original_context(h, state, start_control)
        if context["active"]:
            produced_context, memory_read = output
            record["tokenwise_context_calls"].append(
                {
                    "h_input": _cpu(h),
                    "context": _cpu(produced_context),
                    "memory_read": _cpu(memory_read),
                }
            )
        return output

    object.__setattr__(stage, "_tokenwise_context", MethodType(wrapped_context, stage))
    restorations.append((stage, "_tokenwise_context", original_context))

    def attn_pre_hook(module, args):
        if context["active"]:
            record["attn_calls"].append({"input": _cpu(args[0]), "output": None})

    def attn_hook(module, args, output):
        if context["active"]:
            if not record["attn_calls"]:
                raise RuntimeError("issue594 attention hook order drifted")
            record["attn_calls"][-1]["output"] = _cpu(output)

    handles.append(stage.attn.register_forward_pre_hook(attn_pre_hook))
    handles.append(stage.attn.register_forward_hook(attn_hook))

    def experts_pre_hook(module, args, kwargs):
        if context["active"]:
            record["expert_calls"].append(
                {
                    "input": _cpu(args[0]),
                    "expert_logits": _cpu(args[1]),
                    "count_logits": _cpu(args[2]),
                    "hard": bool(kwargs.get("hard")),
                    "output": None,
                    "chosen_count": None,
                    "route_prob_mean": None,
                }
            )

    def experts_hook(module, args, kwargs, output):
        if context["active"]:
            if not record["expert_calls"]:
                raise RuntimeError("issue594 expert hook order drifted")
            call = record["expert_calls"][-1]
            call["output"] = _cpu(output)
            call["chosen_count"] = _cpu(getattr(module, "last_counts", None))
            call["route_prob_mean"] = _cpu(getattr(module, "last_route_probs", None))

    handles.append(stage.experts.register_forward_pre_hook(experts_pre_hook, with_kwargs=True))
    handles.append(stage.experts.register_forward_hook(experts_hook, with_kwargs=True))

    if not hasattr(stage.experts, "_run_selected"):
        raise TypeError("issue594 requires sparse expert backend with _run_selected")
    original_run_selected = stage.experts._run_selected

    def wrapped_run_selected(this, x, expert_ids):
        if context["active"]:
            record["expert_run_selected_calls"].append(
                {
                    "input": _cpu(x),
                    "expert_ids": _cpu(expert_ids),
                }
            )
        return original_run_selected(x, expert_ids)

    object.__setattr__(
        stage.experts,
        "_run_selected",
        MethodType(wrapped_run_selected, stage.experts),
    )
    restorations.append((stage.experts, "_run_selected", original_run_selected))

    def controller_pre_hook(module, args):
        if context["active"]:
            index = len(record["controller_calls"])
            context["controller_index"] = index
            record["controller_calls"].append(
                {
                    "role": "start" if index == 0 else "end" if index == 1 else f"extra_{index}",
                    "event": _cpu(args[0]),
                    "stream": _cpu(args[1]),
                    "output": None,
                }
            )

    def controller_hook(module, args, output):
        if context["active"]:
            index = context.get("controller_index")
            if index is None:
                raise RuntimeError("issue594 controller hook order drifted")
            record["controller_calls"][int(index)]["output"] = _controller_snapshot(output)
            context["controller_index"] = None

    handles.append(stage.controller.register_forward_pre_hook(controller_pre_hook))
    handles.append(stage.controller.register_forward_hook(controller_hook))

    def proj_pre_hook(module, args):
        if context["active"]:
            index = context.get("controller_index")
            role = "start" if index == 0 else "end" if index == 1 else f"extra_{index}"
            record["controller_proj_calls"].append(
                {
                    "controller_index": index,
                    "role": role,
                    "input": _cpu(args[0]),
                    "output": None,
                }
            )

    def proj_hook(module, args, output):
        if context["active"]:
            if not record["controller_proj_calls"]:
                raise RuntimeError("issue594 controller projection hook order drifted")
            record["controller_proj_calls"][-1]["output"] = _cpu(output)

    handles.append(stage.controller.proj.register_forward_pre_hook(proj_pre_hook))
    handles.append(stage.controller.proj.register_forward_hook(proj_hook))

    try:
        yield record
    finally:
        for handle in reversed(handles):
            handle.remove()
        for obj, name, original in reversed(restorations):
            object.__setattr__(obj, name, original)


def _one(items: list[Any], name: str) -> Any:
    if len(items) != 1:
        raise RuntimeError(f"issue594 expected exactly one {name}, got {len(items)}")
    return items[0]


def _controller_call(record: dict[str, Any], role: str) -> dict[str, Any] | None:
    return next((call for call in record["controller_calls"] if call.get("role") == role), None)


def _proj_call(record: dict[str, Any], role: str) -> dict[str, Any] | None:
    return next((call for call in record["controller_proj_calls"] if call.get("role") == role), None)


def _compare_records(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    boundaries: list[dict[str, Any]] = []

    def add(name: str, left: torch.Tensor | None, right: torch.Tensor | None, *, decision_bearing: bool = False) -> None:
        boundaries.append(
            {
                "name": name,
                "chunk": TARGET_CHUNK,
                "stage": TARGET_STAGE,
                "comparison": _comparison(left, right, decision_bearing=decision_bearing),
            }
        )

    lnorm = _one(reference["norm_outputs"], "reference norm output")
    rnorm = _one(candidate["norm_outputs"], "candidate norm output")
    add("chunk1.stage0.norm.output", lnorm, rnorm)

    lctx = _one(reference["tokenwise_context_calls"], "reference tokenwise context")
    rctx = _one(candidate["tokenwise_context_calls"], "candidate tokenwise context")
    add("chunk1.stage0.tokenwise_context.h_input", lctx["h_input"], rctx["h_input"])
    add("chunk1.stage0.tokenwise_context.context", lctx["context"], rctx["context"])
    add("chunk1.stage0.tokenwise_context.memory_read", lctx["memory_read"], rctx["memory_read"])

    lattn = _one(reference["attn_calls"], "reference attention call")
    rattn = _one(candidate["attn_calls"], "candidate attention call")
    add("chunk1.stage0.post_context.attn_input", lattn["input"], rattn["input"])
    add("chunk1.stage0.attn.output", lattn["output"], rattn["output"])

    lexpert = _one(reference["expert_calls"], "reference expert call")
    rexpert = _one(candidate["expert_calls"], "candidate expert call")
    add("chunk1.stage0.post_attention.experts_input", lexpert["input"], rexpert["input"])
    add("chunk1.stage0.experts.expert_logits", lexpert["expert_logits"], rexpert["expert_logits"])
    add("chunk1.stage0.experts.count_logits", lexpert["count_logits"], rexpert["count_logits"])
    add("chunk1.stage0.experts.chosen_count", lexpert["chosen_count"], rexpert["chosen_count"], decision_bearing=True)

    lselected = reference["expert_run_selected_calls"]
    rselected = candidate["expert_run_selected_calls"]
    boundaries.append(
        {
            "name": "chunk1.stage0.experts.run_selected_call_count",
            "chunk": TARGET_CHUNK,
            "stage": TARGET_STAGE,
            "comparison": _comparison(
                torch.tensor([len(lselected)], dtype=torch.int64),
                torch.tensor([len(rselected)], dtype=torch.int64),
                decision_bearing=True,
            ),
        }
    )
    for index in range(max(len(lselected), len(rselected))):
        left = lselected[index] if index < len(lselected) else None
        right = rselected[index] if index < len(rselected) else None
        boundaries.append(
            {
                "name": f"chunk1.stage0.experts.run_selected{index}.presence",
                "chunk": TARGET_CHUNK,
                "stage": TARGET_STAGE,
                "comparison": _presence_comparison(left is not None, right is not None, decision_bearing=True),
            }
        )
        add(
            f"chunk1.stage0.experts.run_selected{index}.expert_ids",
            left.get("expert_ids") if left else None,
            right.get("expert_ids") if right else None,
            decision_bearing=True,
        )
    add("chunk1.stage0.experts.output", lexpert["output"], rexpert["output"])

    lend = _controller_call(reference, "end")
    rend = _controller_call(candidate, "end")
    add("chunk1.stage0.end_summary", lend.get("event") if lend else None, rend.get("event") if rend else None)
    add("chunk1.stage0.end_controller.stream", lend.get("stream") if lend else None, rend.get("stream") if rend else None)

    lproj = _proj_call(reference, "end")
    rproj = _proj_call(candidate, "end")
    add("chunk1.stage0.end_controller.proj_input", lproj.get("input") if lproj else None, rproj.get("input") if rproj else None)
    add("chunk1.stage0.end_controller.raw", lproj.get("output") if lproj else None, rproj.get("output") if rproj else None)

    lout = lend.get("output", {}) if lend else {}
    rout = rend.get("output", {}) if rend else {}
    for field in sorted(set(lout) | set(rout)):
        add(
            f"chunk1.stage0.end_controller.{field}",
            lout.get(field),
            rout.get(field),
        )

    lroute_end = reference.get("route_end_controls") or {}
    rroute_end = candidate.get("route_end_controls") or {}
    for field in sorted(set(lroute_end) | set(rroute_end)):
        add(
            f"chunk1.stage0.route_returned_end.{field}",
            lroute_end.get(field),
            rroute_end.get(field),
        )

    first_bitwise = issue578._first_item(
        boundaries,
        lambda c: c.get("available") and not c.get("exact", True),
    )
    first_failure = issue578._first_item(
        boundaries,
        lambda c: bool(c.get("failure")),
    )
    first_discrete = issue578._first_item(
        boundaries,
        lambda c: bool(c.get("decision_bearing")) and bool(c.get("failure")),
    )
    failures = [row for row in boundaries if row["comparison"].get("failure")]
    return {
        "boundary_count": len(boundaries),
        "first_bitwise_difference": first_bitwise,
        "first_integrated_tolerance_or_metadata_failure": first_failure,
        "first_discrete_decision_difference": first_discrete,
        "failures": failures,
        "boundaries": boundaries,
        "unavailable_fields": [
            "expert second-call original batch indices are not exposed by production _run_selected",
        ],
    }


@torch.no_grad()
def run_stage0_post_read_amplification_localization(
    *,
    run_dir: str = base.CHECKPOINT_RELATIVE_DIR,
) -> dict[str, Any]:
    if torch.is_inference_mode_enabled():
        raise RuntimeError("issue594 must run under no_grad, not outer inference_mode")
    if not torch.cuda.is_available():
        raise RuntimeError("issue594 localization requires one separately authorized NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    hashes_before = base.checkpoint_hashes(run_dir)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError(f"issue594 checkpoint hash drift before localization: {hashes_before}")

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

        with _capture_stage0_post_read(reference, label="reference") as reference_record:
            reference_output = triage._model_forward(reference, tokens, update_memory=True)
        with _capture_stage0_post_read(candidate, label="candidate") as candidate_record:
            candidate_output = triage._model_forward(candidate, tokens, update_memory=True)

        comparison = _compare_records(reference_record, candidate_record)
        rows[str(batch_size)] = {"comparison": comparison}

        del reference_output, candidate_output, tokens
        torch.cuda.empty_cache()

    versions_unchanged = bool(
        reference_versions == tuple(int(parameter._version) for parameter in reference.parameters())
        and candidate_versions == tuple(int(parameter._version) for parameter in candidate.parameters())
    )
    hashes_after = base.checkpoint_hashes(run_dir)
    hashes_unchanged = hashes_before == hashes_after == CHECKPOINT_HASHES

    return {
        "research_issue": RESEARCH_ISSUE,
        "scope": "stage0_post_read_amplification_localization_after_issue593",
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
