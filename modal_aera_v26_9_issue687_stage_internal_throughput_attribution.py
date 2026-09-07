from __future__ import annotations

"""Issue #687 diagnostic-only stage-internal throughput attribution.

This launcher preserves the exact frozen v26.9 candidate and instruments existing
stage/module boundaries with same-stream CUDA events. It changes no model operation,
parameter, routing decision, FICEM equation, checkpoint, threshold, or scientific seed.
"""

from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import MethodType
from typing import Any, Callable, Iterator

import modal

import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665

APP_NAME = "aera-v26-9-issue687-stage-internal-throughput-attribution"
VOLUME_NAME = issue665.VOLUME_NAME
RESULT_PATH = "/vol/aera-v26/issue687-stage-internal-throughput-attribution/result.json"
SOURCE_ATTRIBUTION_PATH = issue665.RESULT_PATH
MAX_GPU_SECONDS = 300

RESEARCH_ISSUE = 687
SOURCE_MAIN = "bc74136533f27d9f39bf437f52975efd7a929fd4"
SOURCE_TREE = "c8835002261c1b249f90468085ad0c868afb2d90"
SOURCE_TRIGGER = 686
SOURCE_EVIDENCE_COMMENT = 5562153718
SOURCE_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"

ISSUE665_LAUNCHER_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"
SCIENTIFIC_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
RUNTIME_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STAGE_V25_1_BLOB = "1c3456d8040455b4cd1194db4c8586f77d0f3e43"
TOKENWISE_V19_BLOB = "98008bceb8c68af3bc346e5dfcc7a8218875661e"
NOHOST_V25_1_BLOB = "237e5615cf32f644e8675808a6b3e9adaf04fb23"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"

CHECKPOINT_HASHES = dict(issue665.CHECKPOINT_HASHES)
SYSTEM_BATCH_SIZES = tuple(issue665.SYSTEM_BATCH_SIZES)
TOKEN_SEED_BASE = issue665.TOKEN_SEED_BASE
TOKEN_SEED_OFFSET = issue665.TOKEN_SEED_OFFSET
DIAGNOSTIC_WARMUP_CALLS = issue665.DIAGNOSTIC_WARMUP_CALLS
DIAGNOSTIC_MEASURED_CALLS = issue665.DIAGNOSTIC_MEASURED_CALLS

EXPECTED_SOURCE_MEDIANS = {
    "8": {"full_call": 50.84833526611328, "stage_compute_excluding_ficem": 29.77371195331216},
    "64": {"full_call": 117.23713684082031, "stage_compute_excluding_ficem": 59.66491297632456},
}

PRECHECK_MARKER = "AERA_V26_9_ISSUE687_STAGE_INTERNAL_PRECHECK_JSON="
L4_START_MARKER = "AERA_V26_9_ISSUE687_STAGE_INTERNAL_L4_START_JSON="
RESULT_MARKER = "AERA_V26_9_ISSUE687_STAGE_INTERNAL_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_9_ISSUE687_STAGE_INTERNAL_SUMMARY_JSON="

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = issue665.image


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("empty timing sample")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"samples": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "p10_ms": 0.0, "p90_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "samples": float(len(values)),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p10_ms": float(_percentile(values, 0.10)),
        "p90_ms": float(_percentile(values, 0.90)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _verify_source_result() -> dict[str, Any]:
    source_path = Path(SOURCE_ATTRIBUTION_PATH)
    if not source_path.exists():
        raise RuntimeError("issue687 frozen #665 attribution result missing")
    payload = json.loads(source_path.read_text())
    if payload.get("research_issue") != 665:
        raise RuntimeError("issue687 source attribution research issue drift")
    authority = payload.get("source_systems_authority", {})
    if authority.get("decision") != SOURCE_DECISION:
        raise RuntimeError("issue687 source systems decision drift")
    if payload.get("optimization_authorized") is not False:
        raise RuntimeError("issue687 source optimization authority drift")
    rows = payload.get("rows")
    if not isinstance(rows, dict) or set(rows) != {"8", "64"}:
        raise RuntimeError("issue687 source attribution rows drift")
    for batch, expected in EXPECTED_SOURCE_MEDIANS.items():
        measurement = rows[batch].get("measurement", {})
        if measurement.get("dominant_measured_category") != "stage_compute_excluding_ficem":
            raise RuntimeError("issue687 source dominant category drift")
        got_full = measurement.get("full_call", {}).get("median_ms")
        got_stage = measurement.get("exclusive_categories", {}).get("stage_compute_excluding_ficem", {}).get("median_ms")
        if got_full != expected["full_call"] or got_stage != expected["stage_compute_excluding_ficem"]:
            raise RuntimeError(f"issue687 source medians drift for batch {batch}: full={got_full} stage={got_stage}")
    return payload


def _frozen_blob_evidence() -> dict[str, str]:
    import tam_research.aera_hardware_core_v19 as v19
    import tam_research.aera_hardware_core_v25_1 as v25_1
    import tam_research.aera_hardware_core_v25_1_nohost as nohost
    import tam_research.aera_hardware_core_v26 as runtime
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as backend
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems

    blobs = {
        "issue665_launcher": _git_blob_sha(Path(issue665.__file__)),
        "scientific_adapter": _git_blob_sha(Path(systems.__file__)),
        "runtime_interface": _git_blob_sha(Path(runtime.__file__)),
        "stage_v25_1": _git_blob_sha(Path(v25_1.__file__)),
        "tokenwise_v19": _git_blob_sha(Path(v19.__file__)),
        "nohost_v25_1": _git_blob_sha(Path(nohost.__file__)),
        "v26_9_backend": _git_blob_sha(Path(backend.__file__)),
    }
    expected = {
        "issue665_launcher": ISSUE665_LAUNCHER_BLOB,
        "scientific_adapter": SCIENTIFIC_ADAPTER_BLOB,
        "runtime_interface": RUNTIME_INTERFACE_BLOB,
        "stage_v25_1": STAGE_V25_1_BLOB,
        "tokenwise_v19": TOKENWISE_V19_BLOB,
        "nohost_v25_1": NOHOST_V25_1_BLOB,
        "v26_9_backend": V26_9_BACKEND_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue687 frozen blob drift: got={blobs} expected={expected}")
    return blobs


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight() -> dict[str, Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue687 diagnostic result already exists: {RESULT_PATH}")
    _verify_source_result()
    blobs = _frozen_blob_evidence()
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError("issue687 checkpoint hashes drifted")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_trigger": SOURCE_TRIGGER,
        "source_evidence_comment": SOURCE_EVIDENCE_COMMENT,
        "source_decision": SOURCE_DECISION,
        "frozen_blobs": blobs,
        "checkpoint_hashes": hashes,
        "result_absent": True,
        "gpu_used": False,
        "model_constructed": False,
        "new_measurement_performed": False,
        "optimization_authorized": False,
        "systems_pass_earned": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


class _Recorder:
    def __init__(self, torch_module, stage_names: dict[int, str]) -> None:
        self.torch = torch_module
        self.stage_names = stage_names
        self.enabled = False
        self.current_stage: str | None = None
        self.controller_counts: dict[str, int] = {}
        self.events: dict[str, list[tuple[Any, Any]]] = {}

    def reset(self) -> None:
        self.events = {}
        self.controller_counts = {}

    def record(self, label: str, call: Callable[[], Any]) -> Any:
        if not self.enabled:
            return call()
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return call()
        finally:
            end.record()
            self.events.setdefault(label, []).append((start, end))

    def elapsed(self) -> dict[str, list[float]]:
        return {label: [float(a.elapsed_time(b)) for a, b in pairs] for label, pairs in self.events.items()}


def _restore(obj: Any, name: str, had: bool, previous: Any) -> None:
    if had:
        object.__setattr__(obj, name, previous)
    elif name in getattr(obj, "__dict__", {}):
        object.__delattr__(obj, name)


@contextmanager
def _instrument(candidate, torch_module) -> Iterator[_Recorder]:
    import tam_research.aera_hardware_core_v25_1 as v25_1

    stage_names = {id(stage): ("foundation" if i == 0 else f"optional_{i}") for i, stage in enumerate(candidate.stages)}
    recorder = _Recorder(torch_module, stage_names)
    restored: list[Callable[[], None]] = []

    for i, stage in enumerate(candidate.stages):
        stage_name = "foundation" if i == 0 else f"optional_{i}"
        original_forward = stage.forward_chunk
        had_forward = "forward_chunk" in stage.__dict__
        previous_forward = stage.__dict__.get("forward_chunk")

        def wrapped_forward(this, events, state, *, hard, update_memory, _original=original_forward, _name=stage_name):
            previous_stage = recorder.current_stage
            recorder.current_stage = _name
            recorder.controller_counts[_name] = 0
            try:
                return recorder.record(f"stage_forward.{_name}", lambda: _original(events, state, hard=hard, update_memory=update_memory))
            finally:
                recorder.current_stage = previous_stage

        object.__setattr__(stage, "forward_chunk", MethodType(wrapped_forward, stage))
        restored.append(lambda obj=stage, h=had_forward, p=previous_forward: _restore(obj, "forward_chunk", h, p))

        original_context = stage._tokenwise_context
        had_context = "_tokenwise_context" in stage.__dict__
        previous_context = stage.__dict__.get("_tokenwise_context")

        def wrapped_context(this, h, state, start_control, _original=original_context, _name=stage_name):
            return recorder.record(f"context.{_name}", lambda: _original(h, state, start_control))

        object.__setattr__(stage, "_tokenwise_context", MethodType(wrapped_context, stage))
        restored.append(lambda obj=stage, h=had_context, p=previous_context: _restore(obj, "_tokenwise_context", h, p))

        module_specs = (
            ("norm", "norm"),
            ("state_to_chunk", "state_to_chunk"),
            ("attn", "attention"),
            ("experts", "experts"),
            ("reasoner", "reasoner"),
            ("reason_to_chunk", "reason_to_chunk"),
            ("out_norm", "out_norm"),
            ("stream_input_norm", "stream_input_norm"),
            ("stream_cell", "stream_cell"),
            ("pair_write_gate", "pair_write_gate"),
        )
        for attr_name, label in module_specs:
            module = getattr(stage, attr_name)
            original = module.forward
            had = "forward" in module.__dict__
            previous = module.__dict__.get("forward")

            def wrapped_module(this, *args, _original=original, _label=label, _name=stage_name, **kwargs):
                return recorder.record(f"{_label}.{_name}", lambda: _original(*args, **kwargs))

            object.__setattr__(module, "forward", MethodType(wrapped_module, module))
            restored.append(lambda obj=module, h=had, p=previous: _restore(obj, "forward", h, p))

        controller = stage.controller
        original_controller = controller.forward
        had_controller = "forward" in controller.__dict__
        previous_controller = controller.__dict__.get("forward")

        def wrapped_controller(this, *args, _original=original_controller, _name=stage_name, **kwargs):
            index = recorder.controller_counts.get(_name, 0)
            recorder.controller_counts[_name] = index + 1
            label = "controller_start" if index == 0 else "controller_end"
            return recorder.record(f"{label}.{_name}", lambda: _original(*args, **kwargs))

        object.__setattr__(controller, "forward", MethodType(wrapped_controller, controller))
        restored.append(lambda obj=controller, h=had_controller, p=previous_controller: _restore(obj, "forward", h, p))

        backend = stage.memory._execution_backend
        for method_name in ("read", "update", "update_from_projected"):
            original_backend = getattr(backend, method_name)
            had_backend = method_name in getattr(backend, "__dict__", {})
            previous_backend = getattr(backend, "__dict__", {}).get(method_name)

            def wrapped_backend(this, *args, _original=original_backend, _method=method_name, _name=stage_name, **kwargs):
                return recorder.record(f"ficem_{_method}.{_name}", lambda: _original(*args, **kwargs))

            object.__setattr__(backend, method_name, MethodType(wrapped_backend, backend))
            restored.append(lambda obj=backend, name=method_name, h=had_backend, p=previous_backend: _restore(obj, name, h, p))

    original_select = v25_1.select_budgeted_event_pairs

    def wrapped_select(*args, **kwargs):
        stage_name = recorder.current_stage or "unattributed"
        return recorder.record(f"event_pair_select.{stage_name}", lambda: original_select(*args, **kwargs))

    v25_1.select_budgeted_event_pairs = wrapped_select
    try:
        yield recorder
    finally:
        v25_1.select_budgeted_event_pairs = original_select
        for restore in reversed(restored):
            restore()


def _sum_exact(timings: dict[str, list[float]], label: str) -> float:
    return float(sum(timings.get(label, [])))


def _call_decomposition(*, full_call_ms: float, timings: dict[str, list[float]], stage_names: list[str]) -> dict[str, Any]:
    categories = {
        "normalization_and_start_controller": 0.0,
        "context_integration_outside_ficem": 0.0,
        "attention": 0.0,
        "sparse_experts": 0.0,
        "end_controller": 0.0,
        "latent_reasoner": 0.0,
        "reason_to_chunk_and_output_norm": 0.0,
        "recurrent_stream_update": 0.0,
        "write_gate_and_event_pair_selection": 0.0,
        "residual_stage_glue": 0.0,
    }
    per_stage: dict[str, Any] = {}
    total_stage_ex_ficem = 0.0

    for name in stage_names:
        stage_forward = _sum_exact(timings, f"stage_forward.{name}")
        ficem_read = _sum_exact(timings, f"ficem_read.{name}")
        ficem_update = _sum_exact(timings, f"ficem_update.{name}")
        ficem_projected = _sum_exact(timings, f"ficem_update_from_projected.{name}")
        ficem = ficem_read + ficem_update + ficem_projected
        stage_ex_ficem = max(stage_forward - ficem, 0.0)
        norm = _sum_exact(timings, f"norm.{name}")
        start_controller = _sum_exact(timings, f"controller_start.{name}")
        context_inclusive = _sum_exact(timings, f"context.{name}")
        state_to_chunk = _sum_exact(timings, f"state_to_chunk.{name}")
        context_ex_ficem = max(context_inclusive - ficem_read, 0.0)
        attention = _sum_exact(timings, f"attention.{name}")
        experts = _sum_exact(timings, f"experts.{name}")
        end_controller = _sum_exact(timings, f"controller_end.{name}")
        reasoner = _sum_exact(timings, f"reasoner.{name}")
        reason_to_chunk = _sum_exact(timings, f"reason_to_chunk.{name}")
        out_norm = _sum_exact(timings, f"out_norm.{name}")
        stream_input_norm = _sum_exact(timings, f"stream_input_norm.{name}")
        stream_cell = _sum_exact(timings, f"stream_cell.{name}")
        pair_gate = _sum_exact(timings, f"pair_write_gate.{name}")
        pair_select = _sum_exact(timings, f"event_pair_select.{name}")

        exclusive = {
            "normalization_and_start_controller": norm + start_controller,
            "context_integration_outside_ficem": context_ex_ficem,
            "attention": attention,
            "sparse_experts": experts,
            "end_controller": end_controller,
            "latent_reasoner": reasoner,
            "reason_to_chunk_and_output_norm": reason_to_chunk + out_norm,
            "recurrent_stream_update": stream_input_norm + stream_cell,
            "write_gate_and_event_pair_selection": pair_gate + pair_select,
        }
        accounted = sum(exclusive.values())
        residual = max(stage_ex_ficem - accounted, 0.0)
        exclusive["residual_stage_glue"] = residual
        for key, value in exclusive.items():
            categories[key] += value
        total_stage_ex_ficem += stage_ex_ficem

        per_stage[name] = {
            "stage_forward_inclusive_ms": stage_forward,
            "ficem_backend_inclusive_ms": ficem,
            "stage_compute_excluding_ficem_ms": stage_ex_ficem,
            "context_inclusive_ms": context_inclusive,
            "ficem_read_ms": ficem_read,
            "state_to_chunk_child_ms": state_to_chunk,
            "exclusive_stage_internal_ms": exclusive,
            "exclusive_sum_ms": sum(exclusive.values()),
            "exclusive_minus_stage_compute_ms": sum(exclusive.values()) - stage_ex_ficem,
        }

    return {
        "full_call_ms": full_call_ms,
        "stage_compute_excluding_ficem_ms": total_stage_ex_ficem,
        "exclusive_stage_internal_ms": categories,
        "exclusive_stage_internal_sum_ms": sum(categories.values()),
        "exclusive_minus_stage_compute_ms": sum(categories.values()) - total_stage_ex_ficem,
        "per_stage": per_stage,
        "timing_semantics": {
            "same_stream_cuda_events": True,
            "component_level_synchronization": False,
            "synchronize_only_after_complete_diagnostic_call": True,
            "context_inclusive_contains_ficem_read": True,
            "context_exclusive_derived_by_subtracting_ficem_read": True,
            "stage_compute_derived_by_subtracting_all_ficem_backend_time": True,
            "residual_is_parent_minus_disjoint_derived_children": True,
            "naive_sum_of_nested_inclusive_timings_forbidden": True,
        },
    }


def _aggregate(calls: list[dict[str, Any]], stage_names: list[str]) -> dict[str, Any]:
    category_names = tuple(calls[0]["exclusive_stage_internal_ms"])
    categories = {name: _summary([row["exclusive_stage_internal_ms"][name] for row in calls]) for name in category_names}
    stage_compute = _summary([row["stage_compute_excluding_ficem_ms"] for row in calls])
    full_call = _summary([row["full_call_ms"] for row in calls])
    per_stage = {
        stage: {
            "stage_compute_excluding_ficem": _summary([row["per_stage"][stage]["stage_compute_excluding_ficem_ms"] for row in calls]),
            "exclusive_categories": {
                cat: _summary([row["per_stage"][stage]["exclusive_stage_internal_ms"][cat] for row in calls]) for cat in category_names
            },
        }
        for stage in stage_names
    }
    medians = {name: stats["median_ms"] for name, stats in categories.items()}
    dominant = max(medians, key=medians.get)
    return {
        "full_call": full_call,
        "stage_compute_excluding_ficem": stage_compute,
        "exclusive_categories": categories,
        "per_stage": per_stage,
        "dominant_stage_internal_category": dominant,
        "dominant_stage_internal_median_ms": medians[dominant],
        "optimization_authorized": False,
    }


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=MAX_GPU_SECONDS, volumes={"/vol": volume})
def run_diagnostic() -> dict[str, Any]:
    import torch
    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    from tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import IdentityWeightVisibilityTritonFICEMReadWriteBackend

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue687 result already exists: {RESULT_PATH}")
    _verify_source_result()
    _frozen_blob_evidence()
    if not torch.cuda.is_available():
        raise RuntimeError("issue687 requires authorized NVIDIA L4")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    hashes_before = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError("issue687 checkpoint hashes drifted before model load")

    reference, candidate, transformer, backend_names = systems.load_models_v26_9(run_dir=base.CHECKPOINT_RELATIVE_DIR, device=device)
    expected_backend = IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if tuple(backend_names) != tuple(expected_backend for _ in candidate.stages):
        raise RuntimeError("issue687 candidate backend identity drift")
    del reference
    del transformer
    gc.collect()
    torch.cuda.empty_cache()

    stage_names = ["foundation" if i == 0 else f"optional_{i}" for i in range(len(candidate.stages))]
    rows: dict[str, Any] = {}

    with torch.inference_mode(), _instrument(candidate, torch) as recorder:
        for batch_size in SYSTEM_BATCH_SIZES:
            generator = torch.Generator(device="cpu").manual_seed(TOKEN_SEED_BASE + TOKEN_SEED_OFFSET + batch_size)
            tokens = torch.randint(0, triage.VOCAB_SIZE, (batch_size, triage.SEQ_LEN), generator=generator).to(device)
            recorder.enabled = False
            for _ in range(DIAGNOSTIC_WARMUP_CALLS):
                warmup = base._model_call(candidate, tokens, update_memory=True)
                del warmup

            measured: list[dict[str, Any]] = []
            for _ in range(DIAGNOSTIC_MEASURED_CALLS):
                recorder.reset()
                recorder.enabled = True
                outer_start = torch.cuda.Event(enable_timing=True)
                outer_end = torch.cuda.Event(enable_timing=True)
                outer_start.record()
                output = base._model_call(candidate, tokens, update_memory=True)
                outer_end.record()
                torch.cuda.synchronize()
                full_ms = float(outer_start.elapsed_time(outer_end))
                timings = recorder.elapsed()
                recorder.enabled = False
                measured.append(_call_decomposition(full_call_ms=full_ms, timings=timings, stage_names=stage_names))
                del output

            rows[str(batch_size)] = {
                "batch_size": batch_size,
                "token_seed": TOKEN_SEED_BASE + TOKEN_SEED_OFFSET + batch_size,
                "sequence_length": int(triage.SEQ_LEN),
                "route_mode": "hard_sparse",
                "hard": True,
                "update_memory": True,
                "candidate_backend_names": list(backend_names),
                "measurement": _aggregate(measured, stage_names),
                "raw_calls": measured,
            }
            del tokens
            torch.cuda.empty_cache()

    hashes_after = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_after != hashes_before:
        raise RuntimeError("issue687 checkpoint hashes changed")

    dominant_by_batch = {batch: row["measurement"]["dominant_stage_internal_category"] for batch, row in rows.items()}
    unique_dominants = set(dominant_by_batch.values())
    next_target = next(iter(unique_dominants)) if len(unique_dominants) == 1 else None

    result = {
        "scope": "aera_v26_9_issue687_stage_internal_throughput_attribution",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_trigger": SOURCE_TRIGGER,
        "source_evidence_comment": SOURCE_EVIDENCE_COMMENT,
        "source_decision": SOURCE_DECISION,
        "source_decision_changed": False,
        "device": torch.cuda.get_device_name(device),
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "checkpoint_hashes_unchanged": True,
        "candidate_backend_names": list(backend_names),
        "batches": list(SYSTEM_BATCH_SIZES),
        "token_seed_rule": "138471 + 10000 + batch_size",
        "diagnostic_warmup_calls": DIAGNOSTIC_WARMUP_CALLS,
        "diagnostic_measured_calls": DIAGNOSTIC_MEASURED_CALLS,
        "comparative_gate_rerun": False,
        "reference_model_executed": False,
        "transformer_model_executed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "rows": rows,
        "dominant_by_batch": dominant_by_batch,
        "next_target": next_target,
        "systems_pass_earned": False,
        "optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }

    result_path = Path(RESULT_PATH)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    volume.commit()

    summary = {
        "scope": result["scope"],
        "research_issue": RESEARCH_ISSUE,
        "device": result["device"],
        "source_decision": SOURCE_DECISION,
        "checkpoint_hashes_unchanged": True,
        "dominant_by_batch": dominant_by_batch,
        "next_target": next_target,
        "batches": {
            batch: {
                "full_call_median_ms": row["measurement"]["full_call"]["median_ms"],
                "stage_compute_excluding_ficem_median_ms": row["measurement"]["stage_compute_excluding_ficem"]["median_ms"],
                "dominant_stage_internal_category": row["measurement"]["dominant_stage_internal_category"],
                "dominant_stage_internal_median_ms": row["measurement"]["dominant_stage_internal_median_ms"],
                "exclusive_stage_internal_medians_ms": {name: stats["median_ms"] for name, stats in row["measurement"]["exclusive_categories"].items()},
            }
            for batch, row in rows.items()
        },
        "systems_pass_earned": False,
        "optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    print(RESULT_MARKER + json.dumps(summary, sort_keys=True))
    return summary


@app.local_entrypoint()
def main() -> None:
    pre = preflight.remote()
    print(PRECHECK_MARKER + json.dumps(pre, sort_keys=True))
    print(L4_START_MARKER + json.dumps({"research_issue": RESEARCH_ISSUE, "gpu": "L4", "max_gpu_seconds": MAX_GPU_SECONDS, "result_path": RESULT_PATH, "diagnostic_only": True}, sort_keys=True))
    summary = run_diagnostic.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
