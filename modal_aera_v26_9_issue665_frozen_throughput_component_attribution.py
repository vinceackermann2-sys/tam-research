from __future__ import annotations

"""Issue #665 diagnostic-only frozen-throughput component attribution.

This launcher does not rerun the frozen #653/#650 comparative systems gate and
cannot earn systems PASS.  It loads the exact frozen v26.9 candidate only inside
a separately-authorized L4 function, instruments existing execution boundaries
with same-stream CUDA events, and writes one fresh diagnostic-only result.
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

APP_NAME = "aera-v26-9-issue665-frozen-throughput-component-attribution"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue665-frozen-throughput-component-attribution/result.json"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue650-e2e-cli-guard-continuation/result.json"
MAX_GPU_SECONDS = 300

RESEARCH_ISSUE = 665
SOURCE_MAIN = "6a87dd8c3d3f9e73d34aa7a3c1e2ed991b53a002"
SOURCE_TREE = "3daf8e0fce277a65f9ae3daa56d8252dd15ff664"
SCIENTIFIC_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
RUNTIME_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"

SOURCE_TRIGGER = 653
SOURCE_RUN = 34022331841
SOURCE_JOB = 101457058965
SOURCE_ATTEMPT = 1
SOURCE_RESULT_SHA256 = "914615db5267565563dcc9e82bfc31f444a656a68bd560f50447a8fd03588431"
SOURCE_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"

READONLY_TRIGGER = 664
READONLY_RUN = 34039744132
READONLY_JOB = 101504212730
READONLY_ATTEMPT = 1
READONLY_EVIDENCE_COMMENT = 5559943198

CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}

SYSTEM_BATCH_SIZES = (8, 64)
TOKEN_SEED_BASE = 138471
TOKEN_SEED_OFFSET = 10000
DIAGNOSTIC_WARMUP_CALLS = 2
DIAGNOSTIC_MEASURED_CALLS = 12

PRECHECK_MARKER = "AERA_V26_9_ISSUE665_COMPONENT_ATTRIBUTION_PRECHECK_JSON="
L4_START_MARKER = "AERA_V26_9_ISSUE665_COMPONENT_ATTRIBUTION_L4_START_JSON="
RESULT_MARKER = "AERA_V26_9_ISSUE665_COMPONENT_ATTRIBUTION_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_9_ISSUE665_COMPONENT_ATTRIBUTION_SUMMARY_JSON="

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3", "triton>=3.6,<3.7")
    .add_local_python_source("tam_research")
)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty timing sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "samples": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p10_ms": 0.0,
            "p90_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "samples": float(len(values)),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p10_ms": float(_percentile(values, 0.10)),
        "p90_ms": float(_percentile(values, 0.90)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight() -> dict[str, Any]:
    import tam_research.aera_hardware_core_v26 as runtime
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as v26_9
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"issue665 diagnostic result already exists: {RESULT_PATH}")

    source_result = Path(SOURCE_RESULT_PATH)
    if not source_result.exists():
        raise RuntimeError("issue665 immutable #650 source result is missing")
    if _sha256_file(source_result) != SOURCE_RESULT_SHA256:
        raise RuntimeError("issue665 immutable #650 source SHA256 drifted")
    payload = json.loads(source_result.read_text())
    if payload.get("decision") != SOURCE_DECISION or payload.get("overall_pass") is not False:
        raise RuntimeError("issue665 immutable #650 decision drifted")

    blobs = {
        "scientific_adapter": _git_blob_sha(Path(systems.__file__)),
        "runtime_interface": _git_blob_sha(Path(runtime.__file__)),
        "base_systems": _git_blob_sha(Path(base.__file__)),
        "v26_9_backend": _git_blob_sha(Path(v26_9.__file__)),
    }
    expected = {
        "scientific_adapter": SCIENTIFIC_ADAPTER_BLOB,
        "runtime_interface": RUNTIME_INTERFACE_BLOB,
        "base_systems": BASE_SYSTEMS_BLOB,
        "v26_9_backend": V26_9_BACKEND_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue665 frozen blob drift: got={blobs} expected={expected}")

    contract = systems.cpu_contract_preflight_issue643()
    for key in (
        "gpu_authorized_by_cpu_preflight",
        "model_construction_performed",
        "checkpoint_loaded",
        "systems_measurement_performed",
        "scientific_seed_consumed",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"issue665 inherited #643 CPU contract drifted: {key}")

    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError(
            f"issue665 checkpoint hash drift: got={hashes} expected={CHECKPOINT_HASHES}"
        )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_trigger_run_job_attempt": [
            SOURCE_TRIGGER,
            SOURCE_RUN,
            SOURCE_JOB,
            SOURCE_ATTEMPT,
        ],
        "readonly_trigger_run_job_attempt": [
            READONLY_TRIGGER,
            READONLY_RUN,
            READONLY_JOB,
            READONLY_ATTEMPT,
        ],
        "readonly_evidence_comment": READONLY_EVIDENCE_COMMENT,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_decision": SOURCE_DECISION,
        "frozen_blobs": blobs,
        "checkpoint_hashes": hashes,
        "result_absent": True,
        "gpu_used": False,
        "model_constructed": False,
        "new_measurement_performed": False,
        "systems_pass_earned": False,
        "optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


class _CudaEventRecorder:
    """Record nested inclusive GPU intervals without synchronizing in wrappers."""

    def __init__(self, torch_module, stage_names: dict[int, str]) -> None:
        self.torch = torch_module
        self.stage_names = stage_names
        self.enabled = False
        self.current_stage: str | None = None
        self.events: dict[str, list[tuple[Any, Any]]] = {}

    def reset(self) -> None:
        self.events = {}

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

    def elapsed_by_label(self) -> dict[str, list[float]]:
        return {
            label: [float(start.elapsed_time(end)) for start, end in pairs]
            for label, pairs in self.events.items()
        }


def _restore_instance_method(obj: Any, name: str, had_instance: bool, previous: Any) -> None:
    if had_instance:
        object.__setattr__(obj, name, previous)
    elif name in getattr(obj, "__dict__", {}):
        object.__delattr__(obj, name)


@contextmanager
def _instrument_candidate(candidate, torch_module) -> Iterator[_CudaEventRecorder]:
    """Wrap existing runtime boundaries only and restore every wrapper in finally."""

    import tam_research.aera_hardware_core_v26 as runtime

    stage_names = {
        id(stage): ("foundation" if index == 0 else f"optional_{index}")
        for index, stage in enumerate(candidate.stages)
    }
    recorder = _CudaEventRecorder(torch_module, stage_names)
    restored: list[Callable[[], None]] = []

    original_route = candidate._route_one_stage
    had_route = "_route_one_stage" in candidate.__dict__
    previous_route = candidate.__dict__.get("_route_one_stage")

    def wrapped_route(
        this,
        x,
        stage,
        stage_state,
        router,
        *,
        route_mode,
        update_memory,
        _original=original_route,
    ):
        stage_name = stage_names[id(stage)]
        previous_stage = recorder.current_stage
        recorder.current_stage = stage_name
        try:
            return recorder.record(
                f"route.{stage_name}",
                lambda: _original(
                    x,
                    stage,
                    stage_state,
                    router,
                    route_mode=route_mode,
                    update_memory=update_memory,
                ),
            )
        finally:
            recorder.current_stage = previous_stage

    object.__setattr__(candidate, "_route_one_stage", MethodType(wrapped_route, candidate))
    restored.append(
        lambda: _restore_instance_method(
            candidate, "_route_one_stage", had_route, previous_route
        )
    )

    for index, router in enumerate(candidate.stage_routers):
        stage_name = "foundation" if index == 0 else f"optional_{index}"
        original = router.forward
        had = "forward" in router.__dict__
        previous = router.__dict__.get("forward")

        def wrapped_router(this, first_event, stream, *, mode, _original=original, _name=stage_name):
            return recorder.record(
                f"router.{_name}",
                lambda: _original(first_event, stream, mode=mode),
            )

        object.__setattr__(router, "forward", MethodType(wrapped_router, router))
        restored.append(
            lambda obj=router, h=had, p=previous: _restore_instance_method(
                obj, "forward", h, p
            )
        )

    for index, stage in enumerate(candidate.stages):
        stage_name = "foundation" if index == 0 else f"optional_{index}"
        original = stage.forward_chunk
        had = "forward_chunk" in stage.__dict__
        previous = stage.__dict__.get("forward_chunk")

        def wrapped_stage(
            this,
            events,
            state,
            *,
            hard,
            update_memory,
            _original=original,
            _name=stage_name,
        ):
            return recorder.record(
                f"stage_forward.{_name}",
                lambda: _original(
                    events,
                    state,
                    hard=hard,
                    update_memory=update_memory,
                ),
            )

        object.__setattr__(stage, "forward_chunk", MethodType(wrapped_stage, stage))
        restored.append(
            lambda obj=stage, h=had, p=previous: _restore_instance_method(
                obj, "forward_chunk", h, p
            )
        )

        backend = stage.memory._execution_backend
        for method_name in ("read", "update", "update_from_projected"):
            original_backend = getattr(backend, method_name)
            had_backend = method_name in getattr(backend, "__dict__", {})
            previous_backend = getattr(backend, "__dict__", {}).get(method_name)

            def wrapped_backend(
                this,
                *args,
                _original=original_backend,
                _method=method_name,
                _name=stage_name,
                **kwargs,
            ):
                return recorder.record(
                    f"ficem_{_method}.{_name}",
                    lambda: _original(*args, **kwargs),
                )

            object.__setattr__(
                backend,
                method_name,
                MethodType(wrapped_backend, backend),
            )
            restored.append(
                lambda obj=backend, name=method_name, h=had_backend, p=previous_backend:
                    _restore_instance_method(obj, name, h, p)
            )

    original_pack = runtime.pack_ephemeral_epi_state
    original_select = runtime.select_packed_epi_state
    original_merge = runtime.merge_packed_epi_state

    def wrapped_pack(state):
        stage_name = recorder.current_stage or "unattributed"
        return recorder.record(
            f"state_pack.{stage_name}",
            lambda: original_pack(state),
        )

    def wrapped_select(packed, idx):
        stage_name = recorder.current_stage or "unattributed"
        return recorder.record(
            f"state_select.{stage_name}",
            lambda: original_select(packed, idx),
        )

    def wrapped_merge(base, update, idx):
        stage_name = recorder.current_stage or "unattributed"
        return recorder.record(
            f"state_merge.{stage_name}",
            lambda: original_merge(base, update, idx),
        )

    runtime.pack_ephemeral_epi_state = wrapped_pack
    runtime.select_packed_epi_state = wrapped_select
    runtime.merge_packed_epi_state = wrapped_merge

    try:
        yield recorder
    finally:
        runtime.pack_ephemeral_epi_state = original_pack
        runtime.select_packed_epi_state = original_select
        runtime.merge_packed_epi_state = original_merge
        for restore in reversed(restored):
            restore()


def _sum_prefix(timings: dict[str, list[float]], prefix: str) -> float:
    return float(
        sum(sum(values) for label, values in timings.items() if label.startswith(prefix))
    )


def _sum_exact(timings: dict[str, list[float]], label: str) -> float:
    return float(sum(timings.get(label, [])))


def _route_fraction_rows(output: dict[str, object]) -> list[dict[str, Any]]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list):
        raise RuntimeError("issue665 output missing stage_routes")
    rows: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(routes):
        if not isinstance(chunk, list):
            raise RuntimeError("issue665 malformed stage route chunk")
        for stage_index, row in enumerate(chunk):
            if not isinstance(row, dict):
                raise RuntimeError("issue665 malformed stage route row")
            fraction = row.get("executed_fraction")
            if not isinstance(fraction, (float, int)):
                raise RuntimeError("issue665 route row missing executed_fraction")
            rows.append(
                {
                    "chunk": chunk_index,
                    "stage_index": stage_index,
                    "stage": "foundation" if stage_index == 0 else f"optional_{stage_index}",
                    "executed_fraction": float(fraction),
                }
            )
    return rows


def _call_decomposition(
    *,
    full_call_ms: float,
    timings: dict[str, list[float]],
    stage_names: list[str],
) -> dict[str, Any]:
    per_stage: dict[str, Any] = {}
    router_total = 0.0
    movement_total = 0.0
    ficem_total = 0.0
    stage_compute_excluding_ficem_total = 0.0
    route_glue_total = 0.0
    route_total = 0.0

    for stage_name in stage_names:
        route_ms = _sum_exact(timings, f"route.{stage_name}")
        router_ms = _sum_exact(timings, f"router.{stage_name}")
        pack_ms = _sum_exact(timings, f"state_pack.{stage_name}")
        select_ms = _sum_exact(timings, f"state_select.{stage_name}")
        merge_ms = _sum_exact(timings, f"state_merge.{stage_name}")
        movement_ms = pack_ms + select_ms + merge_ms
        stage_forward_ms = _sum_exact(timings, f"stage_forward.{stage_name}")
        ficem_read_ms = _sum_exact(timings, f"ficem_read.{stage_name}")
        ficem_update_ms = _sum_exact(timings, f"ficem_update.{stage_name}")
        ficem_projected_ms = _sum_exact(
            timings, f"ficem_update_from_projected.{stage_name}"
        )
        ficem_ms = ficem_read_ms + ficem_update_ms + ficem_projected_ms
        stage_compute_excluding_ficem_ms = max(stage_forward_ms - ficem_ms, 0.0)
        route_glue_ms = max(
            route_ms - router_ms - movement_ms - stage_forward_ms,
            0.0,
        )
        per_stage[stage_name] = {
            "route_inclusive_ms": route_ms,
            "router_inclusive_ms": router_ms,
            "state_movement_ms": movement_ms,
            "state_pack_ms": pack_ms,
            "state_select_ms": select_ms,
            "state_merge_ms": merge_ms,
            "stage_forward_inclusive_ms": stage_forward_ms,
            "ficem_backend_inclusive_ms": ficem_ms,
            "ficem_read_ms": ficem_read_ms,
            "ficem_update_ms": ficem_update_ms,
            "ficem_projected_update_ms": ficem_projected_ms,
            "derived_stage_compute_excluding_ficem_ms": stage_compute_excluding_ficem_ms,
            "derived_route_glue_excluding_router_state_stage_ms": route_glue_ms,
        }
        route_total += route_ms
        router_total += router_ms
        movement_total += movement_ms
        ficem_total += ficem_ms
        stage_compute_excluding_ficem_total += stage_compute_excluding_ficem_ms
        route_glue_total += route_glue_ms

    outside_routes_ms = max(full_call_ms - route_total, 0.0)
    decomposition_sum = (
        router_total
        + movement_total
        + ficem_total
        + stage_compute_excluding_ficem_total
        + route_glue_total
        + outside_routes_ms
    )
    return {
        "full_call_ms": full_call_ms,
        "per_stage": per_stage,
        "exclusive_decomposition_ms": {
            "router": router_total,
            "optional_state_pack_select_merge": movement_total,
            "ficem_backend": ficem_total,
            "stage_compute_excluding_ficem": stage_compute_excluding_ficem_total,
            "route_glue_excluding_router_state_stage": route_glue_total,
            "model_glue_outside_stage_routes": outside_routes_ms,
        },
        "exclusive_decomposition_sum_ms": decomposition_sum,
        "decomposition_minus_full_call_ms": decomposition_sum - full_call_ms,
        "timing_semantics": {
            "route_stage_ficem_measurements_are_nested_inclusive": True,
            "exclusive_fields_are_derived_by_parent_minus_immediate_children": True,
            "same_stream_cuda_events": True,
            "component_level_synchronization": False,
            "synchronize_only_after_complete_diagnostic_call": True,
            "naive_sum_of_nested_inclusive_timings_forbidden": True,
        },
    }


def _aggregate_calls(calls: list[dict[str, Any]], stage_names: list[str]) -> dict[str, Any]:
    full = [row["full_call_ms"] for row in calls]
    category_names = (
        "router",
        "optional_state_pack_select_merge",
        "ficem_backend",
        "stage_compute_excluding_ficem",
        "route_glue_excluding_router_state_stage",
        "model_glue_outside_stage_routes",
    )
    categories = {
        name: _summary(
            [row["exclusive_decomposition_ms"][name] for row in calls]
        )
        for name in category_names
    }
    stages: dict[str, Any] = {}
    for stage_name in stage_names:
        stages[stage_name] = {
            key: _summary([row["per_stage"][stage_name][key] for row in calls])
            for key in (
                "route_inclusive_ms",
                "router_inclusive_ms",
                "state_movement_ms",
                "stage_forward_inclusive_ms",
                "ficem_backend_inclusive_ms",
                "derived_stage_compute_excluding_ficem_ms",
                "derived_route_glue_excluding_router_state_stage_ms",
            )
        }

    medians = {name: categories[name]["median_ms"] for name in category_names}
    dominant = max(medians, key=medians.get)
    mapping = {
        "router": "routing_control_overhead",
        "optional_state_pack_select_merge": "sparse_state_tensor_movement",
        "ficem_backend": "ficem_memory_work",
        "stage_compute_excluding_ficem": "stage_compute",
        "route_glue_excluding_router_state_stage": "sparse_route_glue_tensor_movement",
        "model_glue_outside_stage_routes": "model_level_nonroute_compute",
    }
    return {
        "full_call": _summary(full),
        "exclusive_categories": categories,
        "per_stage": stages,
        "dominant_measured_category": dominant,
        "diagnostic_next_target_label": mapping[dominant],
        "optimization_authorized": False,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_diagnostic() -> dict[str, Any]:
    import torch

    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    from tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import (
        IdentityWeightVisibilityTritonFICEMReadWriteBackend,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"issue665 diagnostic result already exists: {RESULT_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("issue665 component diagnostic requires the authorized NVIDIA L4")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    hashes_before = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError("issue665 checkpoint hashes drifted before model construction")

    reference, candidate, transformer, candidate_backend_names = systems.load_models_v26_9(
        run_dir=base.CHECKPOINT_RELATIVE_DIR,
        device=device,
    )
    expected_backend = IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if tuple(candidate_backend_names) != tuple(
        expected_backend for _ in candidate.stages
    ):
        raise RuntimeError(
            f"issue665 candidate backend identity drift: {candidate_backend_names}"
        )

    # Reference and Transformer are constructed only to preserve exact #643 loader
    # identity verification.  The diagnostic never executes either model.
    del reference
    del transformer
    gc.collect()
    torch.cuda.empty_cache()

    stage_names = [
        "foundation" if index == 0 else f"optional_{index}"
        for index in range(len(candidate.stages))
    ]
    rows: dict[str, Any] = {}

    with torch.inference_mode(), _instrument_candidate(candidate, torch) as recorder:
        for batch_size in SYSTEM_BATCH_SIZES:
            generator = torch.Generator(device="cpu").manual_seed(
                TOKEN_SEED_BASE + TOKEN_SEED_OFFSET + batch_size
            )
            tokens = torch.randint(
                0,
                triage.VOCAB_SIZE,
                (batch_size, triage.SEQ_LEN),
                generator=generator,
            ).to(device)

            recorder.enabled = False
            for _ in range(DIAGNOSTIC_WARMUP_CALLS):
                warmup = base._model_call(candidate, tokens, update_memory=True)
                del warmup

            measured: list[dict[str, Any]] = []
            route_fraction_samples: list[list[dict[str, Any]]] = []
            routing_accounting_samples: list[dict[str, Any]] = []

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
                timings = recorder.elapsed_by_label()
                recorder.enabled = False

                measured.append(
                    _call_decomposition(
                        full_call_ms=full_ms,
                        timings=timings,
                        stage_names=stage_names,
                    )
                )
                route_fraction_samples.append(_route_fraction_rows(output))
                routing_accounting_samples.append(
                    triage._routing_accounting(output, batch_size)
                )
                del output

            first_routes = route_fraction_samples[0]
            if any(sample != first_routes for sample in route_fraction_samples[1:]):
                raise RuntimeError("issue665 deterministic route fractions drifted across samples")
            first_accounting = routing_accounting_samples[0]
            if any(sample != first_accounting for sample in routing_accounting_samples[1:]):
                raise RuntimeError("issue665 routing accounting drifted across samples")

            rows[str(batch_size)] = {
                "batch_size": batch_size,
                "token_seed": TOKEN_SEED_BASE + TOKEN_SEED_OFFSET + batch_size,
                "sequence_length": int(triage.SEQ_LEN),
                "route_mode": "hard_sparse",
                "hard": True,
                "update_memory": True,
                "candidate_backend_names": list(candidate_backend_names),
                "route_fractions": first_routes,
                "routing_accounting": first_accounting,
                "measurement": _aggregate_calls(measured, stage_names),
                "raw_calls": measured,
            }
            del tokens
            torch.cuda.empty_cache()

    hashes_after = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_after != CHECKPOINT_HASHES or hashes_after != hashes_before:
        raise RuntimeError("issue665 checkpoint hashes changed during diagnostic")

    result = {
        "scope": "aera_v26_9_issue665_frozen_throughput_component_attribution",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_systems_authority": {
            "trigger": SOURCE_TRIGGER,
            "run": SOURCE_RUN,
            "job": SOURCE_JOB,
            "attempt": SOURCE_ATTEMPT,
            "result_path": SOURCE_RESULT_PATH,
            "result_sha256": SOURCE_RESULT_SHA256,
            "decision": SOURCE_DECISION,
            "decision_changed": False,
        },
        "readonly_authority": {
            "trigger": READONLY_TRIGGER,
            "run": READONLY_RUN,
            "job": READONLY_JOB,
            "attempt": READONLY_ATTEMPT,
            "evidence_comment": READONLY_EVIDENCE_COMMENT,
        },
        "device": torch.cuda.get_device_name(device),
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "checkpoint_hashes_unchanged": True,
        "candidate_backend_names": list(candidate_backend_names),
        "batches": list(SYSTEM_BATCH_SIZES),
        "token_seed_rule": "138471 + 10000 + batch_size",
        "diagnostic_warmup_calls": DIAGNOSTIC_WARMUP_CALLS,
        "diagnostic_measured_calls": DIAGNOSTIC_MEASURED_CALLS,
        "frozen_gate_timing_protocol_reused": False,
        "comparative_gate_rerun": False,
        "reference_model_executed": False,
        "transformer_model_executed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_written": False,
        "scientific_seed_consumed": False,
        "rows": rows,
        "systems_pass_earned": False,
        "optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    volume.commit()

    summary = {
        "scope": result["scope"],
        "research_issue": RESEARCH_ISSUE,
        "device": result["device"],
        "source_decision": SOURCE_DECISION,
        "source_decision_changed": False,
        "candidate_backend_names": result["candidate_backend_names"],
        "checkpoint_hashes_unchanged": True,
        "batches": {
            batch: {
                "full_call_median_ms": row["measurement"]["full_call"]["median_ms"],
                "dominant_measured_category": row["measurement"][
                    "dominant_measured_category"
                ],
                "diagnostic_next_target_label": row["measurement"][
                    "diagnostic_next_target_label"
                ],
                "exclusive_category_medians_ms": {
                    name: stats["median_ms"]
                    for name, stats in row["measurement"][
                        "exclusive_categories"
                    ].items()
                },
                "per_stage_route_medians_ms": {
                    name: stats["route_inclusive_ms"]["median_ms"]
                    for name, stats in row["measurement"]["per_stage"].items()
                },
                "routing_accounting": row["routing_accounting"],
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
    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "gpu": "L4",
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "result_path": RESULT_PATH,
                "diagnostic_only": True,
            },
            sort_keys=True,
        )
    )
    summary = run_diagnostic.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
