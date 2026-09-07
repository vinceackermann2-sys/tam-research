from __future__ import annotations

"""Issue #706 zero-GPU preflight + sole-L4 v26.10 optimization microbenchmark."""

import gc
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import modal
import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665

APP_NAME = "aera-v26-10-issue706-latent-depth-sync-coalescing"
VOLUME_NAME = issue665.VOLUME_NAME
RESULT_PATH = "/vol/aera-v26/issue706-v26-10-latent-depth-sync-coalescing/result.json"
PARENT_RESULT_PATH = "/vol/aera-v26/issue702-cuda-timeline-dispatch-gap-attribution/result.json"
SOURCE_MAIN = "4883abe053347379812901c5c2057da70c11e3e7"
SOURCE_TREE = "27be25ba578aa530c1e6d832db1a38be4c8bc22a"
RESEARCH_ISSUE = 706
PARENT_ISSUE = 702
PARENT_TRIGGER = 705
PARENT_EVIDENCE_COMMENT = 5572472746
PARENT_RUN = 34135719600
PARENT_JOB = 101786085046
PARENT_ATTEMPT = 1
SOURCE_DECISION = "FAIL_FROZEN_E2E_SYSTEMS_GATE"
PARENT_NEXT_TARGET = "launch_or_idle_reduction"
ISSUE665_LAUNCHER = "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
EXPECTED_BLOBS = {
    "issue665_launcher": "72f27391ff2f0a7bff8d4532f307ddc4869cf494",
    "v26_10_impl": "d8f691c198eed1fa96bcbb78a4e76cad82d18779",
    "v25_1_nohost": "237e5615cf32f644e8675808a6b3e9adaf04fb23",
    "v26_runtime": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    "v26_9_systems": "512572340cc09e2e7ad6729712258c12cb377ef2",
    "base_systems": "c9731cae7e386f09b2a190b045532591c4fa00be",
    "triage": "e5ce8cdda0777dce97816e2640f4492803a6b191",
    "v26_9_backend": "b81cc209f5d95abbe1fb8bd620c78e87c067bc19",
}
CHECKPOINT_HASHES = {
    "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
    "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
}
BATCHES = (8, 64)
TOKEN_SEED_BASE = 138471 + 10000
WARMUP_CALLS = 3
TIMED_CALLS_PER_CONDITION = 20
PROFILE_CALLS_PER_CONDITION = 1
MIN_LATENCY_IMPROVEMENT = 0.05
MIN_STREAM_SYNCHRONIZE_REDUCTION = 20
INTEGRATED_ATOL = 1e-2
INTEGRATED_RTOL = 1e-2
MAX_GPU_SECONDS = 420
PREAUTH_MARKER = "AERA_V26_10_ISSUE706_PREAUTH_JSON="
L4_START_MARKER = "AERA_V26_10_ISSUE706_L4_START_JSON="
RESULT_MARKER = "AERA_V26_10_ISSUE706_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_10_ISSUE706_SUMMARY_JSON="

image = issue665.image.add_local_file(
    ISSUE665_LAUNCHER, f"/root/{ISSUE665_LAUNCHER}"
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _frozen_blobs() -> dict[str, str]:
    import tam_research.aera_hardware_core_v25_1_nohost as nohost
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_10_latent_depth_sync_coalescing as v2610
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as backend
    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems

    got = {
        "issue665_launcher": _blob(Path(f"/root/{ISSUE665_LAUNCHER}")),
        "v26_10_impl": _blob(Path(v2610.__file__)),
        "v25_1_nohost": _blob(Path(nohost.__file__)),
        "v26_runtime": _blob(Path(v26.__file__)),
        "v26_9_systems": _blob(Path(systems.__file__)),
        "base_systems": _blob(Path(base.__file__)),
        "triage": _blob(Path(triage.__file__)),
        "v26_9_backend": _blob(Path(backend.__file__)),
    }
    if got != EXPECTED_BLOBS:
        raise RuntimeError(f"issue706 frozen blob drift: {got}")
    return got


def _parent_result() -> dict[str, Any]:
    path = Path(PARENT_RESULT_PATH)
    if not path.exists():
        raise RuntimeError("issue706 parent #702 result missing")
    parent = json.loads(path.read_text())
    if parent.get("research_issue") != PARENT_ISSUE:
        raise RuntimeError("issue706 parent research issue drift")
    if parent.get("source_decision") != SOURCE_DECISION:
        raise RuntimeError("issue706 parent systems decision drift")
    if parent.get("next_target") != PARENT_NEXT_TARGET:
        raise RuntimeError("issue706 parent target drift")
    for key in (
        "optimization_authorized",
        "systems_pass_earned",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        if parent.get(key) is not False:
            raise RuntimeError(f"issue706 parent authority drift: {key}")
    rows = parent.get("rows")
    if not isinstance(rows, dict) or set(rows) != {"8", "64"}:
        raise RuntimeError("issue706 parent row drift")
    expected_sync = {"8": 49, "64": 51}
    for batch, expected in expected_sync.items():
        try:
            names = rows[batch]["trace"]["cuda_runtime"]["families"][
                "synchronize_or_wait"
            ]["names"]
            actual = int(names["cudaStreamSynchronize"])
        except Exception as exc:
            raise RuntimeError(f"issue706 parent sync evidence missing: {batch}") from exc
        if actual != expected:
            raise RuntimeError(
                f"issue706 parent cudaStreamSynchronize drift {batch}: {actual}"
            )
    return parent


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight() -> dict[str, Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base
    from tam_research.aera_hardware_core_v26_10_latent_depth_sync_coalescing import (
        latent_depth_sync_coalescing_v26_10_protocol,
    )

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue706 result already exists: {RESULT_PATH}")
    _parent_result()
    blobs = _frozen_blobs()
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError("issue706 checkpoint drift")
    protocol = latent_depth_sync_coalescing_v26_10_protocol()
    if protocol["research_issue"] != RESEARCH_ISSUE:
        raise RuntimeError("issue706 v26.10 protocol drift")
    if protocol["state_dict_schema_changed"] is not False:
        raise RuntimeError("issue706 v26.10 schema authority drift")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "parent_issue": PARENT_ISSUE,
        "parent_trigger": PARENT_TRIGGER,
        "parent_evidence_comment": PARENT_EVIDENCE_COMMENT,
        "parent_run": PARENT_RUN,
        "parent_job": PARENT_JOB,
        "parent_attempt": PARENT_ATTEMPT,
        "source_decision": SOURCE_DECISION,
        "parent_next_target": PARENT_NEXT_TARGET,
        "result_path": RESULT_PATH,
        "parent_result_path": PARENT_RESULT_PATH,
        "frozen_blobs": blobs,
        "checkpoint_hashes": hashes,
        "result_absent": True,
        "gpu_used": False,
        "model_constructed": False,
        "new_measurement_performed": False,
        "optimization_microbenchmark_authorized": False,
        "full_e2e_systems_gate_authorized": False,
        "systems_pass_earned": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(v) for v in values)
    return {
        "samples": float(len(ordered)),
        "mean_ms": float(statistics.fmean(ordered)),
        "median_ms": float(statistics.median(ordered)),
        "min_ms": float(min(ordered)),
        "max_ms": float(max(ordered)),
    }


def _event_timed_call(call: Callable[[], object]) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    del output
    return elapsed


def _capture_decisions(model, call: Callable[[], dict[str, object]]) -> tuple[dict[str, object], dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from tam_research.aera_hardware_core_v25_1_nohost import (
        ExecutionEquivalentNoHostBMMHardSparseExpertBank,
        ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner,
    )
    from tam_research.aera_hardware_core_v26_10_latent_depth_sync_coalescing import (
        CoalescedDepthPrefixLatentReasonerV2610,
    )

    expert_rows: list[dict[str, Any]] = []
    reasoner_rows: list[dict[str, Any]] = []
    handles = []

    for stage_index, stage in enumerate(model.stages):
        experts = stage.experts
        reasoner = stage.reasoner
        if not isinstance(experts, ExecutionEquivalentNoHostBMMHardSparseExpertBank):
            raise RuntimeError("issue706 expert type drift")
        if not isinstance(reasoner, ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner):
            raise RuntimeError("issue706 reasoner type drift")

        def expert_hook(module, args, kwargs, output, *, stage_index=stage_index):
            if len(args) < 3:
                raise RuntimeError("issue706 expert hook argument drift")
            x, expert_logits, count_logits = args[:3]
            route_probs = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
            _, ids = torch.topk(route_probs, module.max_active, dim=-1)
            counts = count_logits.argmax(dim=-1) + 1
            expert_rows.append(
                {
                    "stage": stage_index,
                    "population": int(x.size(0)),
                    "ids": ids.detach().cpu().tolist(),
                    "counts": counts.detach().cpu().tolist(),
                }
            )

        def reasoner_hook(module, args, kwargs, output, *, stage_index=stage_index):
            if len(args) < 2:
                raise RuntimeError("issue706 reasoner hook argument drift")
            summary, depth_logits = args[:2]
            chosen = depth_logits.argmax(dim=-1) + 1
            chosen_cpu = chosen.detach().cpu()
            active = tuple(
                int((chosen_cpu >= step).sum().item())
                for step in range(1, module.max_steps + 1)
            )
            runtime_prefix = None
            dense_masked = None
            if isinstance(module, CoalescedDepthPrefixLatentReasonerV2610):
                runtime_prefix = module.last_active_prefix_counts
                dense_masked = module.last_dense_masked_execution
            reasoner_rows.append(
                {
                    "stage": stage_index,
                    "population": int(summary.size(0)),
                    "chosen": chosen_cpu.tolist(),
                    "active_counts": list(active),
                    "runtime_prefix_counts": None
                    if runtime_prefix is None
                    else list(runtime_prefix),
                    "dense_masked_execution": dense_masked,
                }
            )

        handles.append(experts.register_forward_hook(expert_hook, with_kwargs=True))
        handles.append(reasoner.register_forward_hook(reasoner_hook, with_kwargs=True))

    try:
        output = call()
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(output, dict):
        raise RuntimeError("issue706 correctness call did not return mapping")
    return output, {"experts": expert_rows, "reasoners": reasoner_rows}


def _decision_equivalence(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    expert_exact = baseline["experts"] == candidate["experts"]
    if len(baseline["reasoners"]) != len(candidate["reasoners"]):
        return {
            "pass": False,
            "expert_exact": expert_exact,
            "reasoner_exact": False,
            "candidate_prefix_counts_exact": False,
            "dense_masked_execution": None,
        }
    reasoner_exact = True
    prefix_exact = True
    dense_masked = False
    for ref, cand in zip(baseline["reasoners"], candidate["reasoners"]):
        for key in ("stage", "population", "chosen", "active_counts"):
            reasoner_exact = reasoner_exact and ref[key] == cand[key]
        prefix_exact = prefix_exact and cand["runtime_prefix_counts"] == cand["active_counts"]
        dense_masked = dense_masked or cand["dense_masked_execution"] is not False
    return {
        "pass": bool(expert_exact and reasoner_exact and prefix_exact and not dense_masked),
        "expert_exact": expert_exact,
        "reasoner_exact": reasoner_exact,
        "candidate_prefix_counts_exact": prefix_exact,
        "dense_masked_execution": dense_masked,
        "baseline": baseline,
        "candidate": candidate,
    }


def _route_exact(base_module, baseline_output, candidate_output) -> bool:
    import torch

    left = base_module._route_signature(baseline_output)
    right = base_module._route_signature(candidate_output)
    return bool(
        len(left) == len(right)
        and all(torch.equal(a, b) for a, b in zip(left, right))
    )


def _trace_sync_count(call: Callable[[], object], label: str) -> dict[str, Any]:
    import torch

    path = Path(f"/tmp/{label.replace('.', '-')}.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function(label):
            output = call()
            del output
    torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    path.unlink(missing_ok=True)
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise RuntimeError("issue706 traceEvents missing")
    bounds = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        if event.get("name") == label and str(event.get("cat", "")).lower() == "user_annotation":
            bounds.append((float(event["ts"]), float(event["ts"]) + float(event["dur"])))
    if len(bounds) != 1:
        raise RuntimeError(f"issue706 expected one profile call boundary, got {len(bounds)}")
    start, end = bounds[0]
    runtime: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        if str(event.get("cat", "")).lower() != "cuda_runtime":
            continue
        try:
            ts = float(event["ts"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= ts < end:
            name = str(event.get("name", ""))
            runtime[name] = runtime.get(name, 0) + 1
    return {
        "cuda_runtime_counts": dict(sorted(runtime.items())),
        "cudaStreamSynchronize": int(runtime.get("cudaStreamSynchronize", 0)),
        "profile_timing_credit": False,
    }


def _schemas_and_weights_exact(baseline, candidate) -> dict[str, Any]:
    import torch

    base_schema = tuple((name, tuple(p.shape)) for name, p in baseline.named_parameters())
    cand_schema = tuple((name, tuple(p.shape)) for name, p in candidate.named_parameters())
    base_keys = tuple(baseline.state_dict().keys())
    cand_keys = tuple(candidate.state_dict().keys())
    values_exact = bool(
        base_keys == cand_keys
        and all(torch.equal(baseline.state_dict()[k], candidate.state_dict()[k]) for k in base_keys)
    )
    return {
        "parameter_schema_exact": base_schema == cand_schema,
        "state_dict_keys_exact": base_keys == cand_keys,
        "state_dict_values_exact_before_execution": values_exact,
        "pass": bool(base_schema == cand_schema and base_keys == cand_keys and values_exact),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_microbenchmark() -> dict[str, Any]:
    import torch
    import tam_research.aera_v25_post8471_triage as triage
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
    from tam_research.aera_hardware_core import HardwareAERAState
    from tam_research.aera_hardware_core_v26_10_latent_depth_sync_coalescing import (
        install_latent_depth_sync_coalescing_v26_10,
        latent_depth_sync_coalescing_v26_10_protocol,
    )
    from tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import (
        IdentityWeightVisibilityTritonFICEMReadWriteBackend,
    )

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue706 result already exists: {RESULT_PATH}")
    _parent_result()
    blobs = _frozen_blobs()
    if not torch.cuda.is_available():
        raise RuntimeError("issue706 requires authorized NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    hashes_before = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError("issue706 checkpoint drift before model load")

    root = Path(base.CHECKPOINT_RELATIVE_DIR)
    payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    if payload.get("seed") != base.SOURCE_CHECKPOINT_SEED:
        raise RuntimeError("issue706 AERA checkpoint seed mismatch")

    baseline = base._build_v26(payload, device)
    candidate = base._build_v26(payload, device)
    baseline_backends = systems._install_v26_9_candidate_backend(baseline)
    candidate_backends = systems._install_v26_9_candidate_backend(candidate)
    installed_stages = install_latent_depth_sync_coalescing_v26_10(candidate)
    expected_backend = IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if not all(name == expected_backend for name in baseline_backends + candidate_backends):
        raise RuntimeError("issue706 v26.9 backend identity drift")
    if installed_stages != tuple(range(len(candidate.stages))):
        raise RuntimeError("issue706 v26.10 reasoner installer stage drift")

    schema_check = _schemas_and_weights_exact(baseline, candidate)
    if not schema_check["pass"]:
        raise RuntimeError(f"issue706 schema/weight drift: {schema_check}")
    baseline_versions_before = base._parameter_versions(baseline)
    candidate_versions_before = base._parameter_versions(candidate)

    rows: dict[str, Any] = {}
    with torch.inference_mode():
        for batch in BATCHES:
            generator = torch.Generator(device="cpu").manual_seed(TOKEN_SEED_BASE + batch)
            tokens = torch.randint(
                0,
                triage.VOCAB_SIZE,
                (batch, triage.SEQ_LEN),
                generator=generator,
            ).to(device)
            baseline_call = lambda: base._model_call(baseline, tokens, update_memory=True)
            candidate_call = lambda: base._model_call(candidate, tokens, update_memory=True)

            baseline_output, baseline_decisions = _capture_decisions(baseline, baseline_call)
            candidate_output, candidate_decisions = _capture_decisions(candidate, candidate_call)
            if not isinstance(baseline_output.get("state"), HardwareAERAState) or not isinstance(
                candidate_output.get("state"), HardwareAERAState
            ):
                raise RuntimeError("issue706 correctness output missing HardwareAERAState")
            baseline_logits = baseline_output.get("logits")
            candidate_logits = candidate_output.get("logits")
            if not isinstance(baseline_logits, torch.Tensor) or not isinstance(candidate_logits, torch.Tensor):
                raise RuntimeError("issue706 correctness output missing logits")

            decision_eq = _decision_equivalence(baseline_decisions, candidate_decisions)
            route_exact = _route_exact(base, baseline_output, candidate_output)
            logit_eq = base._logit_equivalence(baseline_logits, candidate_logits)
            state_eq = base._state_equivalence(
                baseline_output["state"], candidate_output["state"]
            )
            finite = bool(base._finite_output(baseline_output) and base._finite_output(candidate_output))
            correctness_pass = bool(
                decision_eq["pass"]
                and route_exact
                and logit_eq["pass"]
                and state_eq["pass"]
                and finite
            )

            # No correctness hooks are present during performance timing.
            for _ in range(WARMUP_CALLS):
                a = baseline_call()
                b = candidate_call()
                del a, b
            samples = {"baseline_v26_9": [], "candidate_v26_10": []}
            for index in range(TIMED_CALLS_PER_CONDITION):
                order = (
                    (("baseline_v26_9", baseline_call), ("candidate_v26_10", candidate_call))
                    if index % 2 == 0
                    else (("candidate_v26_10", candidate_call), ("baseline_v26_9", baseline_call))
                )
                for name, call in order:
                    samples[name].append(_event_timed_call(call))
            timing = {name: _summary(values) for name, values in samples.items()}
            baseline_ms = timing["baseline_v26_9"]["median_ms"]
            candidate_ms = timing["candidate_v26_10"]["median_ms"]
            improvement = (baseline_ms - candidate_ms) / baseline_ms

            baseline_profile = _trace_sync_count(
                baseline_call, f"aera706.baseline.batch{batch}"
            )
            candidate_profile = _trace_sync_count(
                candidate_call, f"aera706.candidate.batch{batch}"
            )
            sync_reduction = (
                baseline_profile["cudaStreamSynchronize"]
                - candidate_profile["cudaStreamSynchronize"]
            )
            latency_pass = improvement >= MIN_LATENCY_IMPROVEMENT
            sync_pass = sync_reduction >= MIN_STREAM_SYNCHRONIZE_REDUCTION
            per_batch_pass = bool(correctness_pass and latency_pass and sync_pass)

            rows[str(batch)] = {
                "batch_size": batch,
                "token_seed": TOKEN_SEED_BASE + batch,
                "route_mode": "hard_sparse",
                "hard": True,
                "update_memory": True,
                "cuda_bf16_autocast_via_frozen_model_call": True,
                "correctness": {
                    "pass": correctness_pass,
                    "route_exact": route_exact,
                    "decision_equivalence": decision_eq,
                    "logit_equivalence": logit_eq,
                    "state_equivalence": state_eq,
                    "finite": finite,
                },
                "timing": timing,
                "candidate_latency_improvement_fraction": improvement,
                "min_latency_improvement_fraction": MIN_LATENCY_IMPROVEMENT,
                "latency_gate_pass": latency_pass,
                "post_timing_profiler": {
                    "baseline": baseline_profile,
                    "candidate": candidate_profile,
                    "cudaStreamSynchronize_reduction": sync_reduction,
                    "min_reduction": MIN_STREAM_SYNCHRONIZE_REDUCTION,
                    "sync_gate_pass": sync_pass,
                    "profiler_timing_credit": False,
                },
                "pass": per_batch_pass,
            }
            del baseline_output, candidate_output, baseline_logits, candidate_logits, tokens
            gc.collect()
            torch.cuda.empty_cache()

    hashes_after = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    checkpoint_unchanged = hashes_after == hashes_before
    parameter_versions_unchanged = bool(
        baseline_versions_before == base._parameter_versions(baseline)
        and candidate_versions_before == base._parameter_versions(candidate)
    )
    all_batches_pass = all(row["pass"] for row in rows.values())
    overall_pass = bool(
        schema_check["pass"]
        and checkpoint_unchanged
        and parameter_versions_unchanged
        and all_batches_pass
    )
    decision = (
        "PASS_V26_10_LATENT_DEPTH_SYNC_COALESCING_MICROBENCH"
        if overall_pass
        else "FAIL_V26_10_LATENT_DEPTH_SYNC_COALESCING_MICROBENCH"
    )
    result = {
        "scope": "aera_v26_10_issue706_latent_depth_sync_coalescing",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "parent_issue": PARENT_ISSUE,
        "parent_trigger": PARENT_TRIGGER,
        "parent_evidence_comment": PARENT_EVIDENCE_COMMENT,
        "parent_run": PARENT_RUN,
        "parent_job": PARENT_JOB,
        "parent_attempt": PARENT_ATTEMPT,
        "source_systems_decision": SOURCE_DECISION,
        "parent_next_target": PARENT_NEXT_TARGET,
        "device": torch.cuda.get_device_name(device),
        "protocol": latent_depth_sync_coalescing_v26_10_protocol(),
        "frozen_blobs": blobs,
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "checkpoint_hashes_unchanged": checkpoint_unchanged,
        "schema_check": schema_check,
        "parameter_versions_unchanged": parameter_versions_unchanged,
        "baseline_backend_names": list(baseline_backends),
        "candidate_backend_names": list(candidate_backends),
        "installed_v26_10_stages": list(installed_stages),
        "batches": list(BATCHES),
        "warmup_calls": WARMUP_CALLS,
        "timed_calls_per_condition": TIMED_CALLS_PER_CONDITION,
        "profile_calls_per_condition": PROFILE_CALLS_PER_CONDITION,
        "min_latency_improvement_fraction": MIN_LATENCY_IMPROVEMENT,
        "min_cudaStreamSynchronize_reduction": MIN_STREAM_SYNCHRONIZE_REDUCTION,
        "rows": rows,
        "overall_pass": overall_pass,
        "decision": decision,
        "optimization_implemented": True,
        "optimization_microbenchmark_pass": overall_pass,
        "fresh_full_e2e_systems_gate_authorized": overall_pass,
        "systems_pass_earned": False,
        "comparative_transformer_gate_executed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "checkpoint_written": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    path = Path(RESULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    volume.commit()
    summary = {
        "scope": result["scope"],
        "research_issue": RESEARCH_ISSUE,
        "device": result["device"],
        "decision": decision,
        "overall_pass": overall_pass,
        "checkpoint_hashes_unchanged": checkpoint_unchanged,
        "schema_check": schema_check,
        "parameter_versions_unchanged": parameter_versions_unchanged,
        "batches": {
            batch: {
                "correctness_pass": row["correctness"]["pass"],
                "route_exact": row["correctness"]["route_exact"],
                "decision_equivalence_pass": row["correctness"]["decision_equivalence"]["pass"],
                "baseline_median_ms": row["timing"]["baseline_v26_9"]["median_ms"],
                "candidate_median_ms": row["timing"]["candidate_v26_10"]["median_ms"],
                "latency_improvement_fraction": row["candidate_latency_improvement_fraction"],
                "latency_gate_pass": row["latency_gate_pass"],
                "baseline_cudaStreamSynchronize": row["post_timing_profiler"]["baseline"]["cudaStreamSynchronize"],
                "candidate_cudaStreamSynchronize": row["post_timing_profiler"]["candidate"]["cudaStreamSynchronize"],
                "cudaStreamSynchronize_reduction": row["post_timing_profiler"]["cudaStreamSynchronize_reduction"],
                "sync_gate_pass": row["post_timing_profiler"]["sync_gate_pass"],
                "pass": row["pass"],
            }
            for batch, row in rows.items()
        },
        "fresh_full_e2e_systems_gate_authorized": overall_pass,
        "systems_pass_earned": False,
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
def preauth_main() -> None:
    print(PREAUTH_MARKER + json.dumps(preflight.remote(), sort_keys=True))


@app.local_entrypoint()
def l4_main() -> None:
    pre = preflight.remote()
    print(PREAUTH_MARKER + json.dumps(pre, sort_keys=True))
    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "gpu": "L4",
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "result_path": RESULT_PATH,
                "optimization_microbenchmark_only": True,
            },
            sort_keys=True,
        )
    )
    summary = run_microbenchmark.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
