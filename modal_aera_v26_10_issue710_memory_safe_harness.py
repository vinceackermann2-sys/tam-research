from __future__ import annotations

"""Issue #710 memory-safe successor for the frozen v26.10 #706 microbenchmark."""

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import modal
import modal_aera_v26_10_issue706_latent_depth_sync_coalescing as frozen706

APP_NAME = "aera-v26-10-issue710-memory-safe-harness-repair1"
VOLUME_NAME = frozen706.VOLUME_NAME
RESULT_PATH = "/vol/aera-v26/issue710-v26-10-memory-safe-harness/result.json"
SOURCE_MAIN = "c2e9c17d7bb34d7652fc6759c80193f2d951f2ce"
SOURCE_TREE = "ff3913de8e605aba10f87839676c9b582304233c"
RESEARCH_ISSUE = 710
PARENT_ISSUE = 706
CONSUMED_TRIGGER = 709
CONSUMED_FAILURE_COMMENT = 5573348027
CONSUMED_PRECISION_COMMENT = 5573367269
CONSUMED_RUN = 34141205064
CONSUMED_JOB = 101803372290
CONSUMED_ATTEMPT = 1
SOURCE_DECISION = frozen706.SOURCE_DECISION
PARENT_NEXT_TARGET = frozen706.PARENT_NEXT_TARGET
ISSUE706_LAUNCHER = "modal_aera_v26_10_issue706_latent_depth_sync_coalescing.py"
ISSUE706_LAUNCHER_BLOB = "e23072aa83b0bc20ccd72e7567f234e9f4a931f6"
V26_10_IMPL_BLOB = "d8f691c198eed1fa96bcbb78a4e76cad82d18779"
CHECKPOINT_HASHES = dict(frozen706.CHECKPOINT_HASHES)
BATCHES = tuple(frozen706.BATCHES)
TOKEN_SEED_BASE = frozen706.TOKEN_SEED_BASE
WARMUP_CALLS = frozen706.WARMUP_CALLS
TIMED_CALLS_PER_CONDITION = frozen706.TIMED_CALLS_PER_CONDITION
PROFILE_CALLS_PER_CONDITION = frozen706.PROFILE_CALLS_PER_CONDITION
MIN_LATENCY_IMPROVEMENT = frozen706.MIN_LATENCY_IMPROVEMENT
MIN_STREAM_SYNCHRONIZE_REDUCTION = frozen706.MIN_STREAM_SYNCHRONIZE_REDUCTION
INTEGRATED_ATOL = frozen706.INTEGRATED_ATOL
INTEGRATED_RTOL = frozen706.INTEGRATED_RTOL
LOGIT_COMPARE_BATCH_CHUNK = 1
MAX_GPU_SECONDS = frozen706.MAX_GPU_SECONDS
PREAUTH_MARKER = "AERA_V26_10_ISSUE710_PREAUTH_JSON="
L4_START_MARKER = "AERA_V26_10_ISSUE710_L4_START_JSON="
RESULT_MARKER = "AERA_V26_10_ISSUE710_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_10_ISSUE710_SUMMARY_JSON="

# frozen706.image already packages the exact root-level issue665 launcher.
image = frozen706.image.add_local_file(
    ISSUE706_LAUNCHER, f"/root/{ISSUE706_LAUNCHER}"
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _frozen_blobs() -> dict[str, str]:
    got = dict(frozen706._frozen_blobs())
    got["issue706_launcher"] = _blob(Path(f"/root/{ISSUE706_LAUNCHER}"))
    if got.get("issue706_launcher") != ISSUE706_LAUNCHER_BLOB:
        raise RuntimeError(f"issue710 frozen #706 launcher drift: {got.get('issue706_launcher')}")
    if got.get("v26_10_impl") != V26_10_IMPL_BLOB:
        raise RuntimeError(f"issue710 frozen v26.10 implementation drift: {got.get('v26_10_impl')}")
    return got


def _chunked_logit_equivalence(reference, candidate) -> dict[str, Any]:
    import torch

    exact_meta = (
        reference.dtype == candidate.dtype
        and reference.device == candidate.device
        and reference.shape == candidate.shape
    )
    if reference.shape != candidate.shape:
        return {
            "pass": False,
            "allclose": False,
            "dtype_device_shape_exact": False,
            "max_abs": float("inf"),
            "atol": INTEGRATED_ATOL,
            "rtol": INTEGRATED_RTOL,
        }
    ref = reference.reshape(1) if reference.ndim == 0 else reference
    cand = candidate.reshape(1) if candidate.ndim == 0 else candidate
    rows = int(ref.shape[0])
    global_max = None
    close = True
    for start in range(0, rows, LOGIT_COMPARE_BATCH_CHUNK):
        end = min(start + LOGIT_COMPARE_BATCH_CHUNK, rows)
        ref32 = ref[start:end].float()
        cand32 = cand[start:end].float()
        chunk_max = (ref32 - cand32).abs().max()
        global_max = chunk_max if global_max is None else torch.maximum(global_max, chunk_max)
        chunk_close = bool(
            torch.allclose(
                ref32,
                cand32,
                atol=INTEGRATED_ATOL,
                rtol=INTEGRATED_RTOL,
            )
        )
        close = close and chunk_close
        del ref32, cand32, chunk_max
    max_abs = float(global_max) if global_max is not None else 0.0
    return {
        "pass": bool(exact_meta and close),
        "allclose": close,
        "dtype_device_shape_exact": exact_meta,
        "max_abs": max_abs,
        "atol": INTEGRATED_ATOL,
        "rtol": INTEGRATED_RTOL,
    }


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight() -> dict[str, Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base
    from tam_research.aera_hardware_core_v26_10_latent_depth_sync_coalescing import (
        latent_depth_sync_coalescing_v26_10_protocol,
    )

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue710 result already exists: {RESULT_PATH}")
    frozen706._parent_result()
    blobs = _frozen_blobs()
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError("issue710 checkpoint drift")
    protocol = latent_depth_sync_coalescing_v26_10_protocol()
    if protocol["research_issue"] != 706:
        raise RuntimeError("issue710 frozen v26.10 protocol drift")
    if protocol["state_dict_schema_changed"] is not False:
        raise RuntimeError("issue710 v26.10 schema drift")
    if LOGIT_COMPARE_BATCH_CHUNK != 1:
        raise RuntimeError("issue710 logit chunk drift")
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "parent_issue": PARENT_ISSUE,
        "consumed_trigger": CONSUMED_TRIGGER,
        "consumed_failure_comment": CONSUMED_FAILURE_COMMENT,
        "consumed_precision_comment": CONSUMED_PRECISION_COMMENT,
        "consumed_run": CONSUMED_RUN,
        "consumed_job": CONSUMED_JOB,
        "consumed_attempt": CONSUMED_ATTEMPT,
        "source_decision": SOURCE_DECISION,
        "parent_next_target": PARENT_NEXT_TARGET,
        "result_path": RESULT_PATH,
        "frozen_blobs": blobs,
        "checkpoint_hashes": hashes,
        "logit_compare_batch_chunk": LOGIT_COMPARE_BATCH_CHUNK,
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
        raise RuntimeError(f"issue710 result already exists: {RESULT_PATH}")
    frozen706._parent_result()
    blobs = _frozen_blobs()
    if not torch.cuda.is_available():
        raise RuntimeError("issue710 requires authorized NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    hashes_before = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes_before != CHECKPOINT_HASHES:
        raise RuntimeError("issue710 checkpoint drift before model load")

    root = Path(base.CHECKPOINT_RELATIVE_DIR)
    payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    if payload.get("seed") != base.SOURCE_CHECKPOINT_SEED:
        raise RuntimeError("issue710 AERA checkpoint seed mismatch")

    baseline = base._build_v26(payload, device)
    candidate = base._build_v26(payload, device)
    baseline_backends = systems._install_v26_9_candidate_backend(baseline)
    candidate_backends = systems._install_v26_9_candidate_backend(candidate)
    installed_stages = install_latent_depth_sync_coalescing_v26_10(candidate)
    expected_backend = IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    if not all(name == expected_backend for name in baseline_backends + candidate_backends):
        raise RuntimeError("issue710 v26.9 backend identity drift")
    if installed_stages != tuple(range(len(candidate.stages))):
        raise RuntimeError("issue710 v26.10 reasoner installer stage drift")

    schema_check = frozen706._schemas_and_weights_exact(baseline, candidate)
    if not schema_check["pass"]:
        raise RuntimeError(f"issue710 schema/weight drift: {schema_check}")
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

            baseline_output, baseline_decisions = frozen706._capture_decisions(baseline, baseline_call)
            candidate_output, candidate_decisions = frozen706._capture_decisions(candidate, candidate_call)
            if not isinstance(baseline_output.get("state"), HardwareAERAState) or not isinstance(
                candidate_output.get("state"), HardwareAERAState
            ):
                raise RuntimeError("issue710 correctness output missing HardwareAERAState")
            baseline_logits = baseline_output.get("logits")
            candidate_logits = candidate_output.get("logits")
            if not isinstance(baseline_logits, torch.Tensor) or not isinstance(candidate_logits, torch.Tensor):
                raise RuntimeError("issue710 correctness output missing logits")

            decision_eq = frozen706._decision_equivalence(baseline_decisions, candidate_decisions)
            route_exact = frozen706._route_exact(base, baseline_output, candidate_output)
            logit_eq = _chunked_logit_equivalence(baseline_logits, candidate_logits)
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

            # Correctness outputs are not timing fixtures. Release them before all warmups/timing.
            del baseline_output, candidate_output, baseline_logits, candidate_logits
            gc.collect()
            torch.cuda.empty_cache()

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
                    samples[name].append(frozen706._event_timed_call(call))
            timing = {name: frozen706._summary(values) for name, values in samples.items()}
            baseline_ms = timing["baseline_v26_9"]["median_ms"]
            candidate_ms = timing["candidate_v26_10"]["median_ms"]
            improvement = (baseline_ms - candidate_ms) / baseline_ms

            baseline_profile = frozen706._trace_sync_count(
                baseline_call, f"aera710.baseline.batch{batch}"
            )
            candidate_profile = frozen706._trace_sync_count(
                candidate_call, f"aera710.candidate.batch{batch}"
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
                    "logit_compare_batch_chunk": LOGIT_COMPARE_BATCH_CHUNK,
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
            del tokens
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
        "scope": "aera_v26_10_issue710_memory_safe_harness",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "parent_issue": PARENT_ISSUE,
        "consumed_trigger": CONSUMED_TRIGGER,
        "consumed_failure_comment": CONSUMED_FAILURE_COMMENT,
        "consumed_precision_comment": CONSUMED_PRECISION_COMMENT,
        "consumed_run": CONSUMED_RUN,
        "consumed_job": CONSUMED_JOB,
        "consumed_attempt": CONSUMED_ATTEMPT,
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
        "logit_compare_batch_chunk": LOGIT_COMPARE_BATCH_CHUNK,
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
        "logit_compare_batch_chunk": LOGIT_COMPARE_BATCH_CHUNK,
        "batches": {
            batch: {
                "correctness_pass": row["correctness"]["pass"],
                "route_exact": row["correctness"]["route_exact"],
                "decision_equivalence_pass": row["correctness"]["decision_equivalence"]["pass"],
                "logit_equivalence": row["correctness"]["logit_equivalence"],
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
                "harness_repair_only": True,
            },
            sort_keys=True,
        )
    )
    summary = run_microbenchmark.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
