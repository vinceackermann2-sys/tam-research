from __future__ import annotations

"""One-shot checkpoint-only CHM-v1 Stage-C STOP postmortem runner (#1008).

This runner may read the immutable Stage-C result/checkpoint and may write only
under the dedicated postmortem result root.  It never trains, updates, resumes,
or saves a model, and it introduces no scientific seed.
"""

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

PHASE = "chm-v1-100m-stage-c-postmortem-1008-v1"
CONTROL_ISSUE = 1008
PREREG_ISSUE = 1002
SOURCE_SCIENTIFIC_ISSUE = 990
SOURCE_SCIENTIFIC_SEED = 977_001

TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-postmortem-1008-v1]"
SOURCE_RESULT_ROOT = "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"
SOURCE_RESULT_PATH = f"{SOURCE_RESULT_ROOT}/RESULT.json"
SOURCE_CHECKPOINT_PATH = f"{SOURCE_RESULT_ROOT}/checkpoints/eiem/step-2048.pt"
SOURCE_CHECKPOINT_METADATA_PATH = f"{SOURCE_RESULT_ROOT}/checkpoints/eiem/step-2048.json"
RESULT_ROOT = "/vol/chm-v1/100m-stage-c-postmortem/issue-1008/v1"

MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
POSTMORTEM_BLOB = "1612dfb17a613508b0604990c94bed6fecd3776d"
SCIENTIFIC_IMPLEMENTATION_BLOB = "fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 7_200
MAX_BILLED_COMPUTE_USD = 3.00
SOFT_TEMPERATURE = 0.10

APP_NAME = "chm-v1-100m-stage-c-postmortem-1008-v1"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2,<3", "tiktoken>=0.9,<1")
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _full_sha(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 40-hex SHA")
    return normalized


def _sha256(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 64-hex SHA-256")
    return normalized


def _validate_bindings(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    postmortem_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> dict[str, str]:
    bindings = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "model_blob_sha": _full_sha(model_sha, "model_sha"),
        "evaluator_blob_sha": _full_sha(evaluator_sha, "evaluator_sha"),
        "postmortem_blob_sha": _full_sha(postmortem_sha, "postmortem_sha"),
        "scientific_implementation_blob_sha": _full_sha(scientific_impl_sha, "scientific_impl_sha"),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
    }
    if bindings["model_blob_sha"] != MODEL_BLOB:
        raise RuntimeError("#1008 model blob drift")
    if bindings["evaluator_blob_sha"] != EVALUATOR_BLOB:
        raise RuntimeError("#1008 evaluator blob drift")
    if bindings["postmortem_blob_sha"] != POSTMORTEM_BLOB:
        raise RuntimeError("#1008 postmortem blob drift")
    if bindings["scientific_implementation_blob_sha"] != SCIENTIFIC_IMPLEMENTATION_BLOB:
        raise RuntimeError("#1008 scientific implementation provenance drift")
    if (CONTROL_ISSUE, PREREG_ISSUE, SOURCE_SCIENTIFIC_ISSUE) != (1008, 1002, 990):
        raise RuntimeError("#1008 issue binding drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 16 * 1024:
        raise RuntimeError("#1008 resource binding drift")
    if MAX_SECONDS != 7_200 or MAX_BILLED_COMPUTE_USD != 3.00:
        raise RuntimeError("#1008 runtime/cost binding drift")
    if SOFT_TEMPERATURE != 0.10:
        raise RuntimeError("#1008 frozen soft temperature drift")
    return bindings


def _evidence(
    bindings: dict[str, str],
    authority_comment_id: int,
    checkpoint_sha256: str | None = None,
    checkpoint_bytes: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "phase": PHASE,
        "control_issue": CONTROL_ISSUE,
        "preregistration_issue": PREREG_ISSUE,
        "source_scientific_issue": SOURCE_SCIENTIFIC_ISSUE,
        "source_scientific_seed": SOURCE_SCIENTIFIC_SEED,
        "source_seed_historical_only": True,
        "trigger_title": TRIGGER_TITLE,
        "source_result_root": SOURCE_RESULT_ROOT,
        "result_root": RESULT_ROOT,
        "final_authority_comment_id": int(authority_comment_id),
        "training_authorized": False,
        "parameter_updates_authorized": False,
        "checkpoint_resume_authorized": False,
        "automatic_retry_authorized": False,
        "stage_d_automatically_authorized": False,
        **bindings,
    }
    if checkpoint_sha256 is not None:
        out["source_checkpoint_sha256"] = _sha256(checkpoint_sha256, "checkpoint_sha256")
    if checkpoint_bytes is not None:
        if int(checkpoint_bytes) <= 0:
            raise ValueError("checkpoint_bytes must be positive")
        out["source_checkpoint_bytes"] = int(checkpoint_bytes)
    return out


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"#1008 missing {name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"#1008 {name} must be a JSON object")
    return payload


def _inspect_source_state() -> dict[str, Any]:
    source_result = _read_json(Path(SOURCE_RESULT_PATH), "source RESULT.json")
    if source_result.get("status") != "COMPLETE":
        raise RuntimeError("#1008 source Stage-C result is not COMPLETE")
    if source_result.get("classification") != "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH":
        raise RuntimeError("#1008 source Stage-C result is not the terminal STOP classification")
    if int(source_result.get("scientific_seed", -1)) != SOURCE_SCIENTIFIC_SEED:
        raise RuntimeError("#1008 source scientific seed provenance drift")
    if source_result.get("scientific_seed_consumed") is not True:
        raise RuntimeError("#1008 source result does not record seed consumption")
    if source_result.get("automatic_retry_authorized") is not False:
        raise RuntimeError("#1008 source result retry semantics drift")
    if source_result.get("checkpoint_resume_authorized") is not False:
        raise RuntimeError("#1008 source result resume semantics drift")
    if source_result.get("stage_d_automatically_authorized") is not False:
        raise RuntimeError("#1008 source result Stage-D authority drift")

    metadata = _read_json(Path(SOURCE_CHECKPOINT_METADATA_PATH), "EIEM final checkpoint metadata")
    required = {
        "kind": "eiem",
        "step": 2048,
        "tokens_seen": 33_554_432,
        "resume_authorized": False,
        "scientific_evaluation_authorized": True,
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"#1008 checkpoint metadata {key} mismatch: {metadata.get(key)!r} != {value!r}"
            )
    checkpoint_sha = _sha256(str(metadata.get("sha256", "")), "checkpoint metadata sha256")
    checkpoint_bytes = int(metadata.get("bytes", 0))
    if checkpoint_bytes <= 0:
        raise RuntimeError("#1008 checkpoint metadata has invalid byte count")

    checkpoint = Path(SOURCE_CHECKPOINT_PATH)
    if not checkpoint.is_file():
        raise RuntimeError("#1008 final EIEM checkpoint file is missing")
    actual_bytes = int(checkpoint.stat().st_size)
    if actual_bytes != checkpoint_bytes:
        raise RuntimeError(
            f"#1008 checkpoint size mismatch: actual={actual_bytes} metadata={checkpoint_bytes}"
        )

    result_root = Path(RESULT_ROOT)
    return {
        "source_result_status": source_result["status"],
        "source_result_classification": source_result["classification"],
        "source_scientific_seed": SOURCE_SCIENTIFIC_SEED,
        "source_scientific_seed_consumed": True,
        "checkpoint_path": SOURCE_CHECKPOINT_PATH,
        "checkpoint_metadata_path": SOURCE_CHECKPOINT_METADATA_PATH,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_file_exists": True,
        "diagnostic_result_namespace_unused": not result_root.exists(),
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def inspect_source(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    postmortem_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        postmortem_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
    )
    volume.reload()
    state = _inspect_source_state()
    payload = {
        "classification": "CHM_V1_100M_STAGE_C_POSTMORTEM_SOURCE_AUDIT",
        **_evidence(bindings, 0),
        **state,
        "gpu_allocated": False,
        "diagnostic_attempt_consumed": False,
        "trigger_authorized_by_runner": False,
    }
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def verify_zero_gpu(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    postmortem_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    expected_checkpoint_sha256: str,
    expected_checkpoint_bytes: int,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        postmortem_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
    )
    authority = int(authority_comment_id)
    if authority <= 0:
        raise RuntimeError("#1008 final authority comment ID must be positive")
    expected_sha = _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    expected_bytes = int(expected_checkpoint_bytes)
    if expected_bytes <= 0:
        raise RuntimeError("#1008 expected checkpoint bytes must be positive")

    volume.reload()
    state = _inspect_source_state()
    if not state["diagnostic_result_namespace_unused"]:
        raise RuntimeError("#1008 diagnostic namespace is already present; fail closed")
    if state["checkpoint_sha256"] != expected_sha or state["checkpoint_bytes"] != expected_bytes:
        raise RuntimeError("#1008 source checkpoint binding differs from final authority")

    root = Path(RESULT_ROOT)
    gate = {
        "status": "PASS",
        "classification": "CHM_V1_100M_STAGE_C_POSTMORTEM_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence(bindings, authority, expected_sha, expected_bytes),
        "source_state": state,
        "gpu_allocated": False,
        "diagnostic_attempt_consumed": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", gate)
    volume.commit()
    return json.dumps(gate, sort_keys=True)


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def reserve_dispatch(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    postmortem_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    expected_checkpoint_sha256: str,
    expected_checkpoint_bytes: int,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        postmortem_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
    )
    authority = int(authority_comment_id)
    expected_sha = _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    expected_bytes = int(expected_checkpoint_bytes)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"

    if not zero_path.is_file():
        raise RuntimeError("#1008 zero-GPU preflight gate is missing")
    if reserve_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1008 diagnostic dispatch/attempt already exists; no redispatch")

    zero = _read_json(zero_path, "zero-GPU gate")
    expected = _evidence(bindings, authority, expected_sha, expected_bytes)
    for key, value in expected.items():
        if zero.get(key) != value:
            raise RuntimeError(f"#1008 zero-GPU gate binding drift at {key}")

    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_100M_STAGE_C_POSTMORTEM_DISPATCH_RESERVED",
        **expected,
        "reserved_unix": time.time(),
        "gpu_allocated": False,
        "diagnostic_attempt_consumed": False,
    }
    _atomic_write(reserve_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=CPU_CORES,
    memory=RAM_MIB,
    timeout=MAX_SECONDS,
    retries=0,
    volumes={"/vol": volume},
)
def run_diagnostic(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    postmortem_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    expected_checkpoint_sha256: str,
    expected_checkpoint_bytes: int,
    live_hourly_resource_usd: float,
) -> str:
    import torch
    import tiktoken

    raw_bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        postmortem_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
    )
    authority = int(authority_comment_id)
    checkpoint_sha = _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    checkpoint_bytes = int(expected_checkpoint_bytes)
    if authority <= 0 or checkpoint_bytes <= 0:
        raise RuntimeError("#1008 invalid final authority/checkpoint binding")

    hourly = float(live_hourly_resource_usd)
    if hourly <= 0.0 or hourly * (MAX_SECONDS / 3600.0) > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError("#1008 live resource rate exceeds frozen diagnostic compute cap")

    volume.reload()
    root = Path(RESULT_ROOT)
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not reserve_path.is_file():
        raise RuntimeError("#1008 durable dispatch reservation is missing")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1008 diagnostic attempt already consumed; no retry/resume/redispatch")

    evidence = _evidence(raw_bindings, authority, checkpoint_sha, checkpoint_bytes)
    consumed = {
        "status": "DIAGNOSTIC_ATTEMPT_CONSUMED",
        "classification": "CHM_V1_100M_STAGE_C_POSTMORTEM_GPU_FUNCTION_BEGAN",
        **evidence,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "diagnostic_attempt_consumed": True,
        "new_scientific_seed_created": False,
        "source_scientific_seed_reused": False,
    }
    # Irreversible one-shot marker is the first durable action inside this GPU function.
    _atomic_write(consumed_path, consumed)
    volume.commit()

    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("#1008 CUDA is unavailable inside the authorized L4 function")
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1008 expected NVIDIA L4, got {device_name!r}")

        checkpoint_path = Path(SOURCE_CHECKPOINT_PATH)
        if int(checkpoint_path.stat().st_size) != checkpoint_bytes:
            raise RuntimeError("#1008 source checkpoint byte count changed before load")
        before_sha = _sha256_file(checkpoint_path)
        if before_sha != checkpoint_sha:
            raise RuntimeError("#1008 source checkpoint SHA-256 changed before load")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        from tam_research.chm_v1_100m_scale import CHMV1100MEIEMLM
        from tam_research.chm_v1_100m_stage_c_eval import (
            CASES_PER_FAMILY,
            GENERATOR_VERSION,
            PROBE_SEED,
            TOTAL_PROBES,
            generate_aligned_probe_suite,
        )
        from tam_research.chm_v1_100m_stage_c_postmortem import (
            classify_diagnostic_signals,
            diagnostic_probe_row,
            summarize_diagnostics,
            validate_checkpoint_payload,
            validate_protocol_manifest,
        )

        manifest = validate_protocol_manifest()
        validate_checkpoint_payload(payload, expected_kind="eiem")

        model = CHMV1100MEIEMLM()
        load_result = model.load_state_dict(payload["model_state_dict"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                f"#1008 strict checkpoint load drift: missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
        del payload
        model.eval()
        model.to(device)

        encoder = tiktoken.get_encoding("gpt2")
        probes = generate_aligned_probe_suite(
            lambda text: encoder.encode(text),
            seed=PROBE_SEED,
            cases_per_family=CASES_PER_FAMILY,
        )
        if len(probes) != TOTAL_PROBES or {p.generator_version for p in probes} != {GENERATOR_VERSION}:
            raise RuntimeError("#1008 frozen probe envelope drift")

        torch.cuda.reset_peak_memory_stats(device)
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for probe in probes:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    rows.append(diagnostic_probe_row(model, probe))

        summary = summarize_diagnostics(rows)
        decision = classify_diagnostic_signals(summary)

        after_sha = _sha256_file(checkpoint_path)
        if after_sha != checkpoint_sha or int(checkpoint_path.stat().st_size) != checkpoint_bytes:
            raise RuntimeError("#1008 source checkpoint changed during diagnostic")

        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        raw_rows_path = root / "RAW_ROWS.json"
        summary_path = root / "SUMMARY.json"
        _atomic_write(raw_rows_path, rows)
        _atomic_write(summary_path, summary)

        result = {
            "status": "COMPLETE",
            "classification": decision["classification"],
            "tags": decision["tags"],
            **evidence,
            "protocol_manifest": manifest,
            "probe_count": len(rows),
            "summary": summary,
            "raw_rows_path": str(raw_rows_path),
            "summary_path": str(summary_path),
            "source_checkpoint_sha256_before": before_sha,
            "source_checkpoint_sha256_after": after_sha,
            "source_checkpoint_unchanged": before_sha == after_sha == checkpoint_sha,
            "wall_seconds": elapsed,
            "peak_vram_bytes": peak,
            "device_name": device_name,
            "live_hourly_resource_usd": hourly,
            "live_worst_case_usd": hourly * (MAX_SECONDS / 3600.0),
            "gpu_allocation_started": True,
            "diagnostic_attempt_consumed": True,
            "new_scientific_seed_created": False,
            "source_scientific_seed_reused": False,
            "stage_c_result_changed": False,
            "stage_d_authorized": False,
            "new_training_authorized": False,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
        }
        _atomic_write(result_path, result)
        volume.commit()
        return json.dumps(result, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "FAILED",
            "classification": "CHM_V1_100M_STAGE_C_POSTMORTEM_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
            **evidence,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_unix": time.time(),
            "gpu_allocation_started": True,
            "diagnostic_attempt_consumed": True,
            "posthoc_interpretation_authorized": False,
            "new_scientific_seed_created": False,
            "source_scientific_seed_reused": False,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
            "stage_d_authorized": False,
        }
        _atomic_write(failure_path, failure)
        volume.commit()
        raise


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def inspect_state(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    postmortem_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    expected_checkpoint_sha256: str,
    expected_checkpoint_bytes: int,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        postmortem_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
    )
    evidence = _evidence(
        bindings,
        int(authority_comment_id),
        expected_checkpoint_sha256,
        int(expected_checkpoint_bytes),
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    names = (
        "ZERO_GPU_GATE.json",
        "DISPATCH_RESERVED.json",
        "ATTEMPT_CONSUMED.json",
        "RESULT.json",
        "ATTEMPT_FAILURE.json",
        "SUMMARY.json",
    )
    files: dict[str, Any] = {}
    for name in names:
        path = root / name
        files[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    state = {**evidence, "root_exists": root.exists(), "files": files}
    return json.dumps(state, sort_keys=True)


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    postmortem_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int = 0,
    expected_checkpoint_sha256: str = "",
    expected_checkpoint_bytes: int = 0,
    live_hourly_resource_usd: float = 0.0,
) -> None:
    base = (
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        postmortem_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
    )
    if phase == "inspect-source":
        payload = inspect_source.remote(*base)
        print(f"CHM_V1_100M_STAGE_C_POSTMORTEM_SOURCE={payload}")
        return

    common = (
        *base,
        int(authority_comment_id),
        expected_checkpoint_sha256,
        int(expected_checkpoint_bytes),
    )
    if phase == "preflight":
        payload = verify_zero_gpu.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_POSTMORTEM_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_POSTMORTEM_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_diagnostic.remote(*common, float(live_hourly_resource_usd))
        print(f"CHM_V1_100M_STAGE_C_POSTMORTEM_RESULT={payload}")
        return
    if phase == "state":
        payload = inspect_state.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_POSTMORTEM_STATE={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
