from __future__ import annotations

"""One-shot checkpoint-only CHM-v1 payload/gate diagnostic (#1017)."""

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

PHASE = "chm-v1-100m-stage-c-payload-gate-1017-v5"
CONTROL_ISSUE = 1017
PREREG_ISSUE = 1014
SOURCE_SCIENTIFIC_ISSUE = 990
SOURCE_POSTMORTEM_ISSUE = 1008
SOURCE_SCIENTIFIC_SEED = 977_001

TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-payload-gate-1017-v5]"
SOURCE_RESULT_ROOT = "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"
SOURCE_RESULT_PATH = f"{SOURCE_RESULT_ROOT}/RESULT.json"
SOURCE_CHECKPOINT_PATH = f"{SOURCE_RESULT_ROOT}/checkpoints/eiem/step-2048.pt"
SOURCE_CHECKPOINT_METADATA_PATH = f"{SOURCE_RESULT_ROOT}/checkpoints/eiem/step-2048.json"
SOURCE_POSTMORTEM_RESULT_PATH = "/vol/chm-v1/100m-stage-c-postmortem/issue-1008/v1/RESULT.json"
RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v5"

MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
PAYLOAD_GATE_BLOB = "32a5ea1baca45705e75fef4c99974668b59d73a6"
SCIENTIFIC_IMPLEMENTATION_BLOB = "fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"
DUAL_ACCOUNT_BLOB = "b10c4a19d3c4114d084a721b1a84235dfef45259"
DUAL_ACCOUNT_CLI_BLOB = "7d9bd073bedcb8bfafcde23e6f03772dddb6d50e"
CHECKPOINT_SHA256 = "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831"
CHECKPOINT_BYTES = 407_424_818

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 7_200
MAX_BILLED_COMPUTE_USD = 3.00

APP_NAME = "chm-v1-100m-stage-c-payload-gate-1017-v5"
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
    value = str(value).strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a full lowercase SHA")
    return value


def _full_sha256(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _modal_identity(
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> dict[str, str]:
    account = str(selected_modal_account).strip().lower()
    if account not in {"primary", "secondary"}:
        raise ValueError("selected_modal_account must be primary or secondary")
    workspace = str(selected_modal_workspace).strip()
    if not workspace:
        raise ValueError("selected_modal_workspace must be non-empty")
    return {
        "selected_modal_account": account,
        "selected_modal_workspace": workspace,
        "account_selection_evidence_sha256": _full_sha256(
            account_selection_evidence_sha256, "account_selection_evidence_sha256"
        ),
        "modal_account_switch_after_reservation_authorized": False,
    }


def _validate_bindings(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    payload_gate_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
) -> dict[str, str]:
    b = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "model_blob_sha": _full_sha(model_sha, "model_sha"),
        "evaluator_blob_sha": _full_sha(evaluator_sha, "evaluator_sha"),
        "payload_gate_blob_sha": _full_sha(payload_gate_sha, "payload_gate_sha"),
        "scientific_implementation_blob_sha": _full_sha(scientific_impl_sha, "scientific_impl_sha"),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
        "dual_account_blob_sha": _full_sha(dual_account_sha, "dual_account_sha"),
        "dual_account_cli_blob_sha": _full_sha(dual_account_cli_sha, "dual_account_cli_sha"),
    }
    if b["model_blob_sha"] != MODEL_BLOB:
        raise RuntimeError("#1017 model blob drift")
    if b["evaluator_blob_sha"] != EVALUATOR_BLOB:
        raise RuntimeError("#1017 evaluator blob drift")
    if b["payload_gate_blob_sha"] != PAYLOAD_GATE_BLOB:
        raise RuntimeError("#1017 payload/gate blob drift")
    if b["scientific_implementation_blob_sha"] != SCIENTIFIC_IMPLEMENTATION_BLOB:
        raise RuntimeError("#1017 scientific implementation drift")
    if b["dual_account_blob_sha"] != DUAL_ACCOUNT_BLOB:
        raise RuntimeError("#1017 dual-account policy blob drift")
    if b["dual_account_cli_blob_sha"] != DUAL_ACCOUNT_CLI_BLOB:
        raise RuntimeError("#1017 dual-account CLI blob drift")
    if (CONTROL_ISSUE, PREREG_ISSUE, SOURCE_SCIENTIFIC_ISSUE, SOURCE_POSTMORTEM_ISSUE) != (1017, 1014, 990, 1008):
        raise RuntimeError("#1017 issue binding drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 16 * 1024:
        raise RuntimeError("#1017 resource binding drift")
    if MAX_SECONDS != 7_200 or MAX_BILLED_COMPUTE_USD != 3.00:
        raise RuntimeError("#1017 cost/runtime drift")
    return b


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"#1017 missing {name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"#1017 {name} is not an object")
    return payload


def _evidence(
    bindings: dict[str, str],
    authority_comment_id: int,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> dict[str, Any]:
    modal_identity = _modal_identity(
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    return {
        "phase": PHASE,
        "control_issue": CONTROL_ISSUE,
        "preregistration_issue": PREREG_ISSUE,
        "source_scientific_issue": SOURCE_SCIENTIFIC_ISSUE,
        "source_postmortem_issue": SOURCE_POSTMORTEM_ISSUE,
        "source_scientific_seed": SOURCE_SCIENTIFIC_SEED,
        "source_seed_historical_only": True,
        "trigger_title": TRIGGER_TITLE,
        "source_result_root": SOURCE_RESULT_ROOT,
        "result_root": RESULT_ROOT,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "source_checkpoint_bytes": CHECKPOINT_BYTES,
        "final_authority_comment_id": int(authority_comment_id),
        "training_authorized": False,
        "parameter_updates_authorized": False,
        "checkpoint_resume_authorized": False,
        "automatic_retry_authorized": False,
        "stage_d_automatically_authorized": False,
        **modal_identity,
        **bindings,
    }


def _inspect_source() -> dict[str, Any]:
    stage_c = _read_json(Path(SOURCE_RESULT_PATH), "Stage-C RESULT.json")
    if stage_c.get("classification") != "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH":
        raise RuntimeError("#1017 #990 terminal STOP drift")
    if stage_c.get("scientific_seed_consumed") is not True:
        raise RuntimeError("#1017 #990 seed consumption drift")

    postmortem = _read_json(Path(SOURCE_POSTMORTEM_RESULT_PATH), "#1008 RESULT.json")
    if postmortem.get("classification") != "CHM_V1_100M_STAGE_C_POSTMORTEM_DIAGNOSTIC_ONLY":
        raise RuntimeError("#1017 #1008 classification drift")
    tags = postmortem.get("tags")
    if tags != ["PAYLOAD_INTEGRATION_WEAK_SIGNAL"]:
        raise RuntimeError(f"#1017 #1008 tag drift: {tags!r}")
    if postmortem.get("source_checkpoint_unchanged") is not True:
        raise RuntimeError("#1017 #1008 did not preserve checkpoint")

    meta = _read_json(Path(SOURCE_CHECKPOINT_METADATA_PATH), "checkpoint metadata")
    required = {
        "kind": "eiem",
        "step": 2048,
        "tokens_seen": 33_554_432,
        "resume_authorized": False,
        "scientific_evaluation_authorized": True,
        "sha256": CHECKPOINT_SHA256,
        "bytes": CHECKPOINT_BYTES,
    }
    for key, value in required.items():
        if meta.get(key) != value:
            raise RuntimeError(f"#1017 checkpoint metadata drift at {key}: {meta.get(key)!r}")

    checkpoint = Path(SOURCE_CHECKPOINT_PATH)
    if not checkpoint.is_file() or int(checkpoint.stat().st_size) != CHECKPOINT_BYTES:
        raise RuntimeError("#1017 checkpoint file missing/size drift")

    return {
        "stage_c_stop_verified": True,
        "postmortem_payload_integration_weak_verified": True,
        "checkpoint_file_exists": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "diagnostic_result_namespace_unused": not Path(RESULT_ROOT).exists(),
    }


@app.function(image=image, cpu=1, memory=1024, timeout=10 * 60, retries=0, volumes={"/vol": volume})
def inspect_source(
    source_sha: str, source_tree: str, model_sha: str, evaluator_sha: str,
    payload_gate_sha: str, scientific_impl_sha: str, runner_sha: str, workflow_sha: str,
    dual_account_sha: str, dual_account_cli_sha: str,
    selected_modal_account: str, selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> str:
    b = _validate_bindings(
        source_sha, source_tree, model_sha, evaluator_sha, payload_gate_sha,
        scientific_impl_sha, runner_sha, workflow_sha, dual_account_sha, dual_account_cli_sha
    )
    volume.reload()
    state = _inspect_source()
    return json.dumps({
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_SOURCE_AUDIT",
        **_evidence(
            b, 0, selected_modal_account, selected_modal_workspace,
            account_selection_evidence_sha256
        ), **state,
        "gpu_allocated": False, "diagnostic_attempt_consumed": False, "trigger_authorized_by_runner": False,
    }, sort_keys=True)


@app.function(image=image, cpu=1, memory=1024, timeout=10 * 60, retries=0, volumes={"/vol": volume})
def verify_zero_gpu(
    source_sha: str, source_tree: str, model_sha: str, evaluator_sha: str,
    payload_gate_sha: str, scientific_impl_sha: str, runner_sha: str, workflow_sha: str,
    dual_account_sha: str, dual_account_cli_sha: str,
    selected_modal_account: str, selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    b = _validate_bindings(
        source_sha, source_tree, model_sha, evaluator_sha, payload_gate_sha,
        scientific_impl_sha, runner_sha, workflow_sha, dual_account_sha, dual_account_cli_sha
    )
    if int(authority_comment_id) <= 0:
        raise RuntimeError("#1017 final authority comment ID must be positive")
    volume.reload()
    state = _inspect_source()
    if not state["diagnostic_result_namespace_unused"]:
        raise RuntimeError("#1017 result namespace already exists")
    gate = {
        "status": "PASS",
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence(
            b, authority_comment_id, selected_modal_account, selected_modal_workspace,
            account_selection_evidence_sha256
        ), **state,
        "gpu_allocated": False, "diagnostic_attempt_consumed": False,
    }
    _atomic_write(Path(RESULT_ROOT) / "ZERO_GPU_GATE.json", gate)
    volume.commit()
    return json.dumps(gate, sort_keys=True)


@app.function(image=image, cpu=1, memory=512, timeout=5 * 60, retries=0, volumes={"/vol": volume})
def reserve_dispatch(
    source_sha: str, source_tree: str, model_sha: str, evaluator_sha: str,
    payload_gate_sha: str, scientific_impl_sha: str, runner_sha: str, workflow_sha: str,
    dual_account_sha: str, dual_account_cli_sha: str,
    selected_modal_account: str, selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    b = _validate_bindings(
        source_sha, source_tree, model_sha, evaluator_sha, payload_gate_sha,
        scientific_impl_sha, runner_sha, workflow_sha, dual_account_sha, dual_account_cli_sha
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    zero = root / "ZERO_GPU_GATE.json"
    reserve = root / "DISPATCH_RESERVED.json"
    consumed = root / "ATTEMPT_CONSUMED.json"
    result = root / "RESULT.json"
    failure = root / "ATTEMPT_FAILURE.json"
    if not zero.is_file():
        raise RuntimeError("#1017 zero-GPU gate missing")
    if reserve.exists() or consumed.exists() or result.exists() or failure.exists():
        raise RuntimeError("#1017 dispatch/attempt already exists")
    zero_payload = _read_json(zero, "zero-GPU gate")
    expected = _evidence(
            b, authority_comment_id, selected_modal_account, selected_modal_workspace,
            account_selection_evidence_sha256
        )
    for key, value in expected.items():
        if zero_payload.get(key) != value:
            raise RuntimeError(f"#1017 zero-GPU binding drift at {key}")
    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_DISPATCH_RESERVED",
        **expected,
        "reserved_unix": time.time(),
        "gpu_allocated": False,
        "diagnostic_attempt_consumed": False,
    }
    _atomic_write(reserve, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


@app.function(image=image, gpu=GPU_CLASS, cpu=CPU_CORES, memory=RAM_MIB, timeout=MAX_SECONDS, retries=0, volumes={"/vol": volume})
def run_diagnostic(
    source_sha: str, source_tree: str, model_sha: str, evaluator_sha: str,
    payload_gate_sha: str, scientific_impl_sha: str, runner_sha: str, workflow_sha: str,
    dual_account_sha: str, dual_account_cli_sha: str,
    selected_modal_account: str, selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int, live_hourly_resource_usd: float,
) -> str:
    import torch
    import tiktoken

    b = _validate_bindings(
        source_sha, source_tree, model_sha, evaluator_sha, payload_gate_sha,
        scientific_impl_sha, runner_sha, workflow_sha, dual_account_sha, dual_account_cli_sha
    )
    authority = int(authority_comment_id)
    hourly = float(live_hourly_resource_usd)
    if authority <= 0:
        raise RuntimeError("#1017 invalid authority")
    if hourly <= 0.0 or hourly * 2.0 > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError("#1017 live resource rate exceeds cap")

    volume.reload()
    root = Path(RESULT_ROOT)
    reserve = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not reserve.is_file():
        raise RuntimeError("#1017 durable reservation missing")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1017 attempt already consumed")

    evidence = _evidence(
        b, authority, selected_modal_account, selected_modal_workspace,
        account_selection_evidence_sha256
    )
    reserve_payload = _read_json(reserve, "dispatch reservation")
    for key, value in evidence.items():
        if reserve_payload.get(key) != value:
            raise RuntimeError(f"#1017 durable reservation binding drift at {key}")

    consumed = {
        "status": "DIAGNOSTIC_ATTEMPT_CONSUMED",
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_GPU_FUNCTION_BEGAN",
        **evidence,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "diagnostic_attempt_consumed": True,
        "new_scientific_seed_created": False,
        "source_scientific_seed_reused": False,
    }
    _atomic_write(consumed_path, consumed)
    volume.commit()

    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("#1017 CUDA unavailable")
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1017 expected L4, got {device_name!r}")

        checkpoint = Path(SOURCE_CHECKPOINT_PATH)
        if int(checkpoint.stat().st_size) != CHECKPOINT_BYTES:
            raise RuntimeError("#1017 checkpoint size changed")
        before_sha = _sha256_file(checkpoint)
        if before_sha != CHECKPOINT_SHA256:
            raise RuntimeError("#1017 checkpoint SHA changed before load")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        from tam_research.chm_v1_100m_scale import CHMV1100MEIEMLM
        from tam_research.chm_v1_100m_stage_c_eval import CASES_PER_FAMILY, LONG_RANGE_FAMILIES, PROBE_SEED, generate_aligned_probe_suite
        from tam_research.chm_v1_100m_stage_c_postmortem import validate_checkpoint_payload
        from tam_research.chm_v1_100m_stage_c_payload_gate_diagnostic import (
            classify_payload_gate_signals,
            gate_state_summary,
            payload_gate_probe_row,
            summarize_payload_gate_diagnostic,
            validate_protocol_manifest,
        )

        manifest = validate_protocol_manifest()
        validate_checkpoint_payload(payload, expected_kind="eiem")

        model = CHMV1100MEIEMLM()
        load = model.load_state_dict(payload["model_state_dict"], strict=True)
        if load.missing_keys or load.unexpected_keys:
            raise RuntimeError(f"#1017 strict checkpoint load drift: {load}")
        del payload
        model.eval().to(device)

        encoder = tiktoken.get_encoding("gpt2")
        probes = generate_aligned_probe_suite(lambda text: encoder.encode(text), seed=PROBE_SEED, cases_per_family=CASES_PER_FAMILY)
        probes = [p for p in probes if p.family in LONG_RANGE_FAMILIES]
        if len(probes) != 384:
            raise RuntimeError(f"#1017 expected 384 long-range probes, got {len(probes)}")

        gate_state = gate_state_summary(model)
        torch.cuda.reset_peak_memory_stats(device)
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for probe in probes:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    rows.append(payload_gate_probe_row(model, probe))

        summary = summarize_payload_gate_diagnostic(rows, gate_state)
        decision = classify_payload_gate_signals(summary)

        after_sha = _sha256_file(checkpoint)
        if after_sha != CHECKPOINT_SHA256 or int(checkpoint.stat().st_size) != CHECKPOINT_BYTES:
            raise RuntimeError("#1017 checkpoint changed during diagnostic")

        _atomic_write(root / "RAW_ROWS.json", rows)
        _atomic_write(root / "SUMMARY.json", summary)
        result = {
            "status": "COMPLETE",
            "classification": decision["classification"],
            "tags": decision["tags"],
            **evidence,
            "protocol_manifest": manifest,
            "probe_count": len(rows),
            "gate_state": gate_state,
            "summary": summary,
            "benefits": decision["benefits"],
            "source_checkpoint_sha256_before": before_sha,
            "source_checkpoint_sha256_after": after_sha,
            "source_checkpoint_unchanged": before_sha == after_sha == CHECKPOINT_SHA256,
            "wall_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "device_name": device_name,
            "live_hourly_resource_usd": hourly,
            "live_worst_case_usd": hourly * 2.0,
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
            "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_GATE_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
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


@app.function(image=image, cpu=1, memory=512, timeout=5 * 60, retries=0, volumes={"/vol": volume})
def inspect_state(
    source_sha: str, source_tree: str, model_sha: str, evaluator_sha: str,
    payload_gate_sha: str, scientific_impl_sha: str, runner_sha: str, workflow_sha: str,
    dual_account_sha: str, dual_account_cli_sha: str,
    selected_modal_account: str, selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    b = _validate_bindings(
        source_sha, source_tree, model_sha, evaluator_sha, payload_gate_sha,
        scientific_impl_sha, runner_sha, workflow_sha, dual_account_sha, dual_account_cli_sha
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    names = ("ZERO_GPU_GATE.json", "DISPATCH_RESERVED.json", "ATTEMPT_CONSUMED.json", "RESULT.json", "ATTEMPT_FAILURE.json", "SUMMARY.json")
    files = {}
    for name in names:
        path = root / name
        files[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return json.dumps({**_evidence(
            b, authority_comment_id, selected_modal_account, selected_modal_workspace,
            account_selection_evidence_sha256
        ), "root_exists": root.exists(), "files": files}, sort_keys=True)


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    payload_gate_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int = 0,
    live_hourly_resource_usd: float = 0.0,
) -> None:
    base = (
        source_sha, source_tree, model_sha, evaluator_sha, payload_gate_sha,
        scientific_impl_sha, runner_sha, workflow_sha, dual_account_sha,
        dual_account_cli_sha, selected_modal_account, selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    if phase == "inspect-source":
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_GATE_SOURCE={inspect_source.remote(*base)}")
        return
    common = (*base, int(authority_comment_id))
    if phase == "preflight":
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_GATE_ZERO_GPU={verify_zero_gpu.remote(*common)}")
        return
    if phase == "reserve":
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_GATE_DISPATCH={reserve_dispatch.remote(*common)}")
        return
    if phase == "run":
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_GATE_RESULT={run_diagnostic.remote(*common, float(live_hourly_resource_usd))}")
        return
    if phase == "state":
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_GATE_STATE={inspect_state.remote(*common)}")
        return
    raise ValueError(f"unknown phase: {phase}")
