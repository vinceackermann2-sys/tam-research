from __future__ import annotations

"""One-shot checkpoint-only CHM-v1 payload/integration decomposition v3 (#1064/#1076).

Reads only the frozen Stage-C checkpoint and authoritative #990/#1008 evidence.
Writes only beneath the dedicated #1064 result root. No training, parameter
updates, checkpoint mutation/resume, new seed, or Stage-D authority.
"""

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import modal

PHASE = "chm-v1-100m-stage-c-payload-integration-1064-v3"
CONTROL_ISSUE = 1064
PREREG_ISSUE = 1037
SOURCE_SCIENTIFIC_ISSUE = 990
SOURCE_POSTMORTEM_ISSUE = 1008
SOURCE_SCIENTIFIC_SEED = 977_001

TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-payload-integration-1064-v3]"
SOURCE_RESULT_ROOT = "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"
SOURCE_RESULT_PATH = f"{SOURCE_RESULT_ROOT}/RESULT.json"
SOURCE_CHECKPOINT_PATH = f"{SOURCE_RESULT_ROOT}/checkpoints/eiem/step-2048.pt"
SOURCE_CHECKPOINT_METADATA_PATH = f"{SOURCE_RESULT_ROOT}/checkpoints/eiem/step-2048.json"
SOURCE_POSTMORTEM_RESULT_PATH = "/vol/chm-v1/100m-stage-c-postmortem/issue-1008/v1/RESULT.json"
RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-integration/issue-1064/v3"

MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
DECOMPOSITION_BLOB = "8ab98658d4c8ced33623c2d28ec899b35339997e"
SCIENTIFIC_IMPLEMENTATION_BLOB = "fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"
DUAL_ACCOUNT_BLOB = "6e1cc39ad8ec87bb7fddccdfbba35eb77f8e7d44"
DUAL_ACCOUNT_CLI_BLOB = "df2d4536962ddb27370b5f5ec397c2e173c7c9ee"
CHECKPOINT_SHA256 = "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831"
CHECKPOINT_BYTES = 407_424_818

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 7_200
MAX_BILLED_COMPUTE_USD = 3.00

APP_NAME = "chm-v1-100m-stage-c-payload-integration-1064-v3"
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
        raise ValueError(f"{name} must be a full lowercase 40-hex SHA")
    return value


def _full_sha256(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a full lowercase 64-hex SHA-256")
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
    decomposition_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
) -> dict[str, str]:
    bindings = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "model_blob_sha": _full_sha(model_sha, "model_sha"),
        "evaluator_blob_sha": _full_sha(evaluator_sha, "evaluator_sha"),
        "decomposition_blob_sha": _full_sha(decomposition_sha, "decomposition_sha"),
        "scientific_implementation_blob_sha": _full_sha(
            scientific_impl_sha, "scientific_impl_sha"
        ),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
        "dual_account_blob_sha": _full_sha(dual_account_sha, "dual_account_sha"),
        "dual_account_cli_blob_sha": _full_sha(
            dual_account_cli_sha, "dual_account_cli_sha"
        ),
    }
    expected = {
        "model_blob_sha": MODEL_BLOB,
        "evaluator_blob_sha": EVALUATOR_BLOB,
        "decomposition_blob_sha": DECOMPOSITION_BLOB,
        "scientific_implementation_blob_sha": SCIENTIFIC_IMPLEMENTATION_BLOB,
        "dual_account_blob_sha": DUAL_ACCOUNT_BLOB,
        "dual_account_cli_blob_sha": DUAL_ACCOUNT_CLI_BLOB,
    }
    for key, value in expected.items():
        if bindings[key] != value:
            raise RuntimeError(f"#1064 immutable blob drift at {key}")
    if (CONTROL_ISSUE, PREREG_ISSUE, SOURCE_SCIENTIFIC_ISSUE, SOURCE_POSTMORTEM_ISSUE) != (
        1064,
        1037,
        990,
        1008,
    ):
        raise RuntimeError("#1064 issue binding drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 16 * 1024:
        raise RuntimeError("#1064 resource binding drift")
    if MAX_SECONDS != 7_200 or MAX_BILLED_COMPUTE_USD != 3.00:
        raise RuntimeError("#1064 runtime/cost binding drift")
    return bindings


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"#1064 missing {name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"#1064 {name} must be a JSON object")
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
    if stage_c.get("status") != "COMPLETE":
        raise RuntimeError("#1064 source Stage-C result is not COMPLETE")
    if stage_c.get("classification") != "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH":
        raise RuntimeError("#1064 source Stage-C classification drift")
    if int(stage_c.get("scientific_seed", -1)) != SOURCE_SCIENTIFIC_SEED:
        raise RuntimeError("#1064 historical seed provenance drift")
    if stage_c.get("scientific_seed_consumed") is not True:
        raise RuntimeError("#1064 source Stage-C result does not record seed consumption")
    if stage_c.get("automatic_retry_authorized") is not False:
        raise RuntimeError("#1064 source Stage-C retry semantics drift")
    if stage_c.get("checkpoint_resume_authorized") is not False:
        raise RuntimeError("#1064 source Stage-C resume semantics drift")
    if stage_c.get("stage_d_automatically_authorized") is not False:
        raise RuntimeError("#1064 source Stage-C Stage-D authority drift")

    first = _read_json(Path(SOURCE_POSTMORTEM_RESULT_PATH), "#1008 RESULT.json")
    if first.get("status") != "COMPLETE":
        raise RuntimeError("#1064 #1008 result is not COMPLETE")
    if first.get("classification") != "CHM_V1_100M_STAGE_C_POSTMORTEM_DIAGNOSTIC_ONLY":
        raise RuntimeError("#1064 #1008 classification drift")
    if first.get("tags") != ["PAYLOAD_INTEGRATION_WEAK_SIGNAL"]:
        raise RuntimeError(f"#1064 #1008 tag drift: {first.get('tags')!r}")
    if first.get("source_checkpoint_unchanged") is not True:
        raise RuntimeError("#1064 #1008 did not preserve checkpoint")

    metadata = _read_json(
        Path(SOURCE_CHECKPOINT_METADATA_PATH), "EIEM final checkpoint metadata"
    )
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
        if metadata.get(key) != value:
            raise RuntimeError(
                f"#1064 checkpoint metadata drift at {key}: {metadata.get(key)!r}"
            )
    checkpoint = Path(SOURCE_CHECKPOINT_PATH)
    if not checkpoint.is_file() or int(checkpoint.stat().st_size) != CHECKPOINT_BYTES:
        raise RuntimeError("#1064 checkpoint file missing/size drift")

    return {
        "stage_c_stop_verified": True,
        "first_postmortem_payload_integration_weak_verified": True,
        "decomposition_preregistration_issue": PREREG_ISSUE,
        "checkpoint_file_exists": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "diagnostic_result_namespace_unused": not Path(RESULT_ROOT).exists(),
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
    decomposition_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        decomposition_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
        dual_account_sha,
        dual_account_cli_sha,
    )
    volume.reload()
    state = _inspect_source()
    return json.dumps(
        {
            "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_SOURCE_AUDIT",
            **_evidence(
                bindings,
                0,
                selected_modal_account,
                selected_modal_workspace,
                account_selection_evidence_sha256,
            ),
            **state,
            "gpu_allocated": False,
            "diagnostic_attempt_consumed": False,
            "trigger_authorized_by_runner": False,
        },
        sort_keys=True,
    )


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
    decomposition_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        decomposition_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
        dual_account_sha,
        dual_account_cli_sha,
    )
    if int(authority_comment_id) <= 0:
        raise RuntimeError("#1064 final authority comment ID must be positive")
    volume.reload()
    state = _inspect_source()
    if not state["diagnostic_result_namespace_unused"]:
        raise RuntimeError("#1064 diagnostic result namespace already exists")
    gate = {
        "status": "PASS",
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence(
            bindings,
            authority_comment_id,
            selected_modal_account,
            selected_modal_workspace,
            account_selection_evidence_sha256,
        ),
        **state,
        "gpu_allocated": False,
        "diagnostic_attempt_consumed": False,
    }
    _atomic_write(Path(RESULT_ROOT) / "ZERO_GPU_GATE.json", gate)
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
    decomposition_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        decomposition_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
        dual_account_sha,
        dual_account_cli_sha,
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file():
        raise RuntimeError("#1064 zero-GPU gate missing")
    if (
        reserve_path.exists()
        or consumed_path.exists()
        or result_path.exists()
        or failure_path.exists()
    ):
        raise RuntimeError("#1064 dispatch/attempt already exists")

    expected = _evidence(
        bindings,
        authority_comment_id,
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    zero = _read_json(zero_path, "zero-GPU gate")
    for key, value in expected.items():
        if zero.get(key) != value:
            raise RuntimeError(f"#1064 zero-GPU binding drift at {key}")
    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_DISPATCH_RESERVED",
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
    decomposition_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
    live_hourly_resource_usd: float,
) -> str:
    import torch
    import tiktoken

    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        decomposition_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
        dual_account_sha,
        dual_account_cli_sha,
    )
    authority = int(authority_comment_id)
    hourly = float(live_hourly_resource_usd)
    if authority <= 0:
        raise RuntimeError("#1064 invalid authority")
    if hourly <= 0.0 or hourly * 2.0 > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError("#1064 live resource rate exceeds cap")

    volume.reload()
    root = Path(RESULT_ROOT)
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not reserve_path.is_file():
        raise RuntimeError("#1064 durable reservation missing")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1064 diagnostic attempt already consumed")

    evidence = _evidence(
        bindings,
        authority,
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    reserve = _read_json(reserve_path, "dispatch reservation")
    for key, value in evidence.items():
        if reserve.get(key) != value:
            raise RuntimeError(f"#1064 durable reservation binding drift at {key}")

    consumed = {
        "status": "DIAGNOSTIC_ATTEMPT_CONSUMED",
        "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_GPU_FUNCTION_BEGAN",
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
            raise RuntimeError("#1064 CUDA unavailable")
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1064 expected L4, got {device_name!r}")

        checkpoint = Path(SOURCE_CHECKPOINT_PATH)
        if int(checkpoint.stat().st_size) != CHECKPOINT_BYTES:
            raise RuntimeError("#1064 checkpoint size changed")
        before_sha = _sha256_file(checkpoint)
        if before_sha != CHECKPOINT_SHA256:
            raise RuntimeError("#1064 checkpoint SHA changed before load")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        from tam_research.chm_v1_100m_scale import CHMV1100MEIEMLM
        from tam_research.chm_v1_100m_stage_c_eval import (
            CASES_PER_FAMILY,
            PROBE_SEED,
            TOTAL_PROBES,
            generate_aligned_probe_suite,
        )
        from tam_research.chm_v1_100m_stage_c_postmortem import (
            validate_checkpoint_payload,
        )
        from tam_research.chm_v1_100m_stage_c_payload_integration import (
            classify_payload_integration,
            gate_statistics,
            payload_integration_probe_row,
            summarize_payload_integration,
            validate_protocol_manifest,
        )

        manifest = validate_protocol_manifest()
        validate_checkpoint_payload(payload, expected_kind="eiem")

        model = CHMV1100MEIEMLM()
        load = model.load_state_dict(payload["model_state_dict"], strict=True)
        if load.missing_keys or load.unexpected_keys:
            raise RuntimeError(f"#1064 strict checkpoint load drift: {load}")
        del payload
        model.eval().to(device)

        encoder = tiktoken.get_encoding("gpt2")
        probes = generate_aligned_probe_suite(
            lambda text: encoder.encode(text),
            seed=PROBE_SEED,
            cases_per_family=CASES_PER_FAMILY,
        )
        if len(probes) != TOTAL_PROBES or len(probes) != 512:
            raise RuntimeError(f"#1064 expected 512 frozen probes, got {len(probes)}")

        gate_stats = gate_statistics(model)
        torch.cuda.reset_peak_memory_stats(device)
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for probe in probes:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    rows.append(payload_integration_probe_row(model, probe))

        summary = summarize_payload_integration(rows, gate_stats)
        decision = classify_payload_integration(summary)

        after_sha = _sha256_file(checkpoint)
        if after_sha != CHECKPOINT_SHA256 or int(checkpoint.stat().st_size) != CHECKPOINT_BYTES:
            raise RuntimeError("#1064 checkpoint changed during diagnostic")

        _atomic_write(root / "RAW_ROWS.json", rows)
        _atomic_write(root / "SUMMARY.json", summary)
        result = {
            "status": "COMPLETE",
            "classification": decision["classification"],
            "tags": decision["tags"],
            **evidence,
            "protocol_manifest": manifest,
            "probe_count": len(rows),
            "gate_statistics": gate_stats,
            "summary": summary,
            "aggregate": summary["aggregate"],
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
            "stage_c_result_changed": decision["stage_c_result_changed"],
            "first_postmortem_result_changed": decision[
                "first_postmortem_result_changed"
            ],
            "new_training_authorized": decision["new_training_authorized"],
            "new_scientific_seed_authorized": decision[
                "new_scientific_seed_authorized"
            ],
            "stage_d_authorized": decision["stage_d_authorized"],
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
        }
        _atomic_write(result_path, result)
        volume.commit()
        return json.dumps(result, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "FAILED",
            "classification": "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
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
    decomposition_sha: str,
    scientific_impl_sha: str,
    runner_sha: str,
    workflow_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_bindings(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        decomposition_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
        dual_account_sha,
        dual_account_cli_sha,
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
        files[name] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    return json.dumps(
        {
            **_evidence(
                bindings,
                authority_comment_id,
                selected_modal_account,
                selected_modal_workspace,
                account_selection_evidence_sha256,
            ),
            "root_exists": root.exists(),
            "files": files,
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    decomposition_sha: str,
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
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        decomposition_sha,
        scientific_impl_sha,
        runner_sha,
        workflow_sha,
        dual_account_sha,
        dual_account_cli_sha,
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    if phase == "inspect-source":
        payload = inspect_source.remote(*base)
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_SOURCE={payload}")
        return

    common = (*base, int(authority_comment_id))
    if phase == "preflight":
        payload = verify_zero_gpu.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_diagnostic.remote(*common, float(live_hourly_resource_usd))
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_RESULT={payload}")
        return
    if phase == "state":
        payload = inspect_state.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_STATE={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
