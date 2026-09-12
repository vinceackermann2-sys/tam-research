from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

PHASE = "preflight-v10-durable-result-readout-v1"
TRIGGER_TITLE = "[modal-cortex-s-v10-durable-result-readout-v1]"
ORIGINAL_PHASE = "preflight-v10-autocast-ce-no-explicit-fp32"
ORIGINAL_TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v10-autocast-ce-no-explicit-fp32]"
ORIGINAL_SOURCE_SHA = "6e2b535a2eb8fdea6579fc27849091ee5125b9ea"
ORIGINAL_SOURCE_TREE = "2a019c7d5785e9c4b7e48eda15c9b6faec30e8f3"
ORIGINAL_HARNESS_SHA = "48c5d5a2ee7ff3688785f8e7943d537e5215358b"
ORIGINAL_ENGINEERING_SEED = 2_026_091_012
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v10-autocast-ce-no-explicit-fp32"
VOLUME_NAME = "tam-research-data"

APP_NAME = "cortex-s-v10-durable-result-readout-v1"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
base_image = modal.Image.debian_slim(python_version="3.11")


def _full_sha(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase SHA")
    return normalized


def _validate_request(
    phase: str,
    original_source_sha: str,
    original_source_tree: str,
    original_harness_sha: str,
) -> None:
    if phase != PHASE:
        raise ValueError(f"phase must be exactly {PHASE}")
    if _full_sha(original_source_sha, "original_source_sha") != ORIGINAL_SOURCE_SHA:
        raise RuntimeError("original source SHA drift")
    if _full_sha(original_source_tree, "original_source_tree") != ORIGINAL_SOURCE_TREE:
        raise RuntimeError("original source tree drift")
    if _full_sha(original_harness_sha, "original_harness_sha") != ORIGINAL_HARNESS_SHA:
        raise RuntimeError("original harness SHA drift")


def _load_required(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"required immutable {label} is missing: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"immutable {label} must be a JSON object")
    return payload, raw


def _load_optional(path: Path) -> tuple[dict[str, Any] | None, bytes | None]:
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("immutable RESULT.json must be a JSON object")
    return payload, raw


def _validate_identity(payload: dict[str, Any], label: str) -> None:
    expected = {
        "phase": ORIGINAL_PHASE,
        "source_sha": ORIGINAL_SOURCE_SHA,
        "source_tree": ORIGINAL_SOURCE_TREE,
        "harness_sha": ORIGINAL_HARNESS_SHA,
        "result_root": RESULT_ROOT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{label} identity mismatch for {key}: {payload.get(key)!r} != {value!r}")


def _validate_records(
    zero_gpu: dict[str, Any],
    dispatch: dict[str, Any],
    result: dict[str, Any] | None,
) -> None:
    _validate_identity(zero_gpu, "ZERO_GPU_GATE.json")
    _validate_identity(dispatch, "H100_DISPATCH_CONSUMED.json")

    if zero_gpu.get("status") != "PASS":
        raise RuntimeError("zero-GPU gate is not PASS")
    if zero_gpu.get("classification") != "ZERO_GPU_AUTOCAST_CE_NO_EXPLICIT_FP32_PREFLIGHT_V10":
        raise RuntimeError("zero-GPU gate classification drift")
    if zero_gpu.get("engineering_seed_reserved") != ORIGINAL_ENGINEERING_SEED:
        raise RuntimeError("zero-GPU reserved engineering seed drift")
    if zero_gpu.get("gpu_allocated") is not False:
        raise RuntimeError("zero-GPU gate incorrectly records GPU allocation")
    if zero_gpu.get("full_training_authorized") is not False or zero_gpu.get("scientific_claim_authorized") is not False:
        raise RuntimeError("zero-GPU gate authority escalation")

    if dispatch.get("status") != "H100_DISPATCH_CONSUMED":
        raise RuntimeError("dispatch marker status drift")
    if dispatch.get("classification") != "ENGINEERING_SINGLE_ATTEMPT_MARKER_V10_AUTOCAST_CE_NO_EXPLICIT_FP32":
        raise RuntimeError("dispatch marker classification drift")
    if dispatch.get("engineering_seed") != ORIGINAL_ENGINEERING_SEED:
        raise RuntimeError("dispatch marker engineering seed drift")
    if dispatch.get("full_training_authorized") is not False or dispatch.get("scientific_claim_authorized") is not False:
        raise RuntimeError("dispatch marker authority escalation")

    if result is not None:
        _validate_identity(result, "RESULT.json")
        if result.get("full_training_authorized") is not False:
            raise RuntimeError("durable result unexpectedly authorizes full training")
        if result.get("scientific_claim_authorized") is not False:
            raise RuntimeError("durable result unexpectedly authorizes a scientific claim")


@app.function(image=base_image, cpu=1, memory=512, timeout=5 * 60, volumes={"/vol": volume})
def read_existing_result(
    phase: str,
    original_source_sha: str,
    original_source_tree: str,
    original_harness_sha: str,
) -> str:
    _validate_request(phase, original_source_sha, original_source_tree, original_harness_sha)
    volume.reload()

    root = Path(RESULT_ROOT)
    zero_gpu, zero_raw = _load_required(root / "ZERO_GPU_GATE.json", "ZERO_GPU_GATE.json")
    dispatch, dispatch_raw = _load_required(root / "H100_DISPATCH_CONSUMED.json", "H100_DISPATCH_CONSUMED.json")
    result, result_raw = _load_optional(root / "RESULT.json")
    _validate_records(zero_gpu, dispatch, result)

    payload = {
        "status": "READOUT_COMPLETE",
        "classification": "READ_ONLY_V10_DURABLE_RESULT_RECOVERY",
        "phase": PHASE,
        "volume_name": VOLUME_NAME,
        "result_root": RESULT_ROOT,
        "original_attempt": {
            "phase": ORIGINAL_PHASE,
            "trigger_title": ORIGINAL_TRIGGER_TITLE,
            "source_sha": ORIGINAL_SOURCE_SHA,
            "source_tree": ORIGINAL_SOURCE_TREE,
            "harness_sha": ORIGINAL_HARNESS_SHA,
            "engineering_seed": ORIGINAL_ENGINEERING_SEED,
        },
        "sha256": {
            "zero_gpu_gate": hashlib.sha256(zero_raw).hexdigest(),
            "dispatch_marker": hashlib.sha256(dispatch_raw).hexdigest(),
            "result": hashlib.sha256(result_raw).hexdigest() if result_raw is not None else None,
        },
        "zero_gpu_gate": zero_gpu,
        "dispatch_marker": dispatch,
        "result_present": result is not None,
        "result": result,
        "read_only": True,
        "gpu_allocated_by_readout": False,
        "new_seed_consumed": False,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@app.local_entrypoint()
def main(
    phase: str,
    original_source_sha: str,
    original_source_tree: str,
    original_harness_sha: str,
) -> None:
    payload = read_existing_result.remote(
        phase,
        original_source_sha,
        original_source_tree,
        original_harness_sha,
    )
    print(f"CORTEX_S_V10_DURABLE_READOUT={payload}")
