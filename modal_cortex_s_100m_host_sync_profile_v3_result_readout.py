from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

PHASE = "host-sync-profile-v3-result-readout-v1"
TRIGGER_TITLE = "[modal-cortex-s-host-sync-profile-v3-result-readout-v1]"
ORIGINAL_PHASE = "host-sync-profile-v3"
ORIGINAL_TRIGGER_TITLE = "[modal-cortex-s-100m-host-sync-profile-v3]"
ORIGINAL_SOURCE_SHA = "22ef4aac4735128219c720c839e6978c424114dc"
ORIGINAL_SOURCE_TREE = "0e89d152aba053b45dcf2c07b878781048f8c23e"
ORIGINAL_HARNESS_SHA = "13b3c3f8cc0f3d3cd60376099a945d6984cb320f"
ORIGINAL_ENGINEERING_SEED = 2_026_091_007
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/host-sync-profile-v3"
VOLUME_NAME = "tam-research-data"

APP_NAME = "cortex-s-v0-100m-host-sync-profile-v3-result-readout-v1"
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
) -> tuple[str, str, str]:
    if phase != PHASE:
        raise ValueError(f"phase must be exactly {PHASE}")
    source = _full_sha(original_source_sha, "original_source_sha")
    tree = _full_sha(original_source_tree, "original_source_tree")
    harness = _full_sha(original_harness_sha, "original_harness_sha")
    if source != ORIGINAL_SOURCE_SHA:
        raise RuntimeError("original source SHA drift")
    if tree != ORIGINAL_SOURCE_TREE:
        raise RuntimeError("original source tree drift")
    if harness != ORIGINAL_HARNESS_SHA:
        raise RuntimeError("original harness SHA drift")
    return source, tree, harness


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"required immutable {label} is missing: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"immutable {label} must be a JSON object")
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
    result: dict[str, Any],
) -> None:
    for label, payload in (("ZERO_GPU_GATE.json", zero_gpu), ("H100_DISPATCH_CONSUMED.json", dispatch), ("RESULT.json", result)):
        _validate_identity(payload, label)

    if zero_gpu.get("status") != "PASS":
        raise RuntimeError("zero-GPU gate is not PASS")
    if zero_gpu.get("classification") != "ZERO_GPU_HOST_SYNC_ATTRIBUTION_V3_GATE":
        raise RuntimeError("zero-GPU gate classification drift")
    if zero_gpu.get("engineering_seed_reserved") != ORIGINAL_ENGINEERING_SEED:
        raise RuntimeError("zero-GPU gate engineering seed drift")
    if zero_gpu.get("gpu_allocated") is not False:
        raise RuntimeError("zero-GPU gate incorrectly records GPU allocation")

    if dispatch.get("status") != "H100_DISPATCH_CONSUMED":
        raise RuntimeError("dispatch marker status drift")
    if dispatch.get("engineering_seed") != ORIGINAL_ENGINEERING_SEED:
        raise RuntimeError("dispatch marker engineering seed drift")
    if dispatch.get("h100_allocation_started") is not True:
        raise RuntimeError("completed v3 result exists without a recorded H100 allocation start")

    if result.get("status") != "PROFILE_COMPLETE":
        raise RuntimeError("durable v3 result is not PROFILE_COMPLETE")
    if result.get("engineering_seed") != ORIGINAL_ENGINEERING_SEED:
        raise RuntimeError("durable v3 result engineering seed drift")
    if result.get("seed_8100_used") is not False:
        raise RuntimeError("durable v3 result unexpectedly records seed 8100 use")
    if result.get("reserved_scientific_seeds_used") is not False:
        raise RuntimeError("durable v3 result unexpectedly records scientific seed use")
    if result.get("full_training_authorized") is not False:
        raise RuntimeError("durable v3 result unexpectedly authorizes full training")
    if result.get("scientific_claim_authorized") is not False:
        raise RuntimeError("durable v3 result unexpectedly authorizes a scientific claim")

    profile = result.get("profile")
    if not isinstance(profile, dict):
        raise RuntimeError("durable v3 result profile is missing")
    profiler = profile.get("profiler")
    if not isinstance(profiler, dict):
        raise RuntimeError("durable v3 profiler payload is missing")
    attribution = profiler.get("attribution")
    if not isinstance(attribution, dict):
        raise RuntimeError("durable v3 attribution payload is missing")
    if attribution.get("classification") != result.get("classification"):
        raise RuntimeError("top-level and attribution classifications disagree")


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
    zero_gpu, zero_raw = _load_json(root / "ZERO_GPU_GATE.json", "ZERO_GPU_GATE.json")
    dispatch, dispatch_raw = _load_json(root / "H100_DISPATCH_CONSUMED.json", "H100_DISPATCH_CONSUMED.json")
    result, result_raw = _load_json(root / "RESULT.json", "RESULT.json")
    _validate_records(zero_gpu, dispatch, result)

    payload = {
        "status": "READOUT_COMPLETE",
        "classification": "READ_ONLY_DURABLE_RESULT_RECOVERY",
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
            "result": hashlib.sha256(result_raw).hexdigest(),
        },
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
    print(f"CORTEX_S_V3_RESULT_READOUT={payload}")
