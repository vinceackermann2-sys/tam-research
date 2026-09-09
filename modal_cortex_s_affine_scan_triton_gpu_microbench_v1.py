from __future__ import annotations

import json
from pathlib import Path
import time

import modal


APP_NAME = "cortex-s-affine-scan-triton-gpu-microbench-v1"
VOLUME_NAME = "tam-research-data"
RESULT_ROOT = "/vol/cortex-s-v0/affine-scan-triton-gpu-microbench-v1"
ENGINEERING_SEED = 2_026_090_907
H100_TIMEOUT_SECONDS = 10 * 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11")
    .add_local_python_source("architectures")
)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={"/vol": volume},
)
def verify_zero_gpu(source_sha: str) -> dict:
    from architectures.cortex_s.experiments.affine_scan_triton_gpu_microbench_v1 import (
        ENGINEERING_SEED as REMOTE_SEED,
        validate_protocol,
    )

    if REMOTE_SEED != ENGINEERING_SEED:
        raise RuntimeError("launcher/core engineering seed drift")
    protocol = validate_protocol()
    if protocol.get("gpu_dispatch_authorized") is not False:
        raise RuntimeError("core protocol unexpectedly self-authorizes GPU dispatch")
    if protocol.get("production_integration_authorized") is not False:
        raise RuntimeError("core protocol unexpectedly authorizes production integration")
    if protocol.get("full_training_authorized") is not False:
        raise RuntimeError("core protocol unexpectedly authorizes full training")

    volume.reload()
    root = Path(RESULT_ROOT)
    gate_path = root / "ZERO_GPU_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if gate_path.exists() or dispatch_path.exists() or result_path.exists():
        raise RuntimeError("scan microbenchmark namespace is already consumed")

    result = {
        "status": "PASS",
        "classification": "ZERO_GPU_PROTOCOL_GATE_ONLY",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "protocol": protocol,
        "gpu_allocated": False,
        "production_integration_authorized": False,
        "full_training_authorized": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"zero_gpu_gate": result}, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=16384,
    timeout=H100_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
)
def h100_scan_microbenchmark(source_sha: str) -> dict:
    from architectures.cortex_s.experiments.affine_scan_triton_gpu_microbench_v1 import (
        ENGINEERING_SEED as REMOTE_SEED,
        run_h100_scan_microbenchmark,
    )

    if REMOTE_SEED != ENGINEERING_SEED:
        raise RuntimeError("launcher/core engineering seed drift")

    volume.reload()
    root = Path(RESULT_ROOT)
    gate_path = root / "ZERO_GPU_GATE.json"
    dispatch_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not gate_path.exists():
        raise RuntimeError("zero-GPU gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("source_sha") != source_sha:
        raise RuntimeError("zero-GPU gate does not bind this exact source SHA")
    if dispatch_path.exists() or result_path.exists():
        raise RuntimeError("scan microbenchmark namespace is already consumed")

    dispatch = {
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "dispatched_unix": time.time(),
        "scientific_seed_used": False,
        "production_integration_authorized": False,
        "full_training_authorized": False,
    }
    dispatch_path.write_text(json.dumps(dispatch, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"h100_dispatch": dispatch}, indent=2), flush=True)

    try:
        result = run_h100_scan_microbenchmark(source_sha=source_sha)
    except Exception as exc:
        failure = {
            "status": "ENGINEERING_RUNTIME_FAIL",
            "scientific_status": "ENGINEERING_GPU_CORRECTNESS_AND_MICROBENCH_ONLY",
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "error": f"{type(exc).__name__}: {exc}",
            "production_integration_authorized": False,
            "full_training_authorized": False,
        }
        result_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        volume.commit()
        print(json.dumps({"failure": failure}, indent=2), flush=True)
        raise

    result = dict(result)
    result["production_integration_authorized"] = False
    result["full_training_authorized"] = False
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"result": result}, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(phase: str = "gpu-microbench-v1", source_sha: str = ""):
    if phase.strip().lower() != "gpu-microbench-v1":
        raise ValueError("only phase='gpu-microbench-v1' is supported")
    source_sha = source_sha.strip().lower()
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("source_sha must be a full lowercase/hex commit SHA")
    gate = verify_zero_gpu.remote(source_sha)
    print(json.dumps({"zero_gpu": gate}, indent=2), flush=True)
    result = h100_scan_microbenchmark.remote(source_sha)
    print(json.dumps({"h100": result}, indent=2), flush=True)
