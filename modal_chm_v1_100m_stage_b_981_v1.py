from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import modal

from tam_research.chm_v1_100m_stage_b import (
    CPU_CORES,
    GPU_CLASS,
    MAX_GPU_SECONDS,
    RAM_GIB,
    SYSTEMS_SEED,
)

PHASE = "chm-v1-100m-stage-b-981-v1"
RESEARCH_ISSUE = 977
RUN_CONTROL_ISSUE = 981
TRIGGER_TITLE = "[modal-chm-v1-100m-stage-b-981-v1]"
RESULT_ROOT = "/vol/chm-v1/100m-stage-b/issue-981/seed-981001-v1"
APP_NAME = "chm-v1-100m-stage-b-981-v1"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2,<3")
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _full_sha(value: str, name: str) -> str:
    value = value.strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be full lowercase 40-hex SHA")
    return value


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _validate(source_sha: str, source_tree: str, harness_sha: str) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    if SYSTEMS_SEED != 981_001:
        raise RuntimeError("#981 systems seed drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_GIB != 16 or MAX_GPU_SECONDS != 900:
        raise RuntimeError("#981 resource envelope drift")
    return source, tree, harness


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=RAM_GIB * 1024,
    timeout=5 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def zero_gpu_preflight(source_sha: str, source_tree: str, harness_sha: str) -> str:
    from tam_research.chm_v1_100m_stage_b import static_preflight, validate_systems_seed

    source, tree, harness = _validate(source_sha, source_tree, harness_sha)
    validate_systems_seed(SYSTEMS_SEED, protocol=True)
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#981 result namespace already exists; fail closed")
    payload = {
        "status": "PASS",
        "classification": "CHM_V1_100M_STAGE_B_ZERO_GPU_PREFLIGHT_PASS",
        "phase": PHASE,
        "research_issue": RESEARCH_ISSUE,
        "run_control_issue": RUN_CONTROL_ISSUE,
        "trigger_title": TRIGGER_TITLE,
        "systems_seed": SYSTEMS_SEED,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "static_preflight": static_preflight(),
        "gpu_allocated": False,
        "systems_seed_consumed": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def reserve_dispatch(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "SYSTEMS_SEED_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file():
        raise RuntimeError("#981 zero-GPU gate missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PASS",
        "systems_seed": SYSTEMS_SEED,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
    }.items():
        if zero.get(key) != expected:
            raise RuntimeError(f"zero-GPU evidence mismatch for {key}")
    if marker_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#981 trigger/result namespace already used")
    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_100M_STAGE_B_PRE_ALLOCATION_MARKER",
        "systems_seed": SYSTEMS_SEED,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "marked_unix": time.time(),
        "gpu_allocation_started": False,
        "systems_seed_consumed": False,
        "automatic_retry_authorized": False,
    }
    _atomic_write(marker_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=CPU_CORES,
    memory=RAM_GIB * 1024,
    timeout=MAX_GPU_SECONDS,
    retries=0,
    volumes={"/vol": volume},
)
def run_stage_b(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    live_hourly_resource_cost_usd: float,
) -> str:
    import torch

    from tam_research.chm_v1_100m_scale import (
        EXPECTED_EIEM_PARAMETERS,
        EXPECTED_LOCAL_PARAMETERS,
        PARAMETER_MISMATCH_LIMIT,
    )
    from tam_research.chm_v1_100m_stage_b import (
        benchmark_training,
        build_stage_b_pair,
        classify_stage_b,
        practical_inference_audit,
        project_screen,
        validate_systems_seed,
    )
    from tam_research.models import parameter_count

    source, tree, harness = _validate(source_sha, source_tree, harness_sha)
    validate_systems_seed(SYSTEMS_SEED, protocol=True)
    if not torch.cuda.is_available():
        raise RuntimeError("#981 L4 allocation started without CUDA")
    if not (0.0 < float(live_hourly_resource_cost_usd) <= 1.4):
        raise RuntimeError("invalid/fail-open live resource hourly rate")

    device = torch.device("cuda")
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "SYSTEMS_SEED_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file() or not marker_path.is_file():
        raise RuntimeError("pre-allocation #981 evidence incomplete")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#981 systems seed already consumed")

    consumed = {
        "status": "SYSTEMS_SEED_CONSUMED",
        "classification": "CHM_V1_100M_STAGE_B_L4_ALLOCATION_STARTED",
        "systems_seed": SYSTEMS_SEED,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "automatic_retry_authorized": False,
        "scientific_credit": False,
    }
    _atomic_write(consumed_path, consumed)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["gpu_allocation_started"] = True
    marker["systems_seed_consumed"] = True
    _atomic_write(marker_path, marker)
    volume.commit()

    try:
        local, eiem = build_stage_b_pair(seed=SYSTEMS_SEED, device=device)
        local_count = parameter_count(local)
        eiem_count = parameter_count(eiem)
        accounting = {
            "local_trainable_parameters": local_count,
            "eiem_trainable_parameters": eiem_count,
            "delta_fraction": abs(eiem_count - local_count) / local_count,
            "expected_local": EXPECTED_LOCAL_PARAMETERS,
            "expected_eiem": EXPECTED_EIEM_PARAMETERS,
            "mismatch_limit": PARAMETER_MISMATCH_LIMIT,
        }

        local_training = benchmark_training(
            "local", local, device=device, seed=SYSTEMS_SEED
        )
        # Release optimizer/temporary tensors from the local loop before EIEM.
        torch.cuda.empty_cache()
        eiem_training = benchmark_training(
            "eiem", eiem, device=device, seed=SYSTEMS_SEED
        )
        torch.cuda.empty_cache()
        inference = practical_inference_audit(eiem, device=device, seed=SYSTEMS_SEED)
        projection = project_screen(
            local_training,
            eiem_training,
            live_hourly_resource_cost_usd=float(live_hourly_resource_cost_usd),
        )
        result: dict[str, Any] = {
            "status": "COMPLETE",
            "phase": PHASE,
            "research_issue": RESEARCH_ISSUE,
            "run_control_issue": RUN_CONTROL_ISSUE,
            "systems_seed": SYSTEMS_SEED,
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "gpu": torch.cuda.get_device_name(device),
            "cuda_available": True,
            "live_hourly_resource_cost_usd": float(live_hourly_resource_cost_usd),
            "parameter_accounting": accounting,
            "training": {"local": local_training, "eiem": eiem_training},
            "inference": inference,
            "projection": projection,
            "systems_seed_consumed": True,
            "automatic_retry_authorized": False,
            "scientific_credit": False,
            "stage_c_authorized_automatically": False,
        }
        result["classification"] = classify_stage_b(result)
        _atomic_write(result_path, result)
        volume.commit()
        return json.dumps(result, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "ATTEMPT_FAILURE",
            "classification": "CHM_V1_100M_STAGE_B_INFRA_OR_RUNTIME_FAILURE_NO_SCIENTIFIC_EVIDENCE",
            "systems_seed": SYSTEMS_SEED,
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "systems_seed_consumed": True,
            "automatic_retry_authorized": False,
            "scientific_credit": False,
        }
        _atomic_write(failure_path, failure)
        volume.commit()
        raise


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    live_hourly_resource_cost_usd: float = 0.0,
) -> None:
    if phase == "preflight":
        value = zero_gpu_preflight.remote(source_sha, source_tree, harness_sha)
        print("CHM_V1_100M_STAGE_B_ZERO_GPU=" + value)
    elif phase == "reserve":
        value = reserve_dispatch.remote(source_sha, source_tree, harness_sha)
        print("CHM_V1_100M_STAGE_B_DISPATCH=" + value)
    elif phase == "run":
        value = run_stage_b.remote(
            source_sha,
            source_tree,
            harness_sha,
            live_hourly_resource_cost_usd,
        )
        print("CHM_V1_100M_STAGE_B_RESULT=" + value)
    else:
        raise ValueError(f"unknown phase: {phase}")
