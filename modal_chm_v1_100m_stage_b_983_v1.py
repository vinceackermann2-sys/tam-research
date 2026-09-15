from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import modal

PHASE = "chm-v1-100m-stage-b-983-v1"
TRIGGER_TITLE = "[modal-chm-v1-100m-stage-b-983-v1]"
PARENT_RESEARCH_ISSUE = 977
SYSTEMS_ISSUE = 983
ENGINEERING_SEED = 977_201
RESERVED_SCIENTIFIC_SEED = 977_001
RESULT_ROOT = "/vol/chm-v1/100m-stage-b/issue-983/seed-977201-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 1_200
MAX_BILLED_COMPUTE_USD = 0.50

APP_NAME = "chm-v1-100m-stage-b-983-v1"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2,<3", "tiktoken>=0.9,<1")
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _full_sha(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 40-hex SHA")
    return normalized


def _validate_source(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    workflow_sha: str,
) -> tuple[str, str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    workflow = _full_sha(workflow_sha, "workflow_sha")
    if ENGINEERING_SEED != 977_201 or RESERVED_SCIENTIFIC_SEED != 977_001:
        raise RuntimeError("#983 engineering/scientific seed binding drift")
    if PARENT_RESEARCH_ISSUE != 977 or SYSTEMS_ISSUE != 983:
        raise RuntimeError("#977/#983 issue binding drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 16 * 1024:
        raise RuntimeError("#983 resource binding drift")
    if MAX_SECONDS != 1_200 or MAX_BILLED_COMPUTE_USD != 0.50:
        raise RuntimeError("#983 runtime/cost binding drift")
    return source, tree, harness, workflow


def _evidence_binding(
    source: str,
    tree: str,
    harness: str,
    workflow: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "parent_research_issue": PARENT_RESEARCH_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "engineering_seed": ENGINEERING_SEED,
        "reserved_scientific_seed_not_authorized": RESERVED_SCIENTIFIC_SEED,
        "execution_code_sha": source,
        "execution_tree_sha": tree,
        "harness_blob_sha": harness,
        "workflow_blob_sha": workflow,
        "result_root": RESULT_ROOT,
    }


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=RAM_MIB,
    timeout=15 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def verify_zero_gpu(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    workflow_sha: str,
) -> str:
    from tam_research.chm_v1_100m_scale import stage_a_preflight
    from tam_research.chm_v1_100m_stage_b import validate_contract
    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus

    source, tree, harness, workflow = _validate_source(
        source_sha, source_tree, harness_sha, workflow_sha
    )
    contract = validate_contract()
    if contract["scientific_execution_authorized"] is not False:
        raise RuntimeError("#983 contract unexpectedly authorizes scientific execution")
    if contract["engineering_seed"] != ENGINEERING_SEED:
        raise RuntimeError("#983 contract engineering seed drift")

    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#983 result namespace already exists; fail closed")

    stage_a = stage_a_preflight()
    fingerprint = fingerprint_frozen_corpus(DATA_DIR)
    payload = {
        "status": "PASS",
        "classification": "CHM_V1_100M_STAGE_B_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence_binding(source, tree, harness, workflow),
        "stage_a": stage_a,
        "stage_b_contract": contract,
        "corpus_fingerprint": fingerprint,
        "gpu_allocated": False,
        "engineering_attempt_consumed": False,
        "scientific_execution": False,
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
def reserve_dispatch(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    workflow_sha: str,
) -> str:
    source, tree, harness, workflow = _validate_source(
        source_sha, source_tree, harness_sha, workflow_sha
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file():
        raise RuntimeError("#983 zero-GPU gate missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "execution_code_sha": source,
        "execution_tree_sha": tree,
        "harness_blob_sha": harness,
        "workflow_blob_sha": workflow,
        "engineering_seed": ENGINEERING_SEED,
        "engineering_attempt_consumed": False,
    }
    for key, value in expected.items():
        if zero.get(key) != value:
            raise RuntimeError(f"#983 zero-GPU evidence mismatch for {key}")
    if marker_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#983 trigger/result namespace was already used")

    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_100M_STAGE_B_DURABLE_PRE_ALLOCATION_MARKER",
        **_evidence_binding(source, tree, harness, workflow),
        "marked_unix": time.time(),
        "gpu_allocation_started": False,
        "engineering_attempt_consumed": False,
        "automatic_retry_authorized": False,
        "scientific_execution": False,
    }
    _atomic_write(marker_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: Any):
    import torch

    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _finite_model(model: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _optimizer(model: Any):
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )


def _benchmark_model(
    *,
    kind: str,
    model: Any,
    train_data: Any,
    device: Any,
    generator_seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_100m_stage_b import (
        GRAD_ACCUM,
        GRAD_CLIP,
        MEASURED_STEPS,
        MEASURED_TOKENS_PER_MODEL,
        MICRO_BATCH,
        SESSION_LEN,
        WARMUP_STEPS,
    )
    from tam_research.chm_v1_small_lm_protocol import (
        eiem_flat_training_session_logits,
        local_session_logits,
    )

    if kind not in {"local", "eiem"}:
        raise ValueError(kind)
    optimizer = _optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)

    def optimizer_step() -> bool:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        finite_loss = True
        for _ in range(GRAD_ACCUM):
            x, y = train_data.batch(MICRO_BATCH, SESSION_LEN, generator, device)
            with _autocast(device):
                logits = (
                    local_session_logits(model, x)
                    if kind == "local"
                    else eiem_flat_training_session_logits(model, x)
                )
                loss = F.cross_entropy(logits.float().reshape(-1, 50_257), y.reshape(-1))
                finite_loss = finite_loss and bool(torch.isfinite(loss).item())
                scaled = loss / GRAD_ACCUM
            scaled.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        return finite_loss

    warmup_finite = True
    for _ in range(WARMUP_STEPS):
        warmup_finite = warmup_finite and optimizer_step()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    measured_finite = True
    started = time.perf_counter()
    for _ in range(MEASURED_STEPS):
        measured_finite = measured_finite and optimizer_step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))

    if MEASURED_TOKENS_PER_MODEL != MEASURED_STEPS * MICRO_BATCH * SESSION_LEN * GRAD_ACCUM:
        raise RuntimeError("#983 measured token accounting drift")
    finite_parameters = _finite_model(model)
    return {
        "kind": kind,
        "trainable_parameters": _count_parameters(model),
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": MEASURED_TOKENS_PER_MODEL,
        "wall_seconds": elapsed,
        "tokens_per_second": MEASURED_TOKENS_PER_MODEL / max(elapsed, 1e-9),
        "peak_vram_bytes": peak,
        "finite_loss": bool(warmup_finite and measured_finite),
        "finite_parameters": finite_parameters,
        "bf16_autocast": True,
        "optimizer": "AdamW-fused",
        "scientific_interpretation_allowed": False,
    }


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=CPU_CORES,
    memory=RAM_MIB,
    timeout=MAX_SECONDS,
    retries=0,
    volumes={"/vol": volume},
)
def run_stage_b(
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    workflow_sha: str,
    live_hourly_resource_usd: float,
) -> str:
    import torch

    from tam_research.chm_v1_100m_scale import (
        CHMV1100MEIEMLM,
        CHMV1100MLocalLM,
        EXPECTED_EIEM_PARAMETERS,
        EXPECTED_LOCAL_PARAMETERS,
    )
    from tam_research.chm_v1_100m_stage_b import classify_stage_b, validate_contract
    from tam_research.chm_v1_corpus_fingerprint import (
        assert_fingerprint_matches,
        fingerprint_frozen_corpus,
    )
    from tam_research.data import TokenBin

    source, tree, harness, workflow = _validate_source(
        source_sha, source_tree, harness_sha, workflow_sha
    )
    validate_contract()
    if float(live_hourly_resource_usd) <= 0:
        raise RuntimeError("#983 live hourly resource rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("#983 L4 function started without CUDA")
    device = torch.device("cuda")

    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file() or not marker_path.is_file():
        raise RuntimeError("#983 pre-allocation evidence incomplete")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#983 engineering attempt already consumed")

    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for payload_name, payload in (("zero", zero), ("marker", marker)):
        for key, expected in {
            "execution_code_sha": source,
            "execution_tree_sha": tree,
            "harness_blob_sha": harness,
            "workflow_blob_sha": workflow,
            "engineering_seed": ENGINEERING_SEED,
        }.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"#983 {payload_name} evidence mismatch for {key}")

    consumed = {
        "status": "ENGINEERING_ATTEMPT_CONSUMED",
        "classification": "CHM_V1_100M_STAGE_B_L4_ALLOCATION_STARTED",
        **_evidence_binding(source, tree, harness, workflow),
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "engineering_attempt_consumed": True,
        "automatic_retry_authorized": False,
        "scientific_execution": False,
    }
    _atomic_write(consumed_path, consumed)
    marker["gpu_allocation_started"] = True
    marker["engineering_attempt_consumed"] = True
    _atomic_write(marker_path, marker)
    volume.commit()

    try:
        actual_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
        assert_fingerprint_matches(actual_fingerprint, zero["corpus_fingerprint"])
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))

        _seed_all(ENGINEERING_SEED)
        local = CHMV1100MLocalLM()
        _seed_all(ENGINEERING_SEED)
        eiem = CHMV1100MEIEMLM()

        if _count_parameters(local) != EXPECTED_LOCAL_PARAMETERS:
            raise RuntimeError("#983 LOCAL exact parameter count drift")
        if _count_parameters(eiem) != EXPECTED_EIEM_PARAMETERS:
            raise RuntimeError("#983 EIEM exact parameter count drift")
        local_state = local.backbone.state_dict()
        eiem_state = eiem.backbone.state_dict()
        if local_state.keys() != eiem_state.keys():
            raise RuntimeError("#983 paired backbone state keys differ")
        for name, value in local_state.items():
            if not torch.equal(value, eiem_state[name]):
                raise RuntimeError(f"#983 paired backbone initialization mismatch at {name}")

        local = local.to(device)
        local_result = _benchmark_model(
            kind="local",
            model=local,
            train_data=train_data,
            device=device,
            generator_seed=ENGINEERING_SEED + 10_000,
        )
        del local
        torch.cuda.empty_cache()

        eiem = eiem.to(device)
        eiem_result = _benchmark_model(
            kind="eiem",
            model=eiem,
            train_data=train_data,
            device=device,
            generator_seed=ENGINEERING_SEED + 20_000,
        )
        del eiem
        torch.cuda.empty_cache()

        decision = classify_stage_b(
            local=local_result,
            eiem=eiem_result,
            live_hourly_resource_usd=float(live_hourly_resource_usd),
        )
        payload = {
            "status": "COMPLETE",
            **_evidence_binding(source, tree, harness, workflow),
            "classification": decision["classification"],
            "passed": decision["passed"],
            "stop_reasons": decision["stop_reasons"],
            "thresholds": decision["thresholds"],
            "projection": decision["projection"],
            "measurements": {
                "local": local_result,
                "eiem_flat": eiem_result,
            },
            "corpus_fingerprint": actual_fingerprint,
            "engineering_attempt_consumed": True,
            "gpu_allocation_started": True,
            "automatic_retry_authorized": False,
            "stage_c_authorized_automatically": False,
            "scientific_execution": False,
            "interpretation_ceiling": decision["interpretation_ceiling"],
        }
        _atomic_write(result_path, payload)
        volume.commit()
        return json.dumps(payload, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "ATTEMPT_FAILED",
            "classification": "CHM_V1_100M_STAGE_B_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
            **_evidence_binding(source, tree, harness, workflow),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_unix": time.time(),
            "engineering_attempt_consumed": True,
            "gpu_allocation_started": True,
            "automatic_retry_authorized": False,
            "scientific_execution": False,
            "scientific_interpretation": False,
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
    harness_sha: str,
    workflow_sha: str,
) -> str:
    source, tree, harness, workflow = _validate_source(
        source_sha, source_tree, harness_sha, workflow_sha
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    names = (
        "ZERO_GPU_GATE.json",
        "DISPATCH_RESERVED.json",
        "ATTEMPT_CONSUMED.json",
        "RESULT.json",
        "ATTEMPT_FAILURE.json",
    )
    state: dict[str, Any] = {
        **_evidence_binding(source, tree, harness, workflow),
        "files": {},
    }
    for name in names:
        path = root / name
        state["files"][name] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    return json.dumps(state, sort_keys=True)


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    harness_sha: str,
    workflow_sha: str,
    live_hourly_resource_usd: float = 0.0,
) -> None:
    if phase == "preflight":
        payload = verify_zero_gpu.remote(source_sha, source_tree, harness_sha, workflow_sha)
        print(f"CHM_V1_100M_STAGE_B_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(source_sha, source_tree, harness_sha, workflow_sha)
        print(f"CHM_V1_100M_STAGE_B_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_stage_b.remote(
            source_sha,
            source_tree,
            harness_sha,
            workflow_sha,
            float(live_hourly_resource_usd),
        )
        print(f"CHM_V1_100M_STAGE_B_RESULT={payload}")
        return
    if phase == "inspect":
        payload = inspect_state.remote(source_sha, source_tree, harness_sha, workflow_sha)
        print(f"CHM_V1_100M_STAGE_B_STATE={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
