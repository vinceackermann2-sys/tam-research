from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from typing import Any

import modal

PHASE = "chm-v2-100m-qva-stage-b-1155-v1"
TRIGGER_TITLE = "[modal-chm-v2-100m-qva-stage-b-parent-1147-v1]"
PARENT_HYPOTHESIS_ISSUE = 1147
SYSTEMS_ISSUE = 1155
ENGINEERING_SEED = 1_147_201
HISTORICAL_CONSUMED_SCIENTIFIC_SEED = 977_001
RESULT_ROOT = "/vol/chm-v2/100m-qva-stage-b/parent-1147/seed-1147201-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

QVA_BLOB = "a34a8dffc2c2c1702a909782c3aba2593e90947c"
V1_MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
V1_PROTOCOL_BLOB = "d5c2e405b5306f556e7fbe70aacd552867e69d3e"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 1_200
MAX_BILLED_COMPUTE_USD = 0.50

APP_NAME = "chm-v2-100m-qva-stage-b-1155-v1"
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
    normalized = str(value).strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 40-hex SHA")
    return normalized


def _validate_source(
    source_sha: str,
    source_tree: str,
    qva_sha: str,
    harness_sha: str,
    workflow_sha: str,
) -> dict[str, str]:
    bindings = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "qva_blob_sha": _full_sha(qva_sha, "qva_sha"),
        "harness_blob_sha": _full_sha(harness_sha, "harness_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
    }
    if bindings["qva_blob_sha"] != QVA_BLOB:
        raise RuntimeError("#1155 QVA blob drift")
    if ENGINEERING_SEED != 1_147_201 or HISTORICAL_CONSUMED_SCIENTIFIC_SEED != 977_001:
        raise RuntimeError("#1155 seed binding drift")
    if PARENT_HYPOTHESIS_ISSUE != 1147 or SYSTEMS_ISSUE != 1155:
        raise RuntimeError("#1155 issue binding drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 16 * 1024:
        raise RuntimeError("#1155 hardware binding drift")
    if MAX_SECONDS != 1_200 or MAX_BILLED_COMPUTE_USD != 0.50:
        raise RuntimeError("#1155 runtime/cost binding drift")
    return bindings


def _evidence(bindings: dict[str, str], authority_comment_id: int) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "parent_hypothesis_issue": PARENT_HYPOTHESIS_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "engineering_seed": ENGINEERING_SEED,
        "historical_consumed_scientific_seed_not_reusable": HISTORICAL_CONSUMED_SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "final_authority_comment_id": int(authority_comment_id),
        "scientific_seed_authorized": False,
        "scientific_execution": False,
        **bindings,
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
    qva_sha: str,
    harness_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
) -> str:
    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus
    from tam_research.chm_v2_100m_qva import stage_a_preflight
    from tam_research.chm_v2_100m_stage_b import validate_contract

    bindings = _validate_source(source_sha, source_tree, qva_sha, harness_sha, workflow_sha)
    authority = int(authority_comment_id)
    if authority <= 0:
        raise RuntimeError("#1155 final launcher authority comment ID must be positive")

    contract = validate_contract()
    if contract["scientific_execution_authorized"] is not False:
        raise RuntimeError("#1155 systems contract unexpectedly authorizes science")
    if contract["engineering_seed"] != ENGINEERING_SEED:
        raise RuntimeError("#1155 contract engineering seed drift")

    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#1155 result namespace already exists; fail closed")

    stage_a = stage_a_preflight()
    fingerprint = fingerprint_frozen_corpus(DATA_DIR)
    payload = {
        "status": "PASS",
        "classification": "CHM_V2_100M_QVA_STAGE_B_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence(bindings, authority),
        "stage_a": stage_a,
        "stage_b_contract": contract,
        "corpus_fingerprint": fingerprint,
        "start_plan_sha256": contract["stream_plan"]["sha256"],
        "gpu_allocated": False,
        "engineering_attempt_consumed": False,
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
    qva_sha: str,
    harness_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_source(source_sha, source_tree, qva_sha, harness_sha, workflow_sha)
    authority = int(authority_comment_id)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file():
        raise RuntimeError("#1155 zero-GPU gate missing")
    if reserve_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1155 trigger/result namespace was already used")

    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    for key, value in _evidence(bindings, authority).items():
        if zero.get(key) != value:
            raise RuntimeError(f"#1155 zero-GPU evidence mismatch for {key}")

    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V2_100M_QVA_STAGE_B_DURABLE_PRE_ALLOCATION_MARKER",
        **_evidence(bindings, authority),
        "marked_unix": time.time(),
        "gpu_allocation_started": False,
        "engineering_attempt_consumed": False,
        "automatic_retry_authorized": False,
    }
    _atomic_write(reserve_path, marker)
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


def _batch_from_starts(train_data: Any, starts_cpu: Any, device: Any):
    import torch

    if starts_cpu.ndim != 1:
        raise ValueError("#1155 microbatch start row must be one-dimensional")
    source = train_data._device_tokens(device)
    starts = starts_cpu.to(device=device, dtype=torch.long)
    offsets = torch.arange(1025, device=device, dtype=torch.long)
    chunks = source[starts[:, None] + offsets[None, :]].long()
    return chunks[:, :-1], chunks[:, 1:]


def _benchmark_model(
    *,
    kind: str,
    model: Any,
    train_data: Any,
    device: Any,
    start_plan: Any,
    start_plan_digest: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_small_lm_protocol import eiem_flat_training_session_logits
    from tam_research.chm_v2_100m_qva import qva_flat_training_session_logits
    from tam_research.chm_v2_100m_stage_b import (
        GRAD_ACCUM,
        GRAD_CLIP,
        MEASURED_STEPS,
        MEASURED_TOKENS_PER_MODEL,
        MICRO_BATCH,
        WARMUP_STEPS,
    )

    if kind not in {"v1", "qva"}:
        raise ValueError(kind)
    optimizer = _optimizer(model)
    losses: list[float] = []

    def optimizer_step(step_index: int) -> bool:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        finite_loss = True
        step_losses: list[float] = []
        for accum_index in range(GRAD_ACCUM):
            starts = start_plan[step_index, accum_index]
            if int(starts.numel()) != MICRO_BATCH:
                raise RuntimeError("#1155 microbatch start-plan width drift")
            x, y = _batch_from_starts(train_data, starts, device)
            with _autocast(device):
                logits = (
                    eiem_flat_training_session_logits(model, x)
                    if kind == "v1"
                    else qva_flat_training_session_logits(model, x)
                )
                loss = F.cross_entropy(logits.float().reshape(-1, 50_257), y.reshape(-1))
                finite_loss = finite_loss and bool(torch.isfinite(loss).item())
                step_losses.append(float(loss.detach().item()))
                scaled = loss / GRAD_ACCUM
            scaled.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(sum(step_losses) / len(step_losses))
        return finite_loss

    warmup_finite = True
    for step_index in range(WARMUP_STEPS):
        warmup_finite = warmup_finite and optimizer_step(step_index)
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    measured_finite = True
    started = time.perf_counter()
    for step_index in range(WARMUP_STEPS, WARMUP_STEPS + MEASURED_STEPS):
        measured_finite = measured_finite and optimizer_step(step_index)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))

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
        "finite_parameters": _finite_model(model),
        "mean_optimizer_step_loss": sum(losses) / len(losses),
        "start_plan_sha256": start_plan_digest,
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
    qva_sha: str,
    harness_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    live_hourly_resource_usd: float,
) -> str:
    import torch

    from tam_research.chm_v1_100m_scale import CHMV1100MEIEMLM, EXPECTED_EIEM_PARAMETERS
    from tam_research.chm_v1_corpus_fingerprint import (
        assert_fingerprint_matches,
        fingerprint_frozen_corpus,
    )
    from tam_research.chm_v2_100m_qva import CHMV2100MEIEMQVA, EXPECTED_QVA_PARAMETERS
    from tam_research.chm_v2_100m_stage_b import (
        build_start_plan,
        classify_stage_b,
        start_plan_sha256,
        validate_contract,
    )
    from tam_research.data import TokenBin

    bindings = _validate_source(source_sha, source_tree, qva_sha, harness_sha, workflow_sha)
    authority = int(authority_comment_id)
    contract = validate_contract()
    hourly = float(live_hourly_resource_usd)
    if authority <= 0:
        raise RuntimeError("#1155 final launcher authority comment ID must be positive")
    if hourly <= 0.0 or hourly * (MAX_SECONDS / 3600.0) > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError("#1155 live resource rate exceeds frozen systems cap")

    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file() or not reserve_path.is_file():
        raise RuntimeError("#1155 pre-allocation evidence incomplete")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1155 engineering attempt already consumed")

    evidence = _evidence(bindings, authority)
    consumed = {
        "status": "ENGINEERING_ATTEMPT_CONSUMED",
        "classification": "CHM_V2_100M_QVA_STAGE_B_L4_ALLOCATION_STARTED",
        **evidence,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "engineering_attempt_consumed": True,
        "automatic_retry_authorized": False,
    }
    _atomic_write(consumed_path, consumed)
    volume.commit()

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("#1155 L4 function started without CUDA")
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1155 expected NVIDIA L4, got {device_name!r}")

        zero = json.loads(zero_path.read_text(encoding="utf-8"))
        reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
        for payload_name, payload in (("zero", zero), ("reserve", reserve)):
            for key, expected in evidence.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"#1155 {payload_name} evidence mismatch for {key}")

        actual_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
        assert_fingerprint_matches(actual_fingerprint, zero["corpus_fingerprint"])
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))

        # Rebuild the deterministic CPU start plan independently for each
        # model, resetting the same frozen CPU generator seed each time.
        v1_start_plan = build_start_plan()
        qva_start_plan = build_start_plan()
        v1_plan_digest = start_plan_sha256(v1_start_plan)
        qva_plan_digest = start_plan_sha256(qva_start_plan)
        expected_plan_digest = contract["stream_plan"]["sha256"]
        if v1_plan_digest != expected_plan_digest or qva_plan_digest != expected_plan_digest:
            raise RuntimeError("#1155 start-plan digest drift")
        if not torch.equal(v1_start_plan, qva_start_plan):
            raise RuntimeError("#1155 regenerated paired start plans differ")

        _seed_all(ENGINEERING_SEED)
        v1 = CHMV1100MEIEMLM()
        _seed_all(ENGINEERING_SEED)
        qva = CHMV2100MEIEMQVA()

        if _count_parameters(v1) != EXPECTED_EIEM_PARAMETERS:
            raise RuntimeError("#1155 CHM-v1 EIEM parameter count drift")
        if _count_parameters(qva) != EXPECTED_QVA_PARAMETERS:
            raise RuntimeError("#1155 CHM-v2 QVA parameter count drift")

        v1_named = dict(v1.named_parameters())
        qva_named = dict(qva.named_parameters())
        for name, value in v1.backbone.state_dict().items():
            if not torch.equal(value, qva.backbone.state_dict()[name]):
                raise RuntimeError(f"#1155 inherited backbone init mismatch at {name}")
        for name in ("query_address.weight", "key_address.weight", "memory_gate_logit"):
            if not torch.equal(v1_named[name], qva_named[name]):
                raise RuntimeError(f"#1155 inherited EIEM init mismatch at {name}")

        v1 = v1.to(device)
        v1_result = _benchmark_model(
            kind="v1",
            model=v1,
            train_data=train_data,
            device=device,
            start_plan=v1_start_plan,
            start_plan_digest=v1_plan_digest,
        )
        del v1
        torch.cuda.empty_cache()

        qva = qva.to(device)
        qva_result = _benchmark_model(
            kind="qva",
            model=qva,
            train_data=train_data,
            device=device,
            start_plan=qva_start_plan,
            start_plan_digest=qva_plan_digest,
        )
        del qva
        torch.cuda.empty_cache()

        decision = classify_stage_b(
            v1=v1_result,
            qva=qva_result,
            live_hourly_resource_usd=hourly,
        )
        payload = {
            "status": "COMPLETE",
            **evidence,
            "classification": decision["classification"],
            "passed": decision["passed"],
            "stop_reasons": decision["stop_reasons"],
            "thresholds": decision["thresholds"],
            "projection": decision["projection"],
            "measurements": {
                "chm_v1_eiem": v1_result,
                "chm_v2_qva": qva_result,
            },
            "start_plan_sha256": expected_plan_digest,
            "paired_start_plan_identical": (
                v1_result["start_plan_sha256"]
                == qva_result["start_plan_sha256"]
                == expected_plan_digest
                and torch.equal(v1_start_plan, qva_start_plan)
            ),
            "corpus_fingerprint": actual_fingerprint,
            "device_name": device_name,
            "live_hourly_resource_usd": hourly,
            "engineering_attempt_consumed": True,
            "gpu_allocation_started": True,
            "automatic_retry_authorized": False,
            "scientific_seed_authorized": False,
            "scientific_execution": False,
            "stage_c_authorized_automatically": False,
            "interpretation_ceiling": decision["interpretation_ceiling"],
        }
        _atomic_write(result_path, payload)
        volume.commit()
        return json.dumps(payload, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "ATTEMPT_FAILED",
            "classification": "CHM_V2_100M_QVA_STAGE_B_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
            **evidence,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_unix": time.time(),
            "engineering_attempt_consumed": True,
            "gpu_allocation_started": True,
            "automatic_retry_authorized": False,
            "scientific_seed_authorized": False,
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
    qva_sha: str,
    harness_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_source(source_sha, source_tree, qva_sha, harness_sha, workflow_sha)
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
        **_evidence(bindings, int(authority_comment_id)),
        "root_exists": root.exists(),
        "files": {},
    }
    for name in names:
        path = root / name
        state["files"][name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return json.dumps(state, sort_keys=True)


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    qva_sha: str,
    harness_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    live_hourly_resource_usd: float = 0.0,
) -> None:
    common = (
        source_sha,
        source_tree,
        qva_sha,
        harness_sha,
        workflow_sha,
        int(authority_comment_id),
    )
    if phase == "preflight":
        payload = verify_zero_gpu.remote(*common)
        print(f"CHM_V2_100M_QVA_STAGE_B_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(*common)
        print(f"CHM_V2_100M_QVA_STAGE_B_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_stage_b.remote(*common, float(live_hourly_resource_usd))
        print(f"CHM_V2_100M_QVA_STAGE_B_RESULT={payload}")
        return
    if phase == "inspect":
        payload = inspect_state.remote(*common)
        print(f"CHM_V2_100M_QVA_STAGE_B_STATE={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
