from __future__ import annotations

"""One-shot CHM-v3 optimized DAEC Stage-B L4 systems runner (#1271).

Non-scientific engineering only. The runner benchmarks LOCAL, RAW-EIEM, and
DAEC on an identical deterministic session-start stream. DAEC uses the #1262
target-only NLL computation proven equivalent to the frozen dense reference. It never reserves,
creates, or consumes a scientific seed and cannot authorize Stage C.
"""

from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from typing import Any

import modal

PHASE = "chm-v3-100m-daec-target-nll-stage-b-1271-v1"
TRIGGER_TITLE = "[modal-chm-v3-100m-daec-stage-b-target-nll-1262-seed-1262201-v1]"
PARENT_HYPOTHESIS_ISSUE = 1234
SYSTEMS_ISSUE = 1271
ENGINEERING_SEED = 1_262_201
RESULT_ROOT = "/vol/chm-v3/100m-daec-stage-b-target-nll/issue-1262/seed-1262201-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

DAEC_BLOB = "77e9ccbff4383be40e6e2865503e1c3926bbda74"
TARGET_NLL_BLOB = "0b0a68f47186f154a48d1d75e3fe3b236188d69d"
V1_MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
V1_PROTOCOL_BLOB = "d5c2e405b5306f556e7fbe70aacd552867e69d3e"
DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"
DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"
RUNTIME_PROBE_BLOB = "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 1_200
MAX_BILLED_COMPUTE_USD = 0.50

APP_NAME = "chm-v3-100m-daec-target-nll-stage-b-1271-v1"
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


def _sha256(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 64-hex SHA-256")
    return normalized


def _validate_source(
    source_sha: str,
    source_tree: str,
    daec_sha: str,
    contract_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> dict[str, str]:
    bindings = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "daec_blob_sha": _full_sha(daec_sha, "daec_sha"),
        "contract_blob_sha": _full_sha(contract_sha, "contract_sha"),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
    }
    if bindings["daec_blob_sha"] != DAEC_BLOB:
        raise RuntimeError("#1271 DAEC blob drift")
    if PARENT_HYPOTHESIS_ISSUE != 1234 or SYSTEMS_ISSUE != 1271:
        raise RuntimeError("#1271 issue binding drift")
    if ENGINEERING_SEED != 1_262_201:
        raise RuntimeError("#1271 engineering seed drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 16 * 1024:
        raise RuntimeError("#1271 hardware binding drift")
    if MAX_SECONDS != 1_200 or MAX_BILLED_COMPUTE_USD != 0.50:
        raise RuntimeError("#1271 runtime/cost binding drift")
    return bindings


def _account_binding(
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> dict[str, str]:
    account = str(selected_modal_account).strip().lower()
    workspace = str(selected_modal_workspace).strip()
    evidence_sha = _sha256(account_selection_evidence_sha256, "account_selection_evidence_sha256")
    if account not in {"primary", "secondary"}:
        raise RuntimeError("#1271 selected Modal account must be primary or secondary")
    if not workspace or workspace.lower() == "none":
        raise RuntimeError("#1271 selected Modal workspace must be non-empty")
    return {
        "selected_modal_account": account,
        "selected_modal_workspace": workspace,
        "account_selection_evidence_sha256": evidence_sha,
    }


def _evidence(
    bindings: dict[str, str],
    authority_comment_id: int,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "parent_hypothesis_issue": PARENT_HYPOTHESIS_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "target_nll_blob_sha": TARGET_NLL_BLOB,
        "final_authority_comment_id": int(authority_comment_id),
        "scientific_seed_authorized": False,
        "scientific_execution": False,
        "stage_c_authorized_automatically": False,
        **_account_binding(
            selected_modal_account,
            selected_modal_workspace,
            account_selection_evidence_sha256,
        ),
        **bindings,
    }


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
    daec_sha: str,
    contract_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> str:
    bindings = _validate_source(
        source_sha, source_tree, daec_sha, contract_sha, runner_sha, workflow_sha
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
    files: dict[str, Any] = {}
    for name in names:
        path = root / name
        files[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return json.dumps(
        {
            **bindings,
            "phase": PHASE,
            "systems_issue": SYSTEMS_ISSUE,
            "engineering_seed": ENGINEERING_SEED,
            "root_exists": root.exists(),
            "files": files,
            "gpu_allocated": False,
            "scientific_seed_authorized": False,
        },
        sort_keys=True,
    )


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
    daec_sha: str,
    contract_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> str:
    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus
    from tam_research.chm_v3_100m_daec import stage_a_preflight
    from tam_research.chm_v3_100m_daec_target_nll_stage_b import validate_contract

    bindings = _validate_source(
        source_sha, source_tree, daec_sha, contract_sha, runner_sha, workflow_sha
    )
    authority = int(authority_comment_id)
    if authority <= 0:
        raise RuntimeError("#1271 final launcher authority comment ID must be positive")
    evidence = _evidence(
        bindings,
        authority,
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )

    contract = validate_contract()
    if contract["scientific_execution_authorized"] is not False:
        raise RuntimeError("#1271 systems contract unexpectedly authorizes science")
    if contract["engineering_seed"] != ENGINEERING_SEED:
        raise RuntimeError("#1271 contract engineering seed drift")

    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#1271 result namespace already exists; fail closed")

    stage_a = stage_a_preflight(instantiate=False)
    fingerprint = fingerprint_frozen_corpus(DATA_DIR)
    payload = {
        "status": "PASS",
        "classification": "CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_ZERO_GPU_PREFLIGHT_PASS",
        **evidence,
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
    daec_sha: str,
    contract_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> str:
    bindings = _validate_source(
        source_sha, source_tree, daec_sha, contract_sha, runner_sha, workflow_sha
    )
    evidence = _evidence(
        bindings,
        int(authority_comment_id),
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file():
        raise RuntimeError("#1271 zero-GPU gate missing")
    if reserve_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1271 trigger/result namespace was already used")

    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    for key, value in evidence.items():
        if zero.get(key) != value:
            raise RuntimeError(f"#1271 zero-GPU evidence mismatch for {key}")

    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_DURABLE_PRE_ALLOCATION_MARKER",
        **evidence,
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
        raise ValueError("#1271 microbatch start row must be one-dimensional")
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

    from tam_research.chm_v1_small_lm_protocol import (
        eiem_flat_training_session_logits,
        local_session_logits,
    )
    from tam_research.chm_v3_100m_daec_target_nll import daec_flat_training_session_nll
    from tam_research.chm_v3_100m_daec_target_nll_stage_b import (
        GRAD_ACCUM,
        GRAD_CLIP,
        MEASURED_STEPS,
        MEASURED_TOKENS_PER_MODEL,
        MICRO_BATCH,
        WARMUP_STEPS,
    )

    if kind not in {"local", "raw", "daec"}:
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
                raise RuntimeError("#1271 microbatch start-plan width drift")
            x, y = _batch_from_starts(train_data, starts, device)
            with _autocast(device):
                if kind == "local":
                    scores = local_session_logits(model, x)
                    loss = F.cross_entropy(scores.float().reshape(-1, 50_257), y.reshape(-1))
                elif kind == "raw":
                    scores = eiem_flat_training_session_logits(model, x)
                    loss = F.cross_entropy(scores.float().reshape(-1, 50_257), y.reshape(-1))
                else:
                    loss = daec_flat_training_session_nll(model, x, y)
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
    daec_sha: str,
    contract_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    live_hourly_resource_usd: float,
) -> str:
    import torch

    from tam_research.chm_v1_100m_scale import (
        CHMV1100MEIEMLM,
        CHMV1100MLocalLM,
        EXPECTED_EIEM_PARAMETERS,
        EXPECTED_LOCAL_PARAMETERS,
    )
    from tam_research.chm_v1_corpus_fingerprint import (
        assert_fingerprint_matches,
        fingerprint_frozen_corpus,
    )
    from tam_research.chm_v3_100m_daec import CHMV3100MDAECLM, EXPECTED_DAEC_PARAMETERS
    from tam_research.chm_v3_100m_daec_target_nll_stage_b import (
        build_start_plan,
        classify_stage_b,
        start_plan_sha256,
        validate_contract,
    )
    from tam_research.data import TokenBin

    bindings = _validate_source(
        source_sha, source_tree, daec_sha, contract_sha, runner_sha, workflow_sha
    )
    evidence = _evidence(
        bindings,
        int(authority_comment_id),
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    contract = validate_contract()
    hourly = float(live_hourly_resource_usd)
    if int(authority_comment_id) <= 0:
        raise RuntimeError("#1271 final launcher authority comment ID must be positive")
    if hourly <= 0.0 or hourly * (MAX_SECONDS / 3600.0) > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError("#1271 live resource rate exceeds frozen systems cap")

    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file() or not reserve_path.is_file():
        raise RuntimeError("#1271 pre-allocation evidence incomplete")
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1271 engineering attempt already consumed")

    consumed = {
        "status": "ENGINEERING_ATTEMPT_CONSUMED",
        "classification": "CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_L4_ALLOCATION_STARTED",
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
            raise RuntimeError("#1271 L4 function started without CUDA")
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1271 expected NVIDIA L4, got {device_name!r}")

        zero = json.loads(zero_path.read_text(encoding="utf-8"))
        reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
        for payload_name, payload in (("zero", zero), ("reserve", reserve)):
            for key, expected in evidence.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"#1271 {payload_name} evidence mismatch for {key}")

        actual_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
        assert_fingerprint_matches(actual_fingerprint, zero["corpus_fingerprint"])
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))

        local_plan = build_start_plan()
        raw_plan = build_start_plan()
        daec_plan = build_start_plan()
        local_digest = start_plan_sha256(local_plan)
        raw_digest = start_plan_sha256(raw_plan)
        daec_digest = start_plan_sha256(daec_plan)
        expected_digest = contract["stream_plan"]["sha256"]
        if {local_digest, raw_digest, daec_digest} != {expected_digest}:
            raise RuntimeError("#1271 start-plan digest drift")
        if not (torch.equal(local_plan, raw_plan) and torch.equal(local_plan, daec_plan)):
            raise RuntimeError("#1271 regenerated model start plans differ")

        _seed_all(ENGINEERING_SEED)
        local = CHMV1100MLocalLM()
        _seed_all(ENGINEERING_SEED)
        raw = CHMV1100MEIEMLM()
        _seed_all(ENGINEERING_SEED)
        daec = CHMV3100MDAECLM()

        if _count_parameters(local) != EXPECTED_LOCAL_PARAMETERS:
            raise RuntimeError("#1271 LOCAL parameter count drift")
        if _count_parameters(raw) != EXPECTED_EIEM_PARAMETERS:
            raise RuntimeError("#1271 RAW-EIEM parameter count drift")
        if _count_parameters(daec) != EXPECTED_DAEC_PARAMETERS:
            raise RuntimeError("#1271 DAEC parameter count drift")

        local_backbone = local.backbone.state_dict()
        raw_backbone = raw.backbone.state_dict()
        daec_backbone = daec.backbone.state_dict()
        if not (local_backbone.keys() == raw_backbone.keys() == daec_backbone.keys()):
            raise RuntimeError("#1271 inherited backbone key drift")
        for name, value in local_backbone.items():
            if not torch.equal(value, raw_backbone[name]) or not torch.equal(value, daec_backbone[name]):
                raise RuntimeError(f"#1271 inherited backbone init mismatch at {name}")

        raw_named = dict(raw.named_parameters())
        daec_named = dict(daec.named_parameters())
        for name in ("query_address.weight", "key_address.weight", "memory_gate_logit"):
            if not torch.equal(raw_named[name], daec_named[name]):
                raise RuntimeError(f"#1271 inherited EIEM init mismatch at {name}")

        local = local.to(device)
        local_result = _benchmark_model(
            kind="local",
            model=local,
            train_data=train_data,
            device=device,
            start_plan=local_plan,
            start_plan_digest=local_digest,
        )
        del local
        torch.cuda.empty_cache()

        raw = raw.to(device)
        raw_result = _benchmark_model(
            kind="raw",
            model=raw,
            train_data=train_data,
            device=device,
            start_plan=raw_plan,
            start_plan_digest=raw_digest,
        )
        del raw
        torch.cuda.empty_cache()

        daec = daec.to(device)
        daec_result = _benchmark_model(
            kind="daec",
            model=daec,
            train_data=train_data,
            device=device,
            start_plan=daec_plan,
            start_plan_digest=daec_digest,
        )
        del daec
        torch.cuda.empty_cache()

        decision = classify_stage_b(
            local=local_result,
            raw=raw_result,
            daec=daec_result,
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
                "local": local_result,
                "raw_eiem": raw_result,
                "daec": daec_result,
            },
            "start_plan_sha256": expected_digest,
            "paired_start_plan_identical": (
                local_result["start_plan_sha256"]
                == raw_result["start_plan_sha256"]
                == daec_result["start_plan_sha256"]
                == expected_digest
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
            "classification": "CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
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
            "stage_c_authorized_automatically": False,
        }
        _atomic_write(failure_path, failure)
        volume.commit()
        raise


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    daec_sha: str,
    contract_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int = 0,
    selected_modal_account: str = "primary",
    selected_modal_workspace: str = "",
    account_selection_evidence_sha256: str = "",
    live_hourly_resource_usd: float = 0.0,
) -> None:
    base = (source_sha, source_tree, daec_sha, contract_sha, runner_sha, workflow_sha)
    if phase == "inspect":
        payload = inspect_state.remote(*base)
        print(f"CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_STATE={payload}")
        return

    common = (
        *base,
        int(authority_comment_id),
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    if phase == "preflight":
        payload = verify_zero_gpu.remote(*common)
        print(f"CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(*common)
        print(f"CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_stage_b.remote(*common, float(live_hourly_resource_usd))
        print(f"CHM_V3_100M_DAEC_TARGET_NLL_STAGE_B_RESULT={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
