from __future__ import annotations

"""One-shot CHM-v2 host-staged L4 engineering preflight (#1133)."""

import json
from pathlib import Path
import random
import time
from typing import Any

import modal

CONTROL_ISSUE = 1133
ENGINEERING_SEED = 1_133_201
CONSUMED_SCIENTIFIC_SEED = 2_011_121
PHASE = "chm-v2-100m-host-staged-l4-preflight-1133-v1"
RESULT_ROOT = "/vol/chm-v2/100m-host-staged-l4-preflight/issue-1133/attempt-1133201-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

V2_MODULE_BLOB = "d0c2186231cead2a7c851d776d21e6ab6f73ec43"
MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
PREP_BLOB = "bc4ed60885aaf991a2d6b9fe8f634ff973f3f722"
DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"
DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"
RUNTIME_PROBE_BLOB = "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 1_800
MAX_BILLED_COMPUTE_USD = 1.00

APP_NAME = "chm-v2-100m-host-staged-l4-preflight-1133-v1"
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
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> dict[str, str]:
    bindings = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "v2_module_blob_sha": _full_sha(v2_module_sha, "v2_module_sha"),
        "model_blob_sha": _full_sha(model_sha, "model_sha"),
        "evaluator_blob_sha": _full_sha(evaluator_sha, "evaluator_sha"),
        "prep_blob_sha": _full_sha(prep_sha, "prep_sha"),
        "dual_account_blob_sha": _full_sha(dual_account_sha, "dual_account_sha"),
        "dual_account_cli_blob_sha": _full_sha(dual_account_cli_sha, "dual_account_cli_sha"),
        "runtime_probe_blob_sha": _full_sha(runtime_probe_sha, "runtime_probe_sha"),
        "core_blob_sha": _full_sha(core_sha, "core_sha"),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
    }
    expected = {
        "v2_module_blob_sha": V2_MODULE_BLOB,
        "model_blob_sha": MODEL_BLOB,
        "evaluator_blob_sha": EVALUATOR_BLOB,
        "prep_blob_sha": PREP_BLOB,
        "dual_account_blob_sha": DUAL_ACCOUNT_BLOB,
        "dual_account_cli_blob_sha": DUAL_ACCOUNT_CLI_BLOB,
        "runtime_probe_blob_sha": RUNTIME_PROBE_BLOB,
    }
    for key, value in expected.items():
        if bindings[key] != value:
            raise RuntimeError(f"#1133 protected blob drift at {key}")
    if (CONTROL_ISSUE, ENGINEERING_SEED, CONSUMED_SCIENTIFIC_SEED) != (
        1133,
        1_133_201,
        2_011_121,
    ):
        raise RuntimeError("#1133 identity drift")
    return bindings


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _finite_model(model: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def _optimizer(model: Any):
    import torch
    from tam_research.chm_v1_100m_stage_c_run_control_prep import BETAS, PEAK_LR, WEIGHT_DECAY

    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )


def _mem_snapshot(device: Any) -> dict[str, int]:
    import torch

    free, total = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _benchmark_model(
    *,
    kind: str,
    model: Any,
    train_data: Any,
    plan: Any,
    plan_digest: str,
    device: Any,
    before_model: dict[str, int],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_100m_scale import VOCAB_SIZE
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context
    from tam_research.chm_v1_100m_stage_c_run_control_prep import (
        GRAD_ACCUM,
        GRAD_CLIP,
        MICRO_BATCH,
        PEAK_LR,
        SESSION_LEN,
    )
    from tam_research.chm_v1_small_lm_protocol import eiem_flat_training_session_logits
    from tam_research.chm_v2_100m_host_staged_preflight import (
        MEASURED_STEPS,
        MEASURED_TOKENS_PER_MODEL,
        WARMUP_STEPS,
        host_staged_gather,
    )
    from tam_research.chm_v2_100m_value_projected_eiem import (
        vp_eiem_flat_training_session_logits,
    )

    if kind not in {"raw_eiem", "vp_eiem"}:
        raise ValueError(kind)
    optimizer = _optimizer(model)
    transferred_bytes_total = 0
    finite_loss = True
    finite_gradients = True
    losses: list[float] = []

    def optimizer_step(step_index: int) -> float:
        nonlocal transferred_bytes_total, finite_loss, finite_gradients
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for micro_index in range(GRAD_ACCUM):
            x, y, transferred = host_staged_gather(
                train_data.data,
                plan[step_index, micro_index],
                seq_len=SESSION_LEN,
                device=device,
            )
            transferred_bytes_total += int(transferred)
            with autocast_context(device):
                logits = (
                    eiem_flat_training_session_logits(model, x)
                    if kind == "raw_eiem"
                    else vp_eiem_flat_training_session_logits(model, x)
                )
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
                scaled = loss / GRAD_ACCUM
            finite_loss = finite_loss and bool(torch.isfinite(loss).item())
            scaled.backward()
            micro_losses.append(float(loss.detach().item()))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        finite_gradients = finite_gradients and bool(torch.isfinite(grad_norm).item())
        for group in optimizer.param_groups:
            group["lr"] = PEAK_LR
        optimizer.step()
        return sum(micro_losses) / len(micro_losses)

    for step in range(WARMUP_STEPS):
        losses.append(optimizer_step(step))
    torch.cuda.synchronize(device)
    after_optimizer_materialized = _mem_snapshot(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    measured_start = time.perf_counter()
    for step in range(WARMUP_STEPS, WARMUP_STEPS + MEASURED_STEPS):
        losses.append(optimizer_step(step))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - measured_start

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    measured_tokens = MEASURED_TOKENS_PER_MODEL

    if train_data._device_cache:
        raise RuntimeError("#1133 full-source CUDA cache populated during host-staged benchmark")

    return {
        "kind": kind,
        "trainable_parameters": _count_parameters(model),
        "training_plan_sha256": plan_digest,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": measured_tokens,
        "wall_seconds": elapsed,
        "tokens_per_second": measured_tokens / max(elapsed, 1e-9),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_allocated / (1024.0**3),
        "peak_reserved_gib": peak_reserved / (1024.0**3),
        "before_model_memory": before_model,
        "after_optimizer_materialized_memory": after_optimizer_materialized,
        "host_to_device_token_bytes_total": transferred_bytes_total,
        "host_staged_transport": True,
        "full_source_cuda_cache_present": bool(train_data._device_cache),
        "finite_loss": finite_loss,
        "finite_gradients": finite_gradients,
        "finite_parameters": _finite_model(model),
        "last_loss": losses[-1],
        "scientific_interpretation_allowed": False,
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=10 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def inspect_zero_gpu(
    source_sha: str,
    source_tree: str,
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> str:
    bindings = _validate_source(
        source_sha, source_tree, v2_module_sha, model_sha, evaluator_sha, prep_sha,
        dual_account_sha, dual_account_cli_sha, runtime_probe_sha, core_sha,
        runner_sha, workflow_sha,
    )
    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus
    from tam_research.chm_v2_100m_host_staged_preflight import (
        engineering_start_plan,
        start_plan_sha256,
        validate_contract,
    )
    from tam_research.data import TokenBin

    validate_contract()
    volume.reload()
    root = Path(RESULT_ROOT)
    train = TokenBin(str(Path(DATA_DIR) / "train.bin"))
    plan = engineering_start_plan(len(train.data))
    payload = {
        "classification": "CHM_V2_100M_HOST_STAGED_CORPUS_L4_ZERO_GPU_INSPECTION",
        **bindings,
        "control_issue": CONTROL_ISSUE,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "result_namespace_unused": not root.exists(),
        "corpus_fingerprint": fingerprint_frozen_corpus(DATA_DIR),
        "training_plan_sha256": start_plan_sha256(plan),
        "train_token_count": int(len(train.data)),
        "tokenbin_device_cache_empty": not bool(train._device_cache),
        "gpu_allocated": False,
        "engineering_attempt_consumed": False,
        "scientific_seed_created": False,
    }
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
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    selected_modal_account: str,
    account_selection_evidence_sha256: str,
) -> str:
    bindings = _validate_source(
        source_sha, source_tree, v2_module_sha, model_sha, evaluator_sha, prep_sha,
        dual_account_sha, dual_account_cli_sha, runtime_probe_sha, core_sha,
        runner_sha, workflow_sha,
    )
    if int(authority_comment_id) <= 0:
        raise RuntimeError("#1133 final authority comment ID must be positive")
    if selected_modal_account not in {"primary", "secondary"}:
        raise RuntimeError("#1133 invalid selected Modal account")
    if len(str(account_selection_evidence_sha256)) != 64:
        raise RuntimeError("#1133 account-selection evidence digest is invalid")

    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#1133 result namespace already exists; no redispatch")
    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V2_100M_HOST_STAGED_CORPUS_L4_DISPATCH_RESERVED",
        **bindings,
        "control_issue": CONTROL_ISSUE,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "final_authority_comment_id": int(authority_comment_id),
        "selected_modal_account": selected_modal_account,
        "account_selection_evidence_sha256": str(account_selection_evidence_sha256),
        "reserved_unix": time.time(),
        "gpu_allocated": False,
        "engineering_attempt_consumed": False,
        "scientific_seed_created": False,
        "automatic_retry_authorized": False,
    }
    _atomic_write(root / "DISPATCH_RESERVED.json", marker)
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
def run_preflight(
    source_sha: str,
    source_tree: str,
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    selected_modal_account: str,
    account_selection_evidence_sha256: str,
    live_hourly_resource_usd: float,
) -> str:
    import torch

    bindings = _validate_source(
        source_sha, source_tree, v2_module_sha, model_sha, evaluator_sha, prep_sha,
        dual_account_sha, dual_account_cli_sha, runtime_probe_sha, core_sha,
        runner_sha, workflow_sha,
    )
    hourly = float(live_hourly_resource_usd)
    if hourly <= 0 or hourly * (MAX_SECONDS / 3600.0) > MAX_BILLED_COMPUTE_USD:
        raise RuntimeError("#1133 live rate exceeds engineering compute envelope")
    if not torch.cuda.is_available():
        raise RuntimeError("#1133 L4 function began without CUDA")

    volume.reload()
    root = Path(RESULT_ROOT)
    reserve = root / "DISPATCH_RESERVED.json"
    consumed = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not reserve.is_file():
        raise RuntimeError("#1133 durable reservation is missing")
    if consumed.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1133 engineering attempt already consumed")

    attempt = {
        "status": "ENGINEERING_ATTEMPT_CONSUMED",
        "classification": "CHM_V2_100M_HOST_STAGED_CORPUS_L4_GPU_FUNCTION_BEGAN",
        **bindings,
        "control_issue": CONTROL_ISSUE,
        "engineering_seed": ENGINEERING_SEED,
        "engineering_seed_is_scientific": False,
        "consumed_scientific_seed_2011121_reused": False,
        "result_root": RESULT_ROOT,
        "final_authority_comment_id": int(authority_comment_id),
        "selected_modal_account": selected_modal_account,
        "account_selection_evidence_sha256": account_selection_evidence_sha256,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "engineering_attempt_consumed": True,
        "scientific_seed_created": False,
        "automatic_retry_authorized": False,
        "stage_d_authorized": False,
    }
    _atomic_write(consumed, attempt)
    volume.commit()

    try:
        from tam_research.chm_v1_100m_scale import (
            CHMV1100MEIEMLM,
            EXPECTED_EIEM_PARAMETERS,
        )
        from tam_research.chm_v1_corpus_fingerprint import (
            assert_fingerprint_matches,
            fingerprint_frozen_corpus,
        )
        from tam_research.chm_v2_100m_host_staged_preflight import (
            classify_systems_preflight,
            engineering_start_plan,
            start_plan_sha256,
            validate_contract,
        )
        from tam_research.chm_v2_100m_value_projected_eiem import (
            CHMV2100MValueProjectedEIEMLM,
            EXPECTED_VP_EIEM_PARAMETERS,
        )
        from tam_research.data import TokenBin

        contract = validate_contract()
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1133 expected NVIDIA L4, got {device_name!r}")

        actual_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
        plan = engineering_start_plan(len(train_data.data))
        plan_digest = start_plan_sha256(plan)
        if train_data._device_cache:
            raise RuntimeError("#1133 TokenBin device cache nonempty before benchmark")

        _seed_all(ENGINEERING_SEED)
        raw = CHMV1100MEIEMLM()
        _seed_all(ENGINEERING_SEED)
        vp = CHMV2100MValueProjectedEIEMLM()
        if _count_parameters(raw) != EXPECTED_EIEM_PARAMETERS:
            raise RuntimeError("#1133 RAW parameter count drift")
        if _count_parameters(vp) != EXPECTED_VP_EIEM_PARAMETERS:
            raise RuntimeError("#1133 VP parameter count drift")

        before_raw = _mem_snapshot(device)
        raw = raw.to(device)
        raw_result = _benchmark_model(
            kind="raw_eiem",
            model=raw,
            train_data=train_data,
            plan=plan,
            plan_digest=plan_digest,
            device=device,
            before_model=before_raw,
        )
        del raw
        torch.cuda.empty_cache()

        before_vp = _mem_snapshot(device)
        vp = vp.to(device)
        vp_result = _benchmark_model(
            kind="vp_eiem",
            model=vp,
            train_data=train_data,
            plan=plan,
            plan_digest=plan_digest,
            device=device,
            before_model=before_vp,
        )
        del vp
        torch.cuda.empty_cache()

        if train_data._device_cache:
            raise RuntimeError("#1133 full-source CUDA cache populated after benchmark")
        final_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
        assert_fingerprint_matches(final_fingerprint, actual_fingerprint)

        decision = classify_systems_preflight(
            raw=raw_result,
            vp=vp_result,
            live_hourly_resource_usd=hourly,
        )
        payload = {
            "status": "COMPLETE",
            **bindings,
            "control_issue": CONTROL_ISSUE,
            "engineering_seed": ENGINEERING_SEED,
            "engineering_seed_is_scientific": False,
            "consumed_scientific_seed_2011121_reused": False,
            "result_root": RESULT_ROOT,
            "final_authority_comment_id": int(authority_comment_id),
            "selected_modal_account": selected_modal_account,
            "account_selection_evidence_sha256": account_selection_evidence_sha256,
            "classification": decision["classification"],
            "passed": decision["passed"],
            "stop_reasons": decision["stop_reasons"],
            "thresholds": decision["thresholds"],
            "projection": decision["projection"],
            "measurements": {"raw_eiem": raw_result, "vp_eiem": vp_result},
            "training_plan_sha256": plan_digest,
            "corpus_fingerprint": actual_fingerprint,
            "contract": contract,
            "device_name": device_name,
            "live_hourly_resource_usd": hourly,
            "gpu_allocation_started": True,
            "engineering_attempt_consumed": True,
            "scientific_interpretation": False,
            "fresh_scientific_seed_authorized": False,
            "automatic_scientific_rerun_authorized": False,
            "stage_d_authorized": False,
        }
        _atomic_write(result_path, payload)
        volume.commit()
        return json.dumps(payload, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "FAILED",
            "classification": "CHM_V2_100M_HOST_STAGED_CORPUS_L4_ENGINEERING_FAILURE",
            **bindings,
            "control_issue": CONTROL_ISSUE,
            "engineering_seed": ENGINEERING_SEED,
            "result_root": RESULT_ROOT,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_unix": time.time(),
            "gpu_allocation_started": True,
            "engineering_attempt_consumed": True,
            "scientific_interpretation": False,
            "fresh_scientific_seed_authorized": False,
            "automatic_retry_authorized": False,
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
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> str:
    bindings = _validate_source(
        source_sha, source_tree, v2_module_sha, model_sha, evaluator_sha, prep_sha,
        dual_account_sha, dual_account_cli_sha, runtime_probe_sha, core_sha,
        runner_sha, workflow_sha,
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    names = ("DISPATCH_RESERVED.json", "ATTEMPT_CONSUMED.json", "RESULT.json", "ATTEMPT_FAILURE.json")
    return json.dumps({
        **bindings,
        "result_root": RESULT_ROOT,
        "root_exists": root.exists(),
        "files": {
            name: json.loads((root / name).read_text(encoding="utf-8")) if (root / name).is_file() else None
            for name in names
        },
    }, sort_keys=True)


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int = 0,
    selected_modal_account: str = "",
    account_selection_evidence_sha256: str = "",
    live_hourly_resource_usd: float = 0.0,
) -> None:
    common = (
        source_sha, source_tree, v2_module_sha, model_sha, evaluator_sha, prep_sha,
        dual_account_sha, dual_account_cli_sha, runtime_probe_sha, core_sha,
        runner_sha, workflow_sha,
    )
    if phase == "inspect":
        print(f"CHM_V2_1133_INSPECT={inspect_zero_gpu.remote(*common)}")
        return
    if phase == "reserve":
        print(
            "CHM_V2_1133_RESERVE="
            + reserve_dispatch.remote(
                *common,
                int(authority_comment_id),
                selected_modal_account,
                account_selection_evidence_sha256,
            )
        )
        return
    if phase == "run":
        print(
            "CHM_V2_1133_RESULT="
            + run_preflight.remote(
                *common,
                int(authority_comment_id),
                selected_modal_account,
                account_selection_evidence_sha256,
                float(live_hourly_resource_usd),
            )
        )
        return
    if phase == "state":
        print(f"CHM_V2_1133_STATE={inspect_state.remote(*common)}")
        return
    raise ValueError(f"unknown phase: {phase}")
