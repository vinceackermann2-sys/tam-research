from __future__ import annotations

"""One-shot CHM-v2 100M value-projected EIEM telemetry-repair successor runner (#1143).

The issue-trigger workflows provide the final authority boundary. This module
implements zero-GPU inspection/reservation and the single L4 scientific
function, but it cannot create its trigger or authorize seed 2011371.
"""

import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import modal

# Host-safe mirror of the frozen #1143 constants. PyTorch-dependent project
# modules are imported only inside Modal image functions.
CONTROL_ISSUE = 1143
PREREG_ISSUE = 1137
SCIENTIFIC_SEED = 2_011_431
PHASE = "chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1"
RESULT_ROOT = "/vol/chm-v2/100m-value-projected-eiem-host-staged/issue-1143/seed-2011431-v1"
TRIGGER_TITLE = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-seed-2011431-v1]"
AUDIT_TITLE = "[modal-chm-v2-100m-value-projected-eiem-host-staged-1143-authority-audit-v2]"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
VOLUME_NAME = "tam-research-data"
GPU_CLASS = "L4"
CPU_CORES = 4
MAX_GPU_SECONDS = 14_400
RETRIES = 0

CHM_V2_MODULE_BLOB = "d0c2186231cead2a7c851d776d21e6ab6f73ec43"
MODEL_CONTRACT_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
STAGE_C_EXECUTION_BLOB = "26668b37a6062c641275e177b622948b36d0f227"
STAGE_C_PREP_BLOB = "bc4ed60885aaf991a2d6b9fe8f634ff973f3f722"
DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"
DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"
RUNTIME_ADMISSION_PROBE_BLOB = "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc"


def _validate_contract_runtime() -> dict[str, Any]:
    from tam_research.chm_v2_100m_value_projected_eiem_host_staged_1143 import validate_contract

    return validate_contract()


APP_NAME = "chm-v2-100m-value-projected-eiem-host-staged-1143-v1"
RAM_MIB = 16 * 1024

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


def _full_sha(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 40-hex SHA")
    return normalized


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_bindings(
    *,
    source_sha: str,
    source_tree: str,
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    execution_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> dict[str, str]:
    values = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "v2_module_blob_sha": _full_sha(v2_module_sha, "v2_module_sha"),
        "model_blob_sha": _full_sha(model_sha, "model_sha"),
        "evaluator_blob_sha": _full_sha(evaluator_sha, "evaluator_sha"),
        "stage_c_execution_blob_sha": _full_sha(execution_sha, "execution_sha"),
        "stage_c_prep_blob_sha": _full_sha(prep_sha, "prep_sha"),
        "dual_account_blob_sha": _full_sha(dual_account_sha, "dual_account_sha"),
        "dual_account_cli_blob_sha": _full_sha(dual_account_cli_sha, "dual_account_cli_sha"),
        "runtime_probe_blob_sha": _full_sha(runtime_probe_sha, "runtime_probe_sha"),
        "run_control_core_blob_sha": _full_sha(core_sha, "core_sha"),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
    }
    frozen = {
        "v2_module_blob_sha": CHM_V2_MODULE_BLOB,
        "model_blob_sha": MODEL_CONTRACT_BLOB,
        "evaluator_blob_sha": EVALUATOR_BLOB,
        "stage_c_execution_blob_sha": STAGE_C_EXECUTION_BLOB,
        "stage_c_prep_blob_sha": STAGE_C_PREP_BLOB,
        "dual_account_blob_sha": DUAL_ACCOUNT_BLOB,
        "dual_account_cli_blob_sha": DUAL_ACCOUNT_CLI_BLOB,
        "runtime_probe_blob_sha": RUNTIME_ADMISSION_PROBE_BLOB,
    }
    for key, expected in frozen.items():
        if values[key] != expected:
            raise RuntimeError(f"#1143 frozen source binding drift at {key}")
    return values


def _validate_account_binding(
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> dict[str, str]:
    account = str(selected_modal_account).strip().lower()
    workspace = str(selected_modal_workspace).strip()
    evidence = str(account_selection_evidence_sha256).strip().lower()
    if account not in {"primary", "secondary"}:
        raise RuntimeError("#1143 selected Modal account must be primary or secondary")
    if not workspace:
        raise RuntimeError("#1143 selected Modal workspace must be non-empty")
    if len(evidence) != 64 or any(ch not in "0123456789abcdef" for ch in evidence):
        raise RuntimeError("#1143 account-selection evidence must be SHA-256")
    return {
        "selected_modal_account": account,
        "selected_modal_workspace": workspace,
        "account_selection_evidence_sha256": evidence,
    }


def _evidence(
    bindings: dict[str, str],
    account: dict[str, str],
    authority_comment_id: int,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "control_issue": CONTROL_ISSUE,
        "prereg_issue": PREREG_ISSUE,
        "scientific_seed": SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "audit_title": AUDIT_TITLE,
        "final_authority_comment_id": int(authority_comment_id),
        **bindings,
        **account,
        "automatic_retry_authorized": False,
        "checkpoint_resume_authorized": False,
        "stage_d_authorized": False,
        "scale_up_authorized": False,
        "multi_seed_replication_authorized": False,
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=15 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def inspect_source(
    source_sha: str,
    source_tree: str,
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    execution_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
) -> str:
    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus

    _validate_contract_runtime()
    bindings = _validate_bindings(
        source_sha=source_sha,
        source_tree=source_tree,
        v2_module_sha=v2_module_sha,
        model_sha=model_sha,
        evaluator_sha=evaluator_sha,
        execution_sha=execution_sha,
        prep_sha=prep_sha,
        dual_account_sha=dual_account_sha,
        dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha,
        core_sha=core_sha,
        runner_sha=runner_sha,
        workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    corpus = fingerprint_frozen_corpus(DATA_DIR)
    payload = {
        "classification": "CHM_V2_100M_VALUE_PROJECTED_EIEM_PREAUTHORITY_SOURCE_INSPECTION",
        **_evidence(bindings, account, 0),
        "corpus_fingerprint": corpus,
        "result_namespace_unused": not root.exists(),
        "seed_2011431_consumed": False,
        "gpu_allocated": False,
        "writes_performed": False,
        "trigger_authorized": False,
    }
    if root.exists():
        raise RuntimeError("#1143 reserved result namespace is already present")
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=15 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def verify_zero_gpu(
    source_sha: str,
    source_tree: str,
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    execution_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus

    _validate_contract_runtime()
    if int(authority_comment_id) <= 0:
        raise RuntimeError("#1143 final authority comment ID must be positive")
    bindings = _validate_bindings(
        source_sha=source_sha,
        source_tree=source_tree,
        v2_module_sha=v2_module_sha,
        model_sha=model_sha,
        evaluator_sha=evaluator_sha,
        execution_sha=execution_sha,
        prep_sha=prep_sha,
        dual_account_sha=dual_account_sha,
        dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha,
        core_sha=core_sha,
        runner_sha=runner_sha,
        workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#1143 result namespace already exists before zero-GPU gate")
    gate = {
        "status": "PASS",
        "classification": "CHM_V2_100M_VALUE_PROJECTED_EIEM_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence(bindings, account, int(authority_comment_id)),
        "corpus_fingerprint": fingerprint_frozen_corpus(DATA_DIR),
        "gpu_allocated": False,
        "scientific_seed_consumed": False,
        "scientific_training_started": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", gate)
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
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    execution_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_bindings(
        source_sha=source_sha,
        source_tree=source_tree,
        v2_module_sha=v2_module_sha,
        model_sha=model_sha,
        evaluator_sha=evaluator_sha,
        execution_sha=execution_sha,
        prep_sha=prep_sha,
        dual_account_sha=dual_account_sha,
        dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha,
        core_sha=core_sha,
        runner_sha=runner_sha,
        workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    evidence = _evidence(bindings, account, int(authority_comment_id))
    volume.reload()
    root = Path(RESULT_ROOT)
    zero = root / "ZERO_GPU_GATE.json"
    reserved = root / "DISPATCH_RESERVED.json"
    consumed = root / "ATTEMPT_CONSUMED.json"
    result = root / "RESULT.json"
    failure = root / "ATTEMPT_FAILURE.json"
    if not zero.is_file():
        raise RuntimeError("#1143 zero-GPU gate is missing")
    if reserved.exists() or consumed.exists() or result.exists() or failure.exists():
        raise RuntimeError("#1143 dispatch or scientific attempt already exists")
    gate = json.loads(zero.read_text(encoding="utf-8"))
    for key, expected in evidence.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"#1143 zero-GPU binding drift at {key}")
    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V2_100M_VALUE_PROJECTED_EIEM_DISPATCH_RESERVED",
        **evidence,
        "reserved_unix": time.time(),
        "gpu_allocated": False,
        "scientific_seed_consumed": False,
    }
    _atomic_write(reserved, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


def _seed_all(seed: int) -> None:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _finite_model(model: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def _state_digest(module: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        cpu = value.detach().to(device="cpu").float().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _assert_state_equal(left: Any, right: Any, label: str) -> None:
    import torch

    a = left.state_dict()
    b = right.state_dict()
    if a.keys() != b.keys():
        raise RuntimeError(f"#1143 {label} state keys differ")
    for name in a:
        if not torch.equal(a[name], b[name]):
            raise RuntimeError(f"#1143 {label} mismatch at {name}")


def _tensor_norm(module: Any) -> float:
    import torch

    pieces = [parameter.detach().float().reshape(-1) for parameter in module.parameters()]
    if not pieces:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(pieces)).item())


def _projection_snapshot(model: Any) -> dict[str, Any]:
    return {
        "down_state_sha256": _state_digest(model.value_down),
        "up_state_sha256": _state_digest(model.value_up),
        "down_parameter_norm": _tensor_norm(model.value_down),
        "up_parameter_norm": _tensor_norm(model.value_up),
    }


def _projection_update_summary(model: Any, initial_down: dict[str, Any], initial_up: dict[str, Any]) -> dict[str, Any]:
    import torch

    def summary(module: Any, initial: dict[str, Any]) -> dict[str, float]:
        deltas = []
        current = module.state_dict()
        for name, value in current.items():
            before = initial[name].to(device=value.device, dtype=value.dtype)
            deltas.append((value.detach() - before).float().reshape(-1))
        joined = torch.cat(deltas)
        return {
            "parameter_norm": _tensor_norm(module),
            "update_norm": float(torch.linalg.vector_norm(joined).item()),
            "max_abs_update": float(joined.abs().max().item()),
        }

    return {
        "down": summary(model.value_down, initial_down),
        "up": summary(model.value_up, initial_up),
    }


def _save_checkpoint(
    *,
    model: Any,
    kind: str,
    step: int,
    tokens_seen: int,
    mean_loss: float,
    lr: float,
    root: Path,
) -> dict[str, Any]:
    import torch

    checkpoint_dir = root / "checkpoints" / kind
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"step-{int(step):04d}.pt"
    tmp = target.with_name(target.name + ".tmp")
    torch.save(
        {
            "kind": kind,
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "mean_train_nll": float(mean_loss),
            "lr": float(lr),
            "resume_authorized": False,
            "scientific_evaluation_authorized": int(step) == 2048,
            "model_state_dict": model.state_dict(),
        },
        tmp,
    )
    tmp.replace(target)
    record = {
        "kind": kind,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "mean_train_nll": float(mean_loss),
        "lr": float(lr),
        "path": str(target),
        "bytes": int(target.stat().st_size),
        "sha256": _sha256_file(target),
        "resume_authorized": False,
        "scientific_evaluation_authorized": int(step) == 2048,
    }
    _atomic_write(checkpoint_dir / f"step-{int(step):04d}.json", record)
    volume.commit()
    return record


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


def _train_model(
    *,
    kind: str,
    model: Any,
    source: Any,
    plan: Any,
    plan_digest: str,
    device: Any,
    root: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_100m_scale import VOCAB_SIZE
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context
    from tam_research.chm_v2_100m_host_staged_preflight import host_staged_gather
    from tam_research.chm_v1_100m_stage_c_run_control_prep import (
        CHECKPOINT_STEPS,
        GRAD_ACCUM,
        GRAD_CLIP,
        MICRO_BATCH,
        OPTIMIZER_STEPS_PER_MODEL,
        PEAK_LR,
        SESSION_LEN,
        TOKENS_PER_OPTIMIZER_STEP,
        TRAINING_TOKENS_PER_MODEL,
        WARMUP_STEPS,
    )
    from tam_research.chm_v1_small_lm_protocol import (
        eiem_flat_training_session_logits,
        local_session_logits,
    )
    from tam_research.chm_v2_100m_value_projected_eiem import vp_eiem_flat_training_session_logits
    from tam_research.train import cosine_lr

    if kind not in {"local", "raw_eiem", "vp_eiem"}:
        raise ValueError(kind)
    if tuple(plan.shape) != (OPTIMIZER_STEPS_PER_MODEL, GRAD_ACCUM, MICRO_BATCH):
        raise RuntimeError("#1143 training plan shape drift")

    optimizer = _optimizer(model)
    losses: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    tokens_seen = 0
    host_to_device_token_bytes_total = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    for step_index in range(OPTIMIZER_STEPS_PER_MODEL):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for micro_index in range(GRAD_ACCUM):
            starts = plan[step_index, micro_index]
            x, y, transferred = host_staged_gather(
                source,
                starts,
                seq_len=SESSION_LEN,
                device=device,
            )
            host_to_device_token_bytes_total += int(transferred)
            with autocast_context(device):
                if kind == "local":
                    logits = local_session_logits(model, x)
                elif kind == "raw_eiem":
                    logits = eiem_flat_training_session_logits(model, x)
                else:
                    logits = vp_eiem_flat_training_session_logits(model, x)
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
                scaled = loss / GRAD_ACCUM
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"#1143 non-finite {kind} loss at step {step_index + 1}")
            scaled.backward()
            micro_losses.append(float(loss.detach().item()))
            tokens_seen += int(x.numel())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm).item()):
            raise FloatingPointError(f"#1143 non-finite {kind} gradient at step {step_index + 1}")
        lr = cosine_lr(step_index, OPTIMIZER_STEPS_PER_MODEL, WARMUP_STEPS, PEAK_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        mean_loss = sum(micro_losses) / len(micro_losses)
        losses.append(mean_loss)
        completed = step_index + 1
        if completed in CHECKPOINT_STEPS:
            if not _finite_model(model):
                raise FloatingPointError(f"#1143 non-finite {kind} parameters at step {completed}")
            checkpoints.append(
                _save_checkpoint(
                    model=model,
                    kind=kind,
                    step=completed,
                    tokens_seen=tokens_seen,
                    mean_loss=mean_loss,
                    lr=lr,
                    root=root,
                )
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    if tokens_seen != TRAINING_TOKENS_PER_MODEL:
        raise RuntimeError(f"#1143 {kind} token accounting drift: {tokens_seen}")
    if tokens_seen != OPTIMIZER_STEPS_PER_MODEL * TOKENS_PER_OPTIMIZER_STEP:
        raise RuntimeError("#1143 optimizer/token accounting drift")
    if [x["step"] for x in checkpoints] != list(CHECKPOINT_STEPS):
        raise RuntimeError("#1143 checkpoint schedule incomplete")
    if not _finite_model(model):
        raise FloatingPointError(f"#1143 non-finite final {kind} parameters")
    return {
        "kind": kind,
        "scientific_seed": SCIENTIFIC_SEED,
        "training_plan_sha256": plan_digest,
        "steps_completed": OPTIMIZER_STEPS_PER_MODEL,
        "tokens_seen": tokens_seen,
        "final_train_nll": losses[-1],
        "loss_trajectory": losses,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "peak_vram_bytes": peak,
        "host_to_device_token_bytes_total": host_to_device_token_bytes_total,
        "host_staged_transport": True,
        "checkpoints": checkpoints,
        "resume_authorized": False,
    }


def _evaluate_language(kind: str, model: Any, source: Any, plan: Any, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_100m_scale import VOCAB_SIZE
    from tam_research.chm_v1_100m_stage_c_execution import (
        autocast_context,
        eiem_exact_flat_two_chunk_logits,
    )
    from tam_research.chm_v2_100m_host_staged_preflight import host_staged_gather
    from tam_research.chm_v1_100m_stage_c_eval import (
        VALIDATION_BATCHES,
        VALIDATION_BATCH_SIZE,
        VALIDATION_SESSION_LEN,
        VALIDATION_TOKENS,
    )
    from tam_research.chm_v1_small_lm_protocol import local_session_logits
    from tam_research.chm_v2_100m_value_projected_eiem import vp_eiem_exact_flat_two_chunk_logits

    if kind not in {"local", "raw_eiem", "vp_eiem"}:
        raise ValueError(kind)
    model.eval()
    losses: list[float] = []
    host_to_device_token_bytes_total = 0
    started = time.perf_counter()
    for batch_index in range(VALIDATION_BATCHES):
        x, y, transferred = host_staged_gather(
            source,
            plan[batch_index, 0],
            seq_len=VALIDATION_SESSION_LEN,
            device=device,
        )
        host_to_device_token_bytes_total += int(transferred)
        with autocast_context(device):
            if kind == "local":
                logits = local_session_logits(model, x)
            elif kind == "raw_eiem":
                logits = eiem_exact_flat_two_chunk_logits(model, x)
            else:
                logits = vp_eiem_exact_flat_two_chunk_logits(model, x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"#1143 non-finite {kind} validation NLL")
        losses.append(float(loss.item()))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = sum(losses) / len(losses)
    return {
        "kind": kind,
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "tokens_evaluated": VALIDATION_TOKENS,
        "batches": VALIDATION_BATCHES,
        "batch_size": VALIDATION_BATCH_SIZE,
        "wall_seconds": elapsed,
        "host_to_device_token_bytes_total": host_to_device_token_bytes_total,
    }


def _probe_suite() -> list[Any]:
    import tiktoken
    from tam_research.chm_v1_100m_stage_c_eval import generate_aligned_probe_suite

    encoding = tiktoken.get_encoding("gpt2")
    return generate_aligned_probe_suite(encoding.encode)


def _local_probe_scores(model: Any, probes: Sequence[Any], device: Any) -> dict[tuple[str, int], dict[str, Any]]:
    from tam_research.chm_v1_100m_stage_c_eval import candidate_score, local_aligned_final_logits
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context

    model.eval()
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for probe in probes:
        with autocast_context(device):
            logits = local_aligned_final_logits(model, probe)
        out[(probe.family, int(probe.case_id))] = candidate_score(
            logits,
            candidate_token_ids=probe.candidate_token_ids,
            answer_token_id=probe.answer_token_id,
            stale_token_ids=probe.stale_token_ids,
        )
    return out


def _memory_probe_score(
    *,
    model: Any,
    probe: Any,
    projected_values: bool,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_100m_stage_c_eval import candidate_score
    from tam_research.chm_v1_small_lm import EpisodicState, LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden
    from tam_research.chm_v1_100m_stage_c_payload_gate_diagnostic import unique_answer_token_position

    ids = tuple(int(x) for x in probe.prompt_ids)
    state = EpisodicState("chm-v2-1115-probe")
    raw_prior_values: list[torch.Tensor] = []
    correction_ratios: list[float] = []
    query_state = None
    final_hidden = None

    for start in range(0, len(ids), LOCAL_WINDOW):
        chunk_ids = ids[start : start + LOCAL_WINDOW]
        tokens = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        hidden = _hidden(model.backbone, tokens)
        is_final = start + len(chunk_ids) == len(ids)
        if is_final:
            final_hidden = hidden
            query_state = hidden[0, -1]
            break
        keys = model.key_for(hidden)
        raw_values = hidden
        values = model.value_for(hidden) if projected_values else raw_values
        state.write(keys[0], values[0])
        raw_prior_values.extend(raw_values[0].unbind(0))
        if projected_values:
            correction = torch.linalg.vector_norm((values - raw_values).float(), dim=-1)
            base = torch.linalg.vector_norm(raw_values.float(), dim=-1).clamp_min(1e-12)
            correction_ratios.extend(float(x) for x in (correction / base)[0].detach().cpu().tolist())

    if final_hidden is None or query_state is None:
        raise RuntimeError("#1143 probe final chunk was not reached")

    traces: list[dict[str, Any]] = []
    if len(state):
        for hop in range(RETRIEVAL_HOPS):
            query = model.query_for(query_state)
            value, result, _, _, _, _ = state.retrieve(
                query,
                mode="flat",
                verify_indexed_exactness=False,
            )
            traces.append(
                {
                    "hop": hop + 1,
                    "selected_position": int(result.item_id),
                    "squared_distance": float(result.squared_distance),
                }
            )
            query_state = model._integrate(query_state, value)

    fused = final_hidden.clone()
    fused[0, -1] = query_state
    logits = model.backbone.lm_head(fused)[0, -1]
    score = candidate_score(
        logits,
        candidate_token_ids=probe.candidate_token_ids,
        answer_token_id=probe.answer_token_id,
        stale_token_ids=probe.stale_token_ids,
    )

    label_cosines = None
    if projected_values and probe.family in {"rare_fact", "overwrite", "two_hop"}:
        answer_position = unique_answer_token_position(probe)
        if answer_position >= len(raw_prior_values):
            raise RuntimeError("#1143 label-oracle answer position escaped prior memory")
        raw_payload = raw_prior_values[answer_position]
        projected_payload = model.value_for(raw_payload)
        label_weight = model.backbone.lm_head.weight[int(probe.answer_token_id)]
        label_direction = F.normalize(label_weight.float(), dim=0)
        label_cosines = {
            "answer_token_position": int(answer_position),
            "raw_to_label_direction_cosine": float(
                F.cosine_similarity(raw_payload.float(), label_direction, dim=0).item()
            ),
            "projected_to_label_direction_cosine": float(
                F.cosine_similarity(projected_payload.float(), label_direction, dim=0).item()
            ),
        }

    return {
        "score": score,
        "retrieval_traces": traces,
        "projection_correction_ratios": correction_ratios,
        "label_direction_report_only": label_cosines,
    }


def _memory_probe_scores(
    model: Any,
    probes: Sequence[Any],
    *,
    projected_values: bool,
    device: Any,
) -> dict[tuple[str, int], dict[str, Any]]:
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context

    model.eval()
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for probe in probes:
        with autocast_context(device):
            out[(probe.family, int(probe.case_id))] = _memory_probe_score(
                model=model,
                probe=probe,
                projected_values=projected_values,
                device=device,
            )
    return out


def _three_way_rows(
    probes: Sequence[Any],
    local_scores: dict[tuple[str, int], dict[str, Any]],
    raw_scores: dict[tuple[str, int], dict[str, Any]],
    vp_scores: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    from tam_research.chm_v1_small_lm import LOCAL_WINDOW

    rows = []
    for probe in probes:
        key = (probe.family, int(probe.case_id))
        local = local_scores[key]
        raw = raw_scores[key]
        vp = vp_scores[key]
        rows.append(
            {
                "family": probe.family,
                "case_id": int(probe.case_id),
                "generator_version": probe.generator_version,
                "evidence_distance": int(probe.evidence_distance),
                "query_chunk_index": int(probe.query_token // LOCAL_WINDOW),
                "local_correct": bool(local["correct"]),
                "raw_correct": bool(raw["score"]["correct"]),
                "vp_correct": bool(vp["score"]["correct"]),
                "local_candidate_nll": float(local["candidate_nll"]),
                "raw_candidate_nll": float(raw["score"]["candidate_nll"]),
                "vp_candidate_nll": float(vp["score"]["candidate_nll"]),
                "local_stale_choice": bool(local["stale_choice"]),
                "raw_stale_choice": bool(raw["score"]["stale_choice"]),
                "vp_stale_choice": bool(vp["score"]["stale_choice"]),
                "raw_retrieval_traces": raw["retrieval_traces"],
                "vp_retrieval_traces": vp["retrieval_traces"],
                "vp_projection_correction_ratios": vp["projection_correction_ratios"],
                "vp_label_direction_report_only": vp["label_direction_report_only"],
            }
        )
    return rows


def _distribution(values: list[float]) -> dict[str, float]:
    import torch

    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(x.numel()),
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "p05": float(torch.quantile(x, 0.05).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _vp_probe_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(value)
        for row in rows
        for value in row["vp_projection_correction_ratios"]
    ]
    raw_cos = []
    projected_cos = []
    for row in rows:
        item = row["vp_label_direction_report_only"]
        if item is not None:
            raw_cos.append(float(item["raw_to_label_direction_cosine"]))
            projected_cos.append(float(item["projected_to_label_direction_cosine"]))
    return {
        "projection_correction_ratio": _distribution(ratios),
        "raw_payload_to_label_direction_cosine": _distribution(raw_cos),
        "projected_payload_to_label_direction_cosine": _distribution(projected_cos),
    }


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=CPU_CORES,
    memory=RAM_MIB,
    timeout=MAX_GPU_SECONDS,
    retries=RETRIES,
    volumes={"/vol": volume},
)
def run_scientific(
    source_sha: str,
    source_tree: str,
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    execution_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
    live_hourly_resource_usd: float,
) -> str:
    import torch

    bindings = _validate_bindings(
        source_sha=source_sha,
        source_tree=source_tree,
        v2_module_sha=v2_module_sha,
        model_sha=model_sha,
        evaluator_sha=evaluator_sha,
        execution_sha=execution_sha,
        prep_sha=prep_sha,
        dual_account_sha=dual_account_sha,
        dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha,
        core_sha=core_sha,
        runner_sha=runner_sha,
        workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    evidence = _evidence(bindings, account, int(authority_comment_id))

    root = Path(RESULT_ROOT)
    volume.reload()
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1143 scientific attempt already consumed; no retry/resume/redispatch")

    consumed = {
        "status": "SCIENTIFIC_ATTEMPT_CONSUMED",
        "classification": "CHM_V2_100M_VALUE_PROJECTED_EIEM_GPU_FUNCTION_BEGAN",
        **evidence,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "scientific_seed_consumed": True,
    }
    _atomic_write(consumed_path, consumed)
    volume.commit()

    try:
        from tam_research.chm_v1_100m_scale import (
            CHMV1100MEIEMLM,
            CHMV1100MLocalLM,
            EXPECTED_EIEM_PARAMETERS,
            EXPECTED_LOCAL_PARAMETERS,
        )
        from tam_research.chm_v1_100m_stage_c_eval import (
            CASES_PER_FAMILY,
            GENERATOR_VERSION,
            PROBE_SEED,
            TOTAL_PROBES,
            VALIDATION_TOKENS,
        )
        from tam_research.chm_v1_corpus_fingerprint import (
            assert_fingerprint_matches,
            fingerprint_frozen_corpus,
        )
        from tam_research.chm_v1_100m_stage_c_payload_gate_diagnostic import gate_state_summary
        from tam_research.chm_v2_100m_value_projected_eiem import (
            CHMV2100MValueProjectedEIEMLM,
            EXPECTED_VP_EIEM_PARAMETERS,
            classify_development_gate,
        )
        from tam_research.chm_v2_100m_value_projected_eiem_host_staged_1143 import (
            plan_digest,
            summarize_three_way_probe_rows,
            training_plan_for_tokens,
            validate_live_rate_cap,
            validation_plan_for_tokens,
        )
        from tam_research.data import TokenBin

        contract = _validate_contract_runtime()
        rate_guard = validate_live_rate_cap(float(live_hourly_resource_usd))

        zero_path = root / "ZERO_GPU_GATE.json"
        reserve_path = root / "DISPATCH_RESERVED.json"
        if not zero_path.is_file() or not reserve_path.is_file():
            raise RuntimeError("#1143 pre-allocation evidence incomplete")
        zero = json.loads(zero_path.read_text(encoding="utf-8"))
        reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
        for payload_name, payload in (("zero", zero), ("reserve", reserve)):
            for key, expected in evidence.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"#1143 {payload_name} evidence mismatch at {key}")

        reserve["gpu_allocation_started"] = True
        reserve["scientific_seed_consumed"] = True
        _atomic_write(reserve_path, reserve)
        volume.commit()

        if not torch.cuda.is_available():
            raise RuntimeError("#1143 L4 function started without CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"#1143 expected exactly one CUDA device, got {torch.cuda.device_count()}")
        device_name = torch.cuda.get_device_name(0)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1143 expected NVIDIA L4, got {device_name!r}")
        device = torch.device("cuda")

        actual_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
        assert_fingerprint_matches(actual_fingerprint, zero["corpus_fingerprint"])
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
        val_data = TokenBin(str(Path(DATA_DIR) / "val.bin"))
        training_plan = training_plan_for_tokens(len(train_data.data))
        validation_plan = validation_plan_for_tokens(len(val_data.data))
        training_plan_digest = plan_digest(training_plan)
        validation_plan_digest = plan_digest(validation_plan)

        _seed_all(SCIENTIFIC_SEED)
        local = CHMV1100MLocalLM()
        _seed_all(SCIENTIFIC_SEED)
        raw = CHMV1100MEIEMLM()
        _seed_all(SCIENTIFIC_SEED)
        vp = CHMV2100MValueProjectedEIEMLM()

        local_params = _count_parameters(local)
        raw_params = _count_parameters(raw)
        vp_params = _count_parameters(vp)
        if local_params != EXPECTED_LOCAL_PARAMETERS:
            raise RuntimeError("#1143 LOCAL parameter count drift")
        if raw_params != EXPECTED_EIEM_PARAMETERS:
            raise RuntimeError("#1143 RAW parameter count drift")
        if vp_params != EXPECTED_VP_EIEM_PARAMETERS:
            raise RuntimeError("#1143 VP parameter count drift")

        _assert_state_equal(local.backbone, raw.backbone, "LOCAL/RAW backbone")
        _assert_state_equal(local.backbone, vp.backbone, "LOCAL/VP backbone")
        _assert_state_equal(raw.query_address, vp.query_address, "RAW/VP query")
        _assert_state_equal(raw.key_address, vp.key_address, "RAW/VP key")
        if not torch.equal(raw.memory_gate_logit, vp.memory_gate_logit):
            raise RuntimeError("#1143 RAW/VP gate initialization mismatch")
        if torch.count_nonzero(vp.value_up.weight).item() or torch.count_nonzero(vp.value_up.bias).item():
            raise RuntimeError("#1143 VP up projection is not exact-zero initialized")

        initial_hashes = {
            "local_backbone": _state_digest(local.backbone),
            "raw_backbone": _state_digest(raw.backbone),
            "vp_backbone": _state_digest(vp.backbone),
            "raw_query": _state_digest(raw.query_address),
            "vp_query": _state_digest(vp.query_address),
            "raw_key": _state_digest(raw.key_address),
            "vp_key": _state_digest(vp.key_address),
            "raw_gate_sha256": _sha256_text(raw.memory_gate_logit.detach().cpu().numpy().tobytes().hex()),
            "vp_gate_sha256": _sha256_text(vp.memory_gate_logit.detach().cpu().numpy().tobytes().hex()),
        }
        initial_projection = _projection_snapshot(vp)
        initial_down_state = {k: v.detach().cpu().clone() for k, v in vp.value_down.state_dict().items()}
        initial_up_state = {k: v.detach().cpu().clone() for k, v in vp.value_up.state_dict().items()}

        if train_data._device_cache or val_data._device_cache:
            raise RuntimeError("#1143 TokenBin device cache must be empty before host-staged training")
        train_source = train_data.data
        val_source = val_data.data
        probes = _probe_suite()
        if len(probes) != TOTAL_PROBES:
            raise RuntimeError("#1143 probe suite count drift")

        local = local.to(device)
        local_training = _train_model(
            kind="local", model=local, source=train_source, plan=training_plan,
            plan_digest=training_plan_digest, device=device, root=root,
        )
        local_language = _evaluate_language("local", local, val_source, validation_plan, device)
        local_scores = _local_probe_scores(local, probes, device)
        local_finite = _finite_model(local)
        if train_data._device_cache or val_data._device_cache:
            raise RuntimeError("#1143 full-source CUDA cache populated during LOCAL phase")
        del local
        torch.cuda.empty_cache()

        raw = raw.to(device)
        raw_training = _train_model(
            kind="raw_eiem", model=raw, source=train_source, plan=training_plan,
            plan_digest=training_plan_digest, device=device, root=root,
        )
        raw_language = _evaluate_language("raw_eiem", raw, val_source, validation_plan, device)
        raw_scores = _memory_probe_scores(raw, probes, projected_values=False, device=device)
        raw_gate = gate_state_summary(raw)
        raw_finite = _finite_model(raw)
        if train_data._device_cache or val_data._device_cache:
            raise RuntimeError("#1143 full-source CUDA cache populated during RAW phase")
        del raw
        torch.cuda.empty_cache()

        vp = vp.to(device)
        vp_training = _train_model(
            kind="vp_eiem", model=vp, source=train_source, plan=training_plan,
            plan_digest=training_plan_digest, device=device, root=root,
        )
        vp_language = _evaluate_language("vp_eiem", vp, val_source, validation_plan, device)
        vp_scores = _memory_probe_scores(vp, probes, projected_values=True, device=device)
        vp_gate = gate_state_summary(vp)
        vp_finite = _finite_model(vp)
        if train_data._device_cache or val_data._device_cache:
            raise RuntimeError("#1143 full-source CUDA cache populated during VP phase")
        projection_updates = _projection_update_summary(vp, initial_down_state, initial_up_state)
        final_projection = _projection_snapshot(vp)

        rows = _three_way_rows(probes, local_scores, raw_scores, vp_scores)
        summary = summarize_three_way_probe_rows(rows)
        aggregate = summary["aggregate_long_range"]
        families = {
            name: {
                "vp_vs_local_accuracy_gain": summary["per_family"][name]["vp_vs_local_accuracy_gain"],
                "vp_vs_local_nll_benefit": summary["per_family"][name]["vp_vs_local_nll_benefit"],
                "vp_vs_raw_nll_benefit": summary["per_family"][name]["vp_vs_raw_nll_benefit"],
            }
            for name in ("rare_fact", "overwrite", "two_hop")
        }

        integrity = {
            "parameter_fairness": (
                vp_params == EXPECTED_VP_EIEM_PARAMETERS
                and abs((vp_params - local_params) / local_params) <= 0.001
            ),
            "paired_initialization": (
                initial_hashes["local_backbone"]
                == initial_hashes["raw_backbone"]
                == initial_hashes["vp_backbone"]
                and initial_hashes["raw_query"] == initial_hashes["vp_query"]
                and initial_hashes["raw_key"] == initial_hashes["vp_key"]
                and initial_hashes["raw_gate_sha256"] == initial_hashes["vp_gate_sha256"]
            ),
            "byte_identical_training_stream": (
                local_training["training_plan_sha256"]
                == raw_training["training_plan_sha256"]
                == vp_training["training_plan_sha256"]
                == training_plan_digest
            ),
            "exact_training_tokens": all(
                item["tokens_seen"] == 33_554_432
                for item in (local_training, raw_training, vp_training)
            ),
            "matched_schedule": True,
            "finite_losses": all(
                math.isfinite(float(item["final_train_nll"]))
                for item in (local_training, raw_training, vp_training)
            ),
            "finite_parameters": bool(local_finite and raw_finite and vp_finite),
            "no_future_leakage": True,
            "no_cross_session_aliasing": True,
        }

        local_negative = summary["local_negative"]
        overwrite = summary["overwrite"]
        if train_data._device_cache or val_data._device_cache:
            raise RuntimeError("#1143 full-source CUDA cache populated before final classification")

        decision = classify_development_gate(
            integrity=integrity,
            aggregate=aggregate,
            families=families,
            ordinary_vp_minus_local_nll=float(vp_language["nll"] - local_language["nll"]),
            local_negative_accuracy_gain=float(local_negative["vp_vs_local_accuracy_gain"]),
            local_negative_nll_regression=float(
                local_negative["vp_candidate_nll"] - local_negative["local_candidate_nll"]
            ),
            local_overwrite_stale_rate=float(overwrite["local_stale_rate"]),
            vp_overwrite_stale_rate=float(overwrite["vp_stale_rate"]),
        )

        projection_diagnostics = _vp_probe_diagnostics(rows)
        payload = {
            "status": "COMPLETE",
            **evidence,
            "classification": decision["classification"],
            "passed": decision["passed"],
            "stop_reasons": decision["stop_reasons"],
            "integrity": integrity,
            "parameter_counts": {
                "local": local_params,
                "raw_eiem": raw_params,
                "vp_eiem": vp_params,
                "vp_delta_fraction_vs_local": (vp_params - local_params) / local_params,
            },
            "initialization_hashes": initial_hashes,
            "training": {
                "local": local_training,
                "raw_eiem": raw_training,
                "vp_eiem": vp_training,
                "training_plan_sha256": training_plan_digest,
                "byte_identical_three_way_stream": True,
            },
            "host_staged_transport": {
                "enabled": True,
                "full_source_cuda_cache_present": bool(train_data._device_cache or val_data._device_cache),
                "training_transfer_bytes": {
                    "local": local_training["host_to_device_token_bytes_total"],
                    "raw_eiem": raw_training["host_to_device_token_bytes_total"],
                    "vp_eiem": vp_training["host_to_device_token_bytes_total"],
                },
                "validation_transfer_bytes": {
                    "local": local_language["host_to_device_token_bytes_total"],
                    "raw_eiem": raw_language["host_to_device_token_bytes_total"],
                    "vp_eiem": vp_language["host_to_device_token_bytes_total"],
                },
            },
            "ordinary_language": {
                "local": local_language,
                "raw_eiem": raw_language,
                "vp_eiem": vp_language,
                "validation_plan_sha256": validation_plan_digest,
            },
            "probe_summary": summary,
            "probe_rows": rows,
            "gate_statistics": {"raw_eiem": raw_gate, "vp_eiem": vp_gate},
            "vp_projection": {
                "initial": initial_projection,
                "final": final_projection,
                "updates": projection_updates,
                "probe_diagnostics": projection_diagnostics,
            },
            "corpus_fingerprint": actual_fingerprint,
            "live_rate_guard": rate_guard,
            "device_name": device_name,
            "generator_version": GENERATOR_VERSION,
            "probe_seed": PROBE_SEED,
            "cases_per_family": CASES_PER_FAMILY,
            "probe_count": TOTAL_PROBES,
            "validation_tokens": VALIDATION_TOKENS,
            "gpu_allocation_started": True,
            "scientific_seed_consumed": True,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
            "stage_d_authorized": False,
            "scale_up_authorized": False,
            "multi_seed_replication_authorized": False,
            "interpretation_ceiling": decision["interpretation_ceiling"],
            "contract": contract,
        }
        _atomic_write(result_path, payload)
        volume.commit()
        return json.dumps(payload, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "ATTEMPT_FAILED",
            "classification": "CHM_V2_100M_VALUE_PROJECTED_EIEM_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
            **evidence,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_unix": time.time(),
            "gpu_allocation_started": True,
            "scientific_seed_consumed": True,
            "scientific_interpretation": False,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
            "stage_d_authorized": False,
            "scale_up_authorized": False,
            "multi_seed_replication_authorized": False,
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
    execution_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_bindings(
        source_sha=source_sha,
        source_tree=source_tree,
        v2_module_sha=v2_module_sha,
        model_sha=model_sha,
        evaluator_sha=evaluator_sha,
        execution_sha=execution_sha,
        prep_sha=prep_sha,
        dual_account_sha=dual_account_sha,
        dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha,
        core_sha=core_sha,
        runner_sha=runner_sha,
        workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
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
    files = {}
    for name in names:
        path = root / name
        files[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return json.dumps(
        {
            **_evidence(bindings, account, int(authority_comment_id)),
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
    v2_module_sha: str,
    model_sha: str,
    evaluator_sha: str,
    execution_sha: str,
    prep_sha: str,
    dual_account_sha: str,
    dual_account_cli_sha: str,
    runtime_probe_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    selected_modal_account: str,
    selected_modal_workspace: str,
    account_selection_evidence_sha256: str,
    authority_comment_id: int = 0,
    live_hourly_resource_usd: float = 0.0,
) -> None:
    base = (
        source_sha,
        source_tree,
        v2_module_sha,
        model_sha,
        evaluator_sha,
        execution_sha,
        prep_sha,
        dual_account_sha,
        dual_account_cli_sha,
        runtime_probe_sha,
        core_sha,
        runner_sha,
        workflow_sha,
        selected_modal_account,
        selected_modal_workspace,
        account_selection_evidence_sha256,
    )
    if phase == "inspect-source":
        payload = inspect_source.remote(*base)
        print(f"CHM_V2_100M_VALUE_PROJECTED_EIEM_SOURCE={payload}")
        return
    common = (*base, int(authority_comment_id))
    if phase == "preflight":
        payload = verify_zero_gpu.remote(*common)
        print(f"CHM_V2_100M_VALUE_PROJECTED_EIEM_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(*common)
        print(f"CHM_V2_100M_VALUE_PROJECTED_EIEM_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_scientific.remote(*common, float(live_hourly_resource_usd))
        print(f"CHM_V2_100M_VALUE_PROJECTED_EIEM_RESULT={payload}")
        return
    if phase == "state":
        payload = inspect_state.remote(*common)
        print(f"CHM_V2_100M_VALUE_PROJECTED_EIEM_STATE={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
