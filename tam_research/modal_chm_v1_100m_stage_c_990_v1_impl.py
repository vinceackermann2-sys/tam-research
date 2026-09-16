from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import modal

PHASE = "chm-v1-100m-stage-c-990-v1"
TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-988-seed-977001-v1]"
PARENT_RESEARCH_ISSUE = 977
PREP_ISSUE = 988
CONTROL_ISSUE = 990
SCIENTIFIC_SEED = 977_001
RESULT_ROOT = "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

MODEL_CONTRACT_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
PREP_BLOB = "bc4ed60885aaf991a2d6b9fe8f634ff973f3f722"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_SECONDS = 43_200
MAX_BILLED_COMPUTE_USD = 25.00

APP_NAME = "chm-v1-100m-stage-c-990-v1"
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
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
) -> dict[str, str]:
    bindings = {
        "source_sha": _full_sha(source_sha, "source_sha"),
        "source_tree": _full_sha(source_tree, "source_tree"),
        "model_blob_sha": _full_sha(model_sha, "model_sha"),
        "evaluator_blob_sha": _full_sha(evaluator_sha, "evaluator_sha"),
        "prep_blob_sha": _full_sha(prep_sha, "prep_sha"),
        "execution_core_blob_sha": _full_sha(core_sha, "core_sha"),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
    }
    if bindings["model_blob_sha"] != MODEL_CONTRACT_BLOB:
        raise RuntimeError("#990 model-contract blob drift")
    if bindings["evaluator_blob_sha"] != EVALUATOR_BLOB:
        raise RuntimeError("#990 evaluator blob drift")
    if bindings["prep_blob_sha"] != PREP_BLOB:
        raise RuntimeError("#990 preparation blob drift")
    if SCIENTIFIC_SEED != 977_001 or (PARENT_RESEARCH_ISSUE, PREP_ISSUE, CONTROL_ISSUE) != (977, 988, 990):
        raise RuntimeError("#990 seed/issue binding drift")
    if GPU_CLASS != "L4" or CPU_CORES != 4 or RAM_MIB != 16 * 1024:
        raise RuntimeError("#990 resource binding drift")
    if MAX_SECONDS != 43_200 or MAX_BILLED_COMPUTE_USD != 25.00:
        raise RuntimeError("#990 runtime/cost binding drift")
    return bindings


def _evidence_binding(bindings: dict[str, str], authority_comment_id: int) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "parent_research_issue": PARENT_RESEARCH_ISSUE,
        "prep_issue": PREP_ISSUE,
        "control_issue": CONTROL_ISSUE,
        "scientific_seed": SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "final_authority_comment_id": int(authority_comment_id),
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
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
) -> str:
    from tam_research.chm_v1_100m_scale import stage_a_preflight
    from tam_research.chm_v1_100m_stage_c_execution import validate_execution_contract
    from tam_research.chm_v1_corpus_fingerprint import fingerprint_frozen_corpus

    bindings = _validate_source(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        prep_sha,
        core_sha,
        runner_sha,
        workflow_sha,
    )
    contract = validate_execution_contract()
    if contract["trigger_authorized_by_module"] is not False:
        raise RuntimeError("#990 execution module unexpectedly grants trigger authority")
    if contract["scientific_seed_consumed_by_module"] is not False:
        raise RuntimeError("#990 execution module unexpectedly consumes the scientific seed")

    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#990 Stage-C result namespace already exists; fail closed")

    stage_a = stage_a_preflight()
    fingerprint = fingerprint_frozen_corpus(DATA_DIR)
    payload = {
        "status": "PASS",
        "classification": "CHM_V1_100M_STAGE_C_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence_binding(bindings, authority_comment_id),
        "stage_a": stage_a,
        "execution_contract": contract,
        "corpus_fingerprint": fingerprint,
        "gpu_allocated": False,
        "scientific_seed_consumed": False,
        "scientific_training_started": False,
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
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_source(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        prep_sha,
        core_sha,
        runner_sha,
        workflow_sha,
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file():
        raise RuntimeError("#990 zero-GPU gate missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "scientific_seed": SCIENTIFIC_SEED,
        "scientific_seed_consumed": False,
        "source_sha": bindings["source_sha"],
        "source_tree": bindings["source_tree"],
        "model_blob_sha": bindings["model_blob_sha"],
        "evaluator_blob_sha": bindings["evaluator_blob_sha"],
        "prep_blob_sha": bindings["prep_blob_sha"],
        "execution_core_blob_sha": bindings["execution_core_blob_sha"],
        "runner_blob_sha": bindings["runner_blob_sha"],
        "workflow_blob_sha": bindings["workflow_blob_sha"],
        "final_authority_comment_id": int(authority_comment_id),
    }
    for key, value in expected.items():
        if zero.get(key) != value:
            raise RuntimeError(f"#990 zero-GPU evidence mismatch for {key}")
    if marker_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#990 trigger/result namespace was already used")

    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V1_100M_STAGE_C_DURABLE_PRE_ALLOCATION_MARKER",
        **_evidence_binding(bindings, authority_comment_id),
        "marked_unix": time.time(),
        "gpu_allocation_started": False,
        "scientific_seed_consumed": False,
        "automatic_retry_authorized": False,
        "checkpoint_resume_authorized": False,
        "stage_d_automatically_authorized": False,
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


def _count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _finite_model(model: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def _backbone_identical(local: Any, eiem: Any) -> bool:
    import torch

    local_state = local.backbone.state_dict()
    eiem_state = eiem.backbone.state_dict()
    if local_state.keys() != eiem_state.keys():
        return False
    return all(torch.equal(value, eiem_state[name]) for name, value in local_state.items())


def _state_digest(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        cpu = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _save_audit_checkpoint(
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

    from tam_research.chm_v1_100m_stage_c_run_control_prep import FINAL_EVALUATION_STEP

    checkpoint_dir = root / "checkpoints" / kind
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"step-{step:04d}.pt"
    tmp = target.with_name(target.name + ".tmp")
    torch.save(
        {
            "kind": kind,
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "mean_train_nll": float(mean_loss),
            "lr": float(lr),
            "resume_authorized": False,
            "scientific_evaluation_authorized": int(step) == FINAL_EVALUATION_STEP,
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
        "scientific_evaluation_authorized": int(step) == FINAL_EVALUATION_STEP,
    }
    _atomic_write(checkpoint_dir / f"step-{step:04d}.json", record)
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
    from tam_research.chm_v1_100m_stage_c_execution import (
        autocast_context,
        gather_batch_from_source,
    )
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
    from tam_research.train import cosine_lr

    if kind not in {"local", "eiem"}:
        raise ValueError(kind)
    if tuple(plan.shape) != (OPTIMIZER_STEPS_PER_MODEL, GRAD_ACCUM, MICRO_BATCH):
        raise RuntimeError("#990 training start-plan shape drift")

    optimizer = _optimizer(model)
    losses: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    tokens_seen = 0
    finite_losses = True
    finite_gradients = True

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step_index in range(OPTIMIZER_STEPS_PER_MODEL):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for micro_index in range(GRAD_ACCUM):
            starts = plan[step_index, micro_index]
            x, y = gather_batch_from_source(source, starts, seq_len=SESSION_LEN)
            with autocast_context(device):
                logits = (
                    local_session_logits(model, x)
                    if kind == "local"
                    else eiem_flat_training_session_logits(model, x)
                )
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
                scaled = loss / GRAD_ACCUM
            if not bool(torch.isfinite(loss).item()):
                finite_losses = False
                raise FloatingPointError(f"#990 non-finite {kind} loss at step {step_index + 1}")
            scaled.backward()
            micro_losses.append(float(loss.detach().item()))
            tokens_seen += int(x.numel())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm).item()):
            finite_gradients = False
            raise FloatingPointError(f"#990 non-finite {kind} gradient norm at step {step_index + 1}")
        lr = cosine_lr(step_index, OPTIMIZER_STEPS_PER_MODEL, WARMUP_STEPS, PEAK_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        mean_loss = sum(micro_losses) / len(micro_losses)
        losses.append(mean_loss)

        completed_step = step_index + 1
        if completed_step in CHECKPOINT_STEPS:
            if not _finite_model(model):
                raise FloatingPointError(f"#990 non-finite {kind} parameters at checkpoint {completed_step}")
            checkpoints.append(
                _save_audit_checkpoint(
                    model=model,
                    kind=kind,
                    step=completed_step,
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
        raise RuntimeError(f"#990 {kind} token accounting drift: {tokens_seen}")
    if tokens_seen != OPTIMIZER_STEPS_PER_MODEL * TOKENS_PER_OPTIMIZER_STEP:
        raise RuntimeError("#990 optimizer-step/token accounting mismatch")
    if [item["step"] for item in checkpoints] != list(CHECKPOINT_STEPS):
        raise RuntimeError("#990 audit checkpoint schedule incomplete")
    finite_parameters = _finite_model(model)
    if not finite_parameters:
        raise FloatingPointError(f"#990 non-finite final {kind} parameters")
    return {
        "kind": kind,
        "scientific_seed": SCIENTIFIC_SEED,
        "training_plan_sha256": plan_digest,
        "steps_completed": OPTIMIZER_STEPS_PER_MODEL,
        "tokens_seen": tokens_seen,
        "final_train_nll": losses[-1],
        "loss_trajectory": losses,
        "finite_losses": finite_losses,
        "finite_gradients": finite_gradients,
        "finite_parameters": finite_parameters,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "peak_vram_bytes": peak,
        "checkpoints": checkpoints,
        "resume_authorized": False,
    }


@torch.no_grad()
def _evaluate_local_language(model: Any, source: Any, plan: Any, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_100m_scale import VOCAB_SIZE
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context, gather_batch_from_source
    from tam_research.chm_v1_100m_stage_c_eval import (
        VALIDATION_BATCHES,
        VALIDATION_BATCH_SIZE,
        VALIDATION_SESSION_LEN,
        VALIDATION_TOKENS,
    )
    from tam_research.chm_v1_small_lm_protocol import local_session_logits

    if tuple(plan.shape) != (VALIDATION_BATCHES, 1, VALIDATION_BATCH_SIZE):
        raise RuntimeError("#990 validation start-plan shape drift")
    model.eval()
    losses: list[float] = []
    started = time.perf_counter()
    for batch_index in range(VALIDATION_BATCHES):
        x, y = gather_batch_from_source(source, plan[batch_index, 0], seq_len=VALIDATION_SESSION_LEN)
        with autocast_context(device):
            logits = local_session_logits(model, x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("#990 non-finite LOCAL validation NLL")
        losses.append(float(loss.item()))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = sum(losses) / len(losses)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "tokens_evaluated": VALIDATION_TOKENS,
        "batches": VALIDATION_BATCHES,
        "batch_size": VALIDATION_BATCH_SIZE,
        "wall_seconds": elapsed,
    }


@torch.no_grad()
def _evaluate_eiem_language(model: Any, source: Any, plan: Any, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from tam_research.chm_v1_100m_scale import VOCAB_SIZE
    from tam_research.chm_v1_100m_stage_c_execution import (
        autocast_context,
        eiem_exact_flat_two_chunk_logits,
        gather_batch_from_source,
    )
    from tam_research.chm_v1_100m_stage_c_eval import (
        VALIDATION_BATCHES,
        VALIDATION_BATCH_SIZE,
        VALIDATION_SESSION_LEN,
        VALIDATION_TOKENS,
    )

    if tuple(plan.shape) != (VALIDATION_BATCHES, 1, VALIDATION_BATCH_SIZE):
        raise RuntimeError("#990 validation start-plan shape drift")
    model.eval()
    losses: list[float] = []
    started = time.perf_counter()
    for batch_index in range(VALIDATION_BATCHES):
        x, y = gather_batch_from_source(source, plan[batch_index, 0], seq_len=VALIDATION_SESSION_LEN)
        with autocast_context(device):
            logits = eiem_exact_flat_two_chunk_logits(model, x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("#990 non-finite EIEM validation NLL")
        losses.append(float(loss.item()))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = sum(losses) / len(losses)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "tokens_evaluated": VALIDATION_TOKENS,
        "batches": VALIDATION_BATCHES,
        "batch_size": VALIDATION_BATCH_SIZE,
        "retrieval": "exhaustive-exact-flat-prior-chunk-only",
        "wall_seconds": elapsed,
    }


def _probe_suite() -> list[Any]:
    import tiktoken

    from tam_research.chm_v1_100m_stage_c_eval import generate_aligned_probe_suite

    encoding = tiktoken.get_encoding("gpt2")
    return generate_aligned_probe_suite(encoding.encode)


@torch.no_grad()
def _local_probe_scores(model: Any, probes: Sequence[Any], device: Any) -> dict[tuple[str, int], dict[str, Any]]:
    from tam_research.chm_v1_100m_stage_c_eval import candidate_score, local_aligned_final_logits
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context

    model.eval()
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for probe in probes:
        with autocast_context(device):
            logits = local_aligned_final_logits(model, probe)
        score = candidate_score(
            logits,
            candidate_token_ids=probe.candidate_token_ids,
            answer_token_id=probe.answer_token_id,
            stale_token_ids=probe.stale_token_ids,
        )
        result[(probe.family, int(probe.case_id))] = score
    return result


@torch.no_grad()
def _paired_probe_rows(
    model: Any,
    probes: Sequence[Any],
    local_scores: dict[tuple[str, int], dict[str, Any]],
    device: Any,
) -> list[dict[str, Any]]:
    from tam_research.chm_v1_100m_stage_c_eval import candidate_score, eiem_flat_final_logits
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context
    from tam_research.chm_v1_small_lm import LOCAL_WINDOW

    model.eval()
    rows: list[dict[str, Any]] = []
    for probe in probes:
        with autocast_context(device):
            logits = eiem_flat_final_logits(model, probe.prompt_ids)
        eiem = candidate_score(
            logits,
            candidate_token_ids=probe.candidate_token_ids,
            answer_token_id=probe.answer_token_id,
            stale_token_ids=probe.stale_token_ids,
        )
        local = local_scores[(probe.family, int(probe.case_id))]
        rows.append(
            {
                "family": probe.family,
                "case_id": int(probe.case_id),
                "generator_version": probe.generator_version,
                "evidence_distance": int(probe.evidence_distance),
                "query_chunk_index": int(probe.query_token // LOCAL_WINDOW),
                "local_correct": bool(local["correct"]),
                "eiem_correct": bool(eiem["correct"]),
                "local_candidate_nll": float(local["candidate_nll"]),
                "eiem_candidate_nll": float(eiem["candidate_nll"]),
                "local_stale_choice": bool(local["stale_choice"]),
                "eiem_stale_choice": bool(eiem["stale_choice"]),
            }
        )
    return rows


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=CPU_CORES,
    memory=RAM_MIB,
    timeout=MAX_SECONDS,
    retries=0,
    volumes={"/vol": volume},
)
def run_stage_c(
    source_sha: str,
    source_tree: str,
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    live_hourly_resource_usd: float,
) -> str:
    import torch

    root = Path(RESULT_ROOT)
    volume.reload()
    root.mkdir(parents=True, exist_ok=True)
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#990 scientific attempt already consumed; no retry/resume/redispatch")

    raw_bindings = {
        "source_sha": str(source_sha),
        "source_tree": str(source_tree),
        "model_blob_sha": str(model_sha),
        "evaluator_blob_sha": str(evaluator_sha),
        "prep_blob_sha": str(prep_sha),
        "execution_core_blob_sha": str(core_sha),
        "runner_blob_sha": str(runner_sha),
        "workflow_blob_sha": str(workflow_sha),
    }
    consumed = {
        "status": "SCIENTIFIC_ATTEMPT_CONSUMED",
        "classification": "CHM_V1_100M_STAGE_C_GPU_ALLOCATION_STARTED",
        "phase": PHASE,
        "scientific_seed": SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "final_authority_comment_id": int(authority_comment_id),
        **raw_bindings,
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "scientific_seed_consumed": True,
        "automatic_retry_authorized": False,
        "checkpoint_resume_authorized": False,
        "stage_d_automatically_authorized": False,
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
            BOOTSTRAP_SEED,
            CASES_PER_FAMILY,
            GENERATOR_VERSION,
            PROBE_SEED,
            TOTAL_PROBES,
            VALIDATION_SEED,
            VALIDATION_TOKENS,
            classify_stage_c,
        )
        from tam_research.chm_v1_100m_stage_c_execution import (
            build_training_start_plan,
            build_validation_start_plan,
            start_plan_sha256,
            validate_execution_contract,
        )
        from tam_research.chm_v1_100m_stage_c_run_control_prep import (
            TRAINING_TOKENS_PER_MODEL,
            validate_live_rate_cap,
        )
        from tam_research.chm_v1_corpus_fingerprint import (
            assert_fingerprint_matches,
            fingerprint_frozen_corpus,
        )
        from tam_research.data import TokenBin

        bindings = _validate_source(
            source_sha,
            source_tree,
            model_sha,
            evaluator_sha,
            prep_sha,
            core_sha,
            runner_sha,
            workflow_sha,
        )
        validate_execution_contract()
        rate_guard = validate_live_rate_cap(float(live_hourly_resource_usd))

        zero_path = root / "ZERO_GPU_GATE.json"
        marker_path = root / "DISPATCH_RESERVED.json"
        if not zero_path.is_file() or not marker_path.is_file():
            raise RuntimeError("#990 pre-allocation evidence incomplete")
        zero = json.loads(zero_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_evidence = {
            "source_sha": bindings["source_sha"],
            "source_tree": bindings["source_tree"],
            "model_blob_sha": bindings["model_blob_sha"],
            "evaluator_blob_sha": bindings["evaluator_blob_sha"],
            "prep_blob_sha": bindings["prep_blob_sha"],
            "execution_core_blob_sha": bindings["execution_core_blob_sha"],
            "runner_blob_sha": bindings["runner_blob_sha"],
            "workflow_blob_sha": bindings["workflow_blob_sha"],
            "scientific_seed": SCIENTIFIC_SEED,
            "final_authority_comment_id": int(authority_comment_id),
        }
        for payload_name, payload in (("zero", zero), ("marker", marker)):
            for key, expected in expected_evidence.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"#990 {payload_name} evidence mismatch for {key}")

        marker["gpu_allocation_started"] = True
        marker["scientific_seed_consumed"] = True
        _atomic_write(marker_path, marker)
        volume.commit()

        if not torch.cuda.is_available():
            raise RuntimeError("#990 L4 function started without CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"#990 expected exactly one CUDA device, got {torch.cuda.device_count()}")
        device_name = torch.cuda.get_device_name(0)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#990 expected NVIDIA L4, got {device_name!r}")
        device = torch.device("cuda")

        actual_fingerprint = fingerprint_frozen_corpus(DATA_DIR)
        assert_fingerprint_matches(actual_fingerprint, zero["corpus_fingerprint"])
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
        val_data = TokenBin(str(Path(DATA_DIR) / "val.bin"))
        training_plan = build_training_start_plan(len(train_data.data))
        validation_plan = build_validation_start_plan(len(val_data.data))
        training_plan_digest = start_plan_sha256(training_plan)
        validation_plan_digest = start_plan_sha256(validation_plan)

        _seed_all(SCIENTIFIC_SEED)
        local = CHMV1100MLocalLM()
        _seed_all(SCIENTIFIC_SEED)
        eiem = CHMV1100MEIEMLM()
        local_params = _count_parameters(local)
        eiem_params = _count_parameters(eiem)
        if local_params != EXPECTED_LOCAL_PARAMETERS or eiem_params != EXPECTED_EIEM_PARAMETERS:
            raise RuntimeError("#990 exact Stage-A parameter count drift")
        backbone_identical = _backbone_identical(local, eiem)
        if not backbone_identical:
            raise RuntimeError("#990 paired backbone initialization mismatch")
        initial_backbone_sha256 = _state_digest(local.backbone)
        if initial_backbone_sha256 != _state_digest(eiem.backbone):
            raise RuntimeError("#990 paired backbone digest mismatch")

        train_source = train_data._device_tokens(device)
        val_source = val_data._device_tokens(device)
        probes = _probe_suite()
        if len(probes) != TOTAL_PROBES:
            raise RuntimeError("#990 probe suite count drift")

        local = local.to(device)
        local_training = _train_model(
            kind="local",
            model=local,
            source=train_source,
            plan=training_plan,
            plan_digest=training_plan_digest,
            device=device,
            root=root,
        )
        local_language = _evaluate_local_language(local, val_source, validation_plan, device)
        local_scores = _local_probe_scores(local, probes, device)
        local_final_finite = _finite_model(local)
        del local
        torch.cuda.empty_cache()

        eiem = eiem.to(device)
        eiem_training = _train_model(
            kind="eiem",
            model=eiem,
            source=train_source,
            plan=training_plan,
            plan_digest=training_plan_digest,
            device=device,
            root=root,
        )
        eiem_language = _evaluate_eiem_language(eiem, val_source, validation_plan, device)
        rows = _paired_probe_rows(eiem, probes, local_scores, device)
        eiem_final_finite = _finite_model(eiem)

        if local_training["training_plan_sha256"] != eiem_training["training_plan_sha256"]:
            raise RuntimeError("#990 LOCAL/EIEM training plan digests differ")
        if local_training["tokens_seen"] != TRAINING_TOKENS_PER_MODEL:
            raise RuntimeError("#990 LOCAL token count drift")
        if eiem_training["tokens_seen"] != TRAINING_TOKENS_PER_MODEL:
            raise RuntimeError("#990 EIEM token count drift")

        integrity = {
            "local_trainable_parameters": local_params,
            "eiem_trainable_parameters": eiem_params,
            "training_tokens_per_model": TRAINING_TOKENS_PER_MODEL,
            "backbone_initialization_identical": backbone_identical,
            "byte_identical_training_stream": (
                local_training["training_plan_sha256"]
                == eiem_training["training_plan_sha256"]
                == training_plan_digest
            ),
            "matched_optimizer_schedule": True,
            "finite_losses": bool(local_training["finite_losses"] and eiem_training["finite_losses"]),
            "finite_parameters": bool(local_final_finite and eiem_final_finite),
            "no_cross_session_state_aliasing": True,
            "no_future_self_leakage": True,
            "generator_version": GENERATOR_VERSION,
            "probe_seed": PROBE_SEED,
            "validation_seed": VALIDATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cases_per_family": CASES_PER_FAMILY,
            "probe_count": TOTAL_PROBES,
            "validation_tokens": VALIDATION_TOKENS,
        }
        decision = classify_stage_c(
            integrity=integrity,
            rows=rows,
            local_language_nll=float(local_language["nll"]),
            eiem_language_nll=float(eiem_language["nll"]),
        )

        payload = {
            "status": "COMPLETE",
            **_evidence_binding(bindings, authority_comment_id),
            "classification": decision["classification"],
            "passed": decision["passed"],
            "stop_reasons": decision["stop_reasons"],
            "metrics": decision["metrics"],
            "integrity": integrity,
            "training": {
                "local": local_training,
                "eiem_flat": eiem_training,
                "training_plan_sha256": training_plan_digest,
                "byte_identical_pair_stream": True,
                "initial_backbone_sha256": initial_backbone_sha256,
            },
            "ordinary_language": {
                "local": local_language,
                "eiem_flat": eiem_language,
                "validation_plan_sha256": validation_plan_digest,
                "byte_identical_pair_stream": True,
            },
            "probe_rows": rows,
            "corpus_fingerprint": actual_fingerprint,
            "live_rate_guard": rate_guard,
            "device_name": device_name,
            "gpu_allocation_started": True,
            "scientific_seed_consumed": True,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
            "stage_d_automatically_authorized": False,
            "interpretation_ceiling": decision["interpretation_ceiling"],
        }
        _atomic_write(result_path, payload)
        volume.commit()
        return json.dumps(payload, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "ATTEMPT_FAILED",
            "classification": "CHM_V1_100M_STAGE_C_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
            "phase": PHASE,
            "scientific_seed": SCIENTIFIC_SEED,
            "result_root": RESULT_ROOT,
            "final_authority_comment_id": int(authority_comment_id),
            **raw_bindings,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_unix": time.time(),
            "gpu_allocation_started": True,
            "scientific_seed_consumed": True,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
            "scientific_interpretation": False,
            "stage_d_automatically_authorized": False,
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
    prep_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
) -> str:
    bindings = _validate_source(
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        prep_sha,
        core_sha,
        runner_sha,
        workflow_sha,
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
        **_evidence_binding(bindings, authority_comment_id),
        "root_exists": root.exists(),
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
    model_sha: str,
    evaluator_sha: str,
    prep_sha: str,
    core_sha: str,
    runner_sha: str,
    workflow_sha: str,
    authority_comment_id: int,
    live_hourly_resource_usd: float = 0.0,
) -> None:
    common = (
        source_sha,
        source_tree,
        model_sha,
        evaluator_sha,
        prep_sha,
        core_sha,
        runner_sha,
        workflow_sha,
        int(authority_comment_id),
    )
    if phase == "preflight":
        payload = verify_zero_gpu.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_stage_c.remote(*common, float(live_hourly_resource_usd))
        print(f"CHM_V1_100M_STAGE_C_RESULT={payload}")
        return
    if phase == "inspect":
        payload = inspect_state.remote(*common)
        print(f"CHM_V1_100M_STAGE_C_STATE={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
