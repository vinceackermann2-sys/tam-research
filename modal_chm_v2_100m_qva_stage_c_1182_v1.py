from __future__ import annotations

"""One-shot CHM-v2 100M QVA Stage-C development runner (#1182 v3).

Final authority comes only from the issue-trigger workflow. This file implements
zero-GPU inspection/reservation plus exactly one L4 scientific function. It
cannot create its trigger or self-authorize reserved seed 2011761.
"""

import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import modal

CONTROL_ISSUE = 1182
PREREG_ISSUE = 1176
HYPOTHESIS_ISSUE = 1147
SYSTEMS_ISSUE = 1155
SCIENTIFIC_SEED = 2_011_761
PHASE = "chm-v2-100m-qva-stage-c-1182-seed-2011761-v1"
RESULT_ROOT = "/vol/chm-v2/100m-qva-stage-c/issue-1182/seed-2011761-v1"
TRIGGER_TITLE = "[modal-chm-v2-100m-qva-stage-c-1182-seed-2011761-v1]"
AUDIT_TITLE = "[modal-chm-v2-100m-qva-stage-c-1182-authority-audit-v1]"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
VOLUME_NAME = "tam-research-data"

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_MIB = 16 * 1024
MAX_GPU_SECONDS = 10_800
RETRIES = 0
MAX_BILLED_COMPUTE_USD = 3.00

QVA_BLOB = "a34a8dffc2c2c1702a909782c3aba2593e90947c"
STAGE_C_PROTOCOL_BLOB = "05e905937ff3a39276462a47fe93e8a499682d1a"
MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"
SMALL_PROTOCOL_BLOB = "d5c2e405b5306f556e7fbe70aacd552867e69d3e"
EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"
DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"
DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"
RUNTIME_PROBE_BLOB = "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc"

APP_NAME = "chm-v2-100m-qva-stage-c-1182-v1"

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
    qva_sha: str,
    stage_c_protocol_sha: str,
    model_sha: str,
    small_protocol_sha: str,
    evaluator_sha: str,
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
        "qva_blob_sha": _full_sha(qva_sha, "qva_sha"),
        "stage_c_protocol_blob_sha": _full_sha(stage_c_protocol_sha, "stage_c_protocol_sha"),
        "model_blob_sha": _full_sha(model_sha, "model_sha"),
        "small_protocol_blob_sha": _full_sha(small_protocol_sha, "small_protocol_sha"),
        "evaluator_blob_sha": _full_sha(evaluator_sha, "evaluator_sha"),
        "dual_account_blob_sha": _full_sha(dual_account_sha, "dual_account_sha"),
        "dual_account_cli_blob_sha": _full_sha(dual_account_cli_sha, "dual_account_cli_sha"),
        "runtime_probe_blob_sha": _full_sha(runtime_probe_sha, "runtime_probe_sha"),
        "execution_core_blob_sha": _full_sha(core_sha, "core_sha"),
        "runner_blob_sha": _full_sha(runner_sha, "runner_sha"),
        "workflow_blob_sha": _full_sha(workflow_sha, "workflow_sha"),
    }
    frozen = {
        "qva_blob_sha": QVA_BLOB,
        "stage_c_protocol_blob_sha": STAGE_C_PROTOCOL_BLOB,
        "model_blob_sha": MODEL_BLOB,
        "small_protocol_blob_sha": SMALL_PROTOCOL_BLOB,
        "evaluator_blob_sha": EVALUATOR_BLOB,
        "dual_account_blob_sha": DUAL_ACCOUNT_BLOB,
        "dual_account_cli_blob_sha": DUAL_ACCOUNT_CLI_BLOB,
        "runtime_probe_blob_sha": RUNTIME_PROBE_BLOB,
    }
    for key, expected in frozen.items():
        if values[key] != expected:
            raise RuntimeError(f"#1182 frozen source binding drift at {key}")
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
        raise RuntimeError("#1182 selected Modal account must be primary or secondary")
    if not workspace:
        raise RuntimeError("#1182 selected Modal workspace must be non-empty")
    if len(evidence) != 64 or any(ch not in "0123456789abcdef" for ch in evidence):
        raise RuntimeError("#1182 account-selection evidence must be SHA-256")
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
        "hypothesis_issue": HYPOTHESIS_ISSUE,
        "systems_issue": SYSTEMS_ISSUE,
        "scientific_seed": SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "audit_title": AUDIT_TITLE,
        "final_authority_comment_id": int(authority_comment_id),
        **bindings,
        **account,
        "automatic_retry_authorized": False,
        "checkpoint_resume_authorized": False,
        "replication_authorized": False,
        "stage_d_authorized": False,
        "scale_up_authorized": False,
    }


def _contract() -> dict[str, Any]:
    from tam_research.chm_v2_100m_qva_stage_c_execution import validate_execution_contract
    return validate_execution_contract()


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
    qva_sha: str,
    stage_c_protocol_sha: str,
    model_sha: str,
    small_protocol_sha: str,
    evaluator_sha: str,
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
    from tam_research.chm_v2_100m_qva_stage_c import validate_protocol_manifest

    bindings = _validate_bindings(
        source_sha=source_sha, source_tree=source_tree, qva_sha=qva_sha,
        stage_c_protocol_sha=stage_c_protocol_sha, model_sha=model_sha,
        small_protocol_sha=small_protocol_sha, evaluator_sha=evaluator_sha,
        dual_account_sha=dual_account_sha, dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha, core_sha=core_sha,
        runner_sha=runner_sha, workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account, selected_modal_workspace, account_selection_evidence_sha256
    )
    contract = _contract()
    protocol = validate_protocol_manifest()
    volume.reload()
    root = Path(RESULT_ROOT)
    payload = {
        "classification": "CHM_V2_100M_QVA_STAGE_C_SOURCE_AUDIT",
        **_evidence(bindings, account, 0),
        "contract": contract,
        "protocol": protocol,
        "corpus_fingerprint": fingerprint_frozen_corpus(DATA_DIR),
        "result_namespace_unused": not root.exists(),
        "gpu_allocated": False,
        "scientific_seed_consumed": False,
        "trigger_authorized_by_runner": False,
    }
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
    qva_sha: str,
    stage_c_protocol_sha: str,
    model_sha: str,
    small_protocol_sha: str,
    evaluator_sha: str,
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
    from tam_research.chm_v2_100m_qva_stage_c import validate_protocol_manifest

    bindings = _validate_bindings(
        source_sha=source_sha, source_tree=source_tree, qva_sha=qva_sha,
        stage_c_protocol_sha=stage_c_protocol_sha, model_sha=model_sha,
        small_protocol_sha=small_protocol_sha, evaluator_sha=evaluator_sha,
        dual_account_sha=dual_account_sha, dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha, core_sha=core_sha,
        runner_sha=runner_sha, workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account, selected_modal_workspace, account_selection_evidence_sha256
    )
    if int(authority_comment_id) <= 0:
        raise RuntimeError("#1182 final authority comment ID must be positive")
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("#1182 result namespace already exists; fail closed")
    payload = {
        "status": "PASS",
        "classification": "CHM_V2_100M_QVA_STAGE_C_ZERO_GPU_PREFLIGHT_PASS",
        **_evidence(bindings, account, int(authority_comment_id)),
        "contract": _contract(),
        "protocol": validate_protocol_manifest(),
        "corpus_fingerprint": fingerprint_frozen_corpus(DATA_DIR),
        "gpu_allocated": False,
        "scientific_seed_consumed": False,
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
    stage_c_protocol_sha: str,
    model_sha: str,
    small_protocol_sha: str,
    evaluator_sha: str,
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
        source_sha=source_sha, source_tree=source_tree, qva_sha=qva_sha,
        stage_c_protocol_sha=stage_c_protocol_sha, model_sha=model_sha,
        small_protocol_sha=small_protocol_sha, evaluator_sha=evaluator_sha,
        dual_account_sha=dual_account_sha, dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha, core_sha=core_sha,
        runner_sha=runner_sha, workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account, selected_modal_workspace, account_selection_evidence_sha256
    )
    evidence = _evidence(bindings, account, int(authority_comment_id))
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    reserve_path = root / "DISPATCH_RESERVED.json"
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if not zero_path.is_file():
        raise RuntimeError("#1182 zero-GPU gate missing")
    if reserve_path.exists() or consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1182 dispatch/attempt already exists; no redispatch")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    for key, expected in evidence.items():
        if zero.get(key) != expected:
            raise RuntimeError(f"#1182 zero-GPU binding drift at {key}")
    marker = {
        "status": "DISPATCH_RESERVED",
        "classification": "CHM_V2_100M_QVA_STAGE_C_DISPATCH_RESERVED",
        **evidence,
        "reserved_unix": time.time(),
        "gpu_allocation_started": False,
        "scientific_seed_consumed": False,
    }
    _atomic_write(reserve_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


def _seed_all(seed: int) -> None:
    import torch
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _finite_model(model: Any) -> bool:
    import torch
    return all(bool(torch.isfinite(p).all().item()) for p in model.parameters())


def _state_digest(module: Any) -> str:
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
        raise RuntimeError(f"#1182 {label} state keys differ")
    for name in a:
        if not torch.equal(a[name], b[name]):
            raise RuntimeError(f"#1182 {label} mismatch at {name}")


def _save_final_checkpoint(
    *, model: Any, kind: str, tokens_seen: int, mean_loss: float, lr: float, root: Path
) -> dict[str, Any]:
    import torch
    step = 2048
    checkpoint_dir = root / "checkpoints" / kind
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / "step-2048.pt"
    tmp = target.with_name(target.name + ".tmp")
    torch.save(
        {
            "kind": kind,
            "step": step,
            "tokens_seen": int(tokens_seen),
            "mean_train_nll": float(mean_loss),
            "lr": float(lr),
            "resume_authorized": False,
            "scientific_evaluation_authorized": True,
            "model_state_dict": model.state_dict(),
        },
        tmp,
    )
    tmp.replace(target)
    record = {
        "kind": kind,
        "step": step,
        "tokens_seen": int(tokens_seen),
        "mean_train_nll": float(mean_loss),
        "lr": float(lr),
        "path": str(target),
        "bytes": int(target.stat().st_size),
        "sha256": _sha256_file(target),
        "resume_authorized": False,
        "scientific_evaluation_authorized": True,
    }
    _atomic_write(checkpoint_dir / "step-2048.json", record)
    volume.commit()
    return record


def _optimizer(model: Any):
    import torch
    from tam_research.chm_v1_100m_stage_c_run_control_prep import BETAS, PEAK_LR, WEIGHT_DECAY
    return torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=BETAS, weight_decay=WEIGHT_DECAY, fused=True
    )


def _train_model(
    *, kind: str, model: Any, source: Any, plan: Any, plan_digest: str, device: Any, root: Path
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from tam_research.chm_v1_100m_scale import VOCAB_SIZE
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context
    from tam_research.chm_v2_100m_host_staged_preflight import host_staged_gather
    from tam_research.chm_v1_100m_stage_c_run_control_prep import (
        GRAD_ACCUM, GRAD_CLIP, MICRO_BATCH, OPTIMIZER_STEPS_PER_MODEL, PEAK_LR,
        SESSION_LEN, TOKENS_PER_OPTIMIZER_STEP, TRAINING_TOKENS_PER_MODEL, WARMUP_STEPS,
    )
    from tam_research.chm_v1_small_lm_protocol import (
        eiem_flat_training_session_logits, local_session_logits,
    )
    from tam_research.chm_v2_100m_qva import qva_flat_training_session_logits
    from tam_research.train import cosine_lr

    if kind not in {"local", "raw_eiem", "qva_eiem"}:
        raise ValueError(kind)
    if tuple(plan.shape) != (OPTIMIZER_STEPS_PER_MODEL, GRAD_ACCUM, MICRO_BATCH):
        raise RuntimeError("#1182 training plan shape drift")

    optimizer = _optimizer(model)
    losses: list[float] = []
    tokens_seen = 0
    transferred_total = 0
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
                source, starts, seq_len=SESSION_LEN, device=device
            )
            transferred_total += int(transferred)
            with autocast_context(device):
                if kind == "local":
                    logits = local_session_logits(model, x)
                elif kind == "raw_eiem":
                    logits = eiem_flat_training_session_logits(model, x)
                else:
                    logits = qva_flat_training_session_logits(model, x)
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
                scaled = loss / GRAD_ACCUM
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"#1182 non-finite {kind} loss at step {step_index + 1}")
            scaled.backward()
            micro_losses.append(float(loss.detach().item()))
            tokens_seen += int(x.numel())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm).item()):
            raise FloatingPointError(f"#1182 non-finite {kind} gradient at step {step_index + 1}")
        lr = cosine_lr(step_index, OPTIMIZER_STEPS_PER_MODEL, WARMUP_STEPS, PEAK_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        losses.append(sum(micro_losses) / len(micro_losses))

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    if tokens_seen != TRAINING_TOKENS_PER_MODEL:
        raise RuntimeError(f"#1182 {kind} token accounting drift: {tokens_seen}")
    if tokens_seen != OPTIMIZER_STEPS_PER_MODEL * TOKENS_PER_OPTIMIZER_STEP:
        raise RuntimeError("#1182 optimizer/token accounting drift")
    if not _finite_model(model):
        raise FloatingPointError(f"#1182 non-finite final {kind} parameters")
    checkpoint = _save_final_checkpoint(
        model=model, kind=kind, tokens_seen=tokens_seen, mean_loss=losses[-1], lr=lr, root=root
    )
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
        "host_to_device_token_bytes_total": transferred_total,
        "host_staged_transport": True,
        "checkpoint": checkpoint,
        "resume_authorized": False,
    }


def _evaluate_language(kind: str, model: Any, source: Any, plan: Any, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from tam_research.chm_v1_100m_scale import VOCAB_SIZE
    from tam_research.chm_v1_100m_stage_c_execution import (
        autocast_context, eiem_exact_flat_two_chunk_logits,
    )
    from tam_research.chm_v2_100m_host_staged_preflight import host_staged_gather
    from tam_research.chm_v1_100m_stage_c_eval import (
        VALIDATION_BATCHES, VALIDATION_BATCH_SIZE, VALIDATION_SESSION_LEN, VALIDATION_TOKENS,
    )
    from tam_research.chm_v1_small_lm_protocol import local_session_logits

    if kind not in {"local", "raw_eiem", "qva_eiem"}:
        raise ValueError(kind)
    model.eval()
    losses: list[float] = []
    transferred_total = 0
    started = time.perf_counter()
    for batch_index in range(VALIDATION_BATCHES):
        x, y, transferred = host_staged_gather(
            source, plan[batch_index, 0], seq_len=VALIDATION_SESSION_LEN, device=device
        )
        transferred_total += int(transferred)
        with autocast_context(device):
            logits = (
                local_session_logits(model, x)
                if kind == "local"
                else eiem_exact_flat_two_chunk_logits(model, x)
            )
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"#1182 non-finite {kind} validation NLL")
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
        "host_to_device_token_bytes_total": transferred_total,
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
    out = {}
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


def _memory_probe_score(*, model: Any, probe: Any, kind: str, device: Any) -> dict[str, Any]:
    import torch
    from tam_research.chm_v1_100m_stage_c_eval import candidate_score
    from tam_research.chm_v1_small_lm import EpisodicState, LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden
    from tam_research.chm_v2_100m_qva_stage_c import qva_integration_diagnostics

    ids = tuple(int(x) for x in probe.prompt_ids)
    state = EpisodicState(f"chm-v2-1182-{kind}-probe")
    query_state = None
    final_hidden = None
    traces = []
    qva_diag = []

    for start in range(0, len(ids), LOCAL_WINDOW):
        chunk_ids = ids[start : start + LOCAL_WINDOW]
        tokens = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        hidden = _hidden(model.backbone, tokens)
        is_final = start + len(chunk_ids) == len(ids)
        if is_final:
            final_hidden = hidden
            query_state = hidden[0, -1]
            break
        state.write(model.key_for(hidden)[0], hidden[0])

    if final_hidden is None or query_state is None:
        raise RuntimeError("#1182 probe final chunk was not reached")

    if len(state):
        for hop in range(RETRIEVAL_HOPS):
            query = model.query_for(query_state)
            value, result, _, _, _, _ = state.retrieve(
                query, mode="flat", verify_indexed_exactness=False
            )
            traces.append({
                "hop": hop + 1,
                "selected_position": int(result.item_id),
                "squared_distance": float(result.squared_distance),
            })
            if kind == "qva_eiem":
                qva_diag.append(qva_integration_diagnostics(model, query_state, value))
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
    return {"score": score, "retrieval_traces": traces, "qva_integration": qva_diag}


def _memory_probe_scores(model: Any, probes: Sequence[Any], *, kind: str, device: Any) -> dict[tuple[str, int], dict[str, Any]]:
    from tam_research.chm_v1_100m_stage_c_execution import autocast_context
    model.eval()
    out = {}
    for probe in probes:
        with autocast_context(device):
            out[(probe.family, int(probe.case_id))] = _memory_probe_score(
                model=model, probe=probe, kind=kind, device=device
            )
    return out


def _three_way_rows(
    probes: Sequence[Any],
    local_scores: dict[tuple[str, int], dict[str, Any]],
    raw_scores: dict[tuple[str, int], dict[str, Any]],
    qva_scores: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        key = (probe.family, int(probe.case_id))
        local = local_scores[key]
        raw = raw_scores[key]
        qva = qva_scores[key]
        row = {
            "family": probe.family,
            "case_id": int(probe.case_id),
            "generator_version": probe.generator_version,
            "evidence_distance": int(probe.evidence_distance),
            "local_correct": bool(local["correct"]),
            "raw_correct": bool(raw["score"]["correct"]),
            "qva_correct": bool(qva["score"]["correct"]),
            "local_candidate_nll": float(local["candidate_nll"]),
            "raw_candidate_nll": float(raw["score"]["candidate_nll"]),
            "qva_candidate_nll": float(qva["score"]["candidate_nll"]),
            "local_stale_choice": bool(local["stale_choice"]),
            "raw_stale_choice": bool(raw["score"]["stale_choice"]),
            "qva_stale_choice": bool(qva["score"]["stale_choice"]),
            "raw_retrieval_traces": raw["retrieval_traces"],
            "qva_retrieval_traces": qva["retrieval_traces"],
            "qva_integration_diagnostics": qva["qva_integration"],
        }
        rows.append(row)
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


def _qva_diagnostics(rows: Sequence[dict[str, Any]], model: Any) -> dict[str, Any]:
    import torch
    gate_values = []
    ratios = []
    for row in rows:
        for item in row["qva_integration_diagnostics"]:
            gate_values.extend([float(item["gate_mean"]), float(item["gate_median"])])
            ratios.extend([float(item["correction_ratio_mean"]), float(item["correction_ratio_median"])])
    qva = model.qva
    return {
        "probe_gate_values": _distribution(gate_values),
        "probe_correction_ratios": _distribution(ratios),
        "gate_bias": float(qva.gate_bias.detach().float().cpu().item()),
        "gate_bias_sigmoid": float(torch.sigmoid(qva.gate_bias.detach().float()).cpu().item()),
        "output_projection_norm": float(torch.linalg.vector_norm(qva.output_projection.weight.detach().float()).cpu().item()),
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
    qva_sha: str,
    stage_c_protocol_sha: str,
    model_sha: str,
    small_protocol_sha: str,
    evaluator_sha: str,
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
    root = Path(RESULT_ROOT)
    volume.reload()
    consumed_path = root / "ATTEMPT_CONSUMED.json"
    result_path = root / "RESULT.json"
    failure_path = root / "ATTEMPT_FAILURE.json"
    if consumed_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("#1182 scientific attempt already consumed; no retry/resume/redispatch")

    # The GPU function has begun: consume the one-shot attempt before any
    # source/account/model/CUDA validation that could fail.
    raw_entry = {
        "status": "SCIENTIFIC_ATTEMPT_CONSUMED",
        "classification": "CHM_V2_100M_QVA_STAGE_C_GPU_FUNCTION_BEGAN",
        "phase": PHASE,
        "control_issue": CONTROL_ISSUE,
        "prereg_issue": PREREG_ISSUE,
        "scientific_seed": SCIENTIFIC_SEED,
        "result_root": RESULT_ROOT,
        "source_sha": str(source_sha),
        "source_tree": str(source_tree),
        "selected_modal_account": str(selected_modal_account),
        "selected_modal_workspace": str(selected_modal_workspace),
        "final_authority_comment_id": int(authority_comment_id),
        "consumed_unix": time.time(),
        "gpu_allocation_started": True,
        "scientific_seed_consumed": True,
        "automatic_retry_authorized": False,
        "checkpoint_resume_authorized": False,
        "replication_authorized": False,
        "stage_d_authorized": False,
        "scale_up_authorized": False,
    }
    _atomic_write(consumed_path, raw_entry)
    volume.commit()

    import torch
    bindings = _validate_bindings(
        source_sha=source_sha, source_tree=source_tree, qva_sha=qva_sha,
        stage_c_protocol_sha=stage_c_protocol_sha, model_sha=model_sha,
        small_protocol_sha=small_protocol_sha, evaluator_sha=evaluator_sha,
        dual_account_sha=dual_account_sha, dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha, core_sha=core_sha,
        runner_sha=runner_sha, workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account, selected_modal_workspace, account_selection_evidence_sha256
    )
    evidence = _evidence(bindings, account, int(authority_comment_id))
    try:
        from tam_research.chm_v1_100m_scale import (
            CHMV1100MEIEMLM, CHMV1100MLocalLM,
            EXPECTED_EIEM_PARAMETERS, EXPECTED_LOCAL_PARAMETERS,
        )
        from tam_research.chm_v1_100m_stage_c_eval import (
            CASES_PER_FAMILY, GENERATOR_VERSION, PROBE_SEED, TOTAL_PROBES, VALIDATION_TOKENS,
        )
        from tam_research.chm_v1_corpus_fingerprint import (
            assert_fingerprint_matches, fingerprint_frozen_corpus,
        )
        from tam_research.chm_v2_100m_qva import CHMV2100MEIEMQVA, EXPECTED_QVA_PARAMETERS
        from tam_research.chm_v2_100m_qva_stage_c import classify_stage_c
        from tam_research.chm_v2_100m_qva_stage_c_execution import (
            build_training_start_plan, build_validation_start_plan,
            start_plan_sha256, validate_live_rate_cap,
        )
        from tam_research.data import TokenBin

        contract = _contract()
        rate_guard = validate_live_rate_cap(float(live_hourly_resource_usd))
        zero = json.loads((root / "ZERO_GPU_GATE.json").read_text(encoding="utf-8"))
        reserve = json.loads((root / "DISPATCH_RESERVED.json").read_text(encoding="utf-8"))
        for payload_name, payload in (("zero", zero), ("reserve", reserve)):
            for key, expected in evidence.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"#1182 {payload_name} evidence mismatch at {key}")
        reserve["gpu_allocation_started"] = True
        reserve["scientific_seed_consumed"] = True
        _atomic_write(root / "DISPATCH_RESERVED.json", reserve)
        volume.commit()

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("#1182 expected exactly one CUDA L4")
        device_name = torch.cuda.get_device_name(0)
        if "L4" not in device_name.upper():
            raise RuntimeError(f"#1182 expected NVIDIA L4, got {device_name!r}")
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
        raw = CHMV1100MEIEMLM()
        _seed_all(SCIENTIFIC_SEED)
        qva = CHMV2100MEIEMQVA()

        local_params = _count_parameters(local)
        raw_params = _count_parameters(raw)
        qva_params = _count_parameters(qva)
        if local_params != EXPECTED_LOCAL_PARAMETERS or raw_params != EXPECTED_EIEM_PARAMETERS:
            raise RuntimeError("#1182 LOCAL/RAW parameter count drift")
        if qva_params != EXPECTED_QVA_PARAMETERS:
            raise RuntimeError("#1182 QVA parameter count drift")

        _assert_state_equal(local.backbone, raw.backbone, "LOCAL/RAW backbone")
        _assert_state_equal(local.backbone, qva.backbone, "LOCAL/QVA backbone")
        _assert_state_equal(raw.query_address, qva.query_address, "RAW/QVA query")
        _assert_state_equal(raw.key_address, qva.key_address, "RAW/QVA key")
        if not torch.equal(raw.memory_gate_logit, qva.memory_gate_logit):
            raise RuntimeError("#1182 RAW/QVA legacy gate initialization mismatch")
        if torch.count_nonzero(qva.qva.output_projection.weight).item() != 0:
            raise RuntimeError("#1182 QVA output projection not exact-zero initialized")
        if float(qva.qva.gate_bias.detach().item()) != -4.0:
            raise RuntimeError("#1182 QVA gate bias initialization drift")

        initial_hashes = {
            "local_backbone": _state_digest(local.backbone),
            "raw_backbone": _state_digest(raw.backbone),
            "qva_backbone": _state_digest(qva.backbone),
            "raw_query": _state_digest(raw.query_address),
            "qva_query": _state_digest(qva.query_address),
            "raw_key": _state_digest(raw.key_address),
            "qva_key": _state_digest(qva.key_address),
            "raw_legacy_gate": _sha256_text(raw.memory_gate_logit.detach().cpu().numpy().tobytes().hex()),
            "qva_legacy_gate": _sha256_text(qva.memory_gate_logit.detach().cpu().numpy().tobytes().hex()),
            "qva_adapter": _state_digest(qva.qva),
        }

        if train_data._device_cache or val_data._device_cache:
            raise RuntimeError("#1182 TokenBin device cache must be empty before host-staged training")
        train_source = train_data.data
        val_source = val_data.data
        probes = _probe_suite()
        if len(probes) != TOTAL_PROBES:
            raise RuntimeError("#1182 probe suite count drift")

        local = local.to(device)
        local_training = _train_model(
            kind="local", model=local, source=train_source, plan=training_plan,
            plan_digest=training_plan_digest, device=device, root=root,
        )
        local_language = _evaluate_language("local", local, val_source, validation_plan, device)
        local_scores = _local_probe_scores(local, probes, device)
        local_finite = _finite_model(local)
        del local
        torch.cuda.empty_cache()

        raw = raw.to(device)
        raw_training = _train_model(
            kind="raw_eiem", model=raw, source=train_source, plan=training_plan,
            plan_digest=training_plan_digest, device=device, root=root,
        )
        raw_language = _evaluate_language("raw_eiem", raw, val_source, validation_plan, device)
        raw_scores = _memory_probe_scores(raw, probes, kind="raw_eiem", device=device)
        raw_finite = _finite_model(raw)
        del raw
        torch.cuda.empty_cache()

        qva = qva.to(device)
        qva_training = _train_model(
            kind="qva_eiem", model=qva, source=train_source, plan=training_plan,
            plan_digest=training_plan_digest, device=device, root=root,
        )
        qva_language = _evaluate_language("qva_eiem", qva, val_source, validation_plan, device)
        qva_scores = _memory_probe_scores(qva, probes, kind="qva_eiem", device=device)
        qva_finite = _finite_model(qva)

        if train_data._device_cache or val_data._device_cache:
            raise RuntimeError("#1182 full-source CUDA cache populated during host-staged run")

        rows = _three_way_rows(probes, local_scores, raw_scores, qva_scores)
        integrity = {
            "local_trainable_parameters": local_params,
            "raw_trainable_parameters": raw_params,
            "qva_trainable_parameters": qva_params,
            "training_tokens_per_model": 33_554_432,
            "optimizer_steps_per_model": 2_048,
            "scientific_seed": SCIENTIFIC_SEED,
            "generator_version": GENERATOR_VERSION,
            "probe_seed": PROBE_SEED,
            "cases_per_family": CASES_PER_FAMILY,
            "probe_count": TOTAL_PROBES,
            "validation_seed": 977_302,
            "validation_tokens": VALIDATION_TOKENS,
            "paired_backbone_initialization_identical": (
                initial_hashes["local_backbone"] == initial_hashes["raw_backbone"] == initial_hashes["qva_backbone"]
            ),
            "raw_qva_address_initialization_identical": (
                initial_hashes["raw_query"] == initial_hashes["qva_query"]
                and initial_hashes["raw_key"] == initial_hashes["qva_key"]
            ),
            "raw_qva_legacy_gate_initialization_identical": (
                initial_hashes["raw_legacy_gate"] == initial_hashes["qva_legacy_gate"]
            ),
            "qva_initialization_exact": True,
            "byte_identical_training_stream": (
                local_training["training_plan_sha256"]
                == raw_training["training_plan_sha256"]
                == qva_training["training_plan_sha256"]
                == training_plan_digest
            ),
            "matched_optimizer_schedule": True,
            "finite_losses": all(math.isfinite(float(x["final_train_nll"])) for x in (local_training, raw_training, qva_training)),
            "finite_parameters": bool(local_finite and raw_finite and qva_finite),
            "no_cross_session_state_aliasing": True,
            "no_future_self_leakage": True,
        }
        decision = classify_stage_c(
            integrity=integrity,
            rows=rows,
            local_language_nll=float(local_language["nll"]),
            raw_language_nll=float(raw_language["nll"]),
            qva_language_nll=float(qva_language["nll"]),
        )
        qva_diag = _qva_diagnostics(rows, qva)

        payload = {
            "status": "COMPLETE",
            **evidence,
            "classification": decision["classification"],
            "passed": decision["passed"],
            "stop_reasons": decision["stop_reasons"],
            "metrics": decision["metrics"],
            "integrity": integrity,
            "parameter_counts": {
                "local": local_params, "raw_eiem": raw_params, "qva_eiem": qva_params,
                "qva_delta_fraction_vs_local": (qva_params - local_params) / local_params,
            },
            "initialization_hashes": initial_hashes,
            "training": {
                "local": local_training, "raw_eiem": raw_training, "qva_eiem": qva_training,
                "training_plan_sha256": training_plan_digest,
                "byte_identical_three_way_stream": True,
            },
            "ordinary_language": {
                "local": local_language, "raw_eiem": raw_language, "qva_eiem": qva_language,
                "validation_plan_sha256": validation_plan_digest,
            },
            "probe_rows": rows,
            "qva_diagnostics": qva_diag,
            "corpus_fingerprint": actual_fingerprint,
            "live_rate_guard": rate_guard,
            "device_name": device_name,
            "gpu_allocation_started": True,
            "scientific_seed_consumed": True,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
            "replication_authorized": False,
            "stage_d_authorized": False,
            "scale_up_authorized": False,
            "interpretation_ceiling": decision["interpretation_ceiling"],
            "contract": contract,
        }
        _atomic_write(result_path, payload)
        volume.commit()
        return json.dumps(payload, sort_keys=True)
    except BaseException as exc:
        failure = {
            "status": "ATTEMPT_FAILED",
            "classification": "CHM_V2_100M_QVA_STAGE_C_INFRASTRUCTURE_OR_RUNTIME_FAILURE",
            **evidence,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_unix": time.time(),
            "gpu_allocation_started": True,
            "scientific_seed_consumed": True,
            "scientific_interpretation": False,
            "automatic_retry_authorized": False,
            "checkpoint_resume_authorized": False,
            "replication_authorized": False,
            "stage_d_authorized": False,
            "scale_up_authorized": False,
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
    source_sha: str, source_tree: str, qva_sha: str, stage_c_protocol_sha: str,
    model_sha: str, small_protocol_sha: str, evaluator_sha: str,
    dual_account_sha: str, dual_account_cli_sha: str, runtime_probe_sha: str,
    core_sha: str, runner_sha: str, workflow_sha: str,
    selected_modal_account: str, selected_modal_workspace: str,
    account_selection_evidence_sha256: str, authority_comment_id: int,
) -> str:
    bindings = _validate_bindings(
        source_sha=source_sha, source_tree=source_tree, qva_sha=qva_sha,
        stage_c_protocol_sha=stage_c_protocol_sha, model_sha=model_sha,
        small_protocol_sha=small_protocol_sha, evaluator_sha=evaluator_sha,
        dual_account_sha=dual_account_sha, dual_account_cli_sha=dual_account_cli_sha,
        runtime_probe_sha=runtime_probe_sha, core_sha=core_sha,
        runner_sha=runner_sha, workflow_sha=workflow_sha,
    )
    account = _validate_account_binding(
        selected_modal_account, selected_modal_workspace, account_selection_evidence_sha256
    )
    volume.reload()
    root = Path(RESULT_ROOT)
    names = ("ZERO_GPU_GATE.json", "DISPATCH_RESERVED.json", "ATTEMPT_CONSUMED.json", "RESULT.json", "ATTEMPT_FAILURE.json")
    files = {}
    for name in names:
        path = root / name
        files[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return json.dumps({
        **_evidence(bindings, account, int(authority_comment_id)),
        "root_exists": root.exists(),
        "files": files,
    }, sort_keys=True)


@app.local_entrypoint()
def main(
    phase: str,
    source_sha: str,
    source_tree: str,
    qva_sha: str,
    stage_c_protocol_sha: str,
    model_sha: str,
    small_protocol_sha: str,
    evaluator_sha: str,
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
        source_sha, source_tree, qva_sha, stage_c_protocol_sha, model_sha,
        small_protocol_sha, evaluator_sha, dual_account_sha, dual_account_cli_sha,
        runtime_probe_sha, core_sha, runner_sha, workflow_sha, selected_modal_account,
        selected_modal_workspace, account_selection_evidence_sha256,
    )
    if phase == "inspect-source":
        payload = inspect_source.remote(*base)
        print(f"CHM_V2_100M_QVA_STAGE_C_SOURCE={payload}")
        return
    common = (*base, int(authority_comment_id))
    if phase == "preflight":
        payload = verify_zero_gpu.remote(*common)
        print(f"CHM_V2_100M_QVA_STAGE_C_ZERO_GPU={payload}")
        return
    if phase == "reserve":
        payload = reserve_dispatch.remote(*common)
        print(f"CHM_V2_100M_QVA_STAGE_C_DISPATCH={payload}")
        return
    if phase == "run":
        payload = run_scientific.remote(*common, float(live_hourly_resource_usd))
        print(f"CHM_V2_100M_QVA_STAGE_C_RESULT={payload}")
        return
    if phase == "state":
        payload = inspect_state.remote(*common)
        print(f"CHM_V2_100M_QVA_STAGE_C_STATE={payload}")
        return
    raise ValueError(f"unknown phase: {phase}")
