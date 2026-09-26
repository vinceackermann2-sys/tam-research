from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count as rlt_parameter_count
from experiments.rlt.train_compiled_pair_1m import CompilableRLT
from tam_research.data import TokenBin
from tam_research.models import ModelConfig, ResearchLM, parameter_count as transformer_parameter_count
from tam_research.train import seed_all

SCIENTIFIC_SEED = 20_261_020
PAIRED_BATCH_SEED = 20_271_020
PAIRED_EVAL_SEED = 20_291_020
COMPILE_PROBE_SEED = 20_301_020
SEQ_LEN = 64
TRAIN_BATCH_SIZE = 768
EVAL_BATCH_SIZE = 64
EVAL_BATCHES = 64
TIME_BUDGET_SECONDS = 60.0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_TIME_RATIO = 0.02
EXPECTED_PARAMETERS = 15_129_344


def rlt_config() -> RLTConfig:
    return RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )


def transformer_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938, architecture="transformer",
    )


def _amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY, fused=True,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _time_lr(elapsed: float) -> float:
    frac = min(max(elapsed / TIME_BUDGET_SECONDS, 0.0), 1.0)
    if frac < WARMUP_TIME_RATIO:
        scale = max(frac / WARMUP_TIME_RATIO, 1e-3)
    else:
        progress = (frac - WARMUP_TIME_RATIO) / (1.0 - WARMUP_TIME_RATIO)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * scale


def _evaluate(
    callable_model: Any,
    parameter_model: nn.Module,
    val_data: TokenBin,
    *,
    vocab_size: int,
) -> dict[str, float]:
    parameter_model.eval()
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(PAIRED_EVAL_SEED)
    losses: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = val_data.batch(EVAL_BATCH_SIZE, SEQ_LEN, gen, device)
            with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
                logits = callable_model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
            losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    nll = sum(losses) / len(losses)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "eval_tokens": EVAL_BATCHES * EVAL_BATCH_SIZE * SEQ_LEN,
        "seconds": time.perf_counter() - started,
    }


def _compile_without_update(
    callable_model: Any,
    parameter_model: nn.Module,
    train_data: TokenBin,
    *,
    vocab_size: int,
) -> dict[str, float]:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED)
    x, y = train_data.batch(TRAIN_BATCH_SIZE, SEQ_LEN, gen, device)
    parameter_model.train()
    parameter_model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
        logits = callable_model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    parameter_model.zero_grad(set_to_none=True)
    return {"seconds": seconds, "loss": float(loss.detach().cpu()), "peak_vram_gb": peak}


def _save_checkpoint(
    path: Path,
    *,
    architecture: str,
    parameter_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    steps: int,
    tokens_seen: int,
    final_eval: dict[str, float],
    train_seconds: float,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": architecture,
        "scientific_seed": SCIENTIFIC_SEED,
        "paired_batch_seed": PAIRED_BATCH_SEED,
        "paired_eval_seed": PAIRED_EVAL_SEED,
        "compute_match": {
            "post_compile_training_time_budget_seconds": TIME_BUDGET_SECONDS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "schedule": "cosine_by_elapsed_post_compile_training_time",
        },
        "steps": steps,
        "tokens_seen": tokens_seen,
        "actual_train_seconds": train_seconds,
        "config": config,
        "model_state_dict": parameter_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_eval": final_eval,
    }, path)
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _train_for_time(
    *,
    name: str,
    callable_model: Any,
    parameter_model: nn.Module,
    train_data: TokenBin,
    val_data: TokenBin,
    vocab_size: int,
    checkpoint_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    device = torch.device("cuda")
    optimizer = _optimizer(parameter_model)
    batch_gen = torch.Generator(device="cpu").manual_seed(PAIRED_BATCH_SEED)
    amp = _amp_dtype()

    initial_eval = _evaluate(callable_model, parameter_model, val_data, vocab_size=vocab_size)
    compile_probe = _compile_without_update(
        callable_model, parameter_model, train_data, vocab_size=vocab_size
    )

    parameter_model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    steps = 0
    tokens_seen = 0
    last_loss = float("nan")
    last_lr = 0.0

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= TIME_BUDGET_SECONDS and steps > 0:
            break
        optimizer.zero_grad(set_to_none=True)
        x, y = train_data.batch(TRAIN_BATCH_SIZE, SEQ_LEN, batch_gen, device)
        last_lr = _time_lr(elapsed)
        for group in optimizer.param_groups:
            group["lr"] = last_lr
        with torch.autocast(device_type="cuda", dtype=amp):
            logits = callable_model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameter_model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        tokens_seen += TRAIN_BATCH_SIZE * SEQ_LEN
        last_loss = float(loss.detach().cpu())

    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
    final_eval = _evaluate(callable_model, parameter_model, val_data, vocab_size=vocab_size)
    checkpoint = _save_checkpoint(
        checkpoint_path,
        architecture=name,
        parameter_model=parameter_model,
        optimizer=optimizer,
        config=config,
        steps=steps,
        tokens_seen=tokens_seen,
        final_eval=final_eval,
        train_seconds=train_seconds,
    )
    return {
        "status": "complete",
        "parameters": sum(p.numel() for p in parameter_model.parameters()),
        "compile_probe": compile_probe,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "steps": steps,
        "tokens_seen": tokens_seen,
        "actual_train_seconds": train_seconds,
        "tokens_per_second": tokens_seen / max(train_seconds, 1e-9),
        "last_train_loss": last_loss,
        "last_lr": last_lr,
        "train_peak_vram_gb": peak_vram,
        "checkpoint": checkpoint,
    }


def train_compute_matched_scale15m_60s_a(
    *,
    data_dir: str,
    checkpoint_dir: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("compute-matched experiment requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))

    # Exact wrapper equivalence at the same scaled configuration.
    cfg = rlt_config()
    seed_all(SCIENTIFIC_SEED)
    reference = RecurrentLoopedTransformer(cfg).to(device)
    seed_all(SCIENTIFIC_SEED)
    candidate_base = RecurrentLoopedTransformer(cfg).to(device)
    candidate_base.load_state_dict(reference.state_dict())
    candidate = CompilableRLT(candidate_base)
    gen = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED + 1)
    x0, y0 = train_data.batch(2, SEQ_LEN, gen, device)
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
        ref_logits = reference(x0)
        cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
    ref_loss.backward(); cand_loss.backward(); torch.cuda.synchronize(device)
    logits_max_abs = float((ref_logits - cand_logits).abs().max().item())
    loss_abs = float((ref_loss - cand_loss).abs().item())
    grad_max_abs = 0.0
    for rp, cp in zip(reference.parameters(), candidate_base.parameters()):
        if rp.grad is None or cp.grad is None:
            if rp.grad is not cp.grad:
                grad_max_abs = float("inf"); break
            continue
        grad_max_abs = max(grad_max_abs, float((rp.grad - cp.grad).abs().max().item()))
    semantic_equivalence = {
        "passed": logits_max_abs == 0.0 and loss_abs == 0.0 and grad_max_abs == 0.0,
        "logits_max_abs": logits_max_abs,
        "loss_abs": loss_abs,
        "grad_max_abs": grad_max_abs,
    }
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()
    if not semantic_equivalence["passed"]:
        raise RuntimeError("exact CompilableRLT semantic equivalence failed")

    seed_all(SCIENTIFIC_SEED)
    rlt_base = RecurrentLoopedTransformer(rlt_config()).to(device)
    if rlt_parameter_count(rlt_base) != EXPECTED_PARAMETERS:
        raise RuntimeError("RLT parameter drift")
    rlt_compiled = torch.compile(CompilableRLT(rlt_base), fullgraph=True, dynamic=False, mode="default")
    rlt = _train_for_time(
        name="rlt",
        callable_model=rlt_compiled,
        parameter_model=rlt_base,
        train_data=train_data,
        val_data=val_data,
        vocab_size=50_257,
        checkpoint_path=Path(checkpoint_dir) / "rlt_final.pt",
        config=asdict(rlt_config()),
    )
    del rlt_compiled, rlt_base
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    seed_all(SCIENTIFIC_SEED)
    tr_base = ResearchLM(transformer_config()).to(device)
    if transformer_parameter_count(tr_base) != EXPECTED_PARAMETERS:
        raise RuntimeError("Transformer parameter drift")
    tr_compiled = torch.compile(tr_base, fullgraph=True, dynamic=False, mode="default")
    transformer = _train_for_time(
        name="transformer",
        callable_model=tr_compiled,
        parameter_model=tr_base,
        train_data=train_data,
        val_data=val_data,
        vocab_size=50_257,
        checkpoint_path=Path(checkpoint_dir) / "transformer_final.pt",
        config=asdict(transformer_config()),
    )

    delta = float(rlt["final_eval"]["nll"] - transformer["final_eval"]["nll"])
    classification = (
        "COMPUTE_MATCHED_SCALE15M_60S_RLT_LOWER_NLL_SINGLE_RUN_HINT"
        if delta < 0
        else "COMPUTE_MATCHED_SCALE15M_60S_NO_RLT_QUALITY_ADVANTAGE"
    )
    return {
        "status": "complete",
        "classification": classification,
        "breakthrough_claim_supported": False,
        "scientific_seed": SCIENTIFIC_SEED,
        "semantic_equivalence": semantic_equivalence,
        "protocol": {
            "post_compile_training_time_budget_seconds_each": TIME_BUDGET_SECONDS,
            "compile_time_excluded_from_compute_budget": True,
            "data_preparation_excluded_from_compute_budget": True,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "learning_rate_schedule": "cosine_by_elapsed_post_compile_training_time",
            "learning_rate": LEARNING_RATE,
            "warmup_time_ratio": WARMUP_TIME_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "eval_batches": EVAL_BATCHES,
            "exact_parameters_each": EXPECTED_PARAMETERS,
        },
        "rlt": rlt,
        "transformer": transformer,
        "derived": {
            "rlt_minus_transformer_final_nll": delta,
            "rlt_minus_transformer_final_perplexity": (
                rlt["final_eval"]["perplexity"] - transformer["final_eval"]["perplexity"]
            ),
            "transformer_tokens_multiple_vs_rlt": (
                transformer["tokens_seen"] / max(rlt["tokens_seen"], 1)
            ),
            "training_time_difference_seconds": (
                rlt["actual_train_seconds"] - transformer["actual_train_seconds"]
            ),
        },
        "interpretation_ceiling": (
            "First steady-state wall-time-matched quality test at exact 15.1M parameter equality. "
            "Each architecture receives 60 seconds of post-compile A100-80GB training at physical "
            "batch 768 with a learning-rate schedule driven by elapsed training-time fraction. "
            "Compilation and data preparation are excluded from the matched budget. One seed is "
            "not sufficient for a general practical-breakthrough claim."
        ),
    }
