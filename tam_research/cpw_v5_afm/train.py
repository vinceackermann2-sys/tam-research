from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.cpw_v4_memory.task import make_associative_batch
from tam_research.cpw_v1.model import CPWV1Config
from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.train import cosine_lr, seed_all

from .model import AFMCPWResearchLM
from .protocol import (
    ARMS,
    GRAD_ACCUM_STEPS,
    MICRO_BATCH_SIZE,
)


def query_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Exact query-position logits without projecting unused earlier positions."""
    if isinstance(model, AFMCPWResearchLM):
        hidden = model.hidden(tokens)
        return model.lm_head(hidden[:, -1])

    t = tokens.size(1)
    cfg = model.cfg
    if t > cfg.max_seq_len:
        raise ValueError("sequence exceeds max_seq_len")
    pos = torch.arange(t, device=tokens.device)
    x = model.token_emb(tokens) + model.pos_emb(pos)[None]

    if isinstance(model, SparseWorldCPWResearchLM):
        workspace = torch.zeros_like(x)
        for block in model.blocks:
            x, workspace = block(x, workspace)
    elif isinstance(model, ResearchLM):
        for block in model.blocks:
            x = block(x)
    else:
        raise TypeError(f"unsupported AFM benchmark model type: {type(model)!r}")

    last = model.norm(x[:, -1])
    return model.lm_head(last)


def build_model(arm: str) -> torch.nn.Module:
    if arm == "transformer":
        return ResearchLM(ModelConfig(architecture="transformer"))
    if arm == "sequence_only":
        return SparseWorldCPWResearchLM("sequence_only", CPWV1Config())
    if arm == "afm_last1":
        return AFMCPWResearchLM(CPWV1Config())
    raise ValueError(f"unknown CPW-v5 arm: {arm}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    *,
    seed: int,
    examples_per_distance: int,
    batch_size: int = 64,
) -> dict[str, object]:
    from tam_research.cpw_v4_memory.protocol import DISTANCES

    model.eval()
    device = next(model.parameters()).device
    by_distance: dict[str, dict[str, float | int]] = {}
    for distance in DISTANCES:
        generator = torch.Generator(device="cpu").manual_seed(
            seed + 1_000_000 + int(distance)
        )
        seen = 0
        correct = 0
        loss_sum = 0.0
        while seen < examples_per_distance:
            current = min(batch_size, examples_per_distance - seen)
            batch = make_associative_batch(
                batch_size=current,
                generator=generator,
                delay=int(distance),
            )
            tokens = batch.tokens.to(device, non_blocking=True)
            targets = batch.targets.to(device, non_blocking=True)
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                logits = query_logits(model, tokens)
            loss = F.cross_entropy(logits.float(), targets, reduction="sum")
            correct += int((logits.argmax(dim=-1) == targets).sum())
            loss_sum += float(loss)
            seen += current
        by_distance[str(distance)] = {
            "examples": seen,
            "accuracy": correct / seen,
            "nll": loss_sum / seen,
        }

    long_keys = ("64", "128", "256")
    return {
        "by_distance": by_distance,
        "long_mean_accuracy": sum(
            float(by_distance[k]["accuracy"]) for k in long_keys
        ) / 3.0,
        "long_mean_nll": sum(
            float(by_distance[k]["nll"]) for k in long_keys
        ) / 3.0,
    }


def train_arm(
    *,
    arm: str,
    seed: int,
    run_root: str,
    steps: int,
    eval_examples_per_distance: int,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown CPW-v5 arm: {arm}")
    if not torch.cuda.is_available():
        raise RuntimeError("CPW-v5 scientific training requires CUDA")

    torch.set_float32_matmul_precision("high")
    seed_all(seed)
    device = torch.device("cuda")
    model = build_model(arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    warmup_steps = max(1, int(steps * 0.02))
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)

    run_dir = Path(run_root) / arm / f"{arm}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    training_seconds = 0.0
    last_loss = float("nan")
    examples_seen = 0

    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            batch = make_associative_batch(
                batch_size=micro_batch_size,
                generator=train_generator,
                delay=None,
            )
            tokens = batch.tokens.to(device, non_blocking=True)
            targets = batch.targets.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = query_logits(model, tokens)
                loss = F.cross_entropy(logits.float(), targets)
                scaled = loss / grad_accum_steps
            scaled.backward()
            accum_loss += float(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, steps, warmup_steps, 3e-4)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize(device)

        training_seconds += max(time.perf_counter() - step_started, 1e-9)
        examples_seen = (step + 1) * micro_batch_size * grad_accum_steps
        last_loss = accum_loss / grad_accum_steps

        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(json.dumps({
                "type": "afm_train",
                "arm": arm,
                "seed": seed,
                "step": step + 1,
                "steps": steps,
                "examples_seen": examples_seen,
                "query_loss": last_loss,
                "lr": lr,
                "examples_per_second": examples_seen / max(training_seconds, 1e-9),
            }, sort_keys=True), flush=True)

    eval_started = time.perf_counter()
    evaluation = evaluate(
        model,
        seed=seed,
        examples_per_distance=eval_examples_per_distance,
    )
    torch.cuda.synchronize(device)
    eval_seconds = max(time.perf_counter() - eval_started, 0.0)
    elapsed = max(time.perf_counter() - started, 1e-9)

    result: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "parameters": parameter_count(model),
        "steps": steps,
        "examples_seen": examples_seen,
        "training_seconds": training_seconds,
        "eval_seconds": eval_seconds,
        "total_compute_seconds": elapsed,
        "examples_per_second": examples_seen / max(training_seconds, 1e-9),
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "last_train_query_loss": last_loss,
        "evaluation": evaluation,
    }
    if isinstance(model, AFMCPWResearchLM):
        result["memory_stats"] = model.memory_stats()

    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result
