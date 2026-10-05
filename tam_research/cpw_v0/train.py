from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from tam_research.data import TokenBin
from tam_research.models import parameter_count
from tam_research.train import cosine_lr, seed_all, train_language_model

from .model import CPWV0Config, CPWV0ResearchLM
from .protocol import AUXILIARY_WEIGHT


@torch.no_grad()
def evaluate_cpw(
    model: CPWV0ResearchLM,
    val: TokenBin,
    *,
    seq_len: int,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    losses: list[float] = []
    for _ in range(batches):
        x, y = val.batch(batch_size, seq_len, g, device)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
        losses.append(float(loss))
    mean = sum(losses) / len(losses)
    return {
        "nll": mean,
        "perplexity": math.exp(min(mean, 20.0)),
        "router": model.router_stats(),
    }


def train_cpw(
    *,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    auxiliary_weight: float = AUXILIARY_WEIGHT,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CPW scientific training requires CUDA")
    torch.set_float32_matmul_precision("high")
    seed_all(seed)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    cfg = replace(CPWV0Config(), max_seq_len=max(1024, seq_len))
    model = CPWV0ResearchLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )

    tokens_per_step = micro_batch_size * seq_len * grad_accum_steps
    total_steps = math.ceil(token_budget / tokens_per_step)
    warmup_steps = max(1, int(total_steps * 0.02))
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)

    run_dir = Path(run_root) / f"cpwv0-25m-eager-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_seconds = 0.0
    running_ce = 0.0
    running_aux = 0.0
    tokens_seen = 0

    for step in range(total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        ce_accum = 0.0
        aux_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_data.batch(
                micro_batch_size,
                seq_len,
                batch_gen,
                device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                ce = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                aux = model.auxiliary_loss()
                loss = (ce + auxiliary_weight * aux) / grad_accum_steps
            loss.backward()
            ce_accum += float(ce)
            aux_accum += float(aux)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, total_steps, warmup_steps, 3e-4)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize(device)
        train_seconds += max(time.perf_counter() - step_started, 1e-9)
        tokens_seen = min(token_budget, (step + 1) * tokens_per_step)
        running_ce = ce_accum / grad_accum_steps
        running_aux = aux_accum / grad_accum_steps
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == total_steps:
            print(json.dumps({
                "type": "train",
                "step": step + 1,
                "tokens_seen": tokens_seen,
                "ce": running_ce,
                "aux": running_aux,
                "lr": lr,
                "training_tokens_per_second": tokens_seen / max(train_seconds, 1e-9),
                "router": model.router_stats(),
            }, sort_keys=True), flush=True)

    eval_started = time.perf_counter()
    mid_eval = evaluate_cpw(
        model,
        val_data,
        seq_len=seq_len,
        batches=20,
        batch_size=max(1, micro_batch_size // 2),
        seed=seed + 20_000,
    )
    final_eval = evaluate_cpw(
        model,
        val_data,
        seq_len=seq_len,
        batches=50,
        batch_size=max(1, micro_batch_size // 2),
        seed=seed + 30_000,
    )
    torch.cuda.synchronize(device)
    eval_seconds = max(time.perf_counter() - eval_started, 0.0)

    checkpoint_started = time.perf_counter()
    torch.save({
        "config": asdict(cfg),
        "architecture": "cpwv0",
        "seed": seed,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "auxiliary_weight": auxiliary_weight,
    }, run_dir / "final.pt")
    torch.cuda.synchronize(device)
    checkpoint_seconds = max(time.perf_counter() - checkpoint_started, 0.0)

    elapsed = max(time.perf_counter() - started, 1e-9)
    result = {
        "run_id": run_dir.name,
        "architecture": "cpwv0",
        "seed": seed,
        "parameters": parameter_count(model),
        "tokens_seen": tokens_seen,
        "steps": total_steps,
        "execution": "eager",
        "compile_seconds": 0.0,
        "elapsed_seconds": elapsed,
        "training_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "total_compute_seconds": elapsed,
        "tokens_per_second": tokens_seen / elapsed,
        "training_tokens_per_second": tokens_seen / max(train_seconds, 1e-9),
        "gpu_name": gpu_name,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "final_eval": final_eval,
        "mid_eval": mid_eval,
        "last_train_ce": running_ce,
        "last_auxiliary_loss": running_aux,
        "config": asdict(cfg),
        "training": {
            "seq_len": seq_len,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "tokens_per_step": tokens_per_step,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "auxiliary_weight": auxiliary_weight,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def train_cpw_v0_candidate(
    *,
    architecture: str,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    architecture = architecture.lower()
    if architecture == "transformer":
        return train_language_model(
            architecture="transformer",
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / "transformer"),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            eval_every_tokens=token_budget,
            checkpoint_every_tokens=token_budget,
            resume=False,
            compile_model=False,
        )
    if architecture == "cpwv0":
        return train_cpw(
            seed=seed,
            data_dir=data_dir,
            run_root=str(Path(run_root) / "cpwv0"),
            token_budget=token_budget,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
        )
    raise ValueError("architecture must be transformer or cpwv0")
