from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from .data import TokenBin
from .models import ModelConfig, ResearchLM, parameter_count


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    floor_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak * (floor_ratio + (1.0 - floor_ratio) * coeff)


@torch.no_grad()
def evaluate(
    model: ResearchLM,
    val: TokenBin,
    seq_len: int,
    batches: int,
    batch_size: int,
    seed: int,
    forward_model: torch.nn.Module | None = None,
) -> dict[str, object]:
    model.eval()
    runner = forward_model if forward_model is not None else model
    runner.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    losses = []
    for _ in range(batches):
        x, y = val.batch(batch_size, seq_len, g, device)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with ctx:
            logits = runner(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        losses.append(float(loss))
    mean = sum(losses) / len(losses)
    return {
        "nll": mean,
        "perplexity": math.exp(min(mean, 20.0)),
        "router": model.router_stats(),
    }


def train_language_model(
    architecture: str,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int = 100_000_000,
    seq_len: int = 512,
    micro_batch_size: int = 8,
    grad_accum_steps: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.02,
    eval_every_tokens: int = 5_000_000,
    checkpoint_every_tokens: int = 10_000_000,
    resume: bool = True,
    compile_model: bool = False,
    compile_mode: str = "max-autotune",
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("serious training run requires CUDA")
    torch.set_float32_matmul_precision("high")
    seed_all(seed)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    cfg = ModelConfig(architecture=architecture, max_seq_len=max(1024, seq_len))
    model = ResearchLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )

    tokens_per_step = micro_batch_size * seq_len * grad_accum_steps
    total_steps = math.ceil(token_budget / tokens_per_step)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))

    execution = "compiled" if compile_model else "eager"
    run_id = f"{architecture}-25m-{execution}-seed{seed}"
    run_dir = Path(run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    latest_path = run_dir / "latest.pt"

    step = 0
    tokens_seen = 0
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    if resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        tokens_seen = int(ckpt["tokens_seen"])
        batch_gen.set_state(ckpt["batch_gen_state"])
        del ckpt

    forward_model: torch.nn.Module = model
    compile_seconds = 0.0
    compile_peak_vram_gb = 0.0
    if compile_model:
        forward_model = torch.compile(model, mode=compile_mode)
        # Trigger forward and backward compilation before timing real training.
        # This does not update weights and uses an independent RNG stream.
        warm_gen = torch.Generator(device="cpu").manual_seed(seed + 99_999)
        warm_x, warm_y = train_data.batch(
            micro_batch_size, seq_len, warm_gen, device
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        compile_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            warm_logits = forward_model(warm_x)
            warm_loss = F.cross_entropy(
                warm_logits.float().reshape(-1, warm_logits.size(-1)),
                warm_y.reshape(-1),
            )
        warm_loss.backward()
        torch.cuda.synchronize(device)
        compile_seconds = max(time.perf_counter() - compile_start, 0.0)
        compile_peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        optimizer.zero_grad(set_to_none=True)
        del warm_x, warm_y, warm_logits, warm_loss

    start_tokens = tokens_seen
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    next_eval = ((tokens_seen // eval_every_tokens) + 1) * eval_every_tokens
    next_ckpt = (
        (tokens_seen // checkpoint_every_tokens) + 1
    ) * checkpoint_every_tokens

    def write_metric(record: dict[str, object]) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def save_checkpoint() -> None:
        payload = {
            "config": asdict(cfg),
            "architecture": architecture,
            "seed": seed,
            "step": step,
            "tokens_seen": tokens_seen,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "batch_gen_state": batch_gen.get_state(),
            "compile_model": compile_model,
            "compile_mode": compile_mode if compile_model else None,
        }
        tmp = run_dir / "latest.tmp.pt"
        torch.save(payload, tmp)
        tmp.replace(latest_path)

    while step < total_steps and tokens_seen < token_budget:
        model.train()
        forward_model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_data.batch(
                micro_batch_size, seq_len, batch_gen, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_model(x)
                loss = (
                    F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )
                    / grad_accum_steps
                )
            loss.backward()
            running += float(loss) * grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(
            step, total_steps, warmup_steps, learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1
        tokens_seen = min(token_budget, step * tokens_per_step)

        if step == 1 or step % 20 == 0:
            torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - start, 1e-6)
            processed = max(0, tokens_seen - start_tokens)
            record = {
                "type": "train",
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": running / grad_accum_steps,
                "lr": lr,
                "tokens_per_second": processed / elapsed,
                "execution": execution,
            }
            print(json.dumps(record), flush=True)
            write_metric(record)

        if tokens_seen >= next_eval or tokens_seen >= token_budget:
            record = {
                "type": "eval",
                "step": step,
                "tokens_seen": tokens_seen,
                **evaluate(
                    model,
                    val_data,
                    seq_len,
                    20,
                    max(1, micro_batch_size // 2),
                    seed + 20_000,
                    forward_model=forward_model,
                ),
            }
            print(json.dumps(record), flush=True)
            write_metric(record)
            next_eval += eval_every_tokens

        if tokens_seen >= next_ckpt or tokens_seen >= token_budget:
            save_checkpoint()
            next_ckpt += checkpoint_every_tokens

    save_checkpoint()
    final_eval = evaluate(
        model,
        val_data,
        seq_len,
        50,
        max(1, micro_batch_size // 2),
        seed + 30_000,
        forward_model=forward_model,
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = max(time.perf_counter() - start, 1e-6)
    processed_tokens = max(0, tokens_seen - start_tokens)
    tokens_per_second = processed_tokens / elapsed_seconds
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    summary = {
        "run_id": run_id,
        "architecture": architecture,
        "seed": seed,
        "parameters": parameter_count(model),
        "tokens_seen": tokens_seen,
        "tokens_processed_this_run": processed_tokens,
        "steps": step,
        "execution": execution,
        "compile_mode": compile_mode if compile_model else None,
        "compile_seconds": compile_seconds,
        "compile_peak_vram_gb": compile_peak_vram_gb,
        "elapsed_seconds": elapsed_seconds,
        "total_compute_seconds": elapsed_seconds + compile_seconds,
        "tokens_per_second": tokens_per_second,
        "gpu_name": gpu_name,
        "peak_vram_gb": peak_vram_gb,
        "final_eval": final_eval,
        "config": asdict(cfg),
        "training": {
            "seq_len": seq_len,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "tokens_per_step": tokens_per_step,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
