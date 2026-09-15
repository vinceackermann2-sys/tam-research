from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from tam_research.data import TokenBin
from tam_research.models import ModelConfig, ResearchLM, parameter_count
from tam_research.train import cosine_lr, seed_all


RLT_REFERENCE_PARAMETERS = 7_007_616
PAIRED_BATCH_SEED = 20_270_920
PAIRED_EVAL_SEED = 20_290_920
EXPECTED_PARAMETERS = 7_040_896


def control_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257,
        d_model=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=128,
        ff_mult=4,
        architecture="transformer",
    )


@torch.no_grad()
def evaluate(
    model: ResearchLM,
    val_data: TokenBin,
    *,
    seq_len: int,
    batch_size: int,
    batches: int = 5,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(PAIRED_EVAL_SEED)
    losses: list[float] = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for _ in range(batches):
        x, y = val_data.batch(batch_size, seq_len, generator, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        losses.append(float(loss.detach().cpu()))
    nll = sum(losses) / len(losses)
    return {"nll": nll, "perplexity": math.exp(min(nll, 20.0))}


def train_transformer_control(
    *,
    seed: int,
    data_dir: str,
    run_root: str,
    token_budget: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.02,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("matched Transformer control requires CUDA")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    torch.set_float32_matmul_precision("high")
    seed_all(seed)
    device = torch.device("cuda")
    cfg = control_config()
    if seq_len > cfg.max_seq_len:
        raise ValueError("sequence length exceeds control max_seq_len")
    model = ResearchLM(cfg).to(device)
    params = parameter_count(model)
    if params != EXPECTED_PARAMETERS:
        raise RuntimeError(f"control parameter count drift: {params} != {EXPECTED_PARAMETERS}")
    delta = (params - RLT_REFERENCE_PARAMETERS) / RLT_REFERENCE_PARAMETERS
    if abs(delta) > 0.005:
        raise RuntimeError(f"parameter match exceeds 0.5%: delta={delta}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))

    tokens_per_step = micro_batch_size * seq_len * grad_accum_steps
    total_steps = math.ceil(token_budget / tokens_per_step)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    batch_gen = torch.Generator(device="cpu").manual_seed(PAIRED_BATCH_SEED)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    run_dir = Path(run_root) / "transformer-control" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    tokens_seen = 0
    last_loss = float("nan")

    for step in range(total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_data.batch(micro_batch_size, seq_len, batch_gen, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
                ) / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.detach().cpu()) * grad_accum_steps

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, total_steps, warmup_steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        last_loss = running / grad_accum_steps
        tokens_seen = min(token_budget, (step + 1) * tokens_per_step)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == total_steps:
            print(json.dumps({
                "architecture": "transformer-control",
                "step": step + 1,
                "steps": total_steps,
                "tokens_seen": tokens_seen,
                "train_loss": last_loss,
                "lr": lr,
            }), flush=True)

    torch.cuda.synchronize(device)
    train_seconds = max(time.perf_counter() - started, 1e-6)
    final_eval = evaluate(
        model,
        val_data,
        seq_len=seq_len,
        batch_size=max(1, micro_batch_size),
        batches=5,
    )
    torch.cuda.synchronize(device)
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    summary: dict[str, object] = {
        "status": "complete",
        "architecture": "matched-decoder-only-transformer-control",
        "seed": seed,
        "parameters": params,
        "rlt_reference_parameters": RLT_REFERENCE_PARAMETERS,
        "parameter_delta_fraction": delta,
        "paired_batch_seed": PAIRED_BATCH_SEED,
        "paired_eval_seed": PAIRED_EVAL_SEED,
        "tokens_seen": tokens_seen,
        "steps": total_steps,
        "last_train_loss": last_loss,
        "final_eval": final_eval,
        "train_seconds": train_seconds,
        "tokens_per_second": tokens_seen / train_seconds,
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": torch.cuda.get_device_name(device),
        "amp_dtype": str(amp_dtype),
        "config": asdict(cfg),
        "training": {
            "seq_len": seq_len,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "token_budget": token_budget,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
