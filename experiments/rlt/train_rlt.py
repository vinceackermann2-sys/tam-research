from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from tam_research.data import TokenBin
from tam_research.train import cosine_lr, seed_all

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count


PROFILES: dict[str, dict[str, int]] = {
    "tiny": {
        "d_model": 128,
        "n_heads": 4,
        "n_stages": 2,
        "ff_mult": 4,
        "swa_window": 32,
    },
    "small": {
        "d_model": 192,
        "n_heads": 6,
        "n_stages": 4,
        "ff_mult": 4,
        "swa_window": 64,
    },
    "paper-depth-probe": {
        "d_model": 128,
        "n_heads": 4,
        "n_stages": 48,
        "ff_mult": 4,
        "swa_window": 64,
    },
}


def config_for_profile(profile: str, seq_len: int) -> RLTConfig:
    try:
        spec = PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown RLT profile: {profile!r}") from exc
    return RLTConfig(
        vocab_size=50_257,
        d_model=spec["d_model"],
        n_heads=spec["n_heads"],
        n_stages=spec["n_stages"],
        ff_mult=spec["ff_mult"],
        swa_window=spec["swa_window"],
        max_seq_len=max(128, seq_len),
    )


@torch.no_grad()
def evaluate(
    model: RecurrentLoopedTransformer,
    val_data: TokenBin,
    *,
    seq_len: int,
    batch_size: int,
    seed: int,
    batches: int = 5,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
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


def train_rlt(
    *,
    profile: str,
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
        raise RuntimeError("RLT training requires CUDA")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    torch.set_float32_matmul_precision("high")
    seed_all(seed)
    device = torch.device("cuda")
    cfg = config_for_profile(profile, seq_len)
    model = RecurrentLoopedTransformer(cfg).to(device)
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
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    run_dir = (
        Path(run_root)
        / profile
        / f"ctx{seq_len}-mb{micro_batch_size}-ga{grad_accum_steps}"
        / f"rlt-seed{seed}"
    )
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
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
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
            print(
                json.dumps(
                    {
                        "architecture": "rlt",
                        "profile": profile,
                        "step": step + 1,
                        "steps": total_steps,
                        "tokens_seen": tokens_seen,
                        "train_loss": last_loss,
                        "lr": lr,
                    }
                ),
                flush=True,
            )

    torch.cuda.synchronize(device)
    train_seconds = max(time.perf_counter() - started, 1e-6)
    final_eval = evaluate(
        model,
        val_data,
        seq_len=seq_len,
        batch_size=max(1, micro_batch_size),
        seed=seed + 30_000,
        batches=5,
    )
    torch.cuda.synchronize(device)
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    summary: dict[str, object] = {
        "status": "complete",
        "architecture": "recurrent-looped-transformer-public-spec-prototype",
        "profile": profile,
        "seed": seed,
        "parameters": parameter_count(model),
        "encoder_stages": cfg.n_stages,
        "decoder_stages": cfg.n_stages,
        "logical_blocks_per_token": 2 * cfg.n_stages,
        "decoder_temporal_depth_at_sequence_end": cfg.n_stages * seq_len,
        "stage_weights_shared_across_encoder_decoder": True,
        "swa_window": cfg.swa_window,
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
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg), "summary": summary},
        run_dir / "final.pt",
    )
    return summary
