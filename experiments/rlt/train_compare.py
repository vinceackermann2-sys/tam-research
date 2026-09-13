from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
import gc
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from tam_research.data import TokenBin
from tam_research.models import ResearchLM
from tam_research.scales import model_config_for_scale
from tam_research.train import cosine_lr, seed_all

from experiments.rlt.model import RecurrentLoopedLM, parameter_count


def build_model(architecture: str, model_scale: str, seq_len: int) -> torch.nn.Module:
    if architecture not in {"rlt", "transformer"}:
        raise ValueError("architecture must be 'rlt' or 'transformer'")
    base_cfg = model_config_for_scale(
        "transformer", model_scale, max_seq_len=max(1024, seq_len)
    )
    if architecture == "rlt":
        return RecurrentLoopedLM(replace(base_cfg, architecture="rlt"))
    return ResearchLM(base_cfg)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    val_data: TokenBin,
    *,
    seq_len: int,
    batch_size: int,
    seed: int,
    batches: int = 20,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for _ in range(batches):
        x, y = val_data.batch(batch_size, seq_len, generator, device)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        losses.append(float(loss))
    nll = sum(losses) / len(losses)
    return {"nll": nll, "perplexity": math.exp(min(nll, 20.0))}


def train_one(
    *,
    architecture: str,
    model_scale: str,
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
    compile_model: bool = False,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("RLT comparison training requires CUDA")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    torch.set_float32_matmul_precision("high")
    seed_all(seed)
    device = torch.device("cuda")
    model = build_model(architecture, model_scale, seq_len).to(device)
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

    forward_model: torch.nn.Module = model
    if compile_model:
        forward_model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    run_dir = (
        Path(run_root)
        / model_scale
        / f"ctx{seq_len}-mb{micro_batch_size}-ga{grad_accum_steps}"
        / f"{architecture}-seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tokens_seen = 0
    last_loss = float("nan")
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(total_steps):
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
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
                ) / grad_accum_steps
            loss.backward()
            running += float(loss) * grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, total_steps, warmup_steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        last_loss = running / grad_accum_steps
        tokens_seen = min(token_budget, (step + 1) * tokens_per_step)

        if step == 0 or (step + 1) % 20 == 0 or step + 1 == total_steps:
            print(
                json.dumps(
                    {
                        "architecture": architecture,
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
    elapsed = max(time.perf_counter() - started, 1e-6)
    final_eval = _evaluate(
        model,
        val_data,
        seq_len=seq_len,
        batch_size=max(1, micro_batch_size // 2),
        seed=seed + 30_000,
        batches=20,
    )
    torch.cuda.synchronize(device)
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    cfg = model.cfg
    summary: dict[str, object] = {
        "architecture": architecture,
        "model_scale_shape": model_scale,
        "seed": seed,
        "parameters": parameter_count(model),
        "effective_depth": cfg.n_layers,
        "physical_blocks": 1 if architecture == "rlt" else cfg.n_layers,
        "tokens_seen": tokens_seen,
        "steps": total_steps,
        "last_train_loss": last_loss,
        "final_eval": final_eval,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_seen / elapsed,
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": torch.cuda.get_device_name(device),
        "config": asdict(cfg),
        "training": {
            "seq_len": seq_len,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "token_budget": token_budget,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "compile_model": compile_model,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "summary": summary,
        },
        run_dir / "final.pt",
    )
    return summary


def run_comparison(**kwargs: object) -> dict[str, object]:
    rlt = train_one(architecture="rlt", **kwargs)
    gc.collect()
    torch.cuda.empty_cache()
    transformer = train_one(architecture="transformer", **kwargs)
    gc.collect()
    torch.cuda.empty_cache()

    rlt_nll = float(rlt["final_eval"]["nll"])  # type: ignore[index]
    transformer_nll = float(transformer["final_eval"]["nll"])  # type: ignore[index]
    return {
        "status": "complete",
        "comparison": "same width, same effective depth, same tokens/data/optimizer",
        "rlt": rlt,
        "transformer": transformer,
        "delta_nll_rlt_minus_transformer": rlt_nll - transformer_nll,
        "parameter_ratio_rlt_over_transformer": float(rlt["parameters"])
        / float(transformer["parameters"]),
    }
