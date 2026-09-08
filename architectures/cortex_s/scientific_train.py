from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tam_research.data import TokenBin
from tam_research.models import ModelConfig, ResearchLM, parameter_count as baseline_parameter_count
from tam_research.train import cosine_lr

try:
    from .language_model import CortexSLM, CortexSLMConfig, parameter_count
except ImportError:
    from language_model import CortexSLM, CortexSLMConfig, parameter_count


SCIENTIFIC_SEED_1 = 48_131
TOKEN_BUDGET = 5_000_000
SEQ_LEN = 512
GLOBAL_BATCH = 128
EVAL_EVERY_TOKENS = 500_000
PROJECT_NAMESPACE = "cortex-s-v0"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(architecture: str, *, max_seq_len: int = 1024) -> torch.nn.Module:
    if architecture == "cortex_s":
        return CortexSLM(CortexSLMConfig(max_seq_len=max_seq_len))
    if architecture == "transformer":
        return ResearchLM(
            ModelConfig(
                architecture="transformer",
                vocab_size=50_257,
                d_model=256,
                n_layers=15,
                n_heads=8,
                max_seq_len=max_seq_len,
                ff_mult=4,
                tamv2_state_size=64,
                tamv3_attn_inner=208,
            )
        )
    raise ValueError(f"unsupported seed1 architecture: {architecture}")


def model_parameter_count(model: torch.nn.Module, architecture: str) -> int:
    if architecture == "cortex_s":
        return parameter_count(model)
    return baseline_parameter_count(model)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_random(
    model: torch.nn.Module,
    architecture: str,
    val: TokenBin,
    *,
    seq_len: int,
    batches: int,
    batch_size: int,
    seed: int,
    disable_world: bool = False,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = val.batch(batch_size, seq_len, generator, device)
        with _autocast(device):
            if architecture == "cortex_s":
                logits = model(x, disable_world=disable_world)
            else:
                logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
        losses.append(float(loss))
    mean = sum(losses) / len(losses)
    return {"nll": mean, "perplexity": math.exp(min(mean, 20.0))}


@torch.no_grad()
def evaluate_cortex_streaming(
    model: CortexSLM,
    val: TokenBin,
    *,
    chunk_len: int = 512,
    chunks: int = 8,
    batch_size: int = 4,
    seed: int = 91_001,
) -> dict[str, float]:
    """Compare carried recurrent state with a reset-state control on identical text.

    This is diagnostic only in seed1 because random-chunk language training does not
    give CORTEX-S privileged cross-chunk context. It tests whether the learned state
    is at least usable when a contiguous stream is presented at evaluation time.
    """

    model.eval()
    device = next(model.parameters()).device
    source = val.data
    total = chunk_len * chunks + 1
    high = len(source) - total
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(0, high, (batch_size,), generator=generator).tolist()
    rows = [np.asarray(source[start : start + total], dtype=np.int64) for start in starts]
    tokens = torch.from_numpy(np.stack(rows)).to(device=device)

    persistent_state = None
    persistent_losses = []
    reset_losses = []
    for chunk_index in range(chunks):
        begin = chunk_index * chunk_len
        x = tokens[:, begin : begin + chunk_len]
        y = tokens[:, begin + 1 : begin + chunk_len + 1]
        with _autocast(device):
            persistent_logits, persistent_state = model(
                x,
                state=persistent_state,
                return_state=True,
            )
            reset_logits = model(x)
            persistent_loss = F.cross_entropy(
                persistent_logits.float().reshape(-1, persistent_logits.size(-1)),
                y.reshape(-1),
            )
            reset_loss = F.cross_entropy(
                reset_logits.float().reshape(-1, reset_logits.size(-1)),
                y.reshape(-1),
            )
        persistent_losses.append(float(persistent_loss))
        reset_losses.append(float(reset_loss))
        persistent_state = tuple(state.detach() for state in persistent_state)

    persistent = sum(persistent_losses) / len(persistent_losses)
    reset = sum(reset_losses) / len(reset_losses)
    return {
        "persistent_nll": persistent,
        "reset_nll": reset,
        "delta_reset_minus_persistent": reset - persistent,
        "chunks": chunks,
        "chunk_len": chunk_len,
        "batch_size": batch_size,
    }


def train_one(
    *,
    architecture: str,
    seed: int,
    data_dir: str,
    output_dir: str,
    token_budget: int = TOKEN_BUDGET,
    seq_len: int = SEQ_LEN,
    micro_batch_size: int = 32,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.02,
) -> dict[str, Any]:
    if architecture not in {"transformer", "cortex_s"}:
        raise ValueError("seed1 supports exactly transformer and cortex_s")
    if micro_batch_size * grad_accum_steps != GLOBAL_BATCH:
        raise ValueError("micro_batch_size * grad_accum_steps must equal 128")
    if not torch.cuda.is_available():
        raise RuntimeError("scientific seed1 requires CUDA")

    torch.set_float32_matmul_precision("high")
    seed_all(seed)
    device = torch.device("cuda")
    model = build_model(architecture, max_seq_len=max(1024, seq_len)).to(device)
    params = model_parameter_count(model, architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )

    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    tokens_per_step = micro_batch_size * seq_len * grad_accum_steps
    total_steps = math.ceil(token_budget / tokens_per_step)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    next_eval = EVAL_EVERY_TOKENS
    curve: list[dict[str, float | int]] = []
    training_seconds = 0.0
    tokens_seen = 0

    torch.cuda.reset_peak_memory_stats(device)
    for step in range(total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        running_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_data.batch(micro_batch_size, seq_len, batch_generator, device)
            with _autocast(device):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                ) / grad_accum_steps
            loss.backward()
            running_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, total_steps, warmup_steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize(device)
        training_seconds += max(time.perf_counter() - started, 0.0)
        tokens_seen = min(token_budget, (step + 1) * tokens_per_step)

        if tokens_seen >= next_eval or tokens_seen >= token_budget:
            evaluation = evaluate_random(
                model,
                architecture,
                val_data,
                seq_len=seq_len,
                batches=10,
                batch_size=max(1, micro_batch_size // 2),
                seed=seed + 20_000,
            )
            curve.append(
                {
                    "tokens_seen": tokens_seen,
                    "training_seconds": training_seconds,
                    "nll": evaluation["nll"],
                    "perplexity": evaluation["perplexity"],
                    "step": step + 1,
                    "last_train_loss": running_loss,
                }
            )
            while next_eval <= tokens_seen:
                next_eval += EVAL_EVERY_TOKENS

    final_eval = evaluate_random(
        model,
        architecture,
        val_data,
        seq_len=seq_len,
        batches=50,
        batch_size=max(1, micro_batch_size // 2),
        seed=seed + 30_000,
    )
    diagnostics: dict[str, Any] = {}
    if architecture == "cortex_s":
        cortex_model = model
        diagnostics["router"] = cortex_model.router_stats()
        diagnostics["streaming_state"] = evaluate_cortex_streaming(cortex_model, val_data)
        diagnostics["world_ablation"] = evaluate_random(
            cortex_model,
            architecture,
            val_data,
            seq_len=seq_len,
            batches=20,
            batch_size=max(1, micro_batch_size // 2),
            seed=seed + 40_000,
            disable_world=True,
        )

    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    training_tokens_per_second = token_budget / max(training_seconds, 1e-9)

    run_dir = Path(output_dir) / architecture
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "final.pt"
    torch.save(
        {
            "architecture": architecture,
            "seed": seed,
            "parameters": params,
            "token_budget": token_budget,
            "seq_len": seq_len,
            "model": model.state_dict(),
        },
        checkpoint_path,
    )

    result = {
        "architecture": architecture,
        "seed": seed,
        "parameters": params,
        "token_budget": token_budget,
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "global_batch": GLOBAL_BATCH,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "training_seconds": training_seconds,
        "training_tokens_per_second": training_tokens_per_second,
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": torch.cuda.get_device_name(device),
        "curve": curve,
        "final_eval": final_eval,
        "diagnostics": diagnostics,
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _curve_at_or_before(result: dict[str, Any], training_seconds: float) -> dict[str, Any]:
    eligible = [point for point in result["curve"] if point["training_seconds"] <= training_seconds]
    if not eligible:
        return result["curve"][0]
    return eligible[-1]


def judge_seed1(transformer: dict[str, Any], cortex: dict[str, Any]) -> dict[str, Any]:
    transformer_params = int(transformer["parameters"])
    cortex_params = int(cortex["parameters"])
    parameter_gap = abs(cortex_params - transformer_params) / transformer_params
    executed_fraction = float(
        cortex["diagnostics"]["router"]["theoretical_executed_fraction"]
    )
    common_training_seconds = min(
        float(transformer["training_seconds"]),
        float(cortex["training_seconds"]),
    )
    transformer_common = _curve_at_or_before(transformer, common_training_seconds)
    cortex_common = _curve_at_or_before(cortex, common_training_seconds)

    gates = {
        "finite": all(
            math.isfinite(float(value))
            for value in (
                transformer["final_eval"]["nll"],
                cortex["final_eval"]["nll"],
                transformer["training_tokens_per_second"],
                cortex["training_tokens_per_second"],
            )
        ),
        "parameter_gap_le_0_5pct": parameter_gap <= 0.005,
        "equal_token_nll_strictly_better": (
            float(cortex["final_eval"]["nll"])
            < float(transformer["final_eval"]["nll"])
        ),
        "equal_time_curve_nll_no_worse": (
            float(cortex_common["nll"]) <= float(transformer_common["nll"])
        ),
        "executed_expert_fraction_le_50pct": executed_fraction <= 0.5,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "parameter_gap_fraction": parameter_gap,
        "executed_expert_fraction": executed_fraction,
        "common_training_seconds": common_training_seconds,
        "equal_time_points": {
            "transformer": transformer_common,
            "cortex_s": cortex_common,
        },
        "final_equal_token_nll": {
            "transformer": transformer["final_eval"]["nll"],
            "cortex_s": cortex["final_eval"]["nll"],
        },
        "throughput_ratio_cortex_over_transformer": (
            float(cortex["training_tokens_per_second"])
            / float(transformer["training_tokens_per_second"])
        ),
        "interpretation": (
            "PASS authorizes only the separately preregistered replication/ablation stage. "
            "FAIL stops additional paid CORTEX-S seeds unless a new hypothesis and protocol are preregistered."
        ),
    }


def run_seed1(
    *,
    seed: int,
    data_dir: str,
    output_root: str,
    token_budget: int = TOKEN_BUDGET,
    seq_len: int = SEQ_LEN,
    micro_batch_size: int = 32,
    grad_accum_steps: int = 4,
) -> dict[str, Any]:
    if seed != SCIENTIFIC_SEED_1:
        raise ValueError(f"seed1 is frozen to {SCIENTIFIC_SEED_1}")
    run_dir = Path(output_root) / f"seed-{seed}"
    if (run_dir / "SUCCESS.json").exists() or (run_dir / "result.json").exists():
        raise RuntimeError("seed1 namespace is already consumed; refusing rerun")
    run_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        seed=seed,
        data_dir=data_dir,
        output_dir=str(run_dir),
        token_budget=token_budget,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
    )
    transformer = train_one(architecture="transformer", **common)
    cortex = train_one(architecture="cortex_s", **common)
    judgement = judge_seed1(transformer, cortex)
    result = {
        "project": PROJECT_NAMESPACE,
        "scientific_seed": seed,
        "scientific_stage": "25m-seed1-falsification",
        "token_budget_per_model": token_budget,
        "models": {"transformer": transformer, "cortex_s": cortex},
        "judgement": judgement,
        "claims": {
            "breakthrough": False,
            "agi": False,
            "ssi": False,
            "safe_superintelligence": False,
        },
    }

    temporary = run_dir / "result.json.tmp"
    final = run_dir / "result.json"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(final)
    success = {
        "seed": seed,
        "result_path": str(final),
        "judgement_status": judgement["status"],
    }
    (run_dir / "SUCCESS.json").write_text(json.dumps(success, indent=2), encoding="utf-8")
    return result
