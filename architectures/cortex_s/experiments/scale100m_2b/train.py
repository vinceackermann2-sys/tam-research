from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from architectures.cortex_s.language_model import CortexSLM, parameter_count
from tam_research.data import TokenBin
from tam_research.train import cosine_lr

from .protocol import (
    BASELINE_TRANSFORMER,
    CALIBRATION_SEED,
    CALIBRATION_STEPS,
    CALIBRATION_WARMUP_STEPS,
    CHECKPOINT_EVERY_TOKENS,
    CORTEX_100M_CONFIG,
    EVAL_EVERY_TOKENS,
    EXPECTED_CORTEX_PARAMS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MAX_PROJECTED_FULL_SECONDS,
    MICRO_BATCH_SIZE,
    PAIRED_SEED,
    SEQ_LEN,
    TRAIN_TOKENS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    projected_full_cost_usd,
    protocol_snapshot,
    validate_protocol,
)

COMPILE_MODE = "max-autotune-no-cudagraphs"
PROJECTION_OVERHEAD_MULTIPLIER = 1.10
MAX_PREFLIGHT_VRAM_GIB = 70.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_cortex_100m() -> CortexSLM:
    model = CortexSLM(CORTEX_100M_CONFIG)
    validate_protocol(actual_cortex_params=parameter_count(model))
    return model


def _enable_dynamic_compile_support() -> None:
    # True sparse routing has data-dependent gather sizes. These options allow
    # Dynamo to capture those output shapes rather than silently forcing dense MoE.
    import torch._dynamo

    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True


def _compile_model(model: torch.nn.Module) -> torch.nn.Module:
    _enable_dynamic_compile_support()
    return torch.compile(model, mode=COMPILE_MODE, fullgraph=False)


def _make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )


def _one_optimizer_step(
    *,
    model: CortexSLM,
    runner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: TokenBin,
    generator: torch.Generator,
    device: torch.device,
    lr: float,
) -> float:
    model.train()
    runner.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        x, y = train_data.batch(MICRO_BATCH_SIZE, SEQ_LEN, generator, device)
        with _autocast(device):
            logits = runner(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
            ) / GRAD_ACCUM_STEPS
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CORTEX-S training loss")
        loss.backward()
        running += float(loss.detach())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite CORTEX-S gradient norm")
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return running


@torch.no_grad()
def evaluate(
    model: CortexSLM,
    runner: torch.nn.Module,
    val_data: TokenBin,
    *,
    batches: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    runner.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for _ in range(batches):
        x, y = val_data.batch(max(1, MICRO_BATCH_SIZE // 2), SEQ_LEN, generator, device)
        with _autocast(device):
            logits = runner(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
        value = float(loss)
        if not math.isfinite(value):
            raise FloatingPointError("non-finite validation loss")
        losses.append(value)
    nll = sum(losses) / len(losses)
    return {"nll": nll, "perplexity": math.exp(min(nll, 20.0))}


def run_h100_calibration(
    *,
    data_dir: str,
    output_path: str,
    source_sha: str,
) -> dict[str, Any]:
    """Consume a tiny engineering seed and estimate full-run wall time.

    The calibration intentionally uses the exact production 100M graph, context,
    microbatch and optimizer. It never consumes seed 8100 or a reserved fresh seed.
    Full 2B training is not launched from this function.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("H100 calibration requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    seed_all(CALIBRATION_SEED)
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))

    def fresh() -> tuple[CortexSLM, torch.nn.Module, torch.optim.Optimizer, torch.Generator, float]:
        seed_all(CALIBRATION_SEED)
        model = build_cortex_100m().to(device)
        optimizer = _make_optimizer(model)
        generator = torch.Generator(device="cpu").manual_seed(CALIBRATION_SEED + 10_000)
        compile_start = time.perf_counter()
        runner = _compile_model(model)
        return model, runner, optimizer, generator, compile_start

    execution = "compiled"
    compile_seconds = 0.0
    compile_error: str | None = None
    try:
        model, runner, optimizer, generator, compile_start = fresh()
        # First step materializes the 2B-token GPU gather cache and triggers compile.
        _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=LEARNING_RATE,
        )
        torch.cuda.synchronize(device)
        compile_seconds = max(time.perf_counter() - compile_start, 0.0)
    except Exception as exc:
        compile_error = f"{type(exc).__name__}: {exc}"
        execution = "eager"
        # A compile failure is an engineering result, not permission to burn the
        # budget retrying compilers. Rebuild once and measure eager as the fallback.
        torch.cuda.empty_cache()
        seed_all(CALIBRATION_SEED)
        model = build_cortex_100m().to(device)
        runner = model
        optimizer = _make_optimizer(model)
        generator = torch.Generator(device="cpu").manual_seed(CALIBRATION_SEED + 10_000)
        _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=LEARNING_RATE,
        )
        torch.cuda.synchronize(device)

    for _ in range(max(0, CALIBRATION_WARMUP_STEPS - 1)):
        _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=LEARNING_RATE,
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_loss = 0.0
    for _ in range(CALIBRATION_STEPS):
        last_loss = _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=LEARNING_RATE,
        )
    torch.cuda.synchronize(device)
    measured_seconds = max(time.perf_counter() - started, 1e-9)
    tokens_per_step = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
    measured_tokens = CALIBRATION_STEPS * tokens_per_step
    training_tps = measured_tokens / measured_seconds
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    projected_training_seconds = TRAIN_TOKENS / training_tps
    projected_full_seconds = (
        projected_training_seconds * PROJECTION_OVERHEAD_MULTIPLIER + compile_seconds
    )
    cost = projected_full_cost_usd(projected_full_seconds)

    gates = {
        "finite_loss": math.isfinite(last_loss),
        "parameter_count_exact": parameter_count(model) == EXPECTED_CORTEX_PARAMS,
        "projected_full_seconds_le_budget": projected_full_seconds <= MAX_PROJECTED_FULL_SECONDS,
        "peak_vram_le_70_gib": peak_vram_gb <= MAX_PREFLIGHT_VRAM_GIB,
        "throughput_positive": training_tps > 0.0,
    }
    status = "PASS" if all(gates.values()) else "ABORT_FULL_RUN"
    result: dict[str, Any] = {
        "status": status,
        "scientific_status": "ENGINEERING_PREFLIGHT_ONLY",
        "source_sha": source_sha,
        "calibration_seed": CALIBRATION_SEED,
        "execution": execution,
        "compile_mode": COMPILE_MODE if execution == "compiled" else None,
        "compile_seconds": compile_seconds,
        "compile_error": compile_error,
        "measured_steps": CALIBRATION_STEPS,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_seconds,
        "training_tokens_per_second": training_tps,
        "projected_training_seconds": projected_training_seconds,
        "projection_overhead_multiplier": PROJECTION_OVERHEAD_MULTIPLIER,
        "projected_full_seconds": projected_full_seconds,
        "projected_cost": cost,
        "peak_vram_gb": peak_vram_gb,
        "last_train_loss": last_loss,
        "gates": gates,
        "protocol": protocol_snapshot(),
        "full_run_authorized": status == "PASS",
        "note": (
            "PASS only authorizes the exact paired 100M/2B seed-8100 run from the same source SHA. "
            "It is not scientific evidence and cannot support a breakthrough claim."
        ),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _save_checkpoint(
    *,
    path: Path,
    model: CortexSLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    generator: torch.Generator,
    source_sha: str,
    execution: str,
) -> None:
    payload = {
        "experiment": "cortex-s-v0-100m-2b-paired8100",
        "source_sha": source_sha,
        "architecture": "cortex_s",
        "seed": PAIRED_SEED,
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "batch_generator_state": generator.get_state(),
        "config": asdict(CORTEX_100M_CONFIG),
        "execution": execution,
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_full_2b(
    *,
    data_dir: str,
    output_dir: str,
    preflight_path: str,
    source_sha: str,
) -> dict[str, Any]:
    """Run exactly one paired CORTEX-S 100M/2B pretraining attempt.

    This function refuses to start without a PASS preflight from the exact source
    SHA. It does not train a second Transformer; issue #140 is the frozen control.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("full 100M/2B run requires CUDA")
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or not preflight.get("full_run_authorized"):
        raise RuntimeError("preflight did not authorize the full run")
    if preflight.get("source_sha") != source_sha:
        raise RuntimeError("source SHA changed after H100 preflight; refusing full run")
    if float(preflight.get("projected_full_seconds", float("inf"))) > MAX_PROJECTED_FULL_SECONDS:
        raise RuntimeError("preflight projection exceeds the frozen budget gate")
    execution = str(preflight.get("execution"))
    if execution not in {"compiled", "eager"}:
        raise RuntimeError("preflight execution mode is invalid")

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "SUCCESS.json").exists() or (run_dir / "ATTEMPT_STARTED.json").exists():
        raise RuntimeError("paired 100M/2B namespace is already consumed; refusing rerun")
    (run_dir / "ATTEMPT_STARTED.json").write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "seed": PAIRED_SEED,
                "tokens": TRAIN_TOKENS,
                "execution": execution,
                "started_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    torch.set_float32_matmul_precision("high")
    seed_all(PAIRED_SEED)
    device = torch.device("cuda")
    model = build_cortex_100m().to(device)
    optimizer = _make_optimizer(model)
    runner: torch.nn.Module = model
    compile_seconds = 0.0
    if execution == "compiled":
        compile_start = time.perf_counter()
        runner = _compile_model(model)
        # Compile/warm the exact train graph without advancing the scientific batch
        # generator. A separate deterministic generator consumes only warmup bytes.
        warm_generator = torch.Generator(device="cpu").manual_seed(PAIRED_SEED + 99_999)
        train_warm = TokenBin(str(Path(data_dir) / "train.bin"))
        _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_warm,
            generator=warm_generator,
            device=device,
            lr=LEARNING_RATE,
        )
        # Warmup must not alter scientific initial weights or optimizer state.
        seed_all(PAIRED_SEED)
        model = build_cortex_100m().to(device)
        optimizer = _make_optimizer(model)
        runner = _compile_model(model)
        torch.cuda.synchronize(device)
        compile_seconds = max(time.perf_counter() - compile_start, 0.0)

    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))
    generator = torch.Generator(device="cpu").manual_seed(PAIRED_SEED + 10_000)
    tokens_per_step = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
    total_steps = math.ceil(TRAIN_TOKENS / tokens_per_step)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    next_eval = EVAL_EVERY_TOKENS
    next_checkpoint = CHECKPOINT_EVERY_TOKENS
    curve: list[dict[str, Any]] = []
    training_seconds = 0.0
    eval_seconds = 0.0
    checkpoint_seconds = 0.0
    tokens_seen = 0
    latest_path = run_dir / "latest.pt"

    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    for step in range(total_steps):
        torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        lr = cosine_lr(step, total_steps, warmup_steps, LEARNING_RATE)
        train_loss = _one_optimizer_step(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=lr,
        )
        torch.cuda.synchronize(device)
        training_seconds += max(time.perf_counter() - train_start, 0.0)
        tokens_seen = min(TRAIN_TOKENS, (step + 1) * tokens_per_step)

        if tokens_seen >= next_eval or tokens_seen >= TRAIN_TOKENS:
            eval_start = time.perf_counter()
            ev = evaluate(
                model,
                runner,
                val_data,
                batches=20,
                seed=PAIRED_SEED + 20_000,
            )
            torch.cuda.synchronize(device)
            eval_seconds += max(time.perf_counter() - eval_start, 0.0)
            curve.append(
                {
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "training_seconds": training_seconds,
                    "wall_seconds": time.perf_counter() - wall_start,
                    "nll": ev["nll"],
                    "perplexity": ev["perplexity"],
                    "last_train_loss": train_loss,
                    "lr": lr,
                }
            )
            while next_eval <= tokens_seen:
                next_eval += EVAL_EVERY_TOKENS

        if tokens_seen >= next_checkpoint or tokens_seen >= TRAIN_TOKENS:
            checkpoint_start = time.perf_counter()
            _save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                tokens_seen=tokens_seen,
                generator=generator,
                source_sha=source_sha,
                execution=execution,
            )
            checkpoint_seconds += max(time.perf_counter() - checkpoint_start, 0.0)
            while next_checkpoint <= tokens_seen:
                next_checkpoint += CHECKPOINT_EVERY_TOKENS

    final_eval_start = time.perf_counter()
    final_eval = evaluate(
        model,
        runner,
        val_data,
        batches=50,
        seed=PAIRED_SEED + 30_000,
    )
    torch.cuda.synchronize(device)
    eval_seconds += max(time.perf_counter() - final_eval_start, 0.0)
    wall_seconds = max(time.perf_counter() - wall_start, 1e-9)
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    router = model.router_stats()
    training_tps = TRAIN_TOKENS / max(training_seconds, 1e-9)

    baseline = dict(BASELINE_TRANSFORMER)
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "scientific_status": "PAIRED_HISTORICAL_CONTROL_ADAPTIVE_EXPERIMENT",
        "breakthrough_claim_allowed": False,
        "source_sha": source_sha,
        "architecture": "cortex_s",
        "parameters": parameter_count(model),
        "seed": PAIRED_SEED,
        "tokens_seen": TRAIN_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "execution": execution,
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "eval_seconds": eval_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "wall_seconds": wall_seconds,
        "total_compute_seconds": wall_seconds + compile_seconds,
        "training_tokens_per_second": training_tps,
        "peak_vram_gb": peak_vram_gb,
        "final_eval": final_eval,
        "curve": curve,
        "router": router,
        "state_policy": "reset_per_random_training_sequence",
        "continual_learning_claim_allowed": False,
        "historical_transformer": baseline,
        "comparison": {
            "equal_token_nll_delta_cortex_minus_transformer": (
                final_eval["nll"] - float(baseline["final_nll"])
            ),
            "equal_token_nll_cortex_better": (
                final_eval["nll"] < float(baseline["final_nll"])
            ),
            "throughput_ratio_cortex_over_transformer": (
                training_tps / float(baseline["training_tokens_per_second"])
            ),
            "parameter_gap_fraction": (
                abs(parameter_count(model) - int(baseline["parameters"]))
                / int(baseline["parameters"])
            ),
            "same_seed_random_window_protocol": True,
            "same_pretraining_bytes": True,
            "same_optimizer_hparams": True,
            "same_context_and_global_batch": True,
            "note": (
                "The Transformer was completed earlier and influenced later architecture development. "
                "This is therefore paired historical-control evidence, not a fresh blind comparison."
            ),
        },
        "protocol": protocol_snapshot(),
        "gpu_name": torch.cuda.get_device_name(device),
        "checkpoint": str(latest_path),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "SUCCESS.json").write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "status": "COMPLETE",
                "final_nll": final_eval["nll"],
                "tokens_seen": TRAIN_TOKENS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
