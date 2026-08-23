from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v8 import MostlyHardRoutingSchedule
from .aera_hardware_core_v10 import HardwareAwareAERATextLMV10
from .data import TokenBin
from .models import ModelConfig, ResearchLM, parameter_count
from .train import cosine_lr

VOCAB_SIZE = 50_257
SEQ_LEN = 512
CHUNK_SIZE = 64
TOKEN_BUDGET = 8_388_608
MICRO_BATCH = 8
GRAD_ACCUM = 4
TOKENS_PER_STEP = MICRO_BATCH * SEQ_LEN * GRAD_ACCUM
TOTAL_STEPS = TOKEN_BUDGET // TOKENS_PER_STEP
CALIBRATION_WARMUP_STEPS = 32
SEED = 8201


def aera_25m_config() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=VOCAB_SIZE,
        d_model=200,
        n_stages=4,
        n_heads=8,
        chunk_size=CHUNK_SIZE,
        n_experts=8,
        max_active_experts=2,
        expert_mult=4,
        memory_dim=50,
        max_reason_steps=4,
        block_size=4,
    )


def transformer_25m_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_layers=15,
        n_heads=8,
        max_seq_len=1024,
        ff_mult=4,
        architecture="transformer",
    )


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV10:
    return HardwareAwareAERATextLMV10(aera_25m_config()).to(device)


def build_transformer(device: torch.device) -> ResearchLM:
    return ResearchLM(transformer_25m_config()).to(device)


def parameter_accounting(
    aera: HardwareAwareAERATextLMV10,
    transformer: ResearchLM,
) -> dict[str, Any]:
    aera_total = sum(p.numel() for p in aera.parameters())
    transformer_total = parameter_count(transformer)
    expert_total = sum(
        stage.experts.w1.numel() + stage.experts.w2.numel() for stage in aera.stages
    )
    return {
        "aera_stored_parameters": aera_total,
        "transformer_parameters": transformer_total,
        "stored_parameter_delta_fraction": (aera_total - transformer_total) / transformer_total,
        "aera_expert_parameters_stored": expert_total,
        "aera_nonexpert_parameters_stored": aera_total - expert_total,
        "aera_predictive_state": aera.predictive_head_accounting(),
    }


def validate_protocol(data_dir: str) -> dict[str, Any]:
    if TOKEN_BUDGET % TOKENS_PER_STEP:
        raise RuntimeError("token budget must divide exactly by tokens/step")
    root = Path(data_dir)
    required = [root / "train.bin", root / "val.bin", root / "meta.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"real-language source data missing: {missing}")

    import json

    meta = json.loads((root / "meta.json").read_text())
    expected = {
        "assembly_version": 3,
        "train_tokens": 2_000_000_000,
        "val_tokens": 5_000_000,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"data metadata mismatch {key}: {meta.get(key)!r} != {value!r}")
    if (root / "train.bin").stat().st_size != 4_000_000_000:
        raise RuntimeError("train.bin is not exact 2B uint16 token stream")
    if (root / "val.bin").stat().st_size != 10_000_000:
        raise RuntimeError("val.bin is not exact 5M uint16 validation stream")

    aera = build_aera(torch.device("cpu"))
    transformer = build_transformer(torch.device("cpu"))
    counts = parameter_accounting(aera, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")
    del aera, transformer
    return {
        "data": expected,
        "token_budget_per_model": TOKEN_BUDGET,
        "tokens_per_step": TOKENS_PER_STEP,
        "optimizer_steps": TOTAL_STEPS,
        "batch_sampling_seed": SEED + 10_000,
        "parameter_accounting": counts,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def aera_route_mode(step: int) -> str:
    if step < CALIBRATION_WARMUP_STEPS:
        return "straight_through"
    return MostlyHardRoutingSchedule(calibration_every=8).mode_for_step(
        step - CALIBRATION_WARMUP_STEPS
    )


def aera_matched_loss(
    model: HardwareAwareAERATextLMV10,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
    mode = aera_route_mode(step)
    hard = mode == "hard_sparse"
    out = model(
        x,
        hard=hard,
        route_mode=mode,
        update_memory=False,
        return_block_logits=False,
    )
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    primary = F.cross_entropy(
        logits.float().reshape(-1, model.cfg.vocab_size), y.reshape(-1)
    )

    if hard:
        aux = model.hard_sparse_task_loss(
            x,
            out,
            event_weight=0.02,
            block_weight=0.0,
            stream_forecast_weight=0.10,
        )
    else:
        aux = model.soft_objective(
            x,
            out,
            event_weight=0.02,
            compute_weight=0.002,
            balance_weight=0.01,
            block_weight=0.0,
            stream_forecast_weight=0.10,
            stage_compute_weight=0.002,
        )
    # The inherited objective's CE is x[t]->x[t+1]. Remove it and insert the
    # exact shared x->y primary target used by the Transformer control.
    total = primary + (aux["total"] - aux["next_token"])
    terms = dict(aux)
    terms["next_token"] = primary
    terms["total"] = total
    return total, terms, mode


@torch.no_grad()
def evaluate_transformer(
    model: ResearchLM,
    val: TokenBin,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for _ in range(batches):
        x, y = val.batch(batch_size, SEQ_LEN, g, device)
        with _autocast(device):
            logits = model(x)
        losses.append(
            float(F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
        )
    mean = sum(losses) / len(losses)
    return {"nll": mean, "perplexity": math.exp(min(mean, 20.0))}


def _route_metrics(output: dict[str, object], n_experts: int) -> dict[str, Any]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list):
        return {}
    executed = 0
    total = 0
    expert_counts: list[float] = []
    depths: list[float] = []
    top1_hist = [0 for _ in range(n_experts)]
    for chunk in routes:
        for item in chunk:
            total += 1
            start = item.get("start")
            end = item.get("end")
            if not isinstance(start, dict) or not isinstance(end, dict):
                continue
            executed += 1
            count_logits = start["expert_count_logits"]
            depth_logits = end["depth_logits"]
            expert_logits = start["expert_logits"]
            expert_counts.extend(
                (count_logits.argmax(dim=-1) + 1).float().detach().cpu().tolist()
            )
            depths.extend(
                (depth_logits.argmax(dim=-1) + 1).float().detach().cpu().tolist()
            )
            for idx in expert_logits.argmax(dim=-1).detach().cpu().tolist():
                top1_hist[int(idx)] += 1
    return {
        "stage_execution_fraction": executed / max(total, 1),
        "mean_active_experts_when_executed": sum(expert_counts) / max(len(expert_counts), 1),
        "mean_reason_steps_when_executed": sum(depths) / max(len(depths), 1),
        "top1_expert_histogram": top1_hist,
    }


@torch.no_grad()
def evaluate_aera(
    model: HardwareAwareAERATextLMV10,
    val: TokenBin,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    carried_losses: list[float] = []
    reset_losses: list[float] = []
    routing: list[dict[str, Any]] = []
    for _ in range(batches):
        x, y = val.batch(batch_size, SEQ_LEN, g, device)
        with _autocast(device):
            out = model(x, hard=True, route_mode="hard_sparse", update_memory=False)
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        carried_losses.append(
            float(F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
        )
        routing.append(_route_metrics(out, model.cfg.n_experts))

        reset_logits: list[torch.Tensor] = []
        for start in range(0, SEQ_LEN, model.cfg.chunk_size):
            chunk = x[:, start : start + model.cfg.chunk_size]
            with _autocast(device):
                reset_out = model(
                    chunk,
                    state=None,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=False,
                )
            chunk_logits = reset_out["logits"]
            assert isinstance(chunk_logits, torch.Tensor)
            reset_logits.append(chunk_logits)
        reset = torch.cat(reset_logits, dim=1)
        reset_losses.append(
            float(F.cross_entropy(reset.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
        )

    carried = sum(carried_losses) / len(carried_losses)
    reset = sum(reset_losses) / len(reset_losses)
    route_summary: dict[str, Any] = {
        "stage_execution_fraction": sum(
            r.get("stage_execution_fraction", 0.0) for r in routing
        ) / len(routing),
        "mean_active_experts_when_executed": sum(
            r.get("mean_active_experts_when_executed", 0.0) for r in routing
        ) / len(routing),
        "mean_reason_steps_when_executed": sum(
            r.get("mean_reason_steps_when_executed", 0.0) for r in routing
        ) / len(routing),
    }
    hist = [0 for _ in range(model.cfg.n_experts)]
    for r in routing:
        for i, count in enumerate(r.get("top1_expert_histogram", [])):
            hist[i] += int(count)
    route_summary["top1_expert_histogram"] = hist
    return {
        "nll": carried,
        "perplexity": math.exp(min(carried, 20.0)),
        "reset_state_nll": reset,
        "state_nll_advantage": reset - carried,
        "routing": route_summary,
    }


def _benchmark_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    aera: bool,
) -> dict[str, float]:
    device = x.device
    model.eval()

    def call() -> None:
        with torch.no_grad(), _autocast(device):
            if aera:
                assert isinstance(model, HardwareAwareAERATextLMV10)
                model(x, hard=True, route_mode="hard_sparse", update_memory=False)
            else:
                model(x)

    for _ in range(3):
        call()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    iters = 10
    for _ in range(iters):
        call()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "ms": 1000.0 * elapsed / iters,
        "tokens_per_second": x.numel() * iters / elapsed,
    }


def _train_one(
    kind: str,
    *,
    train_data: TokenBin,
    val_data: TokenBin,
    device: torch.device,
    seed: int,
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if kind not in {"aera", "transformer"}:
        raise ValueError(kind)
    _seed_all(seed)
    model: torch.nn.Module = build_aera(device) if kind == "aera" else build_transformer(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    warmup = max(1, int(TOTAL_STEPS * 0.02))
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    mode_counts: dict[str, int] = {"hard_sparse": 0, "straight_through": 0}
    last_terms: dict[str, float] = {}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(TOTAL_STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        log_this_step = step == 0 or (step + 1) % 64 == 0
        lm_log: list[float] = []
        total_log: list[float] = []
        for _ in range(GRAD_ACCUM):
            x, y = train_data.batch(MICRO_BATCH, SEQ_LEN, batch_gen, device)
            with _autocast(device):
                if kind == "aera":
                    assert isinstance(model, HardwareAwareAERATextLMV10)
                    total, terms, mode = aera_matched_loss(model, x, y, step=step)
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
                    if log_this_step:
                        last_terms = {
                            k: float(v.detach()) for k, v in terms.items()
                            if isinstance(v, torch.Tensor) and v.numel() == 1
                        }
                else:
                    assert isinstance(model, ResearchLM)
                    logits = model(x)
                    total = F.cross_entropy(
                        logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)
                    )
                scaled = total / GRAD_ACCUM
            scaled.backward()
            if log_this_step:
                total_log.append(float(total.detach()))
                if kind == "aera":
                    lm_log.append(last_terms["next_token"])
                else:
                    lm_log.append(float(total.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, TOTAL_STEPS, warmup, 3e-4)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if log_this_step:
            print(
                {
                    "kind": kind,
                    "step": step + 1,
                    "tokens": (step + 1) * TOKENS_PER_STEP,
                    "lm_loss": sum(lm_log) / len(lm_log),
                    "total_loss": sum(total_log) / len(total_log),
                    "lr": lr,
                    "last_aux": last_terms if kind == "aera" else None,
                },
                flush=True,
            )

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    if kind == "aera":
        assert isinstance(model, HardwareAwareAERATextLMV10)
        final_eval = evaluate_aera(
            model, val_data, batches=20, batch_size=4, seed=seed + 30_000
        )
    else:
        assert isinstance(model, ResearchLM)
        final_eval = evaluate_transformer(
            model, val_data, batches=20, batch_size=4, seed=seed + 30_000
        )
    summary = {
        "kind": kind,
        "seed": seed,
        "tokens_seen": TOKEN_BUDGET,
        "steps": TOTAL_STEPS,
        "training_seconds": training_seconds,
        "training_tokens_per_second": TOKEN_BUDGET / max(training_seconds, 1e-6),
        "peak_vram_gb": peak,
        "final_eval": final_eval,
        "mode_counts": mode_counts if kind == "aera" else None,
        "last_auxiliary_terms": last_terms if kind == "aera" else None,
    }
    torch.save(
        {"kind": kind, "seed": seed, "model": model.state_dict(), "summary": summary},
        checkpoint_path,
    )
    return model, summary


def train_matched_pair(
    *,
    data_dir: str,
    run_dir: str,
    seed: int = SEED,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("real-language gate requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    protocol = validate_protocol(data_dir)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_data = TokenBin(str(Path(data_dir) / "train.bin"))
    val_data = TokenBin(str(Path(data_dir) / "val.bin"))

    transformer, t_summary = _train_one(
        "transformer",
        train_data=train_data,
        val_data=val_data,
        device=device,
        seed=seed,
        checkpoint_path=root / "transformer.pt",
    )
    assert isinstance(transformer, ResearchLM)
    transformer.to("cpu")
    torch.cuda.empty_cache()

    aera, a_summary = _train_one(
        "aera",
        train_data=train_data,
        val_data=val_data,
        device=device,
        seed=seed,
        checkpoint_path=root / "aera.pt",
    )
    assert isinstance(aera, HardwareAwareAERATextLMV10)

    transformer.to(device)
    bench_gen = torch.Generator(device="cpu").manual_seed(seed + 40_000)
    x1, _ = val_data.batch(1, SEQ_LEN, bench_gen, device)
    x8, _ = val_data.batch(8, SEQ_LEN, bench_gen, device)
    inference: dict[str, Any] = {
        "batch1": {
            "transformer": _benchmark_forward(transformer, x1, aera=False),
            "aera": _benchmark_forward(aera, x1, aera=True),
        },
        "batch8": {
            "transformer": _benchmark_forward(transformer, x8, aera=False),
            "aera": _benchmark_forward(aera, x8, aera=True),
        },
    }
    for batch in inference.values():
        batch["aera_vs_transformer_speed"] = (
            batch["aera"]["tokens_per_second"]
            / batch["transformer"]["tokens_per_second"]
        )

    counts = parameter_accounting(aera, transformer)
    a_eval = a_summary["final_eval"]
    t_eval = t_summary["final_eval"]
    routing = a_eval["routing"]
    checks = {
        "quality_gap_nll_le_0_20": a_eval["nll"] - t_eval["nll"] <= 0.20,
        "state_advantage_positive": a_eval["state_nll_advantage"] > 0.0,
        "stage_routing_noncollapsed": 0.15
        <= routing["stage_execution_fraction"]
        <= 0.95,
        "expert_count_adaptive": 1.0
        <= routing["mean_active_experts_when_executed"]
        < 1.95,
        "reasoning_depth_adaptive": 1.0
        <= routing["mean_reason_steps_when_executed"]
        < 3.95,
        "batch8_speed_not_catastrophic": inference["batch8"][
            "aera_vs_transformer_speed"
        ]
        >= 0.60,
    }
    result = {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(device),
        "transformer": t_summary,
        "aera": a_summary,
        "inference": inference,
        "parameter_accounting": counts,
        "checks": checks,
        "seed_pass": all(checks.values()),
        "claims": {
            "one_real_language_seed_only": True,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    import json

    (root / "result.json").write_text(json.dumps(result, indent=2))
    return result
