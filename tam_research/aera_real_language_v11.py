from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig, hardware_parameter_accounting
from .aera_hardware_core_v8 import MostlyHardRoutingSchedule
from .aera_hardware_core_v11 import HardwareAwareAERATextLMV11, initial_logit_stats
from .aera_real_language import (
    VOCAB_SIZE,
    SEQ_LEN,
    TOKEN_BUDGET,
    MICRO_BATCH,
    GRAD_ACCUM,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    _benchmark_forward,
    _route_metrics,
    evaluate_transformer,
    parameter_accounting,
    transformer_25m_config,
)
from .data import TokenBin
from .models import ResearchLM, parameter_count
from .train import cosine_lr

# Development-only seed. It must never be counted as a preregistered breakthrough
# replication seed because v11 was designed after inspecting v10 seed8201.
SEED = 8211
CHUNK_SIZE = 256
DENSE_WARMUP_STEPS = 192
ROUTER_CALIBRATION_END = 320
SPARSE_CALIBRATION_EVERY = 4


def aera_v11_config() -> HardwareAERAConfig:
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


def build_aera(device: torch.device) -> HardwareAwareAERATextLMV11:
    return HardwareAwareAERATextLMV11(aera_v11_config()).to(device)


def build_transformer(device: torch.device) -> ResearchLM:
    return ResearchLM(transformer_25m_config()).to(device)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def phase_for_step(step: int) -> str:
    if step < 0 or step >= TOTAL_STEPS:
        raise ValueError(f"step must be in [0,{TOTAL_STEPS})")
    if step < DENSE_WARMUP_STEPS:
        return "representation_warmup"
    if step < ROUTER_CALIBRATION_END:
        return "router_calibration"
    return "mostly_hard_sparse"


def route_mode_for_step(step: int) -> str:
    phase = phase_for_step(step)
    if phase in {"representation_warmup", "router_calibration"}:
        return "straight_through"
    return MostlyHardRoutingSchedule(
        calibration_every=SPARSE_CALIBRATION_EVERY
    ).mode_for_step(step - ROUTER_CALIBRATION_END)


def set_stage_router_trainable(model: HardwareAwareAERATextLMV11, trainable: bool) -> None:
    for parameter in model.stage_routers.parameters():
        parameter.requires_grad_(trainable)


def _loss_weights(step: int) -> dict[str, float]:
    phase = phase_for_step(step)
    if phase == "representation_warmup":
        # Keep next-token CE overwhelmingly dominant while the representation learns.
        return {
            "event": 0.002,
            "stream": 0.010,
            "compute": 0.0,
            "balance": 0.005,
            "stage": 0.0,
        }
    if phase == "router_calibration":
        return {
            "event": 0.005,
            "stream": 0.020,
            "compute": 0.0005,
            "balance": 0.010,
            "stage": 0.0005,
        }
    return {
        "event": 0.005,
        "stream": 0.020,
        "compute": 0.0005,
        "balance": 0.010,
        "stage": 0.0010,
    }


def aera_matched_loss(
    model: HardwareAwareAERATextLMV11,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str, str]:
    phase = phase_for_step(step)
    mode = route_mode_for_step(step)
    hard = mode == "hard_sparse"
    weights = _loss_weights(step)
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
            event_weight=weights["event"],
            block_weight=0.0,
            stream_forecast_weight=weights["stream"],
        )
    else:
        aux = model.soft_objective(
            x,
            out,
            event_weight=weights["event"],
            compute_weight=weights["compute"],
            balance_weight=weights["balance"],
            block_weight=0.0,
            stream_forecast_weight=weights["stream"],
            stage_compute_weight=weights["stage"],
        )

    # Replace the inherited x[t]->x[t+1] CE with the exact shared x->y target.
    total = primary + (aux["total"] - aux["next_token"])
    terms = dict(aux)
    terms["next_token"] = primary
    terms["total"] = total
    return total, terms, mode, phase


def _initial_cpu_fairness(seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    aera = build_aera(torch.device("cpu")).eval()
    torch.manual_seed(seed)
    transformer = build_transformer(torch.device("cpu")).eval()
    g = torch.Generator().manual_seed(seed + 10_000)
    x = torch.randint(0, VOCAB_SIZE, (2, 32), generator=g)
    y = torch.randint(0, VOCAB_SIZE, (2, 32), generator=g)
    with torch.no_grad():
        aout = aera(x, hard=False, route_mode="soft", update_memory=False)
        alogits = aout["logits"]
        assert isinstance(alogits, torch.Tensor)
        tlogits = transformer(x)
    anll = float(F.cross_entropy(alogits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
    tnll = float(F.cross_entropy(tlogits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
    result = {
        "aera_initial_nll": anll,
        "transformer_initial_nll": tnll,
        "nll_gap": anll - tnll,
        "chance_nll": math.log(VOCAB_SIZE),
        "aera_logits": initial_logit_stats(aera, x),
    }
    if abs(result["nll_gap"]) > 0.50:
        raise RuntimeError(f"initial NLL mismatch exceeds 0.50: {result}")
    if abs(anll - result["chance_nll"]) > 0.75:
        raise RuntimeError(f"AERA initial NLL is not chance-scale: {result}")
    return result


def validate_protocol(data_dir: str) -> dict[str, Any]:
    if TOKEN_BUDGET % TOKENS_PER_STEP:
        raise RuntimeError("token budget must divide exactly by tokens/step")
    if CHUNK_SIZE != 256 or SEQ_LEN % CHUNK_SIZE:
        raise RuntimeError("v11 requires exactly two 256-token chunks per 512-token example")
    if not (0 < DENSE_WARMUP_STEPS < ROUTER_CALIBRATION_END < TOTAL_STEPS):
        raise RuntimeError("invalid progressive routing curriculum boundaries")

    root = Path(data_dir)
    required = [root / "train.bin", root / "val.bin", root / "meta.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"real-language source data missing: {missing}")
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

    torch.manual_seed(SEED)
    aera = build_aera(torch.device("cpu"))
    torch.manual_seed(SEED)
    transformer = build_transformer(torch.device("cpu"))
    counts = parameter_accounting(aera, transformer)
    if abs(counts["stored_parameter_delta_fraction"]) > 0.05:
        raise RuntimeError(f"stored parameter mismatch exceeds 5%: {counts}")
    active = hardware_parameter_accounting(aera, mean_active_experts=1.0)
    del aera, transformer
    initial = _initial_cpu_fairness(SEED)
    return {
        "data": expected,
        "development_seed": SEED,
        "counts_toward_breakthrough_evidence": False,
        "token_budget_per_model": TOKEN_BUDGET,
        "tokens_per_step": TOKENS_PER_STEP,
        "optimizer_steps": TOTAL_STEPS,
        "chunk_size": CHUNK_SIZE,
        "chunks_per_sequence": SEQ_LEN // CHUNK_SIZE,
        "curriculum": {
            "representation_warmup_steps": DENSE_WARMUP_STEPS,
            "router_calibration_end_step": ROUTER_CALIBRATION_END,
            "sparse_calibration_every": SPARSE_CALIBRATION_EVERY,
        },
        "batch_sampling_seed": SEED + 10_000,
        "parameter_accounting": counts,
        "top1_full_stage_active_estimate": active,
        "initialization_fairness": initial,
    }


@torch.no_grad()
def evaluate_aera(
    model: HardwareAwareAERATextLMV11,
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
        "stage_execution_fraction": sum(r.get("stage_execution_fraction", 0.0) for r in routing) / len(routing),
        "mean_active_experts_when_executed": sum(r.get("mean_active_experts_when_executed", 0.0) for r in routing) / len(routing),
        "mean_reason_steps_when_executed": sum(r.get("mean_reason_steps_when_executed", 0.0) for r in routing) / len(routing),
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
    if kind == "aera":
        assert isinstance(model, HardwareAwareAERATextLMV11)
        set_stage_router_trainable(model, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )
    warmup = max(1, int(TOTAL_STEPS * 0.02))
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    mode_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    last_terms: dict[str, float] = {}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(TOTAL_STEPS):
        model.train()
        if kind == "aera" and step == DENSE_WARMUP_STEPS:
            assert isinstance(model, HardwareAwareAERATextLMV11)
            set_stage_router_trainable(model, True)
        optimizer.zero_grad(set_to_none=True)
        log_this_step = step == 0 or (step + 1) % 64 == 0
        lm_log: list[float] = []
        total_log: list[float] = []
        phase_name = "transformer"
        mode_name = "dense"
        for _ in range(GRAD_ACCUM):
            x, y = train_data.batch(MICRO_BATCH, SEQ_LEN, batch_gen, device)
            with _autocast(device):
                if kind == "aera":
                    assert isinstance(model, HardwareAwareAERATextLMV11)
                    total, terms, mode_name, phase_name = aera_matched_loss(model, x, y, step=step)
                    mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
                    phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
                    if log_this_step:
                        last_terms = {
                            k: float(v.detach())
                            for k, v in terms.items()
                            if isinstance(v, torch.Tensor) and v.numel() == 1
                        }
                else:
                    assert isinstance(model, ResearchLM)
                    logits = model(x)
                    total = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
                scaled = total / GRAD_ACCUM
            scaled.backward()
            if log_this_step:
                total_log.append(float(total.detach()))
                lm_log.append(last_terms["next_token"] if kind == "aera" else float(total.detach()))
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
                    "phase": phase_name if kind == "aera" else None,
                    "route_mode": mode_name if kind == "aera" else None,
                    "last_aux": last_terms if kind == "aera" else None,
                },
                flush=True,
            )

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    if kind == "aera":
        assert isinstance(model, HardwareAwareAERATextLMV11)
        final_eval = evaluate_aera(model, val_data, batches=20, batch_size=4, seed=seed + 30_000)
    else:
        assert isinstance(model, ResearchLM)
        final_eval = evaluate_transformer(model, val_data, batches=20, batch_size=4, seed=seed + 30_000)
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
        "phase_counts": phase_counts if kind == "aera" else None,
        "last_auxiliary_terms": last_terms if kind == "aera" else None,
    }
    torch.save(
        {"kind": kind, "seed": seed, "model": model.state_dict(), "summary": summary},
        checkpoint_path,
    )
    return model, summary


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("AERA-v11 real-language development gate requires CUDA")
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
    assert isinstance(aera, HardwareAwareAERATextLMV11)

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
            batch["aera"]["tokens_per_second"] / batch["transformer"]["tokens_per_second"]
        )

    counts = parameter_accounting(aera, transformer)
    a_eval = a_summary["final_eval"]
    t_eval = t_summary["final_eval"]
    routing = a_eval["routing"]
    diagnostics = {
        "quality_gap_nll": a_eval["nll"] - t_eval["nll"],
        "state_advantage_nll": a_eval["state_nll_advantage"],
        "stage_execution_fraction": routing["stage_execution_fraction"],
        "mean_active_experts": routing["mean_active_experts_when_executed"],
        "mean_reason_steps": routing["mean_reason_steps_when_executed"],
        "batch8_speed_ratio": inference["batch8"]["aera_vs_transformer_speed"],
    }
    # Development gate only: these thresholds decide whether v11 is worth another
    # architecture iteration, not whether 100M is authorized.
    checks = {
        "quality_gap_nll_le_1_00": diagnostics["quality_gap_nll"] <= 1.00,
        "state_advantage_nonnegative": diagnostics["state_advantage_nll"] >= 0.0,
        "stage_routing_not_all_on": diagnostics["stage_execution_fraction"] < 0.98,
        "batch8_speed_ratio_ge_0_30": diagnostics["batch8_speed_ratio"] >= 0.30,
    }
    result = {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(device),
        "transformer": t_summary,
        "aera": a_summary,
        "inference": inference,
        "parameter_accounting": counts,
        "diagnostics": diagnostics,
        "development_checks": checks,
        "development_pass": all(checks.values()),
        "claims": {
            "development_seed_only": True,
            "counts_toward_breakthrough_evidence": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    return result
