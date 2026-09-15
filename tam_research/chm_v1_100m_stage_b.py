from __future__ import annotations

"""CHM-v1 ~100M Stage-B GPU systems preflight harness (#981).

Systems-only. Synthetic tokens only. This module deliberately carries no corpus
loader and refuses all scientific/reserved seeds. The sole frozen systems seed
is issue-derived seed 981001.
"""

from contextlib import nullcontext
import math
import random
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F

from .chm_v1_100m_scale import (
    ADDRESS_DIM,
    D_MODEL,
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    FIRST_SCREEN_TOKEN_BUDGET,
    MAX_SEQ_LEN,
    PARAMETER_MISMATCH_LIMIT,
    VOCAB_SIZE,
    CHMV1100MEIEMLM,
    CHMV1100MLocalLM,
)
from .chm_v1_small_lm import LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden
from .models import parameter_count

ISSUE = 981
SYSTEMS_SEED = 981_001
SMOKE_SEED = 981_099
BLOCKED_SEEDS = (
    977_001,
    19_791,
    19_792,
    19_793,
    19_591,
    19_592,
    19_593,
    8_611,
    8_612,
    8_613,
    971_001,
    971_002,
    973_001,
    973_002,
)

GPU_CLASS = "L4"
CPU_CORES = 4
RAM_GIB = 16
MAX_GPU_SECONDS = 900
MAX_STAGE_B_COST_USD = 0.35

BATCH_SIZE = 2
SESSION_LEN = 1_024
CHUNK_LEN = 512
WARMUP_STEPS = 1
MEASURED_STEPS = 20
PEAK_LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
FLAT_TRAIN_TEMPERATURE = 0.10

MAX_PEAK_ALLOCATED_VRAM_BYTES = 22 * 1024**3
MIN_TRAIN_TOKENS_PER_SECOND = 2_000.0
MAX_PROJECTED_COMBINED_SECONDS = 10 * 3600.0
MAX_PROJECTED_COMBINED_COST_USD = 12.0


def validate_systems_seed(seed: int, *, protocol: bool = False) -> int:
    seed = int(seed)
    if seed in BLOCKED_SEEDS:
        raise RuntimeError(f"blocked/scientific seed refused by #981 Stage B: {seed}")
    if protocol and seed != SYSTEMS_SEED:
        raise RuntimeError(f"#981 protocol accepts only systems seed {SYSTEMS_SEED}")
    if not protocol and seed not in (SYSTEMS_SEED, SMOKE_SEED):
        raise RuntimeError(f"unfrozen #981 engineering seed refused: {seed}")
    return seed


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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_stage_b_pair(
    *, seed: int = SYSTEMS_SEED, device: torch.device
) -> tuple[CHMV1100MLocalLM, CHMV1100MEIEMLM]:
    validate_systems_seed(seed, protocol=(seed == SYSTEMS_SEED))
    _seed_all(seed)
    local = CHMV1100MLocalLM().to(device)
    _seed_all(seed)
    eiem = CHMV1100MEIEMLM().to(device)

    local_count = parameter_count(local)
    eiem_count = parameter_count(eiem)
    if local_count != EXPECTED_LOCAL_PARAMETERS:
        raise RuntimeError(f"LOCAL parameter count drift: {local_count}")
    if eiem_count != EXPECTED_EIEM_PARAMETERS:
        raise RuntimeError(f"EIEM parameter count drift: {eiem_count}")
    mismatch = abs(eiem_count - local_count) / local_count
    if mismatch > PARAMETER_MISMATCH_LIMIT:
        raise RuntimeError(f"parameter fairness mismatch: {mismatch}")

    local_state = local.backbone.state_dict()
    eiem_state = eiem.backbone.state_dict()
    if local_state.keys() != eiem_state.keys():
        raise RuntimeError("paired 100M backbone state keys differ")
    for name, local_value in local_state.items():
        if not torch.equal(local_value, eiem_state[name]):
            raise RuntimeError(f"paired 100M backbone init mismatch at {name}")
    return local, eiem


def _chunks(tokens: torch.Tensor):
    if tokens.ndim != 2 or tokens.shape[1] != SESSION_LEN:
        raise ValueError(f"expected [batch,{SESSION_LEN}] sessions")
    for start in range(0, SESSION_LEN, CHUNK_LEN):
        yield tokens[:, start : start + CHUNK_LEN]


def local_session_logits(model: CHMV1100MLocalLM, tokens: torch.Tensor) -> torch.Tensor:
    return torch.cat([model(chunk) for chunk in _chunks(tokens)], dim=1)


def eiem_flat_training_session_logits(
    model: CHMV1100MEIEMLM,
    tokens: torch.Tensor,
    *,
    temperature: float = FLAT_TRAIN_TEMPERATURE,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    memory_keys: torch.Tensor | None = None
    memory_values: torch.Tensor | None = None
    logits: list[torch.Tensor] = []
    for chunk in _chunks(tokens):
        hidden = _hidden(model.backbone, chunk)
        keys = model.key_for(hidden)
        query_state = hidden
        if memory_keys is not None:
            assert memory_values is not None
            for _ in range(RETRIEVAL_HOPS):
                queries = model.query_for(query_state)
                scores = torch.einsum("btd,bsd->bts", queries, memory_keys) / temperature
                weights = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
                memory = torch.matmul(weights, memory_values)
                query_state = model._integrate(query_state, memory)
        logits.append(model.backbone.lm_head(query_state))
        memory_keys = keys if memory_keys is None else torch.cat((memory_keys, keys), dim=1)
        memory_values = hidden if memory_values is None else torch.cat((memory_values, hidden), dim=1)
    return torch.cat(logits, dim=1)


def synthetic_batch(seed: int, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    validate_systems_seed(seed, protocol=(seed == SYSTEMS_SEED))
    generator = torch.Generator(device="cpu").manual_seed(seed + 100_000 + step)
    stream = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, SESSION_LEN + 1),
        generator=generator,
        dtype=torch.long,
    )
    return stream[:, :-1].to(device), stream[:, 1:].to(device)


def _optimizer(model: torch.nn.Module, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": PEAK_LR,
        "betas": BETAS,
        "weight_decay": WEIGHT_DECAY,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _all_finite_parameters(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def _all_finite_grads(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)


def benchmark_training(
    kind: str,
    model: CHMV1100MLocalLM | CHMV1100MEIEMLM,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if kind not in ("local", "eiem"):
        raise ValueError(kind)
    optimizer = _optimizer(model, device)
    total_steps = WARMUP_STEPS + MEASURED_STEPS
    measured: list[float] = []
    losses: list[float] = []
    finite_grads = True

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(total_steps):
        x, y = synthetic_batch(seed, step, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        with _autocast(device):
            if kind == "local":
                assert isinstance(model, CHMV1100MLocalLM)
                logits = local_session_logits(model, x)
            else:
                assert isinstance(model, CHMV1100MEIEMLM)
                logits = eiem_flat_training_session_logits(model, x)
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        finite_grads = finite_grads and _all_finite_grads(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if step >= WARMUP_STEPS:
            measured.append(elapsed)
            losses.append(float(loss.detach()))

    tokens_per_step = BATCH_SIZE * SESSION_LEN
    total_measured_tokens = tokens_per_step * len(measured)
    measured_seconds = sum(measured)
    return {
        "kind": kind,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": len(measured),
        "tokens_per_step": tokens_per_step,
        "measured_tokens": total_measured_tokens,
        "mean_step_seconds": statistics.fmean(measured),
        "median_step_seconds": statistics.median(measured),
        "measured_wall_seconds": measured_seconds,
        "tokens_per_second": total_measured_tokens / max(measured_seconds, 1e-9),
        "last_loss": losses[-1] if losses else float("nan"),
        "loss_finite": all(math.isfinite(x) for x in losses),
        "gradients_finite": finite_grads,
        "parameters_finite": _all_finite_parameters(model),
        "peak_allocated_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_reserved_vram_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "bf16_autocast": device.type == "cuda",
    }


def practical_inference_audit(
    model: CHMV1100MEIEMLM,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    from .chm_v1_batched_eval import forward_session_chunk_batched_transport
    from .chm_v1_optimized_batched_eval import forward_session_chunk_optimized_batched_transport
    from .chm_v1_optimized_replication import OptimizedEpisodicState
    from .chm_v1_small_lm import EpisodicState

    x, _ = synthetic_batch(seed, 50_000, device)
    flat_states = [EpisodicState(f"981-flat-{i}") for i in range(BATCH_SIZE)]
    opt_states = [OptimizedEpisodicState(f"981-opt-{i}") for i in range(BATCH_SIZE)]
    flat_logits: list[torch.Tensor] = []
    opt_logits: list[torch.Tensor] = []
    flat_wall = 0.0
    opt_wall = 0.0
    flat_stats_total = {"reads": 0, "flat_reads": 0, "search": 0.0, "build": 0.0, "wrapper": 0.0}
    opt_stats_total = {"reads": 0, "flat_reads": 0, "search": 0.0, "build": 0.0, "wrapper": 0.0}

    model.eval()
    with torch.no_grad():
        for chunk_no, start in enumerate((0, CHUNK_LEN)):
            chunk = x[:, start : start + CHUNK_LEN]
            _synchronize(device)
            began = time.perf_counter()
            with _autocast(device):
                logits, stats = forward_session_chunk_batched_transport(
                    model,
                    chunk,
                    flat_states,
                    mode="flat",
                    update_memory=True,
                    verify_indexed_exactness=False,
                )
            _synchronize(device)
            flat_wall += time.perf_counter() - began
            flat_logits.append(logits)
            flat_stats_total["reads"] += int(stats.address_vector_reads)
            flat_stats_total["flat_reads"] += int(stats.flat_address_vector_reads)
            flat_stats_total["search"] += float(stats.search_seconds)
            flat_stats_total["build"] += float(stats.index_build_seconds)

            _synchronize(device)
            began = time.perf_counter()
            with _autocast(device):
                logits, stats = forward_session_chunk_optimized_batched_transport(
                    model,
                    chunk,
                    opt_states,
                    update_memory=True,
                    verify_indexed_exactness=True,
                )
            _synchronize(device)
            opt_wall += time.perf_counter() - began
            opt_logits.append(logits)
            opt_stats_total["reads"] += int(stats.address_vector_reads)
            opt_stats_total["flat_reads"] += int(stats.flat_address_vector_reads)
            opt_stats_total["search"] += float(stats.search_seconds)
            opt_stats_total["build"] += float(stats.index_build_seconds)

    flat_joined = torch.cat(flat_logits, dim=1).float()
    opt_joined = torch.cat(opt_logits, dim=1).float()
    max_abs = float((flat_joined - opt_joined).abs().max())
    parity = bool(torch.allclose(flat_joined, opt_joined, rtol=0.0, atol=1e-5))
    opt_stats_total["wrapper"] = sum(float(s.optimized_wrapper_seconds_total) for s in opt_states)
    opt_stats_total["sidecar_build"] = sum(float(s.optimized_sidecar_build_seconds_total) for s in opt_states)
    return {
        "tokens": int(x.numel()),
        "flat_wall_seconds": flat_wall,
        "optimized_wall_seconds": opt_wall,
        "optimized_over_flat_wall_ratio": opt_wall / max(flat_wall, 1e-9),
        "max_abs_logit_delta": max_abs,
        "logits_within_tolerance": parity,
        "flat": flat_stats_total,
        "optimized": opt_stats_total,
    }


def project_screen(
    local_train: dict[str, Any],
    eiem_train: dict[str, Any],
    *,
    live_hourly_resource_cost_usd: float,
) -> dict[str, float]:
    local_seconds = FIRST_SCREEN_TOKEN_BUDGET / float(local_train["tokens_per_second"])
    eiem_seconds = FIRST_SCREEN_TOKEN_BUDGET / float(eiem_train["tokens_per_second"])
    combined_seconds = local_seconds + eiem_seconds
    return {
        "token_budget_per_model": float(FIRST_SCREEN_TOKEN_BUDGET),
        "local_projected_seconds": local_seconds,
        "eiem_projected_seconds": eiem_seconds,
        "combined_projected_seconds": combined_seconds,
        "combined_projected_hours": combined_seconds / 3600.0,
        "live_hourly_resource_cost_usd": live_hourly_resource_cost_usd,
        "combined_projected_cost_usd": combined_seconds / 3600.0 * live_hourly_resource_cost_usd,
    }


def classify_stage_b(result: dict[str, Any]) -> str:
    local = result["training"]["local"]
    eiem = result["training"]["eiem"]
    inference = result["inference"]
    accounting = result["parameter_accounting"]
    correctness = (
        int(accounting["local_trainable_parameters"]) == EXPECTED_LOCAL_PARAMETERS
        and int(accounting["eiem_trainable_parameters"]) == EXPECTED_EIEM_PARAMETERS
        and float(accounting["delta_fraction"]) <= PARAMETER_MISMATCH_LIMIT
        and bool(local["loss_finite"])
        and bool(local["gradients_finite"])
        and bool(local["parameters_finite"])
        and bool(eiem["loss_finite"])
        and bool(eiem["gradients_finite"])
        and bool(eiem["parameters_finite"])
        and bool(inference["logits_within_tolerance"])
    )
    if not correctness:
        return "CHM_V1_100M_STAGE_B_CORRECTNESS_FAIL"

    projection = result["projection"]
    systems_pass = (
        int(local["measured_steps"]) == MEASURED_STEPS
        and int(eiem["measured_steps"]) == MEASURED_STEPS
        and int(local["peak_allocated_vram_bytes"]) <= MAX_PEAK_ALLOCATED_VRAM_BYTES
        and int(eiem["peak_allocated_vram_bytes"]) <= MAX_PEAK_ALLOCATED_VRAM_BYTES
        and float(local["tokens_per_second"]) >= MIN_TRAIN_TOKENS_PER_SECOND
        and float(eiem["tokens_per_second"]) >= MIN_TRAIN_TOKENS_PER_SECOND
        and float(projection["combined_projected_seconds"]) <= MAX_PROJECTED_COMBINED_SECONDS
        and float(projection["combined_projected_cost_usd"]) <= MAX_PROJECTED_COMBINED_COST_USD
    )
    return (
        "CHM_V1_100M_STAGE_B_SYSTEMS_PASS"
        if systems_pass
        else "CHM_V1_100M_STAGE_B_SYSTEMS_STOP"
    )


def static_preflight() -> dict[str, Any]:
    if MAX_SEQ_LEN != SESSION_LEN:
        raise RuntimeError("Stage-A max sequence/session length drift")
    if LOCAL_WINDOW != CHUNK_LEN or SESSION_LEN != 2 * CHUNK_LEN:
        raise RuntimeError("Stage-B two-chunk geometry drift")
    if BATCH_SIZE != 2 or WARMUP_STEPS != 1 or MEASURED_STEPS != 20:
        raise RuntimeError("Stage-B measured workload drift")
    if MAX_GPU_SECONDS != 900 or MAX_STAGE_B_COST_USD != 0.35:
        raise RuntimeError("Stage-B bounded execution envelope drift")
    return {
        "classification": "CHM_V1_100M_STAGE_B_STATIC_PREFLIGHT_PASS",
        "issue": ISSUE,
        "systems_seed": SYSTEMS_SEED,
        "smoke_seed": SMOKE_SEED,
        "gpu": GPU_CLASS,
        "cpu_cores": CPU_CORES,
        "ram_gib": RAM_GIB,
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "max_stage_b_cost_usd": MAX_STAGE_B_COST_USD,
        "batch_size": BATCH_SIZE,
        "session_len": SESSION_LEN,
        "chunk_len": CHUNK_LEN,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "future_screen_token_budget_not_authorized": FIRST_SCREEN_TOKEN_BUDGET,
        "scientific_execution_authorized": False,
    }
