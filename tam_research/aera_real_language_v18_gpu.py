from __future__ import annotations

"""Guarded AERA-v18 real-language memory-integration development harness.

Seed 8351 is development-only. It tests whether the v17 routing/adaptive-compute
candidate remains viable once the existing delta fast memory is actually trained
and used on real language. It cannot count toward independent replication.
"""

from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import aera_real_language_v11 as v11
from . import aera_real_language_v12 as v12
from . import aera_real_language_v18 as v18
from .aera import AERAState
from .aera_hardware_core import HardwareAERAState
from .aera_real_language import SEQ_LEN, TOKEN_BUDGET, VOCAB_SIZE
from .aera_systems_accounting import routing_execution_accounting
from .aera_v14_adaptivity import (
    BATCHES as ADAPTIVITY_BATCHES,
    BATCH_SIZE as ADAPTIVITY_BATCH_SIZE,
    CHUNK_SIZE as ADAPTIVITY_CHUNK_SIZE,
    EXPECTED_CHUNKS,
    _optional_stage_counts,
    summarize_adaptivity,
)
from .data import TokenBin

SEED = 8351
EVAL_SEED = 98_351
MEMORY_EVAL_SEED = 108_351
SYSTEMS_EVAL_SEED = 118_351
TARGET_RATES = torch.tensor([0.50, 1.0 / 3.0, 1.0 / 6.0], dtype=torch.float32)
MEMORY_EVAL_BATCHES = 16
MEMORY_EVAL_BATCH_SIZE = 8
SYSTEM_BATCH_SIZES: tuple[int, ...] = (8, 64)
SYSTEM_WARMUP_ITERS = 3
SYSTEM_TIMED_ITERS = 10

# Frozen development thresholds. These are intentionally preregistered before any
# seed8351 result exists. Failing one is a reason to diagnose v18, not relax it.
QUALITY_GAP_MAX_NLL = 0.50
MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL = 0.005
MEMORY_OVERALL_MIN_ADVANTAGE_NLL = 0.0
WRITE_MEAN_MIN = 0.01
WRITE_MEAN_MAX = 0.95
WRITE_SPREAD_MIN = 0.01
OPTIONAL_STAGE_TARGET_MAE_MAX = 0.12
OPTIONAL_STAGE_MIN_RUN_FRACTION = 0.05
TOTAL_STAGE_EXEC_MIN = 0.35
TOTAL_STAGE_EXEC_MAX = 0.70
BATCH8_MIN_SPEED_RATIO = 0.25
BATCH64_MIN_SPEED_RATIO = 1.25


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _initial_cpu_fairness(seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    aera = v18.build_aera(torch.device("cpu")).eval()
    torch.manual_seed(seed)
    transformer = v18.build_transformer(torch.device("cpu")).eval()
    g = torch.Generator().manual_seed(seed + 10_000)
    x = torch.randint(0, VOCAB_SIZE, (2, 32), generator=g)
    y = torch.randint(0, VOCAB_SIZE, (2, 32), generator=g)
    with torch.no_grad():
        a_out = aera(x, hard=False, route_mode="soft", update_memory=False)
        a_logits = a_out["logits"]
        assert isinstance(a_logits, torch.Tensor)
        t_logits = transformer(x)
    a_nll = float(F.cross_entropy(a_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
    t_nll = float(F.cross_entropy(t_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1)))
    result = {
        "aera_initial_nll": a_nll,
        "transformer_initial_nll": t_nll,
        "nll_gap": a_nll - t_nll,
        "chance_nll": math.log(VOCAB_SIZE),
    }
    if abs(result["nll_gap"]) > 0.50:
        raise RuntimeError(f"initial NLL mismatch exceeds 0.50: {result}")
    return result


def validate_protocol(data_dir: str) -> dict[str, Any]:
    cpu = v18.cpu_preflight()
    data = v12.validate_production_data(data_dir)
    if ADAPTIVITY_CHUNK_SIZE != v18.CHUNK_SIZE:
        raise RuntimeError("v18 held-out adaptivity chunk size mismatch")
    if SEQ_LEN != 2 * v18.CHUNK_SIZE:
        raise RuntimeError("v18 development protocol requires exactly two chunks")
    return {
        **cpu,
        "gpu_authorized": True,
        "gpu_authorization_scope": "one guarded AERA-v18 development seed8351 L4 run only",
        "data": data,
        "development_seed": SEED,
        "counts_toward_independent_replication": False,
        "token_budget_per_model": TOKEN_BUDGET,
        "frozen_optional_stage_target_rates": TARGET_RATES.tolist(),
        "initialization_fairness": _initial_cpu_fairness(SEED),
        "memory_eval": {
            "seed": MEMORY_EVAL_SEED,
            "batches": MEMORY_EVAL_BATCHES,
            "batch_size": MEMORY_EVAL_BATCH_SIZE,
            "memory_advantage_definition": "stream-only NLL minus stream+fast-memory NLL",
            "second_chunk_min_advantage_nll": MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL,
            "overall_min_advantage_nll": MEMORY_OVERALL_MIN_ADVANTAGE_NLL,
            "deployment_update": "detached local delta update; zero base backprop",
        },
        "systems_eval": {
            "seed": SYSTEMS_EVAL_SEED,
            "batch_sizes": list(SYSTEM_BATCH_SIZES),
            "warmup_iterations": SYSTEM_WARMUP_ITERS,
            "timed_iterations": SYSTEM_TIMED_ITERS,
            "batch8_min_aera_vs_transformer": BATCH8_MIN_SPEED_RATIO,
            "batch64_min_aera_vs_transformer": BATCH64_MIN_SPEED_RATIO,
            "aera_memory_enabled": True,
        },
        "development_thresholds": {
            "quality_gap_max_nll": QUALITY_GAP_MAX_NLL,
            "memory_second_chunk_min_advantage_nll": MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL,
            "memory_overall_min_advantage_nll": MEMORY_OVERALL_MIN_ADVANTAGE_NLL,
            "write_mean_range": [WRITE_MEAN_MIN, WRITE_MEAN_MAX],
            "write_p90_minus_p10_min": WRITE_SPREAD_MIN,
            "optional_stage_target_mae_max": OPTIONAL_STAGE_TARGET_MAE_MAX,
            "optional_stage_min_run_fraction": OPTIONAL_STAGE_MIN_RUN_FRACTION,
            "total_stage_execution_range": [TOTAL_STAGE_EXEC_MIN, TOTAL_STAGE_EXEC_MAX],
        },
    }


def _install_v18_harness() -> None:
    v11.build_aera = v18.build_aera
    v11.build_transformer = v18.build_transformer
    v11.aera_matched_loss = v18.aera_matched_loss
    v11.DENSE_WARMUP_STEPS = v18.DENSE_WARMUP_STEPS
    v11.ROUTER_CALIBRATION_END = v18.ROUTER_CALIBRATION_END
    v11.SPARSE_CALIBRATION_EVERY = v18.SPARSE_CALIBRATION_EVERY
    v11.set_stage_router_trainable = v18.set_optional_stage_router_trainable
    v11.validate_protocol = validate_protocol


def _gate_stats(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0, "count": 0.0}
    x = torch.cat([v.detach().float().reshape(-1).cpu() for v in values])
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "p10": float(torch.quantile(x, 0.10)),
        "p90": float(torch.quantile(x, 0.90)),
        "count": float(x.numel()),
    }


def _collect_memory_gates(output: dict[str, object]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list):
        raise RuntimeError("v18 memory eval missing stage_routes")
    reads: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    for chunk in routes:
        if not isinstance(chunk, list):
            raise RuntimeError("invalid v18 stage-route chunk")
        for item in chunk:
            if not isinstance(item, dict):
                raise RuntimeError("invalid v18 route item")
            start = item.get("start")
            end = item.get("end")
            if isinstance(start, dict):
                read = start.get("memory_read")
                if isinstance(read, torch.Tensor):
                    reads.append(read)
            if isinstance(end, dict):
                novelty = end.get("novelty")
                write = end.get("memory_write")
                if isinstance(novelty, torch.Tensor) and isinstance(write, torch.Tensor):
                    writes.append((novelty * write).clamp(0.0, 1.0))
    return reads, writes


def _second_chunk_nll(logits: torch.Tensor, y: torch.Tensor, chunk_size: int) -> float:
    return float(
        F.cross_entropy(
            logits[:, chunk_size:].float().reshape(-1, VOCAB_SIZE),
            y[:, chunk_size:].reshape(-1),
        )
    )


@torch.no_grad()
def _memory_suite(*, data_dir: str, run_dir: str, seed: int) -> dict[str, Any]:
    device = torch.device("cuda")
    root = Path(run_dir)
    a_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    t_payload = torch.load(root / "transformer.pt", map_location="cpu", weights_only=False)
    if a_payload.get("seed") != seed or t_payload.get("seed") != seed:
        raise RuntimeError("v18 memory-suite checkpoint seed mismatch")

    torch.manual_seed(seed)
    aera = v18.build_aera(device).eval()
    torch.manual_seed(seed)
    transformer = v18.build_transformer(device).eval()
    aera.load_state_dict(a_payload["model"], strict=True)
    transformer.load_state_dict(t_payload["model"], strict=True)
    aera.set_memory_pretraining_mode(False)

    parameter_versions_before = [p._version for p in aera.parameters()]
    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(MEMORY_EVAL_SEED)

    t_losses: list[float] = []
    memory_losses: list[float] = []
    stream_losses: list[float] = []
    reset_losses: list[float] = []
    memory_second: list[float] = []
    stream_second: list[float] = []
    reset_second: list[float] = []
    memory_only_second: list[float] = []
    read_values: list[torch.Tensor] = []
    write_values: list[torch.Tensor] = []
    memory_norms: list[torch.Tensor] = []
    state_bytes_per_session: int | None = None

    for _ in range(MEMORY_EVAL_BATCHES):
        x, y = val.batch(MEMORY_EVAL_BATCH_SIZE, SEQ_LEN, g, device)
        with _autocast(device):
            t_logits = transformer(x)
            mem_out = aera(
                x,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
            )
            stream_out = aera(
                x,
                hard=True,
                route_mode="hard_sparse",
                update_memory=False,
            )

        mem_logits = mem_out["logits"]
        stream_logits = stream_out["logits"]
        assert isinstance(mem_logits, torch.Tensor)
        assert isinstance(stream_logits, torch.Tensor)
        t_losses.append(float(F.cross_entropy(t_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))))
        memory_losses.append(float(F.cross_entropy(mem_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))))
        stream_losses.append(float(F.cross_entropy(stream_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))))
        memory_second.append(_second_chunk_nll(mem_logits, y, v18.CHUNK_SIZE))
        stream_second.append(_second_chunk_nll(stream_logits, y, v18.CHUNK_SIZE))

        # Reset both recurrent stream and fast memory at each chunk boundary.
        reset_logits_parts: list[torch.Tensor] = []
        for start in range(0, SEQ_LEN, v18.CHUNK_SIZE):
            chunk = x[:, start : start + v18.CHUNK_SIZE]
            with _autocast(device):
                reset_out = aera(
                    chunk,
                    state=None,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=False,
                )
            chunk_logits = reset_out["logits"]
            assert isinstance(chunk_logits, torch.Tensor)
            reset_logits_parts.append(chunk_logits)
        reset_logits = torch.cat(reset_logits_parts, dim=1)
        reset_losses.append(float(F.cross_entropy(reset_logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))))
        reset_second.append(_second_chunk_nll(reset_logits, y, v18.CHUNK_SIZE))

        # Isolate fast-memory carry: write chunk0, then zero recurrent stream while
        # preserving only the learned session-local memory for chunk1.
        first = x[:, : v18.CHUNK_SIZE]
        second = x[:, v18.CHUNK_SIZE :]
        with _autocast(device):
            first_out = aera(
                first,
                state=None,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
            )
        first_state = first_out.get("state")
        if not isinstance(first_state, HardwareAERAState):
            raise RuntimeError("v18 first-chunk output missing HardwareAERAState")
        memory_only_state = HardwareAERAState(
            [
                AERAState(stream=torch.zeros_like(stage_state.stream), memory=stage_state.memory)
                for stage_state in first_state.stages
            ]
        )
        with _autocast(device):
            memory_only_out = aera(
                second,
                state=memory_only_state,
                hard=True,
                route_mode="hard_sparse",
                update_memory=False,
            )
        memory_only_logits = memory_only_out["logits"]
        assert isinstance(memory_only_logits, torch.Tensor)
        memory_only_second.append(
            float(
                F.cross_entropy(
                    memory_only_logits.float().reshape(-1, VOCAB_SIZE),
                    y[:, v18.CHUNK_SIZE :].reshape(-1),
                )
            )
        )

        reads, writes = _collect_memory_gates(mem_out)
        read_values.extend(reads)
        write_values.extend(writes)
        final_state = mem_out.get("state")
        if not isinstance(final_state, HardwareAERAState):
            raise RuntimeError("v18 memory output missing HardwareAERAState")
        bytes_this_batch = 0
        for stage_state in final_state.stages:
            matrix = stage_state.memory.matrix
            memory_norms.append(
                torch.linalg.vector_norm(matrix.float().reshape(matrix.size(0), -1), dim=1).cpu()
            )
            bytes_this_batch += matrix.numel() * matrix.element_size()
        per_session = bytes_this_batch // MEMORY_EVAL_BATCH_SIZE
        if state_bytes_per_session is None:
            state_bytes_per_session = per_session
        elif state_bytes_per_session != per_session:
            raise RuntimeError("v18 memory state bytes/session changed across batches")

    versions_after = [p._version for p in aera.parameters()]
    base_parameters_unchanged = parameter_versions_before == versions_after

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    t_nll = mean(t_losses)
    mem_nll = mean(memory_losses)
    stream_nll = mean(stream_losses)
    reset_nll = mean(reset_losses)
    mem_second = mean(memory_second)
    stream_second_nll = mean(stream_second)
    reset_second_nll = mean(reset_second)
    memory_only_second_nll = mean(memory_only_second)
    write_stats = _gate_stats(write_values)
    read_stats = _gate_stats(read_values)
    memory_norm = float(torch.cat(memory_norms).mean()) if memory_norms else 0.0

    return {
        "seed": seed,
        "eval_seed": MEMORY_EVAL_SEED,
        "examples": MEMORY_EVAL_BATCHES * MEMORY_EVAL_BATCH_SIZE,
        "transformer_nll": t_nll,
        "stream_plus_memory_nll": mem_nll,
        "stream_only_nll": stream_nll,
        "reset_state_and_memory_nll": reset_nll,
        "quality_gap_nll": mem_nll - t_nll,
        "memory_overall_advantage_nll": stream_nll - mem_nll,
        "stream_over_reset_advantage_nll": reset_nll - stream_nll,
        "second_chunk": {
            "stream_plus_memory_nll": mem_second,
            "stream_only_nll": stream_second_nll,
            "memory_only_nll": memory_only_second_nll,
            "reset_state_and_memory_nll": reset_second_nll,
            "memory_advantage_nll": stream_second_nll - mem_second,
            "stream_advantage_nll": reset_second_nll - stream_second_nll,
            "memory_only_advantage_over_reset_nll": reset_second_nll - memory_only_second_nll,
        },
        "executed_stage_memory_read_gate": read_stats,
        "executed_stage_effective_write_strength": write_stats,
        "mean_final_memory_frobenius_norm": memory_norm,
        "memory_state_bytes_per_session": int(state_bytes_per_session or 0),
        "deployment_base_parameter_versions_unchanged": base_parameters_unchanged,
        "training_performed": False,
        "checkpoint_mutated": False,
    }


@torch.no_grad()
def _heldout_adaptivity(*, data_dir: str, run_dir: str, seed: int) -> dict[str, Any]:
    device = torch.device("cuda")
    root = Path(run_dir)
    a_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    t_payload = torch.load(root / "transformer.pt", map_location="cpu", weights_only=False)
    if a_payload.get("seed") != seed or t_payload.get("seed") != seed:
        raise RuntimeError("v18 held-out checkpoint seed mismatch")

    torch.manual_seed(seed)
    aera = v18.build_aera(device).eval()
    torch.manual_seed(seed)
    transformer = v18.build_transformer(device).eval()
    aera.load_state_dict(a_payload["model"], strict=True)
    transformer.load_state_dict(t_payload["model"], strict=True)
    aera.set_memory_pretraining_mode(False)

    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    all_difficulty: list[torch.Tensor] = []
    all_compute: list[torch.Tensor] = []
    all_positions: list[torch.Tensor] = []
    stage_runs = torch.zeros(3, dtype=torch.float64)

    for _ in range(ADAPTIVITY_BATCHES):
        x, y = val.batch(ADAPTIVITY_BATCH_SIZE, SEQ_LEN, g, device)
        with _autocast(device):
            t_logits = transformer(x)
            a_out = aera(
                x,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
            )
        t_loss = F.cross_entropy(
            t_logits.float().reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            reduction="none",
        ).reshape(ADAPTIVITY_BATCH_SIZE, SEQ_LEN)
        counts = _optional_stage_counts(a_out, ADAPTIVITY_BATCH_SIZE)
        routes = a_out.get("stage_routes")
        if not isinstance(routes, list):
            raise RuntimeError("v18 held-out eval missing stage routes")

        for chunk_index, start in enumerate(range(0, SEQ_LEN, ADAPTIVITY_CHUNK_SIZE)):
            all_difficulty.append(
                t_loss[:, start : start + ADAPTIVITY_CHUNK_SIZE].mean(dim=1).cpu()
            )
            all_compute.append(counts[chunk_index].cpu())
            all_positions.append(
                torch.full((ADAPTIVITY_BATCH_SIZE,), chunk_index, dtype=torch.long)
            )
            chunk = routes[chunk_index]
            if not isinstance(chunk, list) or len(chunk) != 4:
                raise RuntimeError("v18 held-out eval invalid stage routes")
            for stage_index, item in enumerate(chunk[1:]):
                if not isinstance(item, dict):
                    raise RuntimeError("v18 held-out eval invalid route item")
                gate = item.get("stage_route_gate")
                if not isinstance(gate, torch.Tensor):
                    raise RuntimeError("v18 held-out eval missing stage gate")
                stage_runs[stage_index] += float((gate[:, 0] >= 0.5).sum())

    difficulty = torch.cat(all_difficulty)
    compute = torch.cat(all_compute)
    positions = torch.cat(all_positions)
    if difficulty.numel() != EXPECTED_CHUNKS:
        raise RuntimeError(f"v18 held-out sample count {difficulty.numel()} != {EXPECTED_CHUNKS}")
    summary = summarize_adaptivity(difficulty, compute, positions)
    optional_rates = [float(v / EXPECTED_CHUNKS) for v in stage_runs]
    summary.update(
        {
            "seed": seed,
            "eval_seed": EVAL_SEED,
            "difficulty_definition": "matched Transformer per-chunk held-out next-token NLL",
            "compute_definition": "memory-enabled AERA hard optional whole-stage executions per chunk",
            "optional_stage_run_fractions": optional_rates,
            "total_per_example_stage_execution_fraction": (1.0 + sum(optional_rates)) / 4.0,
            "training_performed": False,
            "checkpoint_mutated": False,
            "counts_toward_independent_replication": False,
        }
    )
    return summary


def _benchmark_cuda(call: Callable[[], object], *, device: torch.device) -> dict[str, float]:
    for _ in range(SYSTEM_WARMUP_ITERS):
        call()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(SYSTEM_TIMED_ITERS):
        call()
    end.record()
    torch.cuda.synchronize(device)
    ms = float(start.elapsed_time(end)) / SYSTEM_TIMED_ITERS
    return {
        "ms": ms,
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
    }


@torch.no_grad()
def _systems_suite(*, data_dir: str, run_dir: str, seed: int) -> dict[str, Any]:
    device = torch.device("cuda")
    root = Path(run_dir)
    a_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    t_payload = torch.load(root / "transformer.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(seed)
    aera = v18.build_aera(device).eval()
    torch.manual_seed(seed)
    transformer = v18.build_transformer(device).eval()
    aera.load_state_dict(a_payload["model"], strict=True)
    transformer.load_state_dict(t_payload["model"], strict=True)
    aera.set_memory_pretraining_mode(False)
    val = TokenBin(str(Path(data_dir) / "val.bin"))

    rows: dict[str, Any] = {}
    for batch_size in SYSTEM_BATCH_SIZES:
        g = torch.Generator(device="cpu").manual_seed(SYSTEMS_EVAL_SEED + batch_size)
        x, _ = val.batch(batch_size, SEQ_LEN, g, device)

        def aera_call() -> object:
            with _autocast(device):
                return aera(
                    x,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=True,
                )

        def transformer_call() -> object:
            with _autocast(device):
                return transformer(x)

        with _autocast(device):
            routing_out = aera(
                x,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
            )
        routing = routing_execution_accounting(routing_out)
        del routing_out
        torch.cuda.empty_cache()

        a_t = _benchmark_cuda(aera_call, device=device)
        torch.cuda.empty_cache()
        t_t = _benchmark_cuda(transformer_call, device=device)
        for timing in (a_t, t_t):
            timing["tokens_per_second"] = batch_size * SEQ_LEN * 1000.0 / timing["ms"]
        rows[str(batch_size)] = {
            "aera": a_t,
            "transformer": t_t,
            "aera_vs_transformer_speed": a_t["tokens_per_second"] / t_t["tokens_per_second"],
            "routing": routing,
        }
        del x
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "eval_seed": SYSTEMS_EVAL_SEED,
        "memory_enabled": True,
        "rows": rows,
        "training_performed": False,
        "checkpoint_mutated": False,
    }


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v18 development run is frozen to fresh seed {SEED}")
    _install_v18_harness()
    result = v11.train_matched_pair(data_dir=data_dir, run_dir=run_dir, seed=seed)

    # Release transient training model objects/caches before independent audits.
    gc.collect()
    torch.cuda.empty_cache()

    memory = _memory_suite(data_dir=data_dir, run_dir=run_dir, seed=seed)
    gc.collect()
    torch.cuda.empty_cache()
    adaptivity = _heldout_adaptivity(data_dir=data_dir, run_dir=run_dir, seed=seed)
    gc.collect()
    torch.cuda.empty_cache()
    systems = _systems_suite(data_dir=data_dir, run_dir=run_dir, seed=seed)

    optional_rates = [float(v) for v in adaptivity["optional_stage_run_fractions"]]
    target_rates = TARGET_RATES.tolist()
    target_mae = sum(abs(a - b) for a, b in zip(optional_rates, target_rates)) / len(target_rates)
    write_stats = memory["executed_stage_effective_write_strength"]
    second = memory["second_chunk"]
    b8 = systems["rows"]["8"]["aera_vs_transformer_speed"]
    b64 = systems["rows"]["64"]["aera_vs_transformer_speed"]
    total_stage_execution = float(adaptivity["total_per_example_stage_execution_fraction"])

    checks = {
        "quality_gap_nll_le_0_50": memory["quality_gap_nll"] <= QUALITY_GAP_MAX_NLL,
        "memory_second_chunk_advantage_ge_0_005": second["memory_advantage_nll"] >= MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL,
        "memory_overall_advantage_nonnegative": memory["memory_overall_advantage_nll"] >= MEMORY_OVERALL_MIN_ADVANTAGE_NLL,
        "stream_second_chunk_advantage_nonnegative": second["stream_advantage_nll"] >= 0.0,
        "memory_only_second_advantage_nonnegative": second["memory_only_advantage_over_reset_nll"] >= 0.0,
        "write_gate_not_mean_collapsed": WRITE_MEAN_MIN <= write_stats["mean"] <= WRITE_MEAN_MAX,
        "write_gate_has_variation": (write_stats["p90"] - write_stats["p10"]) >= WRITE_SPREAD_MIN,
        "memory_state_nonzero": memory["mean_final_memory_frobenius_norm"] > 0.0,
        "deployment_base_parameters_unchanged": bool(memory["deployment_base_parameter_versions_unchanged"]),
        "heldout_difficulty_adaptivity_pass": bool(adaptivity["pass"]),
        "all_optional_stages_run_ge_0_05": all(v >= OPTIONAL_STAGE_MIN_RUN_FRACTION for v in optional_rates),
        "optional_stage_rates_shallow_to_deep": optional_rates[0] >= optional_rates[1] >= optional_rates[2],
        "optional_stage_target_mae_le_0_12": target_mae <= OPTIONAL_STAGE_TARGET_MAE_MAX,
        "total_per_example_stage_execution_35_to_70pct": TOTAL_STAGE_EXEC_MIN <= total_stage_execution <= TOTAL_STAGE_EXEC_MAX,
        "batch8_memory_enabled_speed_ge_0_25x": b8 >= BATCH8_MIN_SPEED_RATIO,
        "batch64_memory_enabled_speed_ge_1_25x": b64 >= BATCH64_MIN_SPEED_RATIO,
    }

    result["v18_memory_eval"] = memory
    result["v18_heldout_adaptivity"] = adaptivity
    result["v18_systems_eval"] = systems
    result["diagnostics"]["optional_stage_run_fractions"] = optional_rates
    result["diagnostics"]["optional_stage_target_rate_mae"] = target_mae
    result["diagnostics"]["actual_total_per_example_stage_execution_fraction"] = total_stage_execution
    result["diagnostics"]["memory_enabled_quality_gap_nll"] = memory["quality_gap_nll"]
    result["diagnostics"]["memory_second_chunk_advantage_nll"] = second["memory_advantage_nll"]
    result["diagnostics"]["memory_enabled_batch64_speed_ratio"] = b64
    result["v18_development_checks"] = checks
    result["v18_development_pass"] = all(checks.values())
    result["claims"] = {
        "development_seed_only": True,
        "counts_toward_independent_replication": False,
        "architecture_frozen_for_replication": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    Path(run_dir, "result.json").write_text(json.dumps(result, indent=2))
    return result
