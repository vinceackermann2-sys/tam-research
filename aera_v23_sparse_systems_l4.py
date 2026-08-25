from __future__ import annotations

"""Systems-only L4 benchmark frozen by research issue #333.

No corpus is read, no optimizer step is taken, and no checkpoint is written.
The benchmark measures whether the controlled-PASS AERA-v23 16/255 physical
write path fits the established 512-step real-language development envelope.
"""

import json
from typing import Any, Callable

import torch
import torch.nn.functional as F

from tam_research import aera_hardware_core_v22 as core22
from tam_research import aera_real_language_v23_efficiency as eff
from tam_research.aera_hardware_core_v23 import (
    BudgetedSparseDualDeltaFastMemoryStage,
    sparse_write_budget,
)
from tam_research.aera_real_language import GRAD_ACCUM, MICRO_BATCH, SEQ_LEN, TOTAL_STEPS

BENCH_SEED = 8431
MEMORY_DIM = 50
DENSE_CANDIDATES = 255
SPARSE_CANDIDATES = 16
RECURRENCE_BATCH = 8
EVAL_MARGIN_SECONDS = 150.0
PROJECTION_MAX_SECONDS = 1500.0
MIN_RECURRENCE_SPEEDUP = 4.0


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _recurrence_fixture(
    device: torch.device,
    *,
    candidates: int,
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + candidates)
    matrix = torch.randn(
        RECURRENCE_BATCH,
        MEMORY_DIM,
        MEMORY_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    ) * 0.01
    inverse = torch.eye(MEMORY_DIM, device=device, dtype=torch.bfloat16).expand(
        RECURRENCE_BATCH, -1, -1
    ).clone()
    keys = F.normalize(
        torch.randn(
            RECURRENCE_BATCH,
            candidates,
            MEMORY_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=g,
        ).float(),
        dim=-1,
    ).to(torch.bfloat16)
    targets = torch.tanh(
        torch.randn(
            RECURRENCE_BATCH,
            candidates,
            MEMORY_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=g,
        )
    )
    strengths = torch.sigmoid(
        torch.randn(
            RECURRENCE_BATCH,
            candidates,
            1,
            device=device,
            dtype=torch.bfloat16,
            generator=g,
        )
    )
    return matrix, inverse, keys, targets, strengths


def _measure_recurrence(
    device: torch.device,
    *,
    candidates: int,
    warmups: int = 1,
    iters: int = 2,
) -> dict[str, float]:
    base = _recurrence_fixture(device, candidates=candidates)

    def one(*, timed: bool) -> float:
        inputs = [x.detach().clone().requires_grad_(True) for x in base]
        if timed:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        m, p = core22.interference_corrected_dual_delta_update(*inputs)
        loss = m.float().square().mean() + 0.1 * p.float().square().mean()
        loss.backward()
        if not bool(torch.isfinite(m).all() and torch.isfinite(p).all()):
            raise RuntimeError("nonfinite isolated dual-delta state")
        for value in inputs:
            if value.grad is None or not bool(torch.isfinite(value.grad).all()):
                raise RuntimeError("nonfinite/missing isolated dual-delta gradient")
        return _event_ms(start, end) if timed else 0.0

    for _ in range(warmups):
        one(timed=False)
    torch.cuda.reset_peak_memory_stats()
    samples = [one(timed=True) for _ in range(iters)]
    return {
        "ms": sum(samples) / len(samples),
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
    }


def _random_language_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 100)
    x = torch.randint(
        0,
        50_257,
        (MICRO_BATCH, SEQ_LEN),
        device=device,
        generator=g,
    )
    y = torch.randint(
        0,
        50_257,
        (MICRO_BATCH, SEQ_LEN),
        device=device,
        generator=g,
    )
    return x, y


def _reset_sparse_stats(model) -> None:
    for stage in model.stages:
        if isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
            stage.last_candidate_count = 0
            stage.last_selected_count = 0
            stage.last_selected_indices = None


def _sparse_stats(model) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i, stage in enumerate(model.stages):
        if not isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
            continue
        if stage.last_candidate_count:
            rows.append(
                {
                    "stage": float(i),
                    "candidates": float(stage.last_candidate_count),
                    "selected": float(stage.last_selected_count),
                    "fraction": stage.last_selected_count / stage.last_candidate_count,
                }
            )
    return rows


def _assert_16_of_255(rows: list[dict[str, float]], *, label: str) -> None:
    if not rows:
        raise RuntimeError(f"{label} executed no measurable sparse stage")
    for row in rows:
        if int(row["candidates"]) != DENSE_CANDIDATES:
            raise RuntimeError(f"{label} candidate count mismatch: {rows}")
        if int(row["selected"]) != SPARSE_CANDIDATES:
            raise RuntimeError(f"{label} selected count mismatch: {rows}")


def _measure_stage_path(model, device: torch.device) -> dict[str, Any]:
    stage = model.stages[0]
    if not isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
        raise RuntimeError("v23 benchmark expected sparse stage")
    stage.memory.set_differentiable_pretraining(True)
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 101)
    base_events = torch.randn(
        RECURRENCE_BATCH,
        eff.CHUNK_SIZE,
        model.cfg.d_model,
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    )

    def one(*, timed: bool) -> float:
        events = base_events.detach().clone().requires_grad_(True)
        stage.zero_grad(set_to_none=True)
        if timed:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        out, state, _ = stage.forward_chunk(
            events,
            None,
            hard=False,
            update_memory=True,
        )
        loss = (
            out.float().square().mean()
            + state.memory.matrix.float().square().mean()
            + 0.01 * state.memory.inverse_key_covariance.float().square().mean()
        )
        loss.backward()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite v23 stage sparse path")
        if stage.last_candidate_count != DENSE_CANDIDATES:
            raise RuntimeError("v23 stage path did not expose 255 candidates")
        if stage.last_selected_count != SPARSE_CANDIDATES:
            raise RuntimeError("v23 stage path did not execute 16 writes")
        return _event_ms(start, end) if timed else 0.0

    one(timed=False)
    torch.cuda.reset_peak_memory_stats()
    ms = one(timed=True)
    result = {
        "ms": ms,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
        "candidates": stage.last_candidate_count,
        "selected_writes": stage.last_selected_count,
        "selected_fraction": stage.last_selected_count / stage.last_candidate_count,
    }
    stage.memory.set_differentiable_pretraining(False)
    stage.zero_grad(set_to_none=True)
    return result


def _measure_aux(model, x: torch.Tensor, *, step: int) -> dict[str, float]:
    def one(*, timed: bool) -> float:
        model.zero_grad(set_to_none=True)
        if timed:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            terms = eff.memory_auxiliary_terms(model, x, step=step)
            loss = (
                terms["memory_address_contrastive_loss"]
                + terms["memory_payload_token_loss"]
            ) / GRAD_ACCUM
        loss.backward()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite v23 memory auxiliary")
        return _event_ms(start, end) if timed else 0.0

    one(timed=False)
    torch.cuda.reset_peak_memory_stats()
    ms = one(timed=True)
    return {
        "ms": ms,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
    }


def _measure_full_microbatch(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
) -> dict[str, Any]:
    def one(*, timed: bool) -> tuple[float, list[dict[str, float]]]:
        model.zero_grad(set_to_none=True)
        _reset_sparse_stats(model)
        if timed:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, _, mode, phase = eff.aera_matched_loss(model, x, y, step=step)
            scaled = total / GRAD_ACCUM
        scaled.backward()
        if not bool(torch.isfinite(scaled)):
            raise RuntimeError("nonfinite v23 full microbatch")
        rows = _sparse_stats(model)
        _assert_16_of_255(rows, label=f"full microbatch step={step}")
        ms = _event_ms(start, end) if timed else 0.0
        return ms, rows

    one(timed=False)
    torch.cuda.reset_peak_memory_stats()
    ms, rows = one(timed=True)
    return {
        "ms": ms,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
        "executed_sparse_stages": rows,
        "route_mode": eff.route_mode_for_step(step),
        "phase": eff.phase_for_step(step),
    }


def _mode_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in range(TOTAL_STEPS):
        mode = eff.route_mode_for_step(step)
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _measure_inference(
    call: Callable[[torch.Tensor], None],
    tokens: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        call(tokens)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        call(tokens)
    ms = _event_ms(start, end)
    return {
        "ms": ms,
        "tokens_per_second": tokens.numel() / (ms / 1000.0),
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
    }


def _inference_comparison(device: torch.device, *, batch: int) -> dict[str, Any]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 200 + batch)
    tokens = torch.randint(
        0,
        50_257,
        (batch, eff.CHUNK_SIZE),
        device=device,
        generator=g,
    )
    torch.manual_seed(BENCH_SEED)
    aera = eff.build_aera(device).eval()
    aera_result = _measure_inference(
        lambda z: aera(
            z,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
        ),
        tokens,
    )
    del aera
    torch.cuda.empty_cache()
    torch.manual_seed(BENCH_SEED)
    transformer = eff.build_transformer(device).eval()
    transformer_result = _measure_inference(transformer, tokens)
    del transformer
    torch.cuda.empty_cache()
    return {
        "batch": batch,
        "sequence_length": eff.CHUNK_SIZE,
        "aera_memory_disabled": aera_result,
        "transformer": transformer_result,
        "aera_vs_transformer_throughput_ratio": (
            aera_result["tokens_per_second"] / transformer_result["tokens_per_second"]
        ),
    }


def run_l4_benchmark() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("v23 sparse systems benchmark requires CUDA")
    if sparse_write_budget(DENSE_CANDIDATES) != SPARSE_CANDIDATES:
        raise RuntimeError("frozen v23 write budget is no longer 16/255")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(BENCH_SEED)
    torch.cuda.manual_seed_all(BENCH_SEED)

    dense = _measure_recurrence(device, candidates=DENSE_CANDIDATES)
    sparse = _measure_recurrence(device, candidates=SPARSE_CANDIDATES)
    recurrence_speedup = dense["ms"] / sparse["ms"]

    torch.manual_seed(BENCH_SEED)
    model = eff.build_aera(device).train()
    stage_path = _measure_stage_path(model, device)
    x, y = _random_language_batch(device)
    auxiliary = _measure_aux(model, x, step=0)
    straight = _measure_full_microbatch(model, x, y, step=0)
    hard_step = next(
        step for step in range(TOTAL_STEPS)
        if eff.route_mode_for_step(step) == "hard_sparse"
    )
    hard = _measure_full_microbatch(model, x, y, step=hard_step)
    del model
    torch.cuda.empty_cache()

    counts = _mode_counts()
    projection_training_seconds = 0.0
    for mode, steps in counts.items():
        micro_ms = hard["ms"] if mode == "hard_sparse" else straight["ms"]
        projection_training_seconds += steps * GRAD_ACCUM * micro_ms / 1000.0
    projected_total = projection_training_seconds + EVAL_MARGIN_SECONDS

    inference = {
        "batch8": _inference_comparison(device, batch=8),
        "batch64": _inference_comparison(device, batch=64),
    }

    sparse_count_ok = (
        int(stage_path["candidates"]) == DENSE_CANDIDATES
        and int(stage_path["selected_writes"]) == SPARSE_CANDIDATES
        and all(
            int(row["candidates"]) == DENSE_CANDIDATES
            and int(row["selected"]) == SPARSE_CANDIDATES
            for row in straight["executed_sparse_stages"] + hard["executed_sparse_stages"]
        )
    )
    speedup_ok = recurrence_speedup >= MIN_RECURRENCE_SPEEDUP
    projection_ok = projected_total <= PROJECTION_MAX_SECONDS
    overall_pass = bool(sparse_count_ok and speedup_ok and projection_ok)

    return {
        "scope": "aera_v23_sparse_write_systems_l4_issue_333",
        "scientific_training_performed": False,
        "optimizer_steps_performed": 0,
        "checkpoint_written": False,
        "gpu": torch.cuda.get_device_name(device),
        "production_shape": {
            "memory_dim": MEMORY_DIM,
            "dense_candidates": DENSE_CANDIDATES,
            "selected_writes": SPARSE_CANDIDATES,
            "recurrence_batch": RECURRENCE_BATCH,
            "micro_batch": MICRO_BATCH,
            "seq_len": SEQ_LEN,
            "chunk_size": eff.CHUNK_SIZE,
            "grad_accum": GRAD_ACCUM,
            "aux_events_per_microbatch": eff.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
            "aux_events_per_optimizer_step": eff.MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP,
        },
        "isolated_recurrence": {
            "dense_255": dense,
            "sparse_16": sparse,
            "speedup": recurrence_speedup,
            "minimum_required_speedup": MIN_RECURRENCE_SPEEDUP,
            "passes_speedup_gate": speedup_ok,
        },
        "selector_plus_stage_write": stage_path,
        "corrected_auxiliary": auxiliary,
        "full_microbatch": {
            "straight_through": straight,
            "hard_sparse": hard,
        },
        "route_mode_optimizer_step_counts": counts,
        "projection": {
            "training_seconds": projection_training_seconds,
            "evaluation_margin_seconds": EVAL_MARGIN_SECONDS,
            "train_plus_eval_seconds": projected_total,
            "target_max_seconds": PROJECTION_MAX_SECONDS,
            "passes_budget_projection": projection_ok,
        },
        "inference_descriptive": inference,
        "gates": {
            "physical_16_of_255_every_measured_sparse_path": sparse_count_ok,
            "isolated_recurrence_speedup": speedup_ok,
            "train_plus_eval_projection": projection_ok,
            "overall_pass": overall_pass,
        },
        "decision": (
            "v23_systems_pass_authorize_one_fresh_preregistered_real_language_dev_seed"
            if overall_pass
            else "v23_systems_fail_no_real_language_seed"
        ),
        "claims": {
            "real_language_memory_proven": False,
            "architecture_freeze_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_l4_benchmark()
    print(
        "AERA_V23_SPARSE_SYSTEMS_L4_RESULT_JSON=" + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
