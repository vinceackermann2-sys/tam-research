from __future__ import annotations

"""Single systems-only L4 benchmark preregistered by issue #362.

Random inputs only.  No corpus reader, optimizer step, checkpoint, or scientific
training seed exists here.  The benchmark measures exact AERA-v25 FICEM at the
inherited real-language geometry before any real-language experiment is allowed.
"""

from contextlib import contextmanager
import json
import types
from typing import Any, Callable, Iterator

import torch

from tam_research import aera_v25_systems as sys25
from tam_research.aera_hardware_core_v23 import (
    select_budgeted_event_pairs,
    sparse_write_budget,
)
from tam_research.aera_hardware_core_v25 import (
    FactorizedIdentityContextEpisodicMemoryStage,
    causal_identity_context,
)
from tam_research.aera_real_language import GRAD_ACCUM, MICRO_BATCH, SEQ_LEN, TOTAL_STEPS

BENCH_SEED = 12531
DENSE_CANDIDATES = 255
SELECTED_WRITES = 16
MEMORY_DIM = 50
RECURRENCE_BATCH = 8
EVAL_MARGIN_SECONDS = 150.0
PROJECTION_MAX_SECONDS = 1500.0
BATCH8_WRITE_OVERHEAD_MAX_MS = 25.0
BATCH64_WRITE_OVERHEAD_MAX_MS = 40.0


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _assert_finite_grad(name: str, value: torch.Tensor | None) -> float:
    if value is None or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"missing/nonfinite gradient: {name}")
    total = float(value.abs().sum())
    if total <= 0.0:
        raise RuntimeError(f"zero gradient: {name}")
    return total


def _measure_isolated_ficem(model, device: torch.device) -> dict[str, Any]:
    stage = model.stages[0]
    if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
        raise RuntimeError("v25 isolated benchmark expected FICEM stage0")
    stage.memory.set_differentiable_pretraining(True)
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 1)
    base = torch.randn(
        RECURRENCE_BATCH,
        sys25.CHUNK_SIZE,
        model.cfg.d_model,
        device=device,
        dtype=torch.float32,
        generator=g,
    )

    def one(*, timed: bool) -> tuple[float, dict[str, float]]:
        stage.zero_grad(set_to_none=True)
        events = base.detach().clone().requires_grad_(True)
        if timed:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            identity, context, contextual = causal_identity_context(events)
            address = contextual[:, :-1]
            payload = contextual[:, 1:]
            pair_features = torch.cat((address, payload), dim=-1)
            pair_logits = stage.pair_write_gate(pair_features)
            pair_gate = torch.sigmoid(pair_logits)
            selected = select_budgeted_event_pairs(
                address,
                payload,
                pair_gate,
                pair_logits,
                differentiable_selector=True,
            )
            if selected.hard_count != SELECTED_WRITES:
                raise RuntimeError("isolated v25 selector did not choose 16/255")
            gather = selected.indices.unsqueeze(-1).expand(-1, -1, identity.size(-1))
            selected_identity = identity[:, :-1].gather(1, gather)
            selected_context = context[:, :-1].gather(1, gather)
            state0 = stage.memory.empty_state(
                RECURRENCE_BATCH,
                device,
                events.dtype,
            )
            state1 = stage.memory.update_block(
                selected_identity,
                selected_context,
                selected.payload,
                selected.strength,
                state0,
            )
            recall = stage.memory.read(identity, context, state1)
            loss = (
                recall.float().square().mean()
                + 0.01 * state1.values.float().square().mean()
                + 0.001 * state1.keys.float().square().mean()
            )
        loss.backward()
        if not bool(
            torch.isfinite(state1.keys).all()
            and torch.isfinite(state1.values).all()
            and torch.isfinite(state1.strengths).all()
            and torch.isfinite(loss)
        ):
            raise RuntimeError("nonfinite isolated v25 FICEM state/loss")
        grads = {
            "identity_proj": _assert_finite_grad(
                "identity_proj", stage.memory.identity_proj.weight.grad
            ),
            "context_proj": _assert_finite_grad(
                "context_proj", stage.memory.context_proj.weight.grad
            ),
            "v": _assert_finite_grad("v", stage.memory.v.weight.grad),
            "out": _assert_finite_grad("out", stage.memory.out.weight.grad),
            "pair_write_gate": _assert_finite_grad(
                "pair_write_gate", stage.pair_write_gate.weight.grad
            ),
        }
        ms = _event_ms(start, end) if timed else 0.0
        return ms, grads

    one(timed=False)
    torch.cuda.reset_peak_memory_stats()
    ms, grads = one(timed=True)
    stage.memory.set_differentiable_pretraining(False)
    stage.zero_grad(set_to_none=True)
    return {
        "ms_fwd_bwd_including_post_update_read": ms,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
        "candidates": DENSE_CANDIDATES,
        "selected_writes": SELECTED_WRITES,
        "selected_fraction": SELECTED_WRITES / DENSE_CANDIDATES,
        "vectorized_update_calls": 1,
        "gradient_l1": grads,
    }


def _reset_stats(model) -> None:
    for stage in model.stages:
        if isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
            stage.last_candidate_count = 0
            stage.last_selected_count = 0
            stage.last_selected_indices = None
            stage.last_vectorized_update_calls = 0


def _write_stats(model) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index, stage in enumerate(model.stages):
        if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
            continue
        if stage.last_selected_count:
            rows.append(
                {
                    "stage": float(index),
                    "candidates": float(stage.last_candidate_count),
                    "selected": float(stage.last_selected_count),
                    "vectorized_updates": float(stage.last_vectorized_update_calls),
                }
            )
    return rows


def _assert_write_rows(rows: list[dict[str, float]], *, label: str) -> None:
    if not rows:
        raise RuntimeError(f"{label} executed no measurable memory-writing stage")
    for row in rows:
        if int(row["candidates"]) != DENSE_CANDIDATES:
            raise RuntimeError(f"{label} candidate mismatch: {rows}")
        if int(row["selected"]) != SELECTED_WRITES:
            raise RuntimeError(f"{label} selected-write mismatch: {rows}")
        if int(row["vectorized_updates"]) != 1:
            raise RuntimeError(f"{label} vectorized-update mismatch: {rows}")


@contextmanager
def _zero_memory_reads(model) -> Iterator[None]:
    originals: list[tuple[Any, Any]] = []

    def zero_read(self, identity_source, context_source, state):
        return torch.zeros(
            identity_source.size(0),
            identity_source.size(1),
            self.out.out_features,
            device=identity_source.device,
            dtype=identity_source.dtype,
        )

    try:
        for stage in model.stages:
            if isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
                originals.append((stage.memory, stage.memory.read))
                stage.memory.read = types.MethodType(zero_read, stage.memory)
        yield
    finally:
        for memory, original in originals:
            memory.read = original


def _measure_inference_call(
    model,
    tokens: torch.Tensor,
    *,
    update_memory: bool,
    bypass_reads: bool,
) -> dict[str, Any]:
    def run_once(*, timed: bool) -> tuple[float, list[dict[str, float]]]:
        _reset_stats(model)
        ctx = _zero_memory_reads(model) if bypass_reads else contextmanager(lambda: (yield))()
        with ctx:
            if timed:
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(
                    tokens,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=update_memory,
                    return_block_logits=False,
                )
            ms = _event_ms(start, end) if timed else 0.0
        return ms, _write_stats(model)

    run_once(timed=False)
    torch.cuda.reset_peak_memory_stats()
    ms, rows = run_once(timed=True)
    if update_memory:
        _assert_write_rows(rows, label="inference memory write")
    return {
        "ms": ms,
        "tokens_per_second": tokens.numel() / (ms / 1000.0),
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
        "executed_memory_stages": rows,
    }


def _memory_overhead_decomposition(device: torch.device, *, batch: int) -> dict[str, Any]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 100 + batch)
    tokens = torch.randint(
        0,
        50_257,
        (batch, SEQ_LEN),
        device=device,
        generator=g,
    )
    torch.manual_seed(BENCH_SEED)
    model = sys25.build_aera(device).eval()
    model.set_memory_pretraining_mode(False)
    baseline = _measure_inference_call(
        model,
        tokens,
        update_memory=False,
        bypass_reads=True,
    )
    writes_only = _measure_inference_call(
        model,
        tokens,
        update_memory=True,
        bypass_reads=True,
    )
    full = _measure_inference_call(
        model,
        tokens,
        update_memory=True,
        bypass_reads=False,
    )
    del model
    torch.cuda.empty_cache()
    return {
        "batch": batch,
        "sequence_length": SEQ_LEN,
        "memory_disabled_read_bypass": baseline,
        "writes_enabled_reads_bypassed": writes_only,
        "full_memory": full,
        "write_overhead_ms": writes_only["ms"] - baseline["ms"],
        "read_overhead_ms": full["ms"] - writes_only["ms"],
    }


def _random_language_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 200)
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


def _measure_training_microbatch(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    step: int,
) -> dict[str, Any]:
    def one(*, timed: bool) -> tuple[float, list[dict[str, float]], str, str, float]:
        model.zero_grad(set_to_none=True)
        _reset_stats(model)
        if timed:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, terms, mode, phase = sys25.systems_matched_loss(model, x, y, step=step)
            scaled = total / GRAD_ACCUM
        scaled.backward()
        rows = _write_stats(model)
        _assert_write_rows(rows, label=f"training microbatch step={step}")
        if not bool(torch.isfinite(scaled)):
            raise RuntimeError("nonfinite v25 training-shape loss")
        stage = model.stages[0]
        _assert_finite_grad("training identity_proj", stage.memory.identity_proj.weight.grad)
        _assert_finite_grad("training context_proj", stage.memory.context_proj.weight.grad)
        _assert_finite_grad("training v", stage.memory.v.weight.grad)
        _assert_finite_grad("training out", stage.memory.out.weight.grad)
        _assert_finite_grad("training pair_write_gate", stage.pair_write_gate.weight.grad)
        sampled = float(terms["sampled_payload_events"].detach())
        ms = _event_ms(start, end) if timed else 0.0
        return ms, rows, mode, phase, sampled

    one(timed=False)
    torch.cuda.reset_peak_memory_stats()
    ms, rows, mode, phase, sampled = one(timed=True)
    return {
        "ms": ms,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
        "route_mode": mode,
        "phase": phase,
        "sampled_payload_events": sampled,
        "executed_memory_stages": rows,
    }


def _mode_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in range(TOTAL_STEPS):
        mode = sys25.route_mode_for_step(step)
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _measure_transformer(device: torch.device, *, batch: int) -> dict[str, float]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 300 + batch)
    tokens = torch.randint(0, 50_257, (batch, SEQ_LEN), device=device, generator=g)
    torch.manual_seed(BENCH_SEED)
    model = sys25.build_transformer(device).eval()

    def call(*, timed: bool) -> float:
        if timed:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(tokens)
        return _event_ms(start, end) if timed else 0.0

    call(timed=False)
    torch.cuda.reset_peak_memory_stats()
    ms = call(timed=True)
    result = {
        "ms": ms,
        "tokens_per_second": tokens.numel() / (ms / 1000.0),
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
    }
    del model
    torch.cuda.empty_cache()
    return result


def run_l4_benchmark() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("AERA-v25 FICEM systems benchmark requires CUDA")
    if sparse_write_budget(DENSE_CANDIDATES) != SELECTED_WRITES:
        raise RuntimeError("frozen v25 production write budget is no longer 16/255")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(BENCH_SEED)
    torch.cuda.manual_seed_all(BENCH_SEED)

    preflight = sys25.cpu_preflight()
    torch.manual_seed(BENCH_SEED)
    isolated_model = sys25.build_aera(device).train()
    isolated = _measure_isolated_ficem(isolated_model, device)
    del isolated_model
    torch.cuda.empty_cache()

    batch8 = _memory_overhead_decomposition(device, batch=8)
    batch64 = _memory_overhead_decomposition(device, batch=64)

    torch.manual_seed(BENCH_SEED)
    training_model = sys25.build_aera(device).train()
    x, y = _random_language_batch(device)
    straight = _measure_training_microbatch(training_model, x, y, step=0)
    hard_step = next(
        step for step in range(TOTAL_STEPS)
        if sys25.route_mode_for_step(step) == "hard_sparse"
    )
    hard = _measure_training_microbatch(training_model, x, y, step=hard_step)
    del training_model
    torch.cuda.empty_cache()

    counts = _mode_counts()
    projected_training_seconds = 0.0
    for mode, steps in counts.items():
        micro_ms = hard["ms"] if mode == "hard_sparse" else straight["ms"]
        projected_training_seconds += steps * GRAD_ACCUM * micro_ms / 1000.0
    projected_total = projected_training_seconds + EVAL_MARGIN_SECONDS

    transformer = {
        "batch8": _measure_transformer(device, batch=8),
        "batch64": _measure_transformer(device, batch=64),
    }

    sparse_exact = bool(
        isolated["candidates"] == DENSE_CANDIDATES
        and isolated["selected_writes"] == SELECTED_WRITES
        and isolated["vectorized_update_calls"] == 1
        and all(
            int(row["candidates"]) == DENSE_CANDIDATES
            and int(row["selected"]) == SELECTED_WRITES
            and int(row["vectorized_updates"]) == 1
            for row in (
                batch8["writes_enabled_reads_bypassed"]["executed_memory_stages"]
                + batch8["full_memory"]["executed_memory_stages"]
                + batch64["writes_enabled_reads_bypassed"]["executed_memory_stages"]
                + batch64["full_memory"]["executed_memory_stages"]
                + straight["executed_memory_stages"]
                + hard["executed_memory_stages"]
            )
        )
    )
    batch8_overhead_ok = batch8["write_overhead_ms"] <= BATCH8_WRITE_OVERHEAD_MAX_MS
    batch64_overhead_ok = batch64["write_overhead_ms"] <= BATCH64_WRITE_OVERHEAD_MAX_MS
    projection_ok = projected_total <= PROJECTION_MAX_SECONDS
    no_training = bool(
        preflight["scientific_training_performed"] is False
        and preflight["optimizer_steps_performed"] == 0
        and preflight["checkpoint_written"] is False
        and preflight["corpus_reader_used"] is False
    )
    overall_pass = bool(
        sparse_exact
        and batch8_overhead_ok
        and batch64_overhead_ok
        and projection_ok
        and no_training
    )

    return {
        "scope": "aera_v25_ficem_production_systems_l4_issue_362",
        "scientific_training_performed": False,
        "optimizer_steps_performed": 0,
        "checkpoint_written": False,
        "corpus_reader_used": False,
        "gpu": torch.cuda.get_device_name(device),
        "cpu_preflight": preflight,
        "production_shape": {
            "d_model": 200,
            "n_stages": 4,
            "chunk_size": sys25.CHUNK_SIZE,
            "seq_len": SEQ_LEN,
            "memory_dim": MEMORY_DIM,
            "candidates": DENSE_CANDIDATES,
            "selected_writes": SELECTED_WRITES,
            "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM,
            "payload_events_per_microbatch": sys25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
            "payload_events_per_optimizer_step": sys25.MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP,
        },
        "isolated_ficem": isolated,
        "memory_overhead": {
            "v23_reference_ms": {
                "batch8_write_overhead": 38.44,
                "batch64_write_overhead": 57.48,
            },
            "batch8": batch8,
            "batch64": batch64,
            "thresholds_ms": {
                "batch8_write_overhead_max": BATCH8_WRITE_OVERHEAD_MAX_MS,
                "batch64_write_overhead_max": BATCH64_WRITE_OVERHEAD_MAX_MS,
            },
        },
        "full_training_microbatch": {
            "straight_through": straight,
            "hard_sparse": hard,
        },
        "route_mode_optimizer_step_counts": counts,
        "projection": {
            "training_seconds": projected_training_seconds,
            "evaluation_margin_seconds": EVAL_MARGIN_SECONDS,
            "train_plus_eval_seconds": projected_total,
            "target_max_seconds": PROJECTION_MAX_SECONDS,
            "passes_budget_projection": projection_ok,
        },
        "matched_transformer_descriptive": transformer,
        "gates": {
            "physical_16_of_255_and_one_vectorized_update": sparse_exact,
            "batch8_write_overhead": batch8_overhead_ok,
            "batch64_write_overhead": batch64_overhead_ok,
            "train_plus_eval_projection": projection_ok,
            "no_scientific_training_optimizer_checkpoint_or_corpus": no_training,
            "overall_pass": overall_pass,
        },
        "decision": (
            "v25_systems_pass_authorize_one_fresh_preregistered_small_real_language_dev_seed"
            if overall_pass
            else "v25_systems_fail_no_real_language_seed"
        ),
        "claims": {
            "real_language_memory_proven": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }


def main() -> None:
    result = run_l4_benchmark()
    print(
        "AERA_V25_FICEM_SYSTEMS_L4_RESULT_JSON=" + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
