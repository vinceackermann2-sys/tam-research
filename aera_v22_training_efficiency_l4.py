from __future__ import annotations

"""Systems-only L4 benchmark for issue #327.

No corpus is read, no optimizer step is taken, and no checkpoint is written.  The
benchmark compares the exact eager AERA-v22 dual-delta recurrence with
`torch.compile` applied to that same function, measures the corrected 256-event
memory auxiliary, and times representative random-token training microbatches.
"""

import json
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from tam_research import aera_hardware_core_v22 as core
from tam_research import aera_real_language_v22_efficiency as eff
from tam_research.aera_real_language import GRAD_ACCUM, MICRO_BATCH, SEQ_LEN, TOTAL_STEPS

BENCH_SEED = 8411
MEMORY_DIM = 50
CANDIDATES = 255
RECURRENCE_BATCH = 8
TIMED_ITERS = 2
EVAL_MARGIN_SECONDS = 150.0
PROJECTION_MAX_SECONDS = 1500.0


def _cuda_time(call: Callable[[], None], *, warmups: int = 1, iters: int = TIMED_ITERS) -> dict[str, float]:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    ms = float(start.elapsed_time(end)) / iters
    return {
        "ms": ms,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
    }


def _recurrence_inputs(device: torch.device, *, requires_grad: bool) -> list[torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED)
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
            CANDIDATES,
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
            CANDIDATES,
            MEMORY_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=g,
        )
    )
    strengths = torch.sigmoid(
        torch.randn(
            RECURRENCE_BATCH,
            CANDIDATES,
            1,
            device=device,
            dtype=torch.bfloat16,
            generator=g,
        )
    )
    values = [matrix, inverse, keys, targets, strengths]
    if requires_grad:
        values = [v.detach().clone().requires_grad_(True) for v in values]
    return values


def _recurrence_train_call(fn, device: torch.device) -> None:
    inputs = _recurrence_inputs(device, requires_grad=True)
    m, p = fn(*inputs)
    loss = m.float().square().mean() + 0.1 * p.float().square().mean()
    loss.backward()


def _recurrence_equivalence(eager, compiled, device: torch.device) -> dict[str, float]:
    inputs = _recurrence_inputs(device, requires_grad=False)
    with torch.no_grad():
        em, ep = eager(*[x.clone() for x in inputs])
        cm, cp = compiled(*[x.clone() for x in inputs])
    return {
        "matrix_max_abs_error": float((em.float() - cm.float()).abs().max()),
        "inverse_covariance_max_abs_error": float((ep.float() - cp.float()).abs().max()),
    }


def _random_language_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(BENCH_SEED + 1)
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


def _full_microbatch_call(model, x: torch.Tensor, y: torch.Tensor, *, step: int) -> None:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        total, _, _, _ = eff.aera_matched_loss(model, x, y, step=step)
        scaled = total / GRAD_ACCUM
    scaled.backward()


def _aux_call(model, x: torch.Tensor, *, step: int) -> None:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        terms = eff.memory_auxiliary_terms(model, x, step=step)
        loss = (
            terms["memory_address_contrastive_loss"]
            + terms["memory_payload_token_loss"]
        ) / GRAD_ACCUM
    loss.backward()


def _mode_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in range(TOTAL_STEPS):
        mode = eff.route_mode_for_step(step)
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def run_l4_benchmark() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("v22 efficiency benchmark requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(BENCH_SEED)
    torch.cuda.manual_seed_all(BENCH_SEED)

    eager = core.interference_corrected_dual_delta_update

    eager_recurrence = _cuda_time(
        lambda: _recurrence_train_call(eager, device),
        warmups=1,
    )

    compiled = eff.make_compiled_dual_delta_update(mode="reduce-overhead")
    compile_started = time.perf_counter()
    # First production-shape call performs compilation and is intentionally not
    # counted as steady-state latency.
    _recurrence_train_call(compiled, device)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started
    equivalence = _recurrence_equivalence(eager, compiled, device)
    compiled_recurrence = _cuda_time(
        lambda: _recurrence_train_call(compiled, device),
        warmups=1,
    )

    # Corrected auxiliary cost on the exact ~25M real-language model geometry.
    torch.manual_seed(BENCH_SEED)
    model = eff.build_aera(device).train()
    x, y = _random_language_batch(device)
    aux_timing = _cuda_time(
        lambda: _aux_call(model, x, step=0),
        warmups=1,
    )

    # Full eager v22 random-token training microbatch at straight-through warmup.
    core.interference_corrected_dual_delta_update = eager
    torch.manual_seed(BENCH_SEED)
    eager_model = eff.build_aera(device).train()
    eager_st = _cuda_time(
        lambda: _full_microbatch_call(eager_model, x, y, step=0),
        warmups=1,
        iters=1,
    )

    # Exact same weights and random input with only the recurrence runtime compiled.
    core.interference_corrected_dual_delta_update = compiled
    torch.manual_seed(BENCH_SEED)
    compiled_model = eff.build_aera(device).train()
    compiled_st = _cuda_time(
        lambda: _full_microbatch_call(compiled_model, x, y, step=0),
        warmups=1,
        iters=1,
    )
    hard_step = next(
        step for step in range(TOTAL_STEPS)
        if eff.route_mode_for_step(step) == "hard_sparse"
    )
    compiled_hard = _cuda_time(
        lambda: _full_microbatch_call(compiled_model, x, y, step=hard_step),
        warmups=1,
        iters=1,
    )
    core.interference_corrected_dual_delta_update = eager

    counts = _mode_counts()
    projected_training_seconds = 0.0
    for mode, steps in counts.items():
        if mode == "hard_sparse":
            micro_ms = compiled_hard["ms"]
        else:
            micro_ms = compiled_st["ms"]
        projected_training_seconds += steps * GRAD_ACCUM * micro_ms / 1000.0
    projected_total_seconds = projected_training_seconds + EVAL_MARGIN_SECONDS

    result = {
        "scope": "aera_v22_exact_semantics_training_efficiency_l4",
        "scientific_training_performed": False,
        "optimizer_steps_performed": 0,
        "checkpoint_written": False,
        "gpu": torch.cuda.get_device_name(device),
        "production_shape": {
            "memory_dim": MEMORY_DIM,
            "candidates": CANDIDATES,
            "recurrence_batch": RECURRENCE_BATCH,
            "micro_batch": MICRO_BATCH,
            "seq_len": SEQ_LEN,
            "grad_accum": GRAD_ACCUM,
            "aux_events_per_microbatch": eff.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH,
            "aux_events_per_optimizer_step": (
                eff.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH * GRAD_ACCUM
            ),
        },
        "compiled_recurrence": {
            "compile_seconds": compile_seconds,
            "eager_train_ms": eager_recurrence["ms"],
            "compiled_train_ms": compiled_recurrence["ms"],
            "speedup": eager_recurrence["ms"] / compiled_recurrence["ms"],
            "equivalence": equivalence,
            "peak_allocated_gb": compiled_recurrence["peak_allocated_gb"],
        },
        "corrected_auxiliary": aux_timing,
        "full_microbatch": {
            "eager_straight_through": eager_st,
            "compiled_straight_through": compiled_st,
            "compiled_hard_sparse": compiled_hard,
            "compiled_vs_eager_st_speedup": eager_st["ms"] / compiled_st["ms"],
        },
        "route_mode_optimizer_step_counts": counts,
        "projection": {
            "training_seconds": projected_training_seconds,
            "evaluation_margin_seconds": EVAL_MARGIN_SECONDS,
            "train_plus_eval_seconds": projected_total_seconds,
            "target_max_seconds": PROJECTION_MAX_SECONDS,
            "passes_budget_projection": projected_total_seconds <= PROJECTION_MAX_SECONDS,
        },
        "decision": (
            "exact_semantics_v22_fast_enough_for_fresh_preregistered_dev_seed"
            if projected_total_seconds <= PROJECTION_MAX_SECONDS
            else "exact_semantics_v22_still_too_slow_preregister_sparse_write_v23"
        ),
        "claims": {
            "seed8391_rerun_authorized": False,
            "real_language_memory_proven": False,
            "architecture_freeze_authorized": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
    return result


def main() -> None:
    result = run_l4_benchmark()
    print(
        "AERA_V22_EFFICIENCY_L4_RESULT_JSON=" + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
