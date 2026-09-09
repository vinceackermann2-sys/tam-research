from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable

import torch

from architectures.cortex_s.affine_scan_triton_candidate import (
    affine_scan_triton_candidate,
    candidate_status,
    triton_available,
)
from architectures.cortex_s.language_model import affine_scan


CLASSIFICATION = "ENGINEERING_GPU_CORRECTNESS_AND_MICROBENCH_ONLY"
ENGINEERING_SEED = 2_026_090_907
CONSUMED_ENGINEERING_SEEDS = (
    910_001,
    2_026_090_901,
    2_026_090_902,
    2_026_090_903,
    2_026_090_904,
    2_026_090_905,
    2_026_090_906,
)
FORBIDDEN_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)
SOURCE_PARENT = "15a2cc2cdddfa9f22a632a2c72ad4bf95d94ed75"
TRIGGER_TITLE = "[modal-cortex-s-affine-scan-triton-gpu-microbench-v1]"
RESULT_ROOT = "/vol/cortex-s-v0/affine-scan-triton-gpu-microbench-v1"
PRODUCTION_SHAPE = (64, 512, 128)
FP32_PROBE_SHAPE = (4, 64, 16)
WARMUP_BATCHES = 3
MEASURE_BATCHES = 9
ITERS_PER_BATCH = 5
BF16_ATOL = 0.05
BF16_RTOL = 0.05
FP32_ATOL = 5e-5
FP32_RTOL = 5e-4
STOP_SPEEDUP = 1.20
PROMISING_SPEEDUP = 1.50
MAX_PEAK_GIB = 70.0


def validate_protocol() -> dict[str, Any]:
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("fresh engineering seed collides with consumed/scientific seed")
    if PRODUCTION_SHAPE != (64, 512, 128):
        raise RuntimeError("production scan shape drift")
    if not 1.0 < STOP_SPEEDUP < PROMISING_SPEEDUP:
        raise RuntimeError("speedup thresholds are not ordered")
    status = candidate_status()
    if status.get("production_wired") is not False:
        raise RuntimeError("candidate unexpectedly production-wired")
    if status.get("gpu_benchmark_authorized") is not False:
        raise RuntimeError("candidate module itself must remain non-authorizing")
    if status.get("parameter_count_delta") != 0:
        raise RuntimeError("candidate changes parameter count")
    return {
        "classification": CLASSIFICATION,
        "engineering_seed": ENGINEERING_SEED,
        "consumed_engineering_seeds": list(CONSUMED_ENGINEERING_SEEDS),
        "forbidden_scientific_seeds": list(FORBIDDEN_SCIENTIFIC_SEEDS),
        "source_parent": SOURCE_PARENT,
        "trigger_title": TRIGGER_TITLE,
        "result_root": RESULT_ROOT,
        "production_shape": list(PRODUCTION_SHAPE),
        "fp32_probe_shape": list(FP32_PROBE_SHAPE),
        "warmup_batches": WARMUP_BATCHES,
        "measure_batches": MEASURE_BATCHES,
        "iters_per_batch": ITERS_PER_BATCH,
        "bf16_tolerance": {"atol": BF16_ATOL, "rtol": BF16_RTOL},
        "fp32_tolerance": {"atol": FP32_ATOL, "rtol": FP32_RTOL},
        "stop_speedup": STOP_SPEEDUP,
        "promising_speedup": PROMISING_SPEEDUP,
        "max_peak_gib": MAX_PEAK_GIB,
        "gpu_dispatch_authorized": False,
        "production_integration_authorized": False,
        "full_training_authorized": False,
    }


def _cpu_random(shape: tuple[int, ...], *, seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return scale * torch.randn(*shape, generator=generator, dtype=torch.float32)


def _inputs(
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, state = shape
    # Keep coefficients in (0,1) and updates modest so long recurrent products stay
    # representative and numerically stable while still exercising the real algebra.
    a = torch.sigmoid(_cpu_random(shape, seed=ENGINEERING_SEED + seed_offset)).to(device=device, dtype=dtype)
    b = _cpu_random(shape, seed=ENGINEERING_SEED + seed_offset + 1, scale=0.1).to(device=device, dtype=dtype)
    initial = _cpu_random((batch, state), seed=ENGINEERING_SEED + seed_offset + 2, scale=0.25).to(
        device=device, dtype=dtype
    )
    upstream = _cpu_random(shape, seed=ENGINEERING_SEED + seed_offset + 3, scale=0.1).to(
        device=device, dtype=dtype
    )
    return a, b, initial, upstream


def _forward_backward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    initial: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    aa = a.detach().requires_grad_(True)
    bb = b.detach().requires_grad_(True)
    ii = initial.detach().requires_grad_(True)
    out = fn(aa, bb, ii)
    # FP32 reduction mirrors a stable training-loss reduction while gradients still
    # flow through the candidate in its actual input dtype.
    loss = (out.float() * upstream.float()).sum()
    grads = torch.autograd.grad(loss, (aa, bb, ii), retain_graph=False, create_graph=False)
    return out.detach(), tuple(grad.detach() for grad in grads)


def _comparison(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    delta = (left.float() - right.float()).abs()
    return {
        "allclose": bool(torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol)),
        "finite_left": bool(torch.isfinite(left).all()),
        "finite_right": bool(torch.isfinite(right).all()),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
    }


def _correctness_probe(
    *,
    shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    seed_offset: int,
) -> dict[str, Any]:
    a, b, initial, upstream = _inputs(shape, device=device, dtype=dtype, seed_offset=seed_offset)
    production_out, production_grads = _forward_backward(affine_scan, a, b, initial, upstream)
    candidate_out, candidate_grads = _forward_backward(
        affine_scan_triton_candidate, a, b, initial, upstream
    )
    forward = _comparison(production_out, candidate_out, atol=atol, rtol=rtol)
    grad_a = _comparison(production_grads[0], candidate_grads[0], atol=atol, rtol=rtol)
    grad_b = _comparison(production_grads[1], candidate_grads[1], atol=atol, rtol=rtol)
    grad_initial = _comparison(production_grads[2], candidate_grads[2], atol=atol, rtol=rtol)
    passed = all(
        check["allclose"] and check["finite_left"] and check["finite_right"]
        for check in (forward, grad_a, grad_b, grad_initial)
    )
    return {
        "shape": list(shape),
        "dtype": str(dtype).replace("torch.", ""),
        "atol": atol,
        "rtol": rtol,
        "forward": forward,
        "grad_a": grad_a,
        "grad_b": grad_b,
        "grad_initial": grad_initial,
        "pass": passed,
    }


def _chunk_carry_probe(device: torch.device) -> dict[str, Any]:
    shape = (8, 128, 32)
    a, b, initial, _ = _inputs(shape, device=device, dtype=torch.bfloat16, seed_offset=400)
    whole = affine_scan_triton_candidate(a, b, initial)
    first = affine_scan_triton_candidate(a[:, :47], b[:, :47], initial)
    second = affine_scan_triton_candidate(a[:, 47:], b[:, 47:], first[:, -1])
    joined = torch.cat((first, second), dim=1)
    comparison = _comparison(whole, joined, atol=BF16_ATOL, rtol=BF16_RTOL)
    return {**comparison, "pass": comparison["allclose"] and comparison["finite_left"] and comparison["finite_right"]}


def _timed_batches(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    initial: torch.Tensor,
    upstream: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    def once() -> None:
        _forward_backward(fn, a, b, initial, upstream)

    for _ in range(WARMUP_BATCHES):
        for _ in range(ITERS_PER_BATCH):
            once()
        torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    per_iter: list[float] = []
    for _ in range(MEASURE_BATCHES):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(ITERS_PER_BATCH):
            once()
        torch.cuda.synchronize(device)
        per_iter.append((time.perf_counter() - started) / ITERS_PER_BATCH)
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    return {
        "batch_seconds_per_iter": per_iter,
        "median_seconds_per_iter": statistics.median(per_iter),
        "mean_seconds_per_iter": statistics.fmean(per_iter),
        "min_seconds_per_iter": min(per_iter),
        "max_seconds_per_iter": max(per_iter),
        "peak_allocated_gib": peak_gib,
    }


def run_h100_scan_microbenchmark(*, source_sha: str) -> dict[str, Any]:
    protocol = validate_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("H100 scan microbenchmark requires CUDA")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability < (9, 0):
        raise RuntimeError(f"expected H100-class SM90+, got capability={capability}")
    if not triton_available():
        raise RuntimeError("Triton is unavailable on the GPU worker")

    # Correctness comes first. Timing is invalid if any semantic/gradient gate fails.
    bf16 = _correctness_probe(
        shape=PRODUCTION_SHAPE,
        device=device,
        dtype=torch.bfloat16,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        seed_offset=100,
    )
    fp32 = _correctness_probe(
        shape=FP32_PROBE_SHAPE,
        device=device,
        dtype=torch.float32,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
        seed_offset=200,
    )
    chunk = _chunk_carry_probe(device)
    correctness_pass = bool(bf16["pass"] and fp32["pass"] and chunk["pass"])
    if not correctness_pass:
        return {
            "status": "GPU_CORRECTNESS_FAIL",
            "scientific_status": CLASSIFICATION,
            "source_sha": source_sha,
            "engineering_seed": ENGINEERING_SEED,
            "device_name": torch.cuda.get_device_name(device),
            "cuda_capability": list(capability),
            "torch_version": torch.__version__,
            "bf16_correctness": bf16,
            "fp32_correctness": fp32,
            "chunk_carry": chunk,
            "timing_valid": False,
            "production_integration_authorized": False,
            "full_training_authorized": False,
        }

    a, b, initial, upstream = _inputs(
        PRODUCTION_SHAPE,
        device=device,
        dtype=torch.bfloat16,
        seed_offset=300,
    )
    production = _timed_batches(
        affine_scan,
        a=a,
        b=b,
        initial=initial,
        upstream=upstream,
        device=device,
    )
    candidate = _timed_batches(
        affine_scan_triton_candidate,
        a=a,
        b=b,
        initial=initial,
        upstream=upstream,
        device=device,
    )
    speedup = production["median_seconds_per_iter"] / max(
        candidate["median_seconds_per_iter"], 1e-12
    )
    if not math.isfinite(speedup):
        raise RuntimeError("non-finite isolated scan speedup")
    if candidate["peak_allocated_gib"] > MAX_PEAK_GIB:
        status = "STOP_FUSED_SCAN_PATH"
    elif speedup < STOP_SPEEDUP:
        status = "STOP_FUSED_SCAN_PATH"
    elif speedup < PROMISING_SPEEDUP:
        status = "WEAK_SCAN_GAIN_NEEDS_FULL_GRAPH_PROFILING"
    else:
        status = "PROMISING_SCAN_SYSTEMS_CANDIDATE"

    return {
        "status": status,
        "scientific_status": CLASSIFICATION,
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(capability),
        "torch_version": torch.__version__,
        "bf16_correctness": bf16,
        "fp32_correctness": fp32,
        "chunk_carry": chunk,
        "timing_valid": True,
        "production_affine_scan": production,
        "triton_candidate": candidate,
        "isolated_forward_backward_speedup": speedup,
        "protocol": protocol,
        "production_integration_authorized": False,
        "full_training_authorized": False,
        "note": "Isolated engineering evidence only; no full-graph or scientific authority.",
    }
