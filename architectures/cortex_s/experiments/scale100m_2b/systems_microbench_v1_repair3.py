from __future__ import annotations

from contextlib import nullcontext
import gc
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from architectures.cortex_s.language_model import affine_scan, parameter_count
from architectures.cortex_s.systems_optimization import affine_scan_slice_candidate
from tam_research.data import TokenBin

from .protocol import (
    CORTEX_100M_CONFIG,
    DATA_DIR,
    EXPECTED_CORTEX_PARAMS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    RESERVED_FRESH_SEEDS,
    SEQ_LEN,
    TOKENS_PER_OPTIMIZER_STEP,
)
from .train import _compile_model, _make_optimizer, _one_optimizer_step, build_cortex_100m, seed_all
from .systems_microbench_v1_repair2 import (
    convert_to_stride_aligned_grouped,
    zero_gpu_layout_contract_probe,
)


CLASSIFICATION = "ENGINEERING_SYSTEMS_MICROBENCH_ONLY"
ENGINEERING_SEED = 2_026_090_903
CONSUMED_ENGINEERING_SEEDS = (910_001, 2_026_090_901, 2_026_090_902)
FORBIDDEN_SCIENTIFIC_SEEDS = (8_100, *RESERVED_FRESH_SEEDS)
TRIGGER_TITLE = "[modal-cortex-s-100m-systems-microbench-v1-repair3]"
RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair3"
PARENT_REPAIR2_SOURCE_SHA = "cb4f01233090e7e9fa3c4e3db3915fd5b94480e0"
PARENT_REPAIR2_RUN_ID = 34_329_354_099
PARENT_REPAIR2_JOB_ID = 102_394_032_871

COMPILE_TRIGGER_STEPS = 1
ADDITIONAL_WARMUP_STEPS = 3
MEASURED_STEPS = 20
MIN_PROMISING_SPEEDUP = 1.20
STOP_GROUPED_BELOW_SPEEDUP = 1.10
MAX_PEAK_VRAM_GIB = 70.0
MAX_SEMANTIC_LOSS_DELTA = 0.02
SCAN_WARMUP_ITERS = 3
SCAN_MEASURED_ITERS = 10


def validate_microbench_protocol() -> dict[str, Any]:
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("repair3 engineering seed collides with a consumed/forbidden seed")
    if MICRO_BATCH_SIZE != 64 or SEQ_LEN != 512 or GRAD_ACCUM_STEPS != 2:
        raise RuntimeError("repair3 drifted from production batch shape")
    if CORTEX_100M_CONFIG.top_k != 2 or CORTEX_100M_CONFIG.num_experts != 8:
        raise RuntimeError("repair3 router semantics drifted from top-2/8")
    if CORTEX_100M_CONFIG.expert_hidden != 338:
        raise RuntimeError("repair3 must preserve the production hidden width")
    if not (1.0 < STOP_GROUPED_BELOW_SPEEDUP < MIN_PROMISING_SPEEDUP):
        raise RuntimeError("speedup gates are not ordered")
    return {
        "classification": CLASSIFICATION,
        "engineering_seed": ENGINEERING_SEED,
        "consumed_engineering_seeds": list(CONSUMED_ENGINEERING_SEEDS),
        "trigger_title": TRIGGER_TITLE,
        "result_root": RESULT_ROOT,
        "parent_repair2_source_sha": PARENT_REPAIR2_SOURCE_SHA,
        "parent_repair2_run_id": PARENT_REPAIR2_RUN_ID,
        "parent_repair2_job_id": PARENT_REPAIR2_JOB_ID,
        "compile_trigger_steps": COMPILE_TRIGGER_STEPS,
        "additional_warmup_steps": ADDITIONAL_WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "primary_speedup_gate": MIN_PROMISING_SPEEDUP,
        "stop_below_speedup": STOP_GROUPED_BELOW_SPEEDUP,
        "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
        "max_semantic_loss_delta": MAX_SEMANTIC_LOSS_DELTA,
        "forbidden_seeds": sorted(forbidden),
        "full_training_authorized": False,
        "next_stage_authorized": False,
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def _semantic_probe(device: torch.device) -> dict[str, Any]:
    seed_all(ENGINEERING_SEED)
    legacy = build_cortex_100m().to(device).eval()
    seed_all(ENGINEERING_SEED)
    candidate = convert_to_stride_aligned_grouped(build_cortex_100m().to(device)).eval()
    if parameter_count(legacy) != EXPECTED_CORTEX_PARAMS or parameter_count(candidate) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("repair3 semantic probe parameter count mismatch")

    token_generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 7)
    tokens = torch.randint(0, CORTEX_100M_CONFIG.vocab_size, (2, 64), generator=token_generator).to(device)
    targets = torch.randint(0, CORTEX_100M_CONFIG.vocab_size, (2, 64), generator=token_generator).to(device)
    feature_generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 8)
    features = torch.randn(257, CORTEX_100M_CONFIG.d_model, generator=feature_generator).to(device)
    legacy_router = legacy.blocks[0].moe.router(features).topk(CORTEX_100M_CONFIG.top_k, dim=-1).indices
    candidate_router = candidate.blocks[0].moe.router(features).topk(CORTEX_100M_CONFIG.top_k, dim=-1).indices
    routing_exact = bool(torch.equal(legacy_router, candidate_router))

    with _autocast(device):
        legacy_logits = legacy(tokens)
        candidate_logits = candidate(tokens)
        legacy_loss = F.cross_entropy(legacy_logits.float().reshape(-1, legacy_logits.size(-1)), targets.reshape(-1))
        candidate_loss = F.cross_entropy(candidate_logits.float().reshape(-1, candidate_logits.size(-1)), targets.reshape(-1))
    loss_delta = abs(float(legacy_loss) - float(candidate_loss))
    max_logit_delta = float((legacy_logits.float() - candidate_logits.float()).abs().max())
    mean_logit_delta = float((legacy_logits.float() - candidate_logits.float()).abs().mean())

    del legacy, candidate, tokens, targets, features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "routing_exact": routing_exact,
        "legacy_loss": float(legacy_loss),
        "candidate_loss": float(candidate_loss),
        "absolute_loss_delta": loss_delta,
        "max_absolute_logit_delta": max_logit_delta,
        "mean_absolute_logit_delta": mean_logit_delta,
        "loss_delta_gate": loss_delta <= MAX_SEMANTIC_LOSS_DELTA,
    }


def _benchmark_full_model_variant(*, label: str, train_data: TokenBin, device: torch.device, grouped: bool) -> dict[str, Any]:
    """Production-shape benchmark with the repair2 Python scoping bug removed.

    `import torch._dynamo` inside this function would bind `torch` as a local name
    and make the earlier `torch.Generator` access fail with UnboundLocalError.
    Importing `_dynamo` directly binds only `dynamo`, so `torch` remains the
    module-level global throughout the function.
    """

    seed_all(ENGINEERING_SEED)
    model = build_cortex_100m().to(device)
    if grouped:
        model = convert_to_stride_aligned_grouped(model)
    if parameter_count(model) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"{label} parameter count drift")
    optimizer = _make_optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 10_000)
    runner: torch.nn.Module | None = None

    try:
        from torch import _dynamo as dynamo

        dynamo.reset()
        compile_start = time.perf_counter()
        runner = _compile_model(model)
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
        compile_seconds = time.perf_counter() - compile_start
    except Exception as exc:
        result = {
            "label": label,
            "status": "COMPILE_OR_FIRST_STEP_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "parameter_count": parameter_count(model),
        }
        del runner, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    try:
        for _ in range(ADDITIONAL_WARMUP_STEPS):
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
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(MEASURED_STEPS):
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
    except Exception as exc:
        result = {
            "label": label,
            "status": "WARMUP_OR_MEASURED_STEP_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "compile_seconds": compile_seconds,
            "parameter_count": parameter_count(model),
        }
        del runner, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    measured_seconds = max(time.perf_counter() - started, 1e-9)
    measured_tokens = MEASURED_STEPS * TOKENS_PER_OPTIMIZER_STEP
    tps = measured_tokens / measured_seconds
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    finite = math.isfinite(last_loss)
    result = {
        "label": label,
        "status": "PASS" if finite else "NONFINITE",
        "compile_seconds": compile_seconds,
        "measured_steps": MEASURED_STEPS,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_seconds,
        "training_tokens_per_second": tps,
        "peak_vram_gib": peak_vram,
        "last_loss": last_loss,
        "finite_loss": finite,
        "parameter_count": parameter_count(model),
    }
    del runner, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def zero_gpu_scoping_contract_probe() -> dict[str, Any]:
    """Fail closed on the exact Python-local-shadowing bug from consumed repair2."""

    local_names = set(_benchmark_full_model_variant.__code__.co_varnames)
    if "torch" in local_names:
        raise RuntimeError("repair3 benchmark still binds torch as a local variable")
    if "dynamo" not in local_names:
        raise RuntimeError("repair3 benchmark no longer exposes the direct dynamo binding")
    return {
        "status": "PASS",
        "torch_resolves_from_module_globals": True,
        "dynamo_is_local": True,
        "parent_failure": "UnboundLocalError: cannot access local variable 'torch' where it is not associated with a value",
    }


def zero_gpu_repair3_contract_probe() -> dict[str, Any]:
    layout = zero_gpu_layout_contract_probe()
    scoping = zero_gpu_scoping_contract_probe()
    if layout.get("status") != "PASS" or scoping.get("status") != "PASS":
        raise RuntimeError("repair3 zero-GPU contract probe failed")
    return {"status": "PASS", "layout_contract": layout, "scoping_contract": scoping}


def _scan_once(fn, a: torch.Tensor, b: torch.Tensor, initial: torch.Tensor):
    for tensor in (a, b, initial):
        if tensor.grad is not None:
            tensor.grad = None
    out = fn(a, b, initial)
    loss = out.float().square().mean()
    loss.backward()
    grads = (a.grad.detach().clone(), b.grad.detach().clone(), initial.grad.detach().clone())
    return out.detach(), grads


def _benchmark_scan(device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 30_000)
    a0 = torch.sigmoid(torch.randn(64, 512, 128, generator=gen)).to(device=device, dtype=torch.bfloat16)
    b0 = (0.1 * torch.randn(64, 512, 128, generator=gen)).to(device=device, dtype=torch.bfloat16)
    i0 = torch.randn(64, 128, generator=gen).to(device=device, dtype=torch.bfloat16)

    def fresh():
        return (
            a0.detach().clone().requires_grad_(True),
            b0.detach().clone().requires_grad_(True),
            i0.detach().clone().requires_grad_(True),
        )

    la, lb, li = fresh()
    ca, cb, ci = fresh()
    legacy_out, legacy_grads = _scan_once(affine_scan, la, lb, li)
    candidate_out, candidate_grads = _scan_once(affine_scan_slice_candidate, ca, cb, ci)
    output_max_delta = float((legacy_out.float() - candidate_out.float()).abs().max())
    grad_max_delta = max(
        float((left.float() - right.float()).abs().max())
        for left, right in zip(legacy_grads, candidate_grads)
    )
    del la, lb, li, ca, cb, ci, legacy_out, candidate_out, legacy_grads, candidate_grads

    timings: dict[str, float] = {}
    for name, fn in (("legacy", affine_scan), ("slice_candidate", affine_scan_slice_candidate)):
        for _ in range(SCAN_WARMUP_ITERS):
            a, b, initial = fresh()
            _scan_once(fn, a, b, initial)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(SCAN_MEASURED_ITERS):
            a, b, initial = fresh()
            _scan_once(fn, a, b, initial)
        torch.cuda.synchronize(device)
        timings[name] = (time.perf_counter() - started) / SCAN_MEASURED_ITERS

    return {
        "shape": [64, 512, 128],
        "dtype": "bfloat16",
        "forward_backward_seconds_per_iter": timings,
        "slice_speedup": timings["legacy"] / max(timings["slice_candidate"], 1e-9),
        "output_max_abs_delta": output_max_delta,
        "gradient_max_abs_delta": grad_max_delta,
    }


def run_h100_systems_microbenchmark(*, source_sha: str, data_dir: str = DATA_DIR) -> dict[str, Any]:
    protocol = validate_microbench_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("repair3 systems microbenchmark requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.get_device_capability(device) < (9, 0):
        raise RuntimeError("repair3 systems benchmark requires H100-class SM90+")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("installed PyTorch does not expose torch.nn.functional.grouped_mm")

    zero_gpu_contract = zero_gpu_repair3_contract_probe()
    train_data = TokenBin(f"{data_dir}/train.bin")
    cache_start = time.perf_counter()
    train_data._device_tokens(device)
    torch.cuda.synchronize(device)
    data_cache_seconds = time.perf_counter() - cache_start

    semantic = _semantic_probe(device)
    legacy = _benchmark_full_model_variant(label="legacy", train_data=train_data, device=device, grouped=False)
    grouped = _benchmark_full_model_variant(
        label="grouped_bf16_repair3",
        train_data=train_data,
        device=device,
        grouped=True,
    )
    scan = _benchmark_scan(device)

    gates = {
        "layout_contract": zero_gpu_contract["layout_contract"].get("status") == "PASS",
        "scoping_contract": zero_gpu_contract["scoping_contract"].get("status") == "PASS",
        "semantic_routing_exact": bool(semantic.get("routing_exact")),
        "semantic_loss_delta": bool(semantic.get("loss_delta_gate")),
        "legacy_pass": legacy.get("status") == "PASS",
        "grouped_pass": grouped.get("status") == "PASS",
        "parameter_count_exact": grouped.get("parameter_count") == EXPECTED_CORTEX_PARAMS,
        "peak_vram": float(grouped.get("peak_vram_gib", float("inf"))) <= MAX_PEAK_VRAM_GIB,
    }
    speedup: float | None = None
    if legacy.get("status") == "PASS" and grouped.get("status") == "PASS":
        speedup = float(grouped["training_tokens_per_second"]) / max(
            float(legacy["training_tokens_per_second"]), 1e-9
        )

    if not all(gates.values()) or speedup is None or speedup < STOP_GROUPED_BELOW_SPEEDUP:
        status = "STOP_GROUPED_PATH"
    elif speedup < MIN_PROMISING_SPEEDUP:
        status = "INSUFFICIENT_SYSTEMS_GAIN"
    else:
        status = "PROMISING_SYSTEMS_CANDIDATE"

    return {
        "status": status,
        "scientific_status": CLASSIFICATION,
        "repair_namespace": "microbench-v1-repair3",
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "data_cache_seconds": data_cache_seconds,
        "zero_gpu_contract": zero_gpu_contract,
        "semantic_probe": semantic,
        "legacy": legacy,
        "grouped_bf16_repair3": grouped,
        "grouped_full_model_speedup": speedup,
        "scan_diagnostic": scan,
        "gates": gates,
        "protocol": protocol,
        "full_training_authorized": False,
        "next_stage_authorized": False,
        "note": "Engineering systems evidence only. No repair3 result directly authorizes the 2B run.",
    }
