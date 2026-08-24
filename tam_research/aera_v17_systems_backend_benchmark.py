from __future__ import annotations

"""Checkpoint-only benchmark of the semantics-preserving AERA-v17 systems backend."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch

from . import aera_real_language_v12 as v12
from . import aera_real_language_v17 as v17
from .aera_real_language import SEQ_LEN
from .aera_v17_systems_backend import install_v17_systems_backend, systems_backend_protocol
from .data import TokenBin

CHECKPOINT_SEED = 8331
PROBE_SEED = 118_331
BATCH_SIZES: tuple[int, ...] = (8, 16, 32, 64)
WARMUP_ITERS = 3
TIMED_ITERS = 10
LOGIT_COMPARE_BATCH_SLICE = 1


def validate_protocol(data_dir: str, checkpoint_dir: str) -> dict[str, Any]:
    cpu = v17.cpu_preflight()
    data = v12.validate_production_data(data_dir)
    root = Path(checkpoint_dir)
    required = [root / "aera.pt", root / "transformer.pt", root / "result.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"seed8331 checkpoint artifacts missing: {missing}")
    if tuple(sorted(set(BATCH_SIZES))) != BATCH_SIZES:
        raise RuntimeError("batch sizes must be unique and ascending")
    return {
        **cpu,
        **systems_backend_protocol(),
        "gpu_authorized": True,
        "gpu_authorization_scope": "one checkpoint-only v17 systems-backend benchmark",
        "checkpoint_seed": CHECKPOINT_SEED,
        "probe_seed": PROBE_SEED,
        "batch_sizes": list(BATCH_SIZES),
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "logit_verification": "reference BF16/FP16 logits offloaded to CPU; optimized logits compared one batch row at a time on CPU",
        "logit_compare_batch_slice": LOGIT_COMPARE_BATCH_SLICE,
        "training_performed": False,
        "checkpoint_mutated": False,
        "data": data,
    }


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _benchmark(call: Callable[[], object], *, device: torch.device) -> dict[str, float]:
    for _ in range(WARMUP_ITERS):
        call()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        call()
    end.record()
    torch.cuda.synchronize(device)
    ms = float(start.elapsed_time(end)) / TIMED_ITERS
    return {"ms": ms, "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device))}


def _load_seed(payload_path: Path) -> dict[str, Any]:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if payload.get("seed") != CHECKPOINT_SEED:
        raise RuntimeError(f"checkpoint seed mismatch: {payload.get('seed')!r}")
    return payload


def _compare_logits_memory_bounded(
    reference_logits_cpu: torch.Tensor,
    optimized_logits: torch.Tensor,
    reference_argmax_cpu: torch.Tensor,
    *,
    batch_slice: int = LOGIT_COMPARE_BATCH_SLICE,
) -> tuple[float, float]:
    """Compare full logits without materializing multiple full FP32 tensors on GPU.

    The failed first benchmark kept reference and optimized full-vocabulary logits
    resident on GPU and converted both to FP32 at batch64, requesting a 6.14 GiB
    temporary tensor.  This verifier preserves the exact full-logit comparison but
    moves the already-produced reference logits to CPU in their existing dtype and
    transfers optimized logits in bounded batch slices before converting to FP32.
    Throughput timing is performed separately and is therefore unaffected.
    """
    if reference_logits_cpu.device.type != "cpu":
        raise ValueError("reference_logits_cpu must already be offloaded to CPU")
    if reference_logits_cpu.shape != optimized_logits.shape:
        raise ValueError("reference/optimized logit shape mismatch")
    if batch_slice < 1:
        raise ValueError("batch_slice must be positive")
    if reference_argmax_cpu.device.type != "cpu":
        raise ValueError("reference_argmax_cpu must be on CPU")

    max_delta = 0.0
    agree = 0
    total = int(reference_argmax_cpu.numel())
    for start in range(0, optimized_logits.size(0), batch_slice):
        end = min(start + batch_slice, optimized_logits.size(0))
        opt_slice_cpu = optimized_logits[start:end].detach().to("cpu")
        delta = (
            reference_logits_cpu[start:end].float()
            - opt_slice_cpu.float()
        ).abs().max()
        max_delta = max(max_delta, float(delta))
        opt_argmax_cpu = opt_slice_cpu.argmax(dim=-1)
        agree += int(
            (reference_argmax_cpu[start:end] == opt_argmax_cpu).sum().item()
        )
        del opt_slice_cpu, opt_argmax_cpu, delta
    return max_delta, float(agree / total)


@torch.no_grad()
def run_benchmark(*, data_dir: str, checkpoint_dir: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("systems-backend benchmark requires one CUDA GPU")
    device = torch.device("cuda")
    protocol = validate_protocol(data_dir, checkpoint_dir)
    root = Path(checkpoint_dir)
    aera_payload = _load_seed(root / "aera.pt")
    transformer_payload = _load_seed(root / "transformer.pt")

    torch.manual_seed(CHECKPOINT_SEED)
    reference = v17.build_aera(device).eval()
    torch.manual_seed(CHECKPOINT_SEED)
    optimized = v17.build_aera(device).eval()
    torch.manual_seed(CHECKPOINT_SEED)
    transformer = v17.build_transformer(device).eval()
    reference.load_state_dict(aera_payload["model"], strict=True)
    optimized.load_state_dict(aera_payload["model"], strict=True)
    transformer.load_state_dict(transformer_payload["model"], strict=True)
    install_v17_systems_backend(optimized)

    val = TokenBin(str(Path(data_dir) / "val.bin"))
    rows: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        generator = torch.Generator(device="cpu").manual_seed(PROBE_SEED + batch_size)
        x, _ = val.batch(batch_size, SEQ_LEN, generator, device)

        def ref_call() -> object:
            with _autocast(device):
                return reference(x, hard=True, route_mode="hard_sparse", update_memory=False)

        def opt_call() -> object:
            with _autocast(device):
                return optimized(x, hard=True, route_mode="hard_sparse", update_memory=False)

        def transformer_call() -> object:
            with _autocast(device):
                return transformer(x)

        # Verification is deliberately serialized.  Keep only one full-vocabulary
        # output on GPU at a time; reference logits are offloaded before optimized
        # inference.  This preserves the full comparison at batch64 without making
        # verification itself the GPU-memory bottleneck.
        with _autocast(device):
            ref_out = reference(x, hard=True, route_mode="hard_sparse", update_memory=False)
        ref_logits_cpu = ref_out["logits"].detach().to("cpu")
        ref_argmax_cpu = ref_logits_cpu.argmax(dim=-1)
        del ref_out
        torch.cuda.empty_cache()

        with _autocast(device):
            opt_out = optimized(x, hard=True, route_mode="hard_sparse", update_memory=False)
        max_logit_delta, argmax_agreement = _compare_logits_memory_bounded(
            ref_logits_cpu,
            opt_out["logits"],
            ref_argmax_cpu,
        )
        del opt_out, ref_logits_cpu, ref_argmax_cpu
        torch.cuda.empty_cache()

        ref_t = _benchmark(ref_call, device=device)
        opt_t = _benchmark(opt_call, device=device)
        tr_t = _benchmark(transformer_call, device=device)
        for timing in (ref_t, opt_t, tr_t):
            timing["tokens_per_second"] = batch_size * SEQ_LEN * 1000.0 / timing["ms"]

        expert_kernels = [getattr(stage.experts, "last_kernel", None) for stage in optimized.stages]
        rows.append({
            "batch_size": batch_size,
            "reference_aera": ref_t,
            "optimized_aera": opt_t,
            "transformer": tr_t,
            "optimized_vs_reference_speed": opt_t["tokens_per_second"] / ref_t["tokens_per_second"],
            "optimized_vs_transformer_speed": opt_t["tokens_per_second"] / tr_t["tokens_per_second"],
            "reference_vs_transformer_speed": ref_t["tokens_per_second"] / tr_t["tokens_per_second"],
            "max_logit_delta": max_logit_delta,
            "argmax_agreement": argmax_agreement,
            "optimized_expert_kernels": expert_kernels,
        })
        del x
        torch.cuda.empty_cache()

    return {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(device),
        "rows": rows,
        "claims": {
            "measurement_only": True,
            "training_performed": False,
            "checkpoint_mutated": False,
            "counts_toward_independent_replication": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
