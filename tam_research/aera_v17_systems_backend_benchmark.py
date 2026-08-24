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

        with _autocast(device):
            ref_out = reference(x, hard=True, route_mode="hard_sparse", update_memory=False)
            opt_out = optimized(x, hard=True, route_mode="hard_sparse", update_memory=False)
        max_logit_delta = float((ref_out["logits"].float() - opt_out["logits"].float()).abs().max())
        argmax_agreement = float((ref_out["logits"].argmax(-1) == opt_out["logits"].argmax(-1)).float().mean())

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
        del ref_out, opt_out, x
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
