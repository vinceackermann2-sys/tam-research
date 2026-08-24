from __future__ import annotations

"""Measurement-only GPU systems probe for the frozen AERA-v17 seed8331 checkpoint.

No training, parameter mutation, routing-threshold change, or architecture change is
performed. The probe asks whether AERA's poor batch8 wall-clock speed comes mainly
from tiny selected sub-batches / launch utilization or from a structural fixed cost.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch

from . import aera_real_language_v12 as v12
from . import aera_real_language_v17 as v17
from .aera_real_language import SEQ_LEN
from .aera_systems_accounting import routing_execution_accounting
from .data import TokenBin

CHECKPOINT_SEED = 8331
PROBE_SEED = 108_331
BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
WARMUP_ITERS = 3
TIMED_ITERS = 10


def _require_checkpoint_seed(payload: dict[str, Any], *, expected: int = CHECKPOINT_SEED) -> None:
    if payload.get("seed") != expected:
        raise RuntimeError(
            f"systems probe checkpoint seed mismatch: {payload.get('seed')!r} != {expected}"
        )


def validate_probe_protocol(data_dir: str, checkpoint_dir: str) -> dict[str, Any]:
    cpu = v17.cpu_preflight()
    data = v12.validate_production_data(data_dir)
    root = Path(checkpoint_dir)
    required = [root / "aera.pt", root / "transformer.pt", root / "result.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen seed8331 checkpoint artifacts missing: {missing}")
    if tuple(sorted(set(BATCH_SIZES))) != BATCH_SIZES or BATCH_SIZES[0] != 1:
        raise RuntimeError("batch-size sweep must be unique, ascending, and start at 1")
    return {
        **cpu,
        "gpu_authorized": True,
        "gpu_authorization_scope": "one measurement-only seed8331 checkpoint systems probe",
        "checkpoint_seed": CHECKPOINT_SEED,
        "probe_seed": PROBE_SEED,
        "batch_sizes": list(BATCH_SIZES),
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "training_performed": False,
        "checkpoint_mutated": False,
        "architecture_changed": False,
        "routing_changed": False,
        "data": data,
    }


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _benchmark_cuda(
    call: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = WARMUP_ITERS,
    iterations: int = TIMED_ITERS,
) -> dict[str, float]:
    if device.type != "cuda":
        raise RuntimeError("systems timing probe requires CUDA")
    if warmup < 1 or iterations < 1:
        raise ValueError("benchmark iterations must be positive")
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize(device)
    ms = float(start.elapsed_time(end)) / float(iterations)
    return {
        "ms": ms,
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
    }


@torch.no_grad()
def _representative_stage0_components(
    model,
    x: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Time one fully executed representative stage at the current batch size."""
    chunk = x[:, : model.cfg.chunk_size]
    pos = torch.arange(chunk.size(1), device=device)
    stage = model.stages[0]
    with _autocast(device):
        events = model.token_emb(chunk) + model.local_pos(pos)[None]
        state = stage.empty_state(events)
        h = stage.norm(events)
        start_control = stage.controller(h[:, 0], state.stream)
        memory_read = stage.memory.read(h[:, :1], state.memory).squeeze(1)
        carried = stage.state_to_chunk(state.stream)
        # v3+ uses mandatory carried stream plus controller-gated fast memory.
        h_context = h + (
            carried + start_control["memory_read"] * memory_read
        )[:, None, :]
        attn_once = stage.attn(h_context)
        h_after_attn = h_context + attn_once
        expert_once = stage.experts(
            h_after_attn,
            start_control["expert_logits"],
            start_control["expert_count_logits"],
            hard=True,
        )
        h_after_experts = h_after_attn + expert_once
        end_summary = h_after_experts[:, -1]
        end_control = stage.controller(end_summary, state.stream)
        reasoned_once = stage.reasoner(
            end_summary,
            end_control["depth_logits"],
            hard=True,
        )
        stream_input = stage.stream_input_norm(end_summary + reasoned_once)

    def full_stage() -> object:
        with _autocast(device):
            return stage.forward_chunk(events, state, hard=True, update_memory=False)

    def attention() -> object:
        with _autocast(device):
            return stage.attn(h_context)

    def experts() -> object:
        with _autocast(device):
            return stage.experts(
                h_after_attn,
                start_control["expert_logits"],
                start_control["expert_count_logits"],
                hard=True,
            )

    def reasoner() -> object:
        with _autocast(device):
            return stage.reasoner(
                end_summary,
                end_control["depth_logits"],
                hard=True,
            )

    def stream_gru() -> object:
        with _autocast(device):
            return stage.stream_cell(stream_input, state.stream)

    return {
        "full_stage": _benchmark_cuda(full_stage, device=device),
        "attention": _benchmark_cuda(attention, device=device),
        "experts": _benchmark_cuda(experts, device=device),
        "reasoner": _benchmark_cuda(reasoner, device=device),
        "stream_gru": _benchmark_cuda(stream_gru, device=device),
    }


@torch.no_grad()
def run_probe(*, data_dir: str, checkpoint_dir: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("systems probe requires one CUDA GPU")
    device = torch.device("cuda")
    protocol = validate_probe_protocol(data_dir, checkpoint_dir)
    root = Path(checkpoint_dir)
    aera_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    transformer_payload = torch.load(
        root / "transformer.pt", map_location="cpu", weights_only=False
    )
    _require_checkpoint_seed(aera_payload)
    _require_checkpoint_seed(transformer_payload)

    torch.manual_seed(CHECKPOINT_SEED)
    aera = v17.build_aera(device).eval()
    torch.manual_seed(CHECKPOINT_SEED)
    transformer = v17.build_transformer(device).eval()
    aera.load_state_dict(aera_payload["model"], strict=True)
    transformer.load_state_dict(transformer_payload["model"], strict=True)

    val = TokenBin(str(Path(data_dir) / "val.bin"))
    sweeps: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        generator = torch.Generator(device="cpu").manual_seed(PROBE_SEED + batch_size)
        x, _ = val.batch(batch_size, SEQ_LEN, generator, device)

        with _autocast(device):
            routed = aera(
                x,
                hard=True,
                route_mode="hard_sparse",
                update_memory=False,
            )
        routing = routing_execution_accounting(routed)

        def aera_call() -> object:
            with _autocast(device):
                return aera(
                    x,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=False,
                )

        def transformer_call() -> object:
            with _autocast(device):
                return transformer(x)

        aera_timing = _benchmark_cuda(aera_call, device=device)
        transformer_timing = _benchmark_cuda(transformer_call, device=device)
        aera_timing["tokens_per_second"] = (
            batch_size * SEQ_LEN * 1000.0 / aera_timing["ms"]
        )
        transformer_timing["tokens_per_second"] = (
            batch_size * SEQ_LEN * 1000.0 / transformer_timing["ms"]
        )
        components = _representative_stage0_components(aera, x, device=device)
        sweeps.append(
            {
                "batch_size": batch_size,
                "aera": aera_timing,
                "transformer": transformer_timing,
                "aera_vs_transformer_speed": (
                    aera_timing["tokens_per_second"]
                    / transformer_timing["tokens_per_second"]
                ),
                "routing": routing,
                "representative_stage0_components": components,
            }
        )
        del routed, x
        torch.cuda.empty_cache()

    return {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(device),
        "sweeps": sweeps,
        "claims": {
            "measurement_only": True,
            "training_performed": False,
            "checkpoint_mutated": False,
            "counts_toward_independent_replication": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
