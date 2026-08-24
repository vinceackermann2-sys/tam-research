from __future__ import annotations

"""Measurement-only kernel probe for AERA-v17 production shapes.

This module changes no model architecture, routing policy, checkpoint, or training
state. It measures two known systems hot spots from the frozen seed8331 probe:
expert execution and adaptive latent reasoning.
"""

from contextlib import nullcontext
from typing import Any, Callable

import torch

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from .aera_hardware_core_v6 import BMMHardSparseExpertBank
from .aera_hardware_core_v7 import NativeGroupedMMSparseExpertBank

PROBE_SEED = 109_331
SELECTED_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
WARMUP_ITERS = 3
TIMED_ITERS = 20
D_MODEL = 200
CHUNK_SIZE = 256
N_EXPERTS = 8
EXPERT_MULT = 4
MAX_REASON_STEPS = 4


def production_shape_config() -> HardwareAERAConfig:
    return HardwareAERAConfig(
        vocab_size=50_257,
        d_model=D_MODEL,
        n_stages=4,
        n_heads=8,
        chunk_size=CHUNK_SIZE,
        n_experts=N_EXPERTS,
        max_active_experts=2,
        expert_mult=EXPERT_MULT,
        memory_dim=50,
        max_reason_steps=MAX_REASON_STEPS,
        block_size=4,
    )


def validate_probe_protocol() -> dict[str, Any]:
    cfg = production_shape_config()
    cfg.validate()
    if tuple(sorted(set(SELECTED_SIZES))) != SELECTED_SIZES:
        raise RuntimeError("selected-size sweep must be unique and ascending")
    return {
        "gpu_authorized": True,
        "gpu_authorization_scope": "one measurement-only AERA-v17 kernel opportunity probe",
        "probe_seed": PROBE_SEED,
        "selected_sizes": list(SELECTED_SIZES),
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "shape": {
            "chunk_size": CHUNK_SIZE,
            "d_model": D_MODEL,
            "n_experts": N_EXPERTS,
            "expert_hidden": D_MODEL * EXPERT_MULT,
            "max_reason_steps": MAX_REASON_STEPS,
        },
        "training_performed": False,
        "checkpoint_mutated": False,
        "architecture_changed": False,
        "routing_changed": False,
        "dense_reasoner_is_speed_upper_bound_only": True,
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
        raise RuntimeError("kernel timing probe requires CUDA")
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize(device)
    return {"ms": float(start.elapsed_time(end)) / float(iterations)}


def _expert_controls(batch: int, device: torch.device, *, mixed_top2: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(batch, device=device)
    top1 = ids % N_EXPERTS
    top2 = (top1 + 1) % N_EXPERTS
    logits = torch.full((batch, N_EXPERTS), -8.0, device=device)
    logits.scatter_(1, top1[:, None], 8.0)
    logits.scatter_(1, top2[:, None], 7.0)
    count_logits = torch.empty((batch, 2), device=device)
    count_logits[:, 0] = 3.0
    count_logits[:, 1] = -3.0
    if mixed_top2:
        use_second = (ids % 4) == 0  # representative mean active experts ~=1.25
        count_logits[use_second, 0] = -3.0
        count_logits[use_second, 1] = 3.0
    return logits, count_logits


def dense_masked_reasoner_reference(
    reasoner: DtypeSafeChunkLatentReasoner,
    summary: torch.Tensor,
    depth_logits: torch.Tensor,
) -> torch.Tensor:
    """Semantically matched but compute-dense latency upper bound.

    All rows execute every GRU step and inactive rows discard the result. This is
    intentionally NOT valid sparse execution evidence; it estimates how much of the
    current sparse-reasoner latency is launch/gather/scatter overhead.
    """
    chosen = depth_logits.argmax(dim=-1) + 1
    current = summary
    for step in range(1, reasoner.max_steps + 1):
        updated = reasoner.cell(current, current).to(dtype=current.dtype)
        current = torch.where((chosen >= step)[:, None], updated, current)
    return current


def _depth_logits(batch: int, device: torch.device) -> torch.Tensor:
    chosen = (torch.arange(batch, device=device) % MAX_REASON_STEPS).long()
    logits = torch.full((batch, MAX_REASON_STEPS), -6.0, device=device)
    logits.scatter_(1, chosen[:, None], 6.0)
    return logits


@torch.no_grad()
def run_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("kernel opportunity probe requires one CUDA GPU")
    device = torch.device("cuda")
    protocol = validate_probe_protocol()
    torch.manual_seed(PROBE_SEED)

    cfg = production_shape_config()
    bmm = BMMHardSparseExpertBank(cfg).to(device).eval()
    grouped = NativeGroupedMMSparseExpertBank(cfg).to(device).eval()
    grouped.load_state_dict(bmm.state_dict(), strict=True)
    reasoner = DtypeSafeChunkLatentReasoner(D_MODEL, MAX_REASON_STEPS).to(device).eval()

    expert_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []

    for batch in SELECTED_SIZES:
        g = torch.Generator(device=device).manual_seed(PROBE_SEED + batch)
        x = torch.randn(batch, CHUNK_SIZE, D_MODEL, device=device, generator=g)

        for mixed_top2 in (False, True):
            expert_logits, count_logits = _expert_controls(batch, device, mixed_top2=mixed_top2)

            def bmm_call() -> object:
                with _autocast(device):
                    return bmm(x, expert_logits, count_logits, hard=True)

            def grouped_call() -> object:
                with _autocast(device):
                    return grouped(x, expert_logits, count_logits, hard=True)

            with _autocast(device):
                y_bmm = bmm(x, expert_logits, count_logits, hard=True)
                y_grouped = grouped(x, expert_logits, count_logits, hard=True)
            max_abs = float((y_bmm.float() - y_grouped.float()).abs().max())
            bmm_t = _benchmark_cuda(bmm_call, device=device)
            grouped_t = _benchmark_cuda(grouped_call, device=device)
            expert_rows.append(
                {
                    "selected_batch": batch,
                    "routing_pattern": "mixed_top2_25pct" if mixed_top2 else "top1",
                    "bmm_ms": bmm_t["ms"],
                    "grouped_mm_ms": grouped_t["ms"],
                    "grouped_vs_bmm_speed": bmm_t["ms"] / grouped_t["ms"],
                    "grouped_kernel": grouped.last_kernel,
                    "max_abs_output_delta": max_abs,
                }
            )

        summary = torch.randn(batch, D_MODEL, device=device, generator=g)
        depth_logits = _depth_logits(batch, device)

        def sparse_reasoner() -> object:
            with _autocast(device):
                return reasoner(summary, depth_logits, hard=True)

        def dense_reasoner() -> object:
            with _autocast(device):
                return dense_masked_reasoner_reference(reasoner, summary, depth_logits)

        with _autocast(device):
            sparse_out = reasoner(summary, depth_logits, hard=True)
            dense_out = dense_masked_reasoner_reference(reasoner, summary, depth_logits)
        reason_delta = float((sparse_out.float() - dense_out.float()).abs().max())
        sparse_t = _benchmark_cuda(sparse_reasoner, device=device)
        dense_t = _benchmark_cuda(dense_reasoner, device=device)
        chosen = depth_logits.argmax(dim=-1) + 1
        reason_rows.append(
            {
                "selected_batch": batch,
                "mean_hard_steps": float(chosen.float().mean()),
                "sparse_ms": sparse_t["ms"],
                "dense_masked_upper_bound_ms": dense_t["ms"],
                "dense_vs_sparse_speed": sparse_t["ms"] / dense_t["ms"],
                "max_abs_output_delta": reason_delta,
                "dense_path_counts_as_sparse_evidence": False,
            }
        )

    return {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(device),
        "native_grouped_mm_available": NativeGroupedMMSparseExpertBank.native_grouped_mm_available(
            torch.zeros(1, device=device)
        ),
        "expert_sweep": expert_rows,
        "reasoner_sweep": reason_rows,
        "claims": {
            "measurement_only": True,
            "training_performed": False,
            "checkpoint_mutated": False,
            "counts_toward_independent_replication": False,
            "dense_reasoner_counts_as_sparse_evidence": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
