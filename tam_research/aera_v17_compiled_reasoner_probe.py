from __future__ import annotations

"""Measurement-only probe for compiling AERA's hard sparse latent recurrence.

The candidate executes exactly the same selected GRU transitions as the reference
hard reasoner. It does not use dense masking and does not change selected depth.
The only purpose is to test whether graph capture can remove Python/dynamic dispatch
overhead before considering a custom fused CUDA kernel.
"""

from contextlib import nullcontext
import time
from typing import Any, Callable

import torch
import torch.nn as nn

from .aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from .aera_v17_kernel_probe import (
    D_MODEL,
    MAX_REASON_STEPS,
    PROBE_SEED,
    dense_masked_reasoner_reference,
)

SELECTED_SIZES: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
WARMUP_ITERS = 3
TIMED_ITERS = 30


class CompileFriendlyHardSparseReasoner(nn.Module):
    """Hard sparse recurrence with no statistics copies or data-dependent break.

    Inactive rows are physically absent from each GRUCell call because every step
    gathers only `chosen >= step` rows. Empty selections are allowed and still do
    not execute any useful recurrent assignments.
    """

    def __init__(self, d_model: int = D_MODEL, max_steps: int = MAX_REASON_STEPS):
        super().__init__()
        self.max_steps = max_steps
        self.cell = nn.GRUCell(d_model, d_model)

    def forward(self, summary: torch.Tensor, depth_logits: torch.Tensor) -> torch.Tensor:
        if depth_logits.shape != (summary.size(0), self.max_steps):
            raise ValueError("depth_logits shape mismatch")
        chosen = depth_logits.argmax(dim=-1) + 1
        current = summary
        for step in range(1, self.max_steps + 1):
            idx = torch.nonzero(chosen >= step, as_tuple=False).squeeze(-1)
            selected = current.index_select(0, idx)
            updated = self.cell(selected, selected).to(dtype=current.dtype)
            current = current.index_copy(0, idx, updated)
        return current


def copy_reference_weights(
    reference: DtypeSafeChunkLatentReasoner,
    candidate: CompileFriendlyHardSparseReasoner,
) -> None:
    candidate.cell.load_state_dict(reference.cell.state_dict(), strict=True)


def validate_probe_protocol() -> dict[str, Any]:
    if tuple(sorted(set(SELECTED_SIZES))) != SELECTED_SIZES:
        raise RuntimeError("selected sizes must be unique and ascending")
    return {
        "gpu_authorized": True,
        "gpu_authorization_scope": "one measurement-only compiled hard-sparse reasoner probe",
        "probe_seed": PROBE_SEED + 1,
        "selected_sizes": list(SELECTED_SIZES),
        "d_model": D_MODEL,
        "max_reason_steps": MAX_REASON_STEPS,
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "training_performed": False,
        "architecture_changed": False,
        "routing_changed": False,
        "hard_selected_depth_changed": False,
        "dense_masked_is_latency_upper_bound_only": True,
    }


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _benchmark_cuda(call: Callable[[], object], *, device: torch.device) -> float:
    for _ in range(WARMUP_ITERS):
        call()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        call()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)) / float(TIMED_ITERS)


def _depth_logits(batch: int, device: torch.device) -> torch.Tensor:
    chosen = (torch.arange(batch, device=device) % MAX_REASON_STEPS).long()
    logits = torch.full((batch, MAX_REASON_STEPS), -6.0, device=device)
    logits.scatter_(1, chosen[:, None], 6.0)
    return logits


def _configure_dynamo_dynamic_output_capture() -> None:
    """Configure Dynamo without creating a local `torch` binding in run_probe."""
    from torch import _dynamo

    _dynamo.config.capture_dynamic_output_shape_ops = True


@torch.no_grad()
def run_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("compiled sparse reasoner probe requires one CUDA GPU")
    device = torch.device("cuda")
    protocol = validate_probe_protocol()
    torch.manual_seed(PROBE_SEED + 1)

    reference = DtypeSafeChunkLatentReasoner(D_MODEL, MAX_REASON_STEPS).to(device).eval()
    candidate = CompileFriendlyHardSparseReasoner().to(device).eval()
    copy_reference_weights(reference, candidate)

    compile_error: str | None = None
    compile_setup_seconds: float | None = None
    compiled = None
    try:
        _configure_dynamo_dynamic_output_capture()
        compiled = torch.compile(
            candidate,
            mode="reduce-overhead",
            dynamic=True,
            fullgraph=False,
        )
        warm_summary = torch.randn(8, D_MODEL, device=device)
        warm_logits = _depth_logits(8, device)
        started = time.perf_counter()
        with _autocast(device):
            compiled(warm_summary, warm_logits)
        torch.cuda.synchronize(device)
        compile_setup_seconds = time.perf_counter() - started
    except Exception as exc:  # measurement outcome, not a reason to rerun
        compile_error = f"{type(exc).__name__}: {exc}"
        compiled = None

    rows: list[dict[str, Any]] = []
    for batch in SELECTED_SIZES:
        g = torch.Generator(device=device).manual_seed(PROBE_SEED + 10_000 + batch)
        summary = torch.randn(batch, D_MODEL, device=device, generator=g)
        logits = _depth_logits(batch, device)

        def reference_call() -> object:
            with _autocast(device):
                return reference(summary, logits, hard=True)

        def candidate_call() -> object:
            with _autocast(device):
                return candidate(summary, logits)

        def dense_call() -> object:
            with _autocast(device):
                return dense_masked_reasoner_reference(reference, summary, logits)

        with _autocast(device):
            ref_out = reference(summary, logits, hard=True)
            cand_out = candidate(summary, logits)
            dense_out = dense_masked_reasoner_reference(reference, summary, logits)
        candidate_delta = float((ref_out.float() - cand_out.float()).abs().max())
        dense_delta = float((ref_out.float() - dense_out.float()).abs().max())

        ref_ms = _benchmark_cuda(reference_call, device=device)
        cand_ms = _benchmark_cuda(candidate_call, device=device)
        dense_ms = _benchmark_cuda(dense_call, device=device)

        compiled_ms: float | None = None
        compiled_delta: float | None = None
        compiled_error_for_shape: str | None = None
        if compiled is not None:
            try:
                def compiled_call() -> object:
                    with _autocast(device):
                        return compiled(summary, logits)

                with _autocast(device):
                    comp_out = compiled(summary, logits)
                compiled_delta = float((ref_out.float() - comp_out.float()).abs().max())
                compiled_ms = _benchmark_cuda(compiled_call, device=device)
            except Exception as exc:
                compiled_error_for_shape = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "selected_batch": batch,
                "mean_hard_steps": float((logits.argmax(dim=-1) + 1).float().mean()),
                "reference_sparse_ms": ref_ms,
                "compile_friendly_eager_ms": cand_ms,
                "compiled_sparse_ms": compiled_ms,
                "dense_masked_upper_bound_ms": dense_ms,
                "eager_candidate_vs_reference_speed": ref_ms / cand_ms,
                "compiled_vs_reference_speed": (ref_ms / compiled_ms) if compiled_ms else None,
                "dense_vs_reference_speed": ref_ms / dense_ms,
                "candidate_max_abs_delta": candidate_delta,
                "compiled_max_abs_delta": compiled_delta,
                "dense_max_abs_delta": dense_delta,
                "compiled_shape_error": compiled_error_for_shape,
            }
        )

    return {
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(device),
        "compile_setup_seconds": compile_setup_seconds,
        "compile_error": compile_error,
        "rows": rows,
        "claims": {
            "measurement_only": True,
            "hard_sparse_semantics_required": True,
            "dense_masked_counts_as_sparse_evidence": False,
            "training_performed": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
