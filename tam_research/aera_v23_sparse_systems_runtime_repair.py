from __future__ import annotations

"""Implementation-only runtime repair for the #335 systems harness.

Issue #335 failed before producing any systems measurement because its standalone
stage probe constructed BF16 activations and called a FP32 LayerNorm outside the
CUDA autocast context used by the actual real-language training path.  This file
patches only that benchmark probe: the model remains FP32, probe activations are
FP32, and stage execution occurs under BF16 autocast exactly like the full
microbatch benchmark.  AERA-v23 architecture, M/P equations, sparse budget,
losses, routing, thresholds and all other benchmark measurements are unchanged.
"""

from types import ModuleType
from typing import Any

import torch

from .aera_hardware_core_v23 import BudgetedSparseDualDeltaFastMemoryStage


def _stage_probe_forward(
    stage: BudgetedSparseDualDeltaFastMemoryStage,
    events: torch.Tensor,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    """Run the sparse stage in the same mixed-precision context as training."""
    if events.dtype != torch.float32:
        raise RuntimeError(
            f"stage benchmark probe must enter autocast as FP32, got {events.dtype}"
        )
    with torch.autocast(
        device_type=events.device.type,
        dtype=torch.bfloat16,
        enabled=events.device.type in {"cuda", "cpu"},
    ):
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
    return out, state, loss


def patch_stage_measurement(benchmark: ModuleType) -> None:
    """Replace only #335's broken standalone stage measurement wrapper."""

    def repaired_measure_stage_path(model, device: torch.device) -> dict[str, Any]:
        stage = model.stages[0]
        if not isinstance(stage, BudgetedSparseDualDeltaFastMemoryStage):
            raise RuntimeError("v23 benchmark expected sparse stage")
        stage.memory.set_differentiable_pretraining(True)
        generator = torch.Generator(device=device).manual_seed(benchmark.BENCH_SEED + 101)
        # Actual training keeps model parameters/inputs in FP32 and uses BF16
        # autocast around the forward.  The failed #335 probe incorrectly created
        # BF16 inputs before entering any autocast context.
        base_events = torch.randn(
            benchmark.RECURRENCE_BATCH,
            benchmark.eff.CHUNK_SIZE,
            model.cfg.d_model,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

        def one(*, timed: bool) -> float:
            events = base_events.detach().clone().requires_grad_(True)
            stage.zero_grad(set_to_none=True)
            if timed:
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            _, state, loss = _stage_probe_forward(stage, events)
            loss.backward()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite v23 stage sparse path")
            if events.grad is None or not bool(torch.isfinite(events.grad).all()):
                raise RuntimeError("nonfinite/missing v23 stage probe input gradient")
            if stage.last_candidate_count != benchmark.DENSE_CANDIDATES:
                raise RuntimeError("v23 stage path did not expose 255 candidates")
            if stage.last_selected_count != benchmark.SPARSE_CANDIDATES:
                raise RuntimeError("v23 stage path did not execute 16 writes")
            return benchmark._event_ms(start, end) if timed else 0.0

        one(timed=False)
        torch.cuda.reset_peak_memory_stats()
        ms = one(timed=True)
        result = {
            "ms": ms,
            "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
            "candidates": stage.last_candidate_count,
            "selected_writes": stage.last_selected_count,
            "selected_fraction": stage.last_selected_count / stage.last_candidate_count,
            "precision_context": "fp32_inputs_fp32_model_cuda_bf16_autocast",
            "implementation_repair": "post_issue_335_stage_probe_dtype_only",
        }
        stage.memory.set_differentiable_pretraining(False)
        stage.zero_grad(set_to_none=True)
        return result

    benchmark._measure_stage_path = repaired_measure_stage_path
