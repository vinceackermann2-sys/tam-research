"""CPU-only model-cost diagnostics; NEVER a scientific quality/compute match.

The profiler counts only operators PyTorch knows how to count and reports a
PARTIAL forward FLOP estimate. It omits parts of custom recurrent, gating,
routing and control-plane execution; CPU elapsed time is not GPU efficiency.
This is a shape-only developer instrument and NEVER loads a heldout world.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Mapping

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile


@dataclass(frozen=True)
class DiagnosticCPUCost:
    architecture: str
    tokens: int
    total_parameters: int
    parameter_bytes: int
    partial_forward_flops: int
    profiled_operator_keys: int
    keys_without_flop_estimates: int
    elapsed_cpu_ns: int
    flops_complete: bool
    equal_active_compute_verified: bool


def profile_public_shape(
    architecture: str, model: nn.Module, *,
    tokens: int = 8,
) -> DiagnosticCPUCost:
    """One forward call with SYNTHETIC zero-valued public feature shapes.

    No reward labels, no learning step, no hidden-world information or seed.
    Inputs are identical across models. Numbers are diagnostics, not
    performance scores or evidence of active-FLOP matching.
    """
    if type(architecture) is not str or not architecture:
        raise ValueError("architecture label required")
    if not isinstance(model, nn.Module):
        raise TypeError("expected PyTorch module")
    if type(tokens) is not int or not 1 <= tokens <= 16:
        raise ValueError("invalid bounded development sequence shape")
    obs = torch.zeros(1, tokens, 3, dtype=torch.float32)
    acts = torch.zeros(1, tokens, 4, dtype=torch.float32)
    modes = torch.zeros(1, tokens, dtype=torch.long)
    modes[0, -1] = 1
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            start = perf_counter_ns()
            with profile(activities=[ProfilerActivity.CPU],
                         with_flops=True, record_shapes=True) as recorder:
                output = model(obs, acts, modes)
            elapsed = perf_counter_ns() - start
        if not isinstance(output, torch.Tensor) or output.shape != (1,) \
                or not torch.isfinite(output).all():
            raise ValueError("nonfinite or wrong-shape factual reward forecast")
        stats = recorder.key_averages()
        partial_flops = sum(int(max(0, x.flops or 0)) for x in stats)
        absent = sum(bool(x.self_cpu_time_total) and not x.flops for x in stats)
        return DiagnosticCPUCost(
            architecture=architecture, tokens=tokens,
            total_parameters=sum(p.numel() for p in model.parameters()),
            parameter_bytes=sum(p.numel() * p.element_size()
                                for p in model.parameters()),
            partial_forward_flops=partial_flops,
            profiled_operator_keys=len(stats),
            keys_without_flop_estimates=absent,
            elapsed_cpu_ns=elapsed,
            flops_complete=False,  # Custom ops and profiler gaps.
            equal_active_compute_verified=False,
        )
    finally:
        model.train(was_training)


def summarize_nominal_parity(
    reports: Mapping[str, DiagnosticCPUCost],
) -> dict[str, object]:
    """Fail closed: identical tensors/params never certify equal active work."""
    if set(reports) != {"hybrid", "transformer", "gru"}:
        raise ValueError("requires SI hybrid, Transformer and GRU cost reports")
    if len({r.tokens for r in reports.values()}) != 1:
        raise ValueError("different context lengths cannot be compared")
    if any(r.flops_complete or r.equal_active_compute_verified
           for r in reports.values()):
        raise ValueError("cannot certify active FLOPs from partial CPU profiles")
    a = reports["hybrid"].total_parameters
    b = reports["transformer"].total_parameters
    if b <= 0:
        raise ValueError("invalid Transformer parameter count")
    return {
        "same_synthetic_input_length": True,
        "hybrid_parameters": a,
        "transformer_parameters": b,
        "compact_gru_parameters": reports["gru"].total_parameters,
        "hybrid_vs_transformer_relative_parameter_difference": abs(a-b)/b,
        "hybrid_vs_transformer_below_0_1pct_parameter_difference": abs(a-b)/b <= 0.001,
        "partial_forward_flops": {
            k: v.partial_forward_flops for k, v in reports.items()
        },
        "unattributed_operator_key_counts": {
            k: v.keys_without_flop_estimates for k, v in reports.items()
        },
        "active_flops_matched": False,
        "matched_scientific_training_compute": False,
        "valid_scientific_quality_result": False,
    }
