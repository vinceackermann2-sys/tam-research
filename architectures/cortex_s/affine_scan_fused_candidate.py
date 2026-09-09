from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class AffineScanKernelContract:
    recurrence: str = "s_t = a_t * s_(t-1) + b_t"
    forward_parallel_lanes: str = "batch * state"
    sequence_loop: str = "one fused device program loops left-to-right over sequence"
    backward_recurrence: str = "lambda_t = grad_t + a_(t+1) * lambda_(t+1)"
    parameter_count_delta: int = 0
    scientific_semantics_changed: bool = False


def affine_scan_sequential_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Simple recurrence reference used only to validate a future fused kernel.

    This is intentionally not proposed as the production implementation: its Python
    sequence loop is unsuitable for the 100M H100 training path. It is the semantic
    oracle for a one-kernel implementation that would parallelize over batch*state
    lanes and perform the 512 recurrent steps inside each device program.
    """

    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must have matching [batch, sequence, state] shapes")
    batch, length, state_size = a.shape
    if initial is None:
        state = torch.zeros(batch, state_size, dtype=a.dtype, device=a.device)
    else:
        if initial.shape != (batch, state_size):
            raise ValueError("initial state has wrong shape")
        state = initial.to(device=a.device, dtype=a.dtype)
    states = []
    for index in range(length):
        state = a[:, index] * state + b[:, index]
        states.append(state)
    if not states:
        return a.new_empty((batch, 0, state_size))
    return torch.stack(states, dim=1)


def affine_scan_analytic_backward_reference(
    a: torch.Tensor,
    states: torch.Tensor,
    grad_states: torch.Tensor,
    initial: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit reverse recurrence for the exact affine scan gradients.

    For s_t = a_t*s_(t-1)+b_t and total upstream gradient g_t on each returned
    state, define lambda_t = dL/ds_t after accounting for all future recurrence
    uses. Then lambda_t = g_t + a_(t+1)*lambda_(t+1), db_t=lambda_t,
    da_t=lambda_t*s_(t-1), and d_initial=a_0*lambda_0.

    This CPU/reference implementation exists so a future fused CUDA/Triton backward
    must prove equality before any paid benchmark or production integration.
    """

    if a.shape != states.shape or grad_states.shape != states.shape or a.ndim != 3:
        raise ValueError("a, states and grad_states must share [batch, sequence, state]")
    batch, length, state_size = a.shape
    if initial is None:
        initial_state = torch.zeros(batch, state_size, dtype=a.dtype, device=a.device)
    else:
        if initial.shape != (batch, state_size):
            raise ValueError("initial state has wrong shape")
        initial_state = initial.to(device=a.device, dtype=a.dtype)

    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(a)
    carry = torch.zeros(batch, state_size, dtype=a.dtype, device=a.device)
    for index in range(length - 1, -1, -1):
        lam = grad_states[:, index] + carry
        previous = initial_state if index == 0 else states[:, index - 1]
        grad_a[:, index] = lam * previous
        grad_b[:, index] = lam
        carry = lam * a[:, index]
    return grad_a, grad_b, carry


def kernel_contract() -> dict[str, object]:
    contract = AffineScanKernelContract()
    return {
        "recurrence": contract.recurrence,
        "forward_parallel_lanes": contract.forward_parallel_lanes,
        "sequence_loop": contract.sequence_loop,
        "backward_recurrence": contract.backward_recurrence,
        "parameter_count_delta": contract.parameter_count_delta,
        "scientific_semantics_changed": contract.scientific_semantics_changed,
        "production_wired": False,
        "gpu_benchmark_authorized": False,
        "note": (
            "This module freezes the exact forward/backward semantics for a future fused "
            "device kernel. It deliberately contains no CUDA/Triton dispatch yet."
        ),
    }
