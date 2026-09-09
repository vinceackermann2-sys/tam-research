from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from architectures.cortex_s.affine_scan_fused_candidate import (
    affine_scan_analytic_backward_reference,
    affine_scan_sequential_reference,
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU CI may not ship Triton.
    triton = None
    tl = None


@dataclass(frozen=True)
class TritonAffineScanStatus:
    production_wired: bool = False
    gpu_benchmark_authorized: bool = False
    recurrence: str = "s_t = a_t * s_(t-1) + b_t"
    backward: str = "lambda_t = grad_t + a_(t+1) * lambda_(t+1)"
    intended_shape: tuple[int, int, int] = (64, 512, 128)
    kernels_per_scan: int = 2


if triton is not None and tl is not None:

    @triton.jit
    def _affine_scan_forward_kernel(
        a_ptr,
        b_ptr,
        initial_ptr,
        out_ptr,
        n_lanes,
        LENGTH: tl.constexpr,
        STATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        lanes = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = lanes < n_lanes
        batch = lanes // STATE
        state_index = lanes - batch * STATE
        initial_offset = batch * STATE + state_index
        state = tl.load(initial_ptr + initial_offset, mask=mask, other=0.0)

        # Deliberately sequential only along time. Independent batch*state lanes run
        # in parallel and intermediate states never round-trip through nine global
        # Hillis-Steele stages.
        for t in tl.range(0, LENGTH):
            offset = (batch * LENGTH + t) * STATE + state_index
            a_t = tl.load(a_ptr + offset, mask=mask, other=0.0)
            b_t = tl.load(b_ptr + offset, mask=mask, other=0.0)
            state = a_t * state + b_t
            tl.store(out_ptr + offset, state, mask=mask)


    @triton.jit
    def _affine_scan_backward_kernel(
        a_ptr,
        states_ptr,
        grad_states_ptr,
        initial_ptr,
        grad_a_ptr,
        grad_b_ptr,
        grad_initial_ptr,
        n_lanes,
        LENGTH: tl.constexpr,
        STATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        lanes = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = lanes < n_lanes
        batch = lanes // STATE
        state_index = lanes - batch * STATE
        initial_offset = batch * STATE + state_index
        initial_value = tl.load(initial_ptr + initial_offset, mask=mask, other=0.0)
        carry = tl.zeros((BLOCK,), dtype=tl.float32)

        for reverse_index in tl.range(0, LENGTH):
            t = LENGTH - 1 - reverse_index
            offset = (batch * LENGTH + t) * STATE + state_index
            upstream = tl.load(grad_states_ptr + offset, mask=mask, other=0.0)
            lam = upstream + carry

            previous_offset = offset - STATE
            previous_state = tl.load(
                states_ptr + previous_offset,
                mask=mask & (t > 0),
                other=0.0,
            )
            previous = tl.where(t == 0, initial_value, previous_state)
            a_t = tl.load(a_ptr + offset, mask=mask, other=0.0)

            tl.store(grad_a_ptr + offset, lam * previous, mask=mask)
            tl.store(grad_b_ptr + offset, lam, mask=mask)
            carry = lam * a_t

        tl.store(grad_initial_ptr + initial_offset, carry, mask=mask)


def triton_available() -> bool:
    return triton is not None and tl is not None


def _validate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: Optional[torch.Tensor],
) -> None:
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must have matching [batch, sequence, state] shapes")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("candidate requires a and b to share device and dtype")
    if initial is not None and initial.shape != (a.size(0), a.size(2)):
        raise ValueError("initial state has wrong shape")


def _triton_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    if not triton_available() or not a.is_cuda:
        raise RuntimeError("Triton CUDA path is unavailable")
    batch, length, state_size = a.shape
    a_work = a.contiguous()
    b_work = b.contiguous()
    initial_work = initial.contiguous()
    output = torch.empty_like(a_work)
    n_lanes = batch * state_size
    block = 256
    grid = (triton.cdiv(n_lanes, block),)
    _affine_scan_forward_kernel[grid](
        a_work,
        b_work,
        initial_work,
        output,
        n_lanes,
        LENGTH=length,
        STATE=state_size,
        BLOCK=block,
        num_warps=4,
    )
    return output


def _triton_backward(
    a: torch.Tensor,
    states: torch.Tensor,
    grad_states: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not triton_available() or not a.is_cuda:
        raise RuntimeError("Triton CUDA path is unavailable")
    batch, length, state_size = a.shape
    a_work = a.contiguous()
    states_work = states.contiguous()
    grad_work = grad_states.contiguous()
    initial_work = initial.contiguous()
    grad_a = torch.empty_like(a_work)
    grad_b = torch.empty_like(a_work)
    grad_initial = torch.empty_like(initial_work)
    n_lanes = batch * state_size
    block = 256
    grid = (triton.cdiv(n_lanes, block),)
    _affine_scan_backward_kernel[grid](
        a_work,
        states_work,
        grad_work,
        initial_work,
        grad_a,
        grad_b,
        grad_initial,
        n_lanes,
        LENGTH=length,
        STATE=state_size,
        BLOCK=block,
        num_warps=4,
    )
    return grad_a, grad_b, grad_initial


class _AffineScanTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        initial: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _validate_inputs(a, b, initial)
        had_initial = initial is not None
        if initial is None:
            initial_work = torch.zeros(
                a.size(0), a.size(2), device=a.device, dtype=a.dtype
            )
            initial_device = a.device
            initial_dtype = a.dtype
        else:
            initial_device = initial.device
            initial_dtype = initial.dtype
            initial_work = initial.to(device=a.device, dtype=a.dtype)

        use_triton = bool(a.is_cuda and triton_available())
        if use_triton:
            states = _triton_forward(a, b, initial_work)
        else:
            states = affine_scan_sequential_reference(a, b, initial_work)

        ctx.save_for_backward(a.contiguous(), states, initial_work.contiguous())
        ctx.had_initial = had_initial
        ctx.initial_device = initial_device
        ctx.initial_dtype = initial_dtype
        ctx.use_triton = use_triton
        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor):
        a, states, initial = ctx.saved_tensors
        if ctx.use_triton:
            grad_a, grad_b, grad_initial = _triton_backward(
                a, states, grad_states, initial
            )
        else:
            grad_a, grad_b, grad_initial = affine_scan_analytic_backward_reference(
                a, states, grad_states, initial
            )

        if ctx.had_initial:
            grad_initial_out = grad_initial.to(
                device=ctx.initial_device,
                dtype=ctx.initial_dtype,
            )
        else:
            grad_initial_out = None
        return grad_a, grad_b, grad_initial_out


def affine_scan_triton_candidate(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Guarded fused-scan candidate with CPU semantic fallback.

    This is intentionally *not* wired into ``PersistentWorldState``. On CUDA with
    Triton available it dispatches one forward and one analytic-backward kernel per
    scan. On CPU/unsupported environments it executes the frozen semantic oracle so
    ordinary CI can validate the custom-autograd contract without pretending to
    measure GPU speed.
    """

    return _AffineScanTritonFunction.apply(a, b, initial)


def candidate_status() -> dict[str, object]:
    status = TritonAffineScanStatus()
    return {
        "triton_import_available": triton_available(),
        "production_wired": status.production_wired,
        "gpu_benchmark_authorized": status.gpu_benchmark_authorized,
        "recurrence": status.recurrence,
        "backward": status.backward,
        "intended_shape": list(status.intended_shape),
        "kernels_per_scan": status.kernels_per_scan,
        "parameter_count_delta": 0,
        "scientific_semantics_changed": False,
        "note": "Candidate implementation only; GPU correctness/performance remains unmeasured.",
    }
