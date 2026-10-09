"""Parameter-free chunked carry for the gated RLT affine recurrence.

This module implements only the scan primitive. It does NOT change encoder
context, cross attention, token-position handling, decoder KV cache, or the
max-context behavior of the RLT language model. It is NOT a streaming LM.

State rule:
    s_t = gate_t * s_(t-1) + write_t
State is carried from each chunk's last output into the next chunk. It is
never detached: gradients propagate through all chunks (full BPTT).
"""
from __future__ import annotations

import torch

from experiments.rlt.model_gated_scan import associative_affine_scan


def chunked_affine_scan(
    gates: torch.Tensor,
    writes: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run affine scan over arbitrary-size chunks with differentiable carry.

    Args:
      gates, writes: [batch, time, width], identical dtype and device.
      initial_state: [width] or [batch, width]; converted to gates dtype/device
        consistently with the original full-sequence associative scan.
      chunk_size: positive integer; final chunk can be shorter.

    Returns: (all_states [batch,time,width], final_state [batch,width]).
    The final_state is a differentiable view into the last chunk output.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if gates.ndim != 3 or gates.shape != writes.shape:
        raise ValueError("gates and writes must share [batch,time,width]")
    if gates.size(1) < 1:
        raise ValueError("scan requires at least one token")
    if not gates.dtype.is_floating_point:
        raise TypeError("gates and writes must use floating-point dtype")
    if gates.dtype != writes.dtype or gates.device != writes.device:
        raise ValueError("gates and writes dtype/device mismatch")
    batch, time, width = gates.shape
    if initial_state.shape not in {(width,), (batch, width)}:
        raise ValueError("initial_state must be [width] or [batch,width]")

    state = initial_state.to(device=gates.device, dtype=gates.dtype)
    outputs: list[torch.Tensor] = []
    for lo in range(0, time, chunk_size):
        hi = min(lo + chunk_size, time)
        chunk_states = associative_affine_scan(
            gates[:, lo:hi, :], writes[:, lo:hi, :], state,
        )
        outputs.append(chunk_states)
        state = chunk_states[:, -1, :]
    return torch.cat(outputs, dim=1), state
