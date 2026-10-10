"""Differentiable, fixed-width state carry for the RLT associative scan.

This is a CPU-validated streaming *primitive*, NOT an end-to-end streaming RLT
model: the existing model encoder and decoder can still attend to the entire
prefix. A bounded-attention streaming architecture must be specified and tested
independently before any architectural capability claims.
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
    collect_states: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Carry final state across chunks, preserving all BPTT gradients.

    Recurrence: s[t] = gates[t] * s[t-1] + writes[t], with supplied s[-1].

    In training, collecting states and retaining the graph requires memory that
    grows with sequence length; this does NOT implement constant-memory BPTT.
    With torch.no_grad() and collect_states=False, only a fixed-width carried
    state is retained across the chunk boundaries (plus per-chunk operations).
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if gates.ndim != 3 or writes.shape != gates.shape:
        raise ValueError("gates and writes must share [batch,time,width]")
    batch, time_steps, width = gates.shape
    if time_steps < 1:
        raise ValueError("at least one timestep required")
    if initial_state.shape not in {(width,), (batch, width)}:
        raise ValueError("initial state must be [width] or [batch,width]")
    if gates.dtype != writes.dtype or gates.device != writes.device:
        raise ValueError("gates/writes dtype and device mismatch")

    carried = initial_state
    outputs: list[torch.Tensor] = []
    for start in range(0, time_steps, chunk_size):
        end = min(start + chunk_size, time_steps)
        local_states = associative_affine_scan(
            gates[:, start:end, :],
            writes[:, start:end, :],
            carried,
        )
        carried = local_states[:, -1, :]
        if collect_states:
            outputs.append(local_states)
    stacked = torch.cat(outputs, dim=1) if collect_states else None
    return stacked, carried


def chunked_affine_scan_inference(
    gates: torch.Tensor,
    writes: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """No-grad path carrying a [batch,width] state but no history output."""
    with torch.no_grad():
        _, final_state = chunked_affine_scan(
            gates, writes, initial_state,
            chunk_size=chunk_size,
            collect_states=False,
        )
    return final_state
