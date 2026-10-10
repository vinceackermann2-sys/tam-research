"""Stage-2B3: deterministic analytical attention-core and workspace counts.

Counts are algorithmic structural operations only, NOT runtime measurements,
training FLOPs, equal-compute comparisons, gradient FLOPs or profiler evidence.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionOperationCounts:
    batch: int
    tokens: int
    heads: int
    head_width: int
    chunks: int
    full_dense_pair_ops: int
    full_causal_pair_ops: int
    chunk_dense_pair_ops: int
    chunk_causal_pair_ops: int
    full_dense_qk_av_macs: int
    chunk_dense_qk_av_macs: int
    pgw_selected_write_calls: int
    pgw_read_slot_key_comparisons: int
    pgw_write_slot_key_comparisons: int


def attention_operation_counts(
    *, batch: int, tokens: int, heads: int = 4, width: int = 32,
    chunk: int = 8, workspace_slots: int = 4, selected_events: int = 2,
) -> AttentionOperationCounts:
    """Two attention pair ops: one QK score and one AV weighted value.

    Dense matrix implementation may compute masked pairs; causal count is an
    ideal useful lower-triangular comparison only. Each pair op costs
    head_width multiply-accumulates, ignoring softmax and projection cost.
    """
    values = (batch, tokens, heads, width, chunk, workspace_slots, selected_events)
    if any(type(x) is not int for x in values):
        raise ValueError("all geometry fields must be integers")
    if (
        batch < 1 or tokens < 16 or chunk != 8 or tokens % chunk
        or heads != 4 or width != 32
        or workspace_slots != 4 or selected_events != 2
    ):
        raise ValueError("frozen geometry: B>=1 T>=16 multiple8, H4 d32, K4 selected2")
    chunks = tokens // chunk
    head_width = width // heads
    full_dense = 2 * batch * heads * tokens * tokens
    full_causal = 2 * batch * heads * tokens * (tokens + 1) // 2
    chunk_dense = 2 * batch * heads * chunks * chunk * chunk
    chunk_causal = 2 * batch * heads * chunks * chunk * (chunk + 1) // 2
    return AttentionOperationCounts(
        batch=batch,
        tokens=tokens,
        heads=heads,
        head_width=head_width,
        chunks=chunks,
        full_dense_pair_ops=full_dense,
        full_causal_pair_ops=full_causal,
        chunk_dense_pair_ops=chunk_dense,
        chunk_causal_pair_ops=chunk_causal,
        full_dense_qk_av_macs=full_dense * head_width,
        chunk_dense_qk_av_macs=chunk_dense * head_width,
        pgw_selected_write_calls=batch * chunks * selected_events,
        pgw_read_slot_key_comparisons=batch * tokens * workspace_slots,
        pgw_write_slot_key_comparisons=batch * chunks * selected_events * workspace_slots,
    )
