from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import CausalSelfAttention, diagonal_affine_scan


def _affine_pair_scan(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele scan over affine maps s -> a*s + b."""
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    t = a.size(1)
    pa, pb = a, b
    offset = 1
    while offset < t:
        ones = torch.ones_like(pa[:, :offset])
        zeros = torch.zeros_like(pb[:, :offset])
        a_shift = torch.cat((ones, pa[:, :-offset]), dim=1)
        b_shift = torch.cat((zeros, pb[:, :-offset]), dim=1)
        old_a = pa
        pb = pb + old_a * b_shift
        pa = old_a * a_shift
        offset <<= 1
    return pa, pb


def chunked_diagonal_affine_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Exact two-level scan for s_t = a_t*s_(t-1)+b_t.

    The mathematical recurrence is unchanged. We first scan independently inside
    short chunks, scan only the chunk summaries, then apply each preceding chunk's
    terminal state to the local prefixes. This reduces full-sequence tensor traffic
    from O(log T) stages to O(log C + 1 + log(T/C)/C) full-tensor equivalents.
    """
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.ndim != 3:
        raise ValueError("expected [batch, time, state] tensors")
    if chunk_size < 2 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a power of two >= 2")

    batch, time, state_size = a.shape
    if time == 0:
        return b

    pad = (-time) % chunk_size
    if pad:
        a = torch.cat(
            (
                a,
                torch.ones(
                    batch,
                    pad,
                    state_size,
                    device=a.device,
                    dtype=a.dtype,
                ),
            ),
            dim=1,
        )
        b = torch.cat(
            (
                b,
                torch.zeros(
                    batch,
                    pad,
                    state_size,
                    device=b.device,
                    dtype=b.dtype,
                ),
            ),
            dim=1,
        )

    padded_time = a.size(1)
    chunks = padded_time // chunk_size
    a_chunks = a.reshape(batch * chunks, chunk_size, state_size)
    b_chunks = b.reshape(batch * chunks, chunk_size, state_size)

    local_a, local_b = _affine_pair_scan(a_chunks, b_chunks)
    local_a = local_a.reshape(batch, chunks, chunk_size, state_size)
    local_b = local_b.reshape(batch, chunks, chunk_size, state_size)

    chunk_a = local_a[:, :, -1, :]
    chunk_b = local_b[:, :, -1, :]
    _, chunk_end = _affine_pair_scan(chunk_a, chunk_b)

    previous_chunk_state = torch.cat(
        (torch.zeros_like(chunk_end[:, :1]), chunk_end[:, :-1]),
        dim=1,
    )
    state = local_b + local_a * previous_chunk_state[:, :, None, :]
    state = state.reshape(batch, padded_time, state_size)
    return state[:, :time]


class ChunkedWorldState(nn.Module):
    """RecurrentWorldState with identical math and a chunked scan implementation."""

    def __init__(self, d_model: int, state_size: int, chunk_size: int):
        super().__init__()
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.candidate = nn.Linear(d_model, state_size, bias=False)
        self.keep = nn.Linear(d_model, state_size, bias=True)
        self.out = nn.Linear(state_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        state = chunked_diagonal_affine_scan(
            keep,
            (1.0 - keep) * candidate,
            chunk_size=self.chunk_size,
        )
        return self.out(state), state


class ChunkedTAMV3Mixer(nn.Module):
    """Timing-only TAM v3 mixer using the exact chunked world-state scan."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_inner: int,
        state_size: int,
        chunk_size: int,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, attention_inner)
        self.world = ChunkedWorldState(d_model, state_size, chunk_size)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(x)
        world, _ = self.world(x)
        gate = torch.sigmoid(self.gate_logit).to(x.dtype)
        return 2.0 * ((1.0 - gate) * attn + gate * world)


class ChunkedTAMV3Block(nn.Module):
    """Timing-only parameter-identical TAM v3 block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_inner: int,
        state_size: int,
        ff_mult: int,
        chunk_size: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mixer = ChunkedTAMV3Mixer(
            d_model,
            n_heads,
            attention_inner,
            state_size,
            chunk_size,
        )
        ff = ff_mult * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class ProjectionFusedTAMV3Mixer(nn.Module):
    """Function-equivalent TAM v3 mixer with five projections collapsed to two.

    The input projection concatenates attention QKV, world candidate, and world
    keep weights. The output projection concatenates the attention and world output
    matrices. The scalar TAM v3 gate is applied to the two activation slices before
    the joint output projection, making the operation algebraically identical to
    summing the two separately projected branches.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_inner: int,
        state_size: int,
        chunk_size: int | None = None,
    ):
        super().__init__()
        if attention_inner % n_heads:
            raise ValueError("attention_inner must be divisible by n_heads")
        self.n_heads = n_heads
        self.attention_inner = attention_inner
        self.head_dim = attention_inner // n_heads
        self.state_size = state_size
        self.chunk_size = chunk_size

        joint_input = 3 * attention_inner + 2 * state_size
        self.in_proj = nn.Linear(d_model, joint_input, bias=False)
        self.keep_bias = nn.Parameter(torch.zeros(state_size))
        self.out_proj = nn.Linear(attention_inner + state_size, d_model, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        projected = self.in_proj(x)
        q, k, v, candidate_raw, keep_raw = projected.split(
            (
                self.attention_inner,
                self.attention_inner,
                self.attention_inner,
                self.state_size,
                self.state_size,
            ),
            dim=-1,
        )

        q = q.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(
            batch,
            time,
            self.attention_inner,
        )

        candidate = torch.tanh(candidate_raw)
        keep = torch.sigmoid(keep_raw + self.keep_bias)
        affine_b = (1.0 - keep) * candidate
        if self.chunk_size is None:
            state = diagonal_affine_scan(keep, affine_b)
        else:
            state = chunked_diagonal_affine_scan(
                keep,
                affine_b,
                chunk_size=self.chunk_size,
            )

        gate = torch.sigmoid(self.gate_logit).to(x.dtype)
        joint_output = torch.cat(
            (
                2.0 * (1.0 - gate) * attn,
                2.0 * gate * state,
            ),
            dim=-1,
        )
        return self.out_proj(joint_output)


class ProjectionFusedTAMV3Block(nn.Module):
    """Timing-only full TAM v3 block with exact projection fusion."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_inner: int,
        state_size: int,
        ff_mult: int,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mixer = ProjectionFusedTAMV3Mixer(
            d_model,
            n_heads,
            attention_inner,
            state_size,
            chunk_size,
        )
        ff = ff_mult * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))
