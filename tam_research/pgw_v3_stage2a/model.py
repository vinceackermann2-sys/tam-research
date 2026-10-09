"""PGW-v3 Stage-2A causal answer interface; structural CPU tests only.

No optimizer, training loop, scientific seed, GPU, or historical model edits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from tam_research.pgw_v3_stage0.model import (
    PGWV3Stage0Config,
    ROUTE_MODES,
    UtilityAddressedWorkspace,
)
from tam_research.pgw_v3_stage1b.oracle import (
    NOT_FOUND,
    QUERY,
    READ,
    PositionIndependentExample,
    oracle_latest_value_or_not_found,
)


@dataclass(frozen=True)
class PGWV3Stage2AConfig:
    vocab_size: int = 256
    d_model: int = 32
    chunk_size: int = 8
    heads: int = 4
    ff_width: int = 64
    route_mode: str = "hybrid"

    def __post_init__(self) -> None:
        if self.vocab_size != 256 or self.d_model != 32 or self.chunk_size != 8:
            raise ValueError("Stage-2A geometry is frozen to 256/32/8")
        if self.heads != 4 or self.ff_width != 64:
            raise ValueError("Stage-2A causal encoder heads/FFN are frozen to 4/64")
        if self.route_mode not in ROUTE_MODES:
            raise ValueError("unsupported frozen Stage-0 route mode")


class _CausalLocalBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_width: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, ff_width)
        self.ff_out = nn.Linear(ff_width, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        future_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        attended, _ = self.attn(x, x, x, attn_mask=future_mask, need_weights=False)
        hidden = self.norm1(x + attended)
        return self.norm2(hidden + self.ff_out(F.gelu(self.ff_in(hidden))))


class PGWV3Stage2A(nn.Module):
    """Strictly chunk-causal token adapter to the untouched PGW-v3 workspace.

    Returns class logits 0..32 for output token IDs 32..64 inclusive.
    The READ marker must occupy the final chunk's offset zero, QUERY offset two.
    """
    def __init__(self, cfg: PGWV3Stage2AConfig = PGWV3Stage2AConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.local_position = nn.Embedding(cfg.chunk_size, cfg.d_model)
        self.local = _CausalLocalBlock(cfg.d_model, cfg.heads, cfg.ff_width)
        self.workspace = UtilityAddressedWorkspace(
            PGWV3Stage0Config(route_mode=cfg.route_mode)
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.answer_head = nn.Linear(cfg.d_model, 33)

    def forward(self, tokens: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor[B,T]")
        batch, length = tokens.shape
        if batch < 1 or length < 16 or length % self.cfg.chunk_size:
            raise ValueError("batch must be nonempty; time must contain complete chunks")
        if torch.any(tokens < 0).item() or torch.any(tokens >= self.cfg.vocab_size).item():
            raise ValueError("token outside vocabulary")
        if anchors.ndim != 1 or anchors.shape[0] != batch or anchors.dtype != torch.long:
            raise ValueError("anchors must be LongTensor[B]")
        if torch.any(anchors != length - self.cfg.chunk_size + 2).item():
            raise ValueError("QUERY anchors must be at offset two of final READ chunk")
        rows = torch.arange(batch, device=tokens.device)
        if torch.any(tokens[rows, anchors] != QUERY).item():
            raise ValueError("QUERY marker missing at anchor")
        if torch.any(tokens[rows, anchors - 2] != READ).item():
            raise ValueError("READ marker missing at start of final chunk")

        chunks = length // self.cfg.chunk_size
        positions = torch.arange(self.cfg.chunk_size, device=tokens.device).repeat(chunks)
        local_in = self.token_embedding(tokens) + self.local_position(positions).unsqueeze(0)
        local_out = self.local(local_in.reshape(batch * chunks, self.cfg.chunk_size, -1))
        local_out = local_out.reshape(batch, length, self.cfg.d_model)
        memory_residual = self.workspace(local_out)
        features = self.out_norm(local_out + memory_residual)
        return self.answer_head(features[rows, anchors])


def batch_verified_examples(
    examples: tuple[PositionIndependentExample, ...] | list[PositionIndependentExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fixed-length batches, with independent answer-oracle verification.

    Does not use hidden metadata as a feature. Callers must bucket by delay.
    """
    if not examples:
        raise ValueError("examples must not be empty")
    length = len(examples[0].tokens)
    sequences: list[tuple[int, ...]] = []
    anchors: list[int] = []
    targets: list[int] = []
    for e in examples:
        if not isinstance(e, PositionIndependentExample):
            raise TypeError("expected frozen Stage-1B example")
        if len(e.tokens) != length:
            raise ValueError("mixed lengths require explicit bucketing; no padding")
        if hashlib.sha256(bytes(e.tokens)).hexdigest() != e.sha256:
            raise ValueError("stream fingerprint mismatch")
        if e.query_anchor != length - 8 + 2 or e.tokens[e.query_anchor] != QUERY:
            raise ValueError("invalid final QUERY anchor")
        answer = oracle_latest_value_or_not_found(e.tokens)
        if answer != e.expected_value or not (32 <= answer <= NOT_FOUND):
            raise ValueError("external target conflicts with causal oracle")
        sequences.append(e.tokens)
        anchors.append(e.query_anchor)
        targets.append(answer - 32)
    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(anchors, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def instantiated_parameter_count(model: nn.Module) -> int:
    return sum(t.numel() for t in model.parameters() if t.requires_grad)
