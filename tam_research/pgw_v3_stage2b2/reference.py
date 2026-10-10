"""PGW-v3 Stage-2B2 count-near causal references, CPU STRUCTURE only.

These reference models are NOT trained, not GPU-enabled and not active/compute
matched to PGW. Frozen in issue #1400. Historical PGW models are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tam_research.pgw_v3_stage1b.oracle import QUERY, READ, WRITE


@dataclass(frozen=True)
class CausalReferenceConfig:
    attention_scope: str = "full"
    vocab_size: int = 256
    d_model: int = 32
    chunk_size: int = 8
    heads: int = 4
    ff_width: int = 90
    predictor_rank: int = 8

    def __post_init__(self) -> None:
        if self.attention_scope not in ("full", "chunk"):
            raise ValueError("attention_scope must be full or chunk")
        if (
            self.vocab_size, self.d_model, self.chunk_size,
            self.heads, self.ff_width, self.predictor_rank,
        ) != (256, 32, 8, 4, 90, 8):
            raise ValueError("Stage-2B2 reference geometry is frozen")


class CausalReference(nn.Module):
    """Single-layer full-causal Transformer or chunk-only negative control.

    Both include a LIVE 512-parameter predictive auxiliary head, not inert
    capacity padding; all 20,347 instantiated parameters are structurally used.
    Same token embedding/answer classes as PGWV3Stage2A.
    """

    def __init__(self, cfg: CausalReferenceConfig = CausalReferenceConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.token_embedding = nn.Embedding(cfg.vocab_size, d)
        self.local_position = nn.Embedding(cfg.chunk_size, d)
        self.attn = nn.MultiheadAttention(d, cfg.heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, cfg.ff_width)
        self.ff_out = nn.Linear(cfg.ff_width, d)
        self.norm2 = nn.LayerNorm(d)
        self.out_norm = nn.LayerNorm(d)
        self.answer_head = nn.Linear(d, 33)
        self.predict_down = nn.Linear(d, cfg.predictor_rank, bias=False)
        self.predict_up = nn.Linear(cfg.predictor_rank, d, bias=False)

    def _validate(
        self, tokens: torch.Tensor, anchors: torch.Tensor
    ) -> tuple[int, int]:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor[B,T]")
        if tokens.device.type != "cpu":
            raise ValueError("reference probe is CPU-only")
        batch, length = tokens.shape
        if batch < 1 or length < 80 or length % self.cfg.chunk_size:
            raise ValueError("expected eight WRITE chunks, delay, final READ; no padding")
        if bool(((tokens < 0) | (tokens >= self.cfg.vocab_size)).any()):
            raise ValueError("tokens outside frozen vocabulary")
        if not isinstance(anchors, torch.Tensor) or anchors.ndim != 1 or anchors.dtype != torch.long:
            raise ValueError("anchors must be LongTensor[B]")
        if anchors.shape != (batch,) or anchors.device.type != "cpu":
            raise ValueError("anchors must be on CPU with one index per sequence")
        if bool((anchors != length - self.cfg.chunk_size + 2).any()):
            raise ValueError("anchor must be final READ chunk offset2")
        if bool((tokens[:, -self.cfg.chunk_size] != READ).any()):
            raise ValueError("final READ marker is missing")
        if bool((tokens[:, -self.cfg.chunk_size + 2] != QUERY).any()):
            raise ValueError("final QUERY marker is missing")
        if bool((tokens[:, :64:8] != WRITE).any()):
            raise ValueError("first eight chunks must be complete WRITE events")
        return batch, length

    def input_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Public leaf for the CPU-only input-position gradient causal test."""
        length = tokens.shape[1]
        position_ids = torch.arange(length, device=tokens.device) % self.cfg.chunk_size
        return self.token_embedding(tokens) + self.local_position(position_ids)[None, :, :]

    def encode_embeddings(self, embedded: torch.Tensor) -> torch.Tensor:
        """Causal attention; full and chunk scopes differ ONLY in visible history."""
        if embedded.ndim != 3 or embedded.shape[-1] != self.cfg.d_model:
            raise ValueError("embedded tensor must be [B,T,32]")
        batch, length, width = embedded.shape
        if batch < 1 or length == 0 or length % self.cfg.chunk_size:
            raise ValueError("embedded sequence requires complete 8-token chunks")
        if self.cfg.attention_scope == "chunk":
            width_t = self.cfg.chunk_size
            x = embedded.reshape(batch * (length // width_t), width_t, width)
        else:
            x = embedded
            width_t = length
        causal_mask = torch.triu(
            torch.ones(width_t, width_t, device=embedded.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff_out(F.gelu(self.ff_in(x))))
        return x.reshape(batch, length, width)

    def answer_from_hidden(self, hidden: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return self.answer_head(self.out_norm(hidden[rows, anchors]))

    def auxiliary_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Same frozen Stage-2B0 latent-prediction objective on 8 WRITE chunks."""
        if hidden.ndim != 3 or hidden.shape[0] < 1 or hidden.shape[1] < 64 or hidden.shape[2] != 32:
            raise ValueError("hidden tensor lacks eight event chunks")
        x = hidden[:, :64].reshape(hidden.shape[0], 8, 8, 32)
        z = self.predict_up(F.gelu(self.predict_down(x[:, :, :-1])))
        predicted = F.layer_norm(z.float(), (32,), eps=1e-5)
        target = F.layer_norm(x[:, :, 1:].detach().float(), (32,), eps=1e-5)
        return (predicted - target).square().sum(dim=-1).mean()

    def forward(self, tokens: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        self._validate(tokens, anchors)
        hidden = self.encode_embeddings(self.input_embeddings(tokens))
        return self.answer_from_hidden(hidden, anchors)

    def forward_with_aux(
        self, tokens: torch.Tensor, anchors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """No lambda/optimizer: caller gets uncombined logits and auxiliary loss."""
        self._validate(tokens, anchors)
        h = self.encode_embeddings(self.input_embeddings(tokens))
        return self.answer_from_hidden(h, anchors), self.auxiliary_from_hidden(h)


def count_instantiated(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_modules(model: CausalReference) -> dict[str, int]:
    """Account for every genuinely used trainable tensor exactly once."""
    groups = {
        "shared_embeddings": sum(p.numel() for m in (model.token_embedding, model.local_position) for p in m.parameters()),
        "causal_attention": sum(p.numel() for p in model.attn.parameters()),
        "encoder_ffn_norms": sum(p.numel() for m in (model.norm1, model.norm2, model.ff_in, model.ff_out) for p in m.parameters()),
        "answer": sum(p.numel() for m in (model.out_norm, model.answer_head) for p in m.parameters()),
        "predictor_auxiliary": sum(p.numel() for m in (model.predict_down, model.predict_up) for p in m.parameters()),
    }
    if sum(groups.values()) != count_instantiated(model):
        raise ValueError("unaccounted reference parameters")
    return groups
