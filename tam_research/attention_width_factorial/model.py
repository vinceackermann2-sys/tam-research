from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from tam_research.models import CausalSelfAttention, parameter_count
from tam_research.pgw_v1.model import ChunkLocalAttention

from .protocol import (
    CHUNK_SIZE,
    D_MODEL,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    REDUCED_ATTN_INNER,
    REDUCED_FF_HIDDEN,
    VOCAB_SIZE,
)


@dataclass(frozen=True)
class WidthFactorialConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    max_seq_len: int = MAX_SEQ_LEN
    attention_inner: int = REDUCED_ATTN_INNER
    ff_hidden: int = REDUCED_FF_HIDDEN
    chunk_size: int = CHUNK_SIZE
    locality: str = "global"
    architecture: str = "attention_width_factorial"


class WidthFactorialBlock(nn.Module):
    def __init__(self, cfg: WidthFactorialConfig):
        super().__init__()
        if cfg.locality not in {"global", "local"}:
            raise ValueError("locality must be 'global' or 'local'")
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

        if cfg.locality == "global":
            self.mixer = CausalSelfAttention(
                cfg.d_model,
                cfg.n_heads,
                inner=cfg.attention_inner,
            )
        else:
            class LocalCfg:
                d_model = cfg.d_model
                n_heads = cfg.n_heads
                local_attn_inner = cfg.attention_inner
                chunk_size = cfg.chunk_size

            self.mixer = ChunkLocalAttention(LocalCfg())

        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_hidden, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ff_hidden, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class WidthFactorialResearchLM(nn.Module):
    def __init__(self, cfg: WidthFactorialConfig):
        super().__init__()
        if cfg.architecture != "attention_width_factorial":
            raise ValueError(
                "WidthFactorialResearchLM requires architecture='attention_width_factorial'"
            )
        if cfg.attention_inner % cfg.n_heads:
            raise ValueError("attention_inner must be divisible by n_heads")
        if cfg.locality == "local" and cfg.max_seq_len < cfg.chunk_size:
            raise ValueError("max_seq_len must be >= chunk_size")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            WidthFactorialBlock(cfg) for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if self.cfg.locality == "local" and t % self.cfg.chunk_size:
            raise ValueError("local sequence length must be divisible by chunk_size")

        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def router_stats(self) -> None:
        return None


def reduced_width_parameter_count(locality: str) -> int:
    return parameter_count(
        WidthFactorialResearchLM(
            WidthFactorialConfig(locality=locality)
        )
    )
