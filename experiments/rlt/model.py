from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from tam_research.models import Block, ModelConfig


class RecurrentLoopedLM(nn.Module):
    """Recurrent Looped Transformer language model.

    The token/position embeddings are applied once. A single Transformer block
    F_theta is then reused ``n_layers`` times:
        h_0 = Embed(tokens)
        h_{t+1} = F_theta(h_t)
    This keeps physical block parameters constant as effective depth changes.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_layers < 1:
            raise ValueError("RLT requires at least one recurrent loop")
        if cfg.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")

        self.cfg = replace(cfg, architecture="rlt")
        block_cfg = replace(cfg, architecture="transformer")
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.shared_block = Block(block_cfg)
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

    @property
    def loop_steps(self) -> int:
        return self.cfg.n_layers

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        pos = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_emb(tokens) + self.pos_emb(pos)[None]
        for _ in range(self.loop_steps):
            hidden = self.shared_block(hidden)
        return self.lm_head(self.norm(hidden))

    @torch.no_grad()
    def router_stats(self) -> None:
        return None


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
