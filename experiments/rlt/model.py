from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RLTConfig:
    vocab_size: int = 50_257
    d_model: int = 256
    n_heads: int = 8
    n_stages: int = 4
    max_seq_len: int = 512
    ff_mult: int = 4
    swa_window: int = 64
    rms_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * scale.to(x.dtype)) * self.weight


class SharedSelfAttention(nn.Module):
    """One attention module used in both the causal encoder and recurrent decoder."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward_causal(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        b, _, t, _ = y.shape
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(y)

    def step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        window: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        if k.size(2) > window:
            k = k[:, :, -window:, :]
            v = v[:, :, -window:, :]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, 1, self.d_model)
        return self.out(y), (k, v)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def precompute(
        self, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.kv(memory).chunk(2, dim=-1)
        return self._split(k), self._split(v)

    def step(
        self,
        x: torch.Tensor,
        memory_kv: tuple[torch.Tensor, torch.Tensor],
        prefix_len: int,
    ) -> torch.Tensor:
        q = self._split(self.q(x))
        k = memory_kv[0][:, :, :prefix_len, :]
        v = memory_kv[1][:, :, :prefix_len, :]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, 1, self.d_model)
        return self.out(y)


class SharedStage(nn.Module):
    """Encoder stage whose self-attention/FFN weights are reused by the decoder."""

    def __init__(self, cfg: RLTConfig):
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = SharedSelfAttention(cfg.d_model, cfg.n_heads)
        self.cross_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.cross_attn = CrossAttention(cfg.d_model, cfg.n_heads)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.forward_causal(self.self_norm(x))
        return x + self.ff(self.ff_norm(x))

    def decode_step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        memory_kv: tuple[torch.Tensor, torch.Tensor],
        prefix_len: int,
        window: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        local, new_cache = self.self_attn.step(self.self_norm(x), cache, window)
        x = x + local
        x = x + self.cross_attn.step(
            self.cross_norm(x), memory_kv, prefix_len
        )
        x = x + self.ff(self.ff_norm(x))
        return x, new_cache


class RecurrentLoopedTransformer(nn.Module):
    """Reference prototype of the public RLT architecture.

    This implements the public equations/execution semantics: a causal encoder
    builds global prefix memory; a token-recurrent decoder carries s_(t-1) and
    layerwise SWA KV; decoder cross-attention sees encoder memory only through t;
    compatible self-attention/FFN weights are shared between encoder/decoder
    stages; and recurrent state/cache tensors remain differentiable for full BPTT.

    The public project page leaves Merge(e_t, s_(t-1)) abstract. This prototype
    instantiates Merge as a learned linear projection of their concatenation.
    """

    def __init__(self, cfg: RLTConfig):
        super().__init__()
        if cfg.n_stages < 1:
            raise ValueError("n_stages must be positive")
        if cfg.swa_window < 1:
            raise ValueError("swa_window must be positive")
        if cfg.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.stages = nn.ModuleList(SharedStage(cfg) for _ in range(cfg.n_stages))
        self.encoder_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.merge = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.merge_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.start_state = nn.Parameter(torch.zeros(cfg.d_model))
        self.output_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.last_cache_lengths: tuple[int, ...] = ()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        pos = torch.arange(t, device=tokens.device)
        hidden = self.token_emb(tokens) + self.pos_emb(pos)[None]
        for stage in self.stages:
            hidden = stage.encode(hidden)
        return self.encoder_norm(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        memory = self.encode(tokens)
        memory_kv = [stage.cross_attn.precompute(memory) for stage in self.stages]

        state = self.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.stages
        ]
        logits: list[torch.Tensor] = []

        for token_index in range(t):
            merged = torch.cat((memory[:, token_index, :], state), dim=-1)
            hidden = self.merge_norm(self.merge(merged)).unsqueeze(1)
            prefix_len = token_index + 1
            for layer_index, stage in enumerate(self.stages):
                hidden, caches[layer_index] = stage.decode_step(
                    hidden,
                    caches[layer_index],
                    memory_kv[layer_index],
                    prefix_len,
                    self.cfg.swa_window,
                )
            state = hidden[:, 0, :]
            logits.append(self.lm_head(self.output_norm(state)))

        self.last_cache_lengths = tuple(
            0 if cache is None else int(cache[0].size(2)) for cache in caches
        )
        return torch.stack(logits, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
