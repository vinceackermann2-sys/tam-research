"""AX: independent causal Transformer with true bounded sliding-window KV caches.

No recurrent state, cross-attention, or encoder memory. Each layer holds at
most W recent K/V pairs, including the current token. Importantly, L stacked
layers can propagate information across L*(W-1) previous positions, even when
each per-layer cache is bounded to W. Therefore the last W tokens are NOT the
complete model-level information bottleneck.

Inference cache is O(L*W*D); full training activation memory is not constant.
Phase positions repeat globally every W tokens (never reset by API partition).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SlidingWindowConfig:
    vocab_size: int = 16
    width: int = 32
    heads: int = 4
    layers: int = 2
    window: int = 8
    ff_inner: int = 145
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.width, self.heads, self.layers, self.window, self.ff_inner) < 1:
            raise ValueError("positive configuration fields required")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype) * self.weight


class WindowAttention(nn.Module):
    def __init__(self, cfg: SlidingWindowConfig) -> None:
        super().__init__()
        self.h = cfg.heads
        self.d = cfg.width // cfg.heads
        self.window = cfg.window
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.h, self.d).transpose(1, 2)

    def combine(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, time, dim = x.shape
        return x.transpose(1, 2).contiguous().reshape(batch, time, heads * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel causal finite-window reference; no precomputed memory."""
        q, k, v = (self.split(t) for t in self.qkv(x).chunk(3, -1))
        time = x.size(1)
        rows = torch.arange(time, device=x.device)
        offsets = rows[:, None] - rows[None, :]
        allow = (offsets >= 0) & (offsets < self.window)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        return self.proj(self.combine(out))

    def step(
        self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Append current token; attend to it and at most W-1 earlier KVs."""
        if x.shape[1] != 1:
            raise ValueError("step expects exactly one token")
        q, k, v = (self.split(t) for t in self.qkv(x).chunk(3, -1))
        if cache is not None:
            old_k, old_v = cache
            if old_k.shape != old_v.shape or old_k.shape[0] != x.shape[0]:
                raise ValueError("invalid cache shape")
            if old_k.shape[1] != self.h or old_k.shape[-1] != self.d or old_k.size(2) > self.window:
                raise ValueError("invalid cache dimensions")
            k = torch.cat((old_k, k), 2)
            v = torch.cat((old_v, v), 2)
        k, v = k[:, :, -self.window:, :], v[:, :, -self.window:, :]
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(self.combine(out)), (k, v)


class WindowBlock(nn.Module):
    def __init__(self, cfg: SlidingWindowConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.width, cfg.eps)
        self.attn = WindowAttention(cfg)
        self.norm2 = RMSNorm(cfg.width, cfg.eps)
        self.ff = nn.Sequential(
            nn.Linear(cfg.width, cfg.ff_inner, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ff_inner, cfg.width, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))

    def step(self, x: torch.Tensor, cache):
        attn, cache = self.attn.step(self.norm1(x), cache)
        x = x + attn
        return x + self.ff(self.norm2(x)), cache


@dataclass
class WindowStreamState:
    caches: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...]
    total_tokens: int = 0


class SlidingWindowTransformer(nn.Module):
    def __init__(self, cfg: SlidingWindowConfig = SlidingWindowConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.width)
        self.pos = nn.Embedding(cfg.window, cfg.width)
        self.blocks = nn.ModuleList(WindowBlock(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.width, cfg.eps)
        self.out = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.out.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(layer: nn.Module) -> None:
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            nn.init.normal_(layer.weight, mean=0, std=0.02)

    @property
    def receptive_distance(self) -> int:
        """Maximum earlier token distance via L stacked W-window layers."""
        return self.cfg.layers * (self.cfg.window - 1)

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.dtype != torch.long or min(x.shape) < 1:
            raise ValueError("expected nonempty int64 tokens [batch,time]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        idx = torch.arange(x.size(1), device=x.device) % self.cfg.window
        hidden = self.tok(x) + self.pos(idx)[None]
        for block in self.blocks:
            hidden = block(hidden)
        return self.out(self.norm(hidden))

    def init_stream(self, batch_size: int) -> WindowStreamState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return WindowStreamState((None,) * self.cfg.layers)

    def forward_chunk(
        self, x: torch.Tensor, state: WindowStreamState,
    ) -> tuple[torch.Tensor, WindowStreamState]:
        self._validate(x)
        if len(state.caches) != self.cfg.layers or state.total_tokens < 0:
            raise ValueError("invalid stream state")
        logits = []
        caches = list(state.caches)
        for t in range(x.size(1)):
            global_index = state.total_tokens + t
            pos = self.pos.weight[global_index % self.cfg.window]
            h = self.tok(x[:, t:t + 1]) + pos[None, None]
            for i, block in enumerate(self.blocks):
                h, caches[i] = block.step(h, caches[i])
            logits.append(self.out(self.norm(h)))
        return torch.cat(logits, dim=1), WindowStreamState(tuple(caches), state.total_tokens + x.size(1))


def tiny_sliding_window_config() -> SlidingWindowConfig:
    """27,680 parameters, 32 fewer than AW carry RLT (27,712; -0.115%)."""
    return SlidingWindowConfig()
