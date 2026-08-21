from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TAMV2_ARCHITECTURES = {
    "tamv2",
    "tamv2_nomem",
    "tamv2_noworld",
    "tamv2_fixed",
}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50_257
    d_model: int = 256
    n_layers: int = 15
    n_heads: int = 8
    max_seq_len: int = 1024
    ff_mult: int = 4
    architecture: str = "transformer"
    tamv2_branch_inner: int = 104
    tamv2_state_size: int = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, inner: int | None = None):
        super().__init__()
        inner = inner or d_model
        if inner % n_heads:
            raise ValueError("attention inner width must be divisible by n_heads")
        self.n_heads = n_heads
        self.inner = inner
        self.head_dim = inner // n_heads
        self.qkv = nn.Linear(d_model, 3 * inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, self.inner)
        return self.out(y)


class ATAMMixer(nn.Module):
    """Causal temporal associative memory over learned predecessor→successor bindings."""

    def __init__(self, d_model: int, n_heads: int, inner: int | None = None):
        super().__init__()
        inner = inner or d_model
        if inner % n_heads:
            raise ValueError("ATAM inner width must be divisible by n_heads")
        self.n_heads = n_heads
        self.inner = inner
        self.head_dim = inner // n_heads
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)
        self.write = nn.Linear(2 * d_model, n_heads, bias=True)
        self.temporal_alpha = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t == 1:
            return torch.zeros_like(x)

        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        prev, curr = x[:, :-1], x[:, 1:]
        k = self.k(prev).view(b, t - 1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(curr).view(b, t - 1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        write_gate = torch.sigmoid(self.write(torch.cat((prev, curr), dim=-1)))
        write_gate = write_gate.permute(0, 2, 1).unsqueeze(2)
        scores = scores + torch.log(write_gate.clamp_min(1e-6))

        qi = torch.arange(t, device=x.device).view(t, 1)
        ji = torch.arange(t - 1, device=x.device).view(1, t - 1)
        valid = ji < qi
        rel = (ji.float() - (qi.float() - 1).clamp_min(0)) / max(t - 1, 1)
        scores = scores + self.temporal_alpha.view(1, self.n_heads, 1, 1) * rel.view(1, 1, t, t - 1)
        scores = scores.masked_fill(~valid.view(1, 1, t, t - 1), float("-inf"))
        scores[:, :, 0, :] = 0
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        y = torch.matmul(weights, v)
        y[:, :, 0, :] = 0
        y = y.transpose(1, 2).contiguous().view(b, t, self.inner)
        return self.out(y)


class TAMMixer(nn.Module):
    """Parameter/FLOP-matched v1 hybrid: half-width attention + half-width ATAM."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if n_heads % 2:
            raise ValueError("TAM needs an even head count")
        h = n_heads // 2
        inner = d_model // 2
        self.attn = CausalSelfAttention(d_model, h, inner)
        self.mem = ATAMMixer(d_model, h, inner)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logit)
        return 2.0 * ((1.0 - g) * self.attn(x) + g * self.mem(x))


def diagonal_affine_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive parallel scan for s_t = a_t * s_(t-1) + b_t, s_-1 = 0."""
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
    return pb


class RecurrentWorldState(nn.Module):
    def __init__(self, d_model: int, state_size: int):
        super().__init__()
        self.state_size = state_size
        self.candidate = nn.Linear(d_model, state_size, bias=False)
        self.keep = nn.Linear(d_model, state_size, bias=True)
        self.out = nn.Linear(state_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        state = diagonal_affine_scan(keep, (1.0 - keep) * candidate)
        return self.out(state), state


class TAMV2Mixer(nn.Module):
    """Token-wise router over causal attention, ATAM, and recurrent world-state.

    Ablation modes retain all modules/parameters and still execute every branch so
    parameter count and branch compute stay matched. They only change which branch
    outputs the router is allowed to use.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        branch_inner: int,
        state_size: int,
        mode: str = "full",
    ):
        super().__init__()
        if n_heads % 2:
            raise ValueError("TAM v2 needs an even head count")
        if mode not in {"full", "nomem", "noworld", "fixed"}:
            raise ValueError(f"unknown TAM v2 mode: {mode}")
        branch_heads = n_heads // 2
        if branch_inner % branch_heads:
            raise ValueError("branch_inner must be divisible by half the head count")
        self.mode = mode
        self.attn = CausalSelfAttention(d_model, branch_heads, branch_inner)
        self.mem = ATAMMixer(d_model, branch_heads, branch_inner)
        self.world = RecurrentWorldState(d_model, state_size)
        self.router = nn.Linear(d_model + state_size, 3, bias=True)
        self.temperature_log = nn.Parameter(torch.zeros(()))
        self.last_route: torch.Tensor | None = None
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(x)
        mem = self.mem(x)
        world, state = self.world(x)

        if self.mode == "fixed":
            route = torch.full(
                (*x.shape[:2], 3),
                1.0 / 3.0,
                device=x.device,
                dtype=x.dtype,
            )
            output_scale = 3.0
        else:
            temperature = self.temperature_log.exp().clamp(0.25, 4.0)
            logits = self.router(torch.cat((x, state), dim=-1)).float() / temperature
            if self.mode == "nomem":
                logits[..., 1] = float("-inf")
                output_scale = 2.0
            elif self.mode == "noworld":
                logits[..., 2] = float("-inf")
                output_scale = 2.0
            else:
                output_scale = 3.0
            route = torch.softmax(logits, dim=-1).to(x.dtype)

        self.last_route = route.detach()
        return output_scale * (
            route[..., 0:1] * attn
            + route[..., 1:2] * mem
            + route[..., 2:3] * world
        )


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        if cfg.architecture == "transformer":
            self.mixer = CausalSelfAttention(cfg.d_model, cfg.n_heads)
        elif cfg.architecture == "tam":
            self.mixer = TAMMixer(cfg.d_model, cfg.n_heads)
        elif cfg.architecture in TAMV2_ARCHITECTURES:
            mode = {
                "tamv2": "full",
                "tamv2_nomem": "nomem",
                "tamv2_noworld": "noworld",
                "tamv2_fixed": "fixed",
            }[cfg.architecture]
            self.mixer = TAMV2Mixer(
                cfg.d_model,
                cfg.n_heads,
                cfg.tamv2_branch_inner,
                cfg.tamv2_state_size,
                mode=mode,
            )
        else:
            raise ValueError(f"unknown architecture: {cfg.architecture}")
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class ResearchLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        if cfg.architecture in TAMV2_ARCHITECTURES:
            for block in self.blocks:
                nn.init.zeros_(block.mixer.router.weight)
                nn.init.zeros_(block.mixer.router.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def router_stats(self) -> dict[str, object] | None:
        if self.cfg.architecture not in TAMV2_ARCHITECTURES:
            return None
        per_layer = []
        for block in self.blocks:
            route = block.mixer.last_route
            if route is None:
                continue
            per_layer.append(route.float().mean(dim=(0, 1)).cpu().tolist())
        if not per_layer:
            return None
        mean = [sum(row[i] for row in per_layer) / len(per_layer) for i in range(3)]
        return {"mean": {"attention": mean[0], "memory": mean[1], "world": mean[2]}, "per_layer": per_layer}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
