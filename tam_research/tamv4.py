from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import diagonal_affine_scan


@dataclass(frozen=True)
class TAMV4Config:
    vocab_size: int = 50_257
    d_model: int = 256
    n_layers: int = 15
    n_heads: int = 8
    max_local_positions: int = 1024
    ff_mult: int = 4
    state_size: int = 64
    attention_inner: int = 208
    attention_window: int = 512


@dataclass
class TAMV4State:
    """External per-stream recurrent state.

    State is intentionally explicit so callers cannot accidentally treat it as
    global model memory. A fresh unrelated sequence must start with ``None``.
    """

    layers: tuple[torch.Tensor, ...]

    def detach(self) -> "TAMV4State":
        return TAMV4State(tuple(x.detach() for x in self.layers))


class LocalCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, inner: int, window: int):
        super().__init__()
        if inner % n_heads:
            raise ValueError("attention inner width must be divisible by n_heads")
        if window < 1:
            raise ValueError("attention window must be positive")
        self.n_heads = n_heads
        self.inner = inner
        self.head_dim = inner // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        qi = torch.arange(t, device=x.device).view(t, 1)
        kj = torch.arange(t, device=x.device).view(1, t)
        allowed = (kj <= qi) & ((qi - kj) < self.window)
        mask = torch.full((t, t), float("-inf"), device=x.device, dtype=q.dtype)
        mask = mask.masked_fill(allowed, 0.0)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, t, self.inner)
        return self.out(y)


class PersistentWorldState(nn.Module):
    """Parallel affine recurrence with explicit incoming state and novelty writes."""

    def __init__(self, d_model: int, state_size: int):
        super().__init__()
        self.state_size = state_size
        self.candidate = nn.Linear(d_model, state_size, bias=False)
        self.keep = nn.Linear(d_model, state_size, bias=True)
        self.novelty = nn.Linear(d_model, 1, bias=True)
        self.out = nn.Linear(state_size, d_model, bias=False)
        self.last_write_rate: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, _ = x.shape
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        novelty = torch.sigmoid(self.novelty(x))
        effective_keep = 1.0 - novelty * (1.0 - keep)
        update = (1.0 - effective_keep) * candidate

        state = diagonal_affine_scan(effective_keep, update)
        if initial_state is not None:
            if initial_state.shape != (b, self.state_size):
                raise ValueError(
                    f"initial_state must be {(b, self.state_size)}, got {tuple(initial_state.shape)}"
                )
            prefix_keep = torch.cumprod(effective_keep, dim=1)
            state = state + prefix_keep * initial_state[:, None, :]

        self.last_write_rate = (1.0 - effective_keep).detach()
        final_state = state[:, -1]
        return self.out(state), final_state


class TAMV4Mixer(nn.Module):
    """Dynamic attention/world-state hybrid.

    The reference path computes both branches and learns a token-dependent soft
    gate. It is a correctness/training reference, not yet a sparse-kernel speed
    claim. Hard grouped routing is specified in docs/TAM_V4_ARCHITECTURE.md.
    """

    def __init__(self, cfg: TAMV4Config):
        super().__init__()
        self.attn = LocalCausalSelfAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.attention_inner,
            cfg.attention_window,
        )
        self.world = PersistentWorldState(cfg.d_model, cfg.state_size)
        self.router = nn.Linear(cfg.d_model, 1, bias=True)
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = self.attn(x)
        world, final_state = self.world(x, initial_state)
        gate = torch.sigmoid(self.router(x)).to(x.dtype)
        self.last_gate = gate.detach()
        mixed = 2.0 * ((1.0 - gate) * attn + gate * world)
        return mixed, final_state


class TAMV4Block(nn.Module):
    def __init__(self, cfg: TAMV4Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = TAMV4Mixer(cfg)
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed, final_state = self.mixer(self.norm1(x), initial_state)
        x = x + mixed
        x = x + self.ff(self.norm2(x))
        return x, final_state


class TAMV4LM(nn.Module):
    """Reference TAM v4 language model with explicit cross-chunk state."""

    def __init__(self, cfg: TAMV4Config = TAMV4Config()):
        super().__init__()
        if cfg.attention_window > cfg.max_local_positions:
            raise ValueError("attention_window cannot exceed max_local_positions")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_local_positions, cfg.d_model)
        self.blocks = nn.ModuleList(TAMV4Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.mixer.router.weight)
            nn.init.zeros_(block.mixer.router.bias)
            nn.init.zeros_(block.mixer.world.novelty.weight)
            nn.init.zeros_(block.mixer.world.novelty.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        state: TAMV4State | None = None,
    ) -> tuple[torch.Tensor, TAMV4State]:
        b, t = tokens.shape
        if t > self.cfg.max_local_positions:
            raise ValueError(
                f"chunk length {t} exceeds max_local_positions={self.cfg.max_local_positions}"
            )
        if state is not None and len(state.layers) != len(self.blocks):
            raise ValueError("state layer count does not match model layer count")

        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        next_states: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            incoming = None if state is None else state.layers[i]
            x, outgoing = block(x, incoming)
            next_states.append(outgoing)
        logits = self.lm_head(self.norm(x))
        return logits, TAMV4State(tuple(next_states))

    def forward_stream(
        self,
        tokens: torch.Tensor,
        chunk_size: int,
        state: TAMV4State | None = None,
        *,
        detach_between_chunks: bool = False,
    ) -> tuple[torch.Tensor, TAMV4State]:
        if chunk_size < 1 or chunk_size > self.cfg.max_local_positions:
            raise ValueError("invalid chunk_size")
        outputs: list[torch.Tensor] = []
        current = state
        for start in range(0, tokens.size(1), chunk_size):
            chunk = tokens[:, start : start + chunk_size]
            logits, current = self(chunk, current)
            outputs.append(logits)
            if detach_between_chunks:
                current = current.detach()
        if current is None:
            raise RuntimeError("empty stream is not supported")
        return torch.cat(outputs, dim=1), current

    @torch.no_grad()
    def router_stats(self) -> dict[str, object] | None:
        gates: list[float] = []
        writes: list[float] = []
        for block in self.blocks:
            gate = block.mixer.last_gate
            write = block.mixer.world.last_write_rate
            if gate is not None:
                gates.append(float(gate.float().mean().cpu()))
            if write is not None:
                writes.append(float(write.float().mean().cpu()))
        if not gates:
            return None
        mean_world = sum(gates) / len(gates)
        return {
            "mean": {
                "attention": 1.0 - mean_world,
                "world": mean_world,
                "write_rate": sum(writes) / len(writes) if writes else None,
            },
            "per_layer_world_gate": gates,
            "per_layer_write_rate": writes,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
