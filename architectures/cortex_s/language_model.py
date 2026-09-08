from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CortexSLMConfig:
    """Frozen 25M-class CORTEX-S v0 language configuration.

    At the defaults this model has 25,022,776 trainable parameters versus
    24,940,288 for the repository's 25M Transformer: +0.3307%.
    """

    vocab_size: int = 50_257
    d_model: int = 256
    n_layers: int = 15
    n_heads: int = 8
    max_seq_len: int = 1024
    state_size: int = 64
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 168
    attention_every: int = 5

    def validate(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.state_size <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.attention_every <= 0:
            raise ValueError("attention_every must be positive")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(y)


def affine_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inclusive parallel affine scan for s_t = a_t*s_(t-1) + b_t.

    This is O(log sequence_length) sequential Python stages instead of one Python
    stage per token, while preserving an explicit state that can be carried across
    chunks. The arithmetic is differentiable and causal.
    """

    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must have matching [batch, sequence, state] shapes")
    length = a.size(1)
    pa, pb = a, b
    offset = 1
    while offset < length:
        ones = torch.ones_like(pa[:, :offset])
        zeros = torch.zeros_like(pb[:, :offset])
        a_shift = torch.cat((ones, pa[:, :-offset]), dim=1)
        b_shift = torch.cat((zeros, pb[:, :-offset]), dim=1)
        old_a = pa
        pb = pb + old_a * b_shift
        pa = old_a * a_shift
        offset <<= 1
    if initial is not None:
        if initial.shape != (a.size(0), a.size(2)):
            raise ValueError("initial state has wrong shape")
        pb = pb + pa * initial[:, None, :]
    return pb


class PersistentWorldState(nn.Module):
    """Small causal latent state with an explicit chunk-to-chunk carry."""

    def __init__(self, d_model: int, state_size: int):
        super().__init__()
        self.state_size = state_size
        self.candidate = nn.Linear(d_model, state_size, bias=False)
        self.keep = nn.Linear(d_model, state_size, bias=True)
        self.out = nn.Linear(state_size, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        if state is None:
            state = torch.zeros(
                x.size(0),
                self.state_size,
                device=x.device,
                dtype=x.dtype,
            )
        states = affine_scan(keep, (1.0 - keep) * candidate, state)
        return self.out(states), states[:, -1], states


class TrulySparseMoE(nn.Module):
    """Token-routed MoE that executes only the selected expert-token pairs.

    Unlike the tiny reference block in ``model.py``, this implementation does not
    evaluate every expert for every token and gather afterwards. For each expert it
    gathers only tokens routed to that expert, executes that expert on the selected
    rows, and scatters weighted outputs back. With 8 experts/top-2, exactly 25% of
    possible expert-token assignments are executed (ignoring empty-route overhead).

    This is deliberately simple rather than claiming an optimal fused GPU kernel.
    Throughput is measured rather than assumed.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=True)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, d_model, bias=False),
            )
            for _ in range(num_experts)
        )
        self.last_counts: Optional[list[int]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)

        output = torch.zeros_like(flat)
        counts: list[int] = []
        for expert_index, expert in enumerate(self.experts):
            hits = (top_indices == expert_index).nonzero(as_tuple=False)
            counts.append(int(hits.size(0)))
            if hits.numel() == 0:
                continue
            token_indices = hits[:, 0]
            slots = hits[:, 1]
            selected = flat.index_select(0, token_indices)
            expert_output = expert(selected)
            weighted = expert_output * weights[token_indices, slots, None]
            output.index_add_(0, token_indices, weighted)

        self.last_counts = counts
        return output.view(original_shape)

    @property
    def theoretical_executed_fraction(self) -> float:
        return self.top_k / self.num_experts


class CortexSBlock(nn.Module):
    def __init__(self, cfg: CortexSLMConfig, index: int):
        super().__init__()
        self.has_attention = (index + 1) % cfg.attention_every == 0
        self.norm_world = nn.LayerNorm(cfg.d_model)
        self.world = PersistentWorldState(cfg.d_model, cfg.state_size)
        if self.has_attention:
            self.norm_attention = nn.LayerNorm(cfg.d_model)
            self.attention = CausalSelfAttention(cfg.d_model, cfg.n_heads)
        self.norm_moe = nn.LayerNorm(cfg.d_model)
        self.moe = TrulySparseMoE(
            cfg.d_model,
            cfg.num_experts,
            cfg.top_k,
            cfg.expert_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor],
        *,
        disable_world: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_delta, new_state, _ = self.world(self.norm_world(x), state)
        if not disable_world:
            x = x + world_delta
        if self.has_attention:
            x = x + self.attention(self.norm_attention(x))
        x = x + self.moe(self.norm_moe(x))
        return x, new_state


class CortexSLM(nn.Module):
    """CORTEX-S v0 autoregressive language model.

    Architecture bundle:
      * explicit per-layer recurrent world state;
      * state can persist across chunks during streaming evaluation/inference;
      * true top-k token-level conditional expert execution;
      * only every fifth layer uses full causal attention (3/15 at 25M);
      * tied token embedding/output weights.

    Language training intentionally uses ordinary next-token loss so improvements
    cannot be attributed to extra supervision unavailable to matched baselines.
    """

    def __init__(self, cfg: CortexSLMConfig = CortexSLMConfig()):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(CortexSBlock(cfg, index) for index in range(cfg.n_layers))
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

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, ...]:
        parameter = next(self.parameters())
        resolved_device = parameter.device if device is None else device
        resolved_dtype = parameter.dtype if dtype is None else dtype
        return tuple(
            torch.zeros(
                batch_size,
                self.cfg.state_size,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            for _ in range(self.cfg.n_layers)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        state: Optional[tuple[torch.Tensor, ...]] = None,
        return_state: bool = False,
        disable_world: bool = False,
    ):
        batch, length = tokens.shape
        if length > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {length} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        if state is None:
            state = self.initial_state(batch, device=tokens.device, dtype=self.token_emb.weight.dtype)
        if len(state) != self.cfg.n_layers:
            raise ValueError("state tuple length does not match n_layers")

        positions = torch.arange(length, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        next_state = []
        for block, block_state in zip(self.blocks, state):
            x, block_next = block(x, block_state, disable_world=disable_world)
            next_state.append(block_next)
        logits = self.lm_head(self.norm(x))
        if return_state:
            return logits, tuple(next_state)
        return logits

    @torch.no_grad()
    def router_stats(self) -> dict[str, object]:
        per_layer = []
        for index, block in enumerate(self.blocks):
            counts = block.moe.last_counts
            if counts is None:
                continue
            total = sum(counts)
            fractions = [count / total if total else 0.0 for count in counts]
            per_layer.append({"layer": index, "counts": counts, "fractions": fractions})
        return {
            "theoretical_executed_fraction": self.cfg.top_k / self.cfg.num_experts,
            "per_layer": per_layer,
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
