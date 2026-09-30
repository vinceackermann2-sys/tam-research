from __future__ import annotations

import torch
import torch.nn as nn

from experiments.rlt.model import (
    RLTConfig,
    RecurrentLoopedTransformer,
    RMSNorm,
    SharedSelfAttention,
    CrossAttention,
)


ONE_STAGE_FF_INNER = 3073


class WideOneStage(nn.Module):
    """Single shared encoder/decoder stage with a widened FFN."""

    def __init__(self, cfg: RLTConfig):
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = SharedSelfAttention(cfg.d_model, cfg.n_heads)
        self.cross_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.cross_attn = CrossAttention(cfg.d_model, cfg.n_heads)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ONE_STAGE_FF_INNER, bias=False),
            nn.GELU(),
            nn.Linear(ONE_STAGE_FF_INNER, cfg.d_model, bias=False),
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
        x = x + self.cross_attn.step(self.cross_norm(x), memory_kv, prefix_len)
        x = x + self.ff(self.ff_norm(x))
        return x, new_cache


class OneStageRecurrentLoopedTransformer(RecurrentLoopedTransformer):
    """Token-recurrent RLT with one sequential stage instead of two.

    The token-level recurrence, causal encoder memory, cross-attention prefix rule,
    SWA cache, shared encoder/decoder stage weights, tied LM head, and full BPTT
    are preserved. Capacity removed with the second stage is reallocated into a
    wider FFN. A learned per-channel state gate contributes exactly d_model
    parameters and actively modulates the recurrent state before Merge.

    At the established d_model=256 / max_seq_len=128 configuration:
      ff_inner=3073 + state_gate[256] => exactly 15,129,344 parameters.
    """

    def __init__(self, cfg: RLTConfig):
        if cfg.n_stages != 1:
            raise ValueError("OneStageRecurrentLoopedTransformer requires n_stages=1")
        if cfg.d_model != 256:
            raise ValueError("exact-matched prototype is frozen at d_model=256")
        super().__init__(cfg)

        # Replace the default 4*d FF stage created by the reference constructor
        # with the exact-parameter one-stage capacity allocation.
        self.stages = nn.ModuleList([WideOneStage(cfg)])
        self.state_gate = nn.Parameter(torch.ones(cfg.d_model))
        self.stages.apply(self._init_weights)

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
            gated_state = state * self.state_gate.to(state.dtype)
            merged = torch.cat((memory[:, token_index, :], gated_state), dim=-1)
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
