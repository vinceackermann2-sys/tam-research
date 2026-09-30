from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer


class CarrierRecurrentLoopedTransformer(RecurrentLoopedTransformer):
    """Parameter-neutral RLT variant with a cheap token-recurrent carrier.

    The causal encoder is unchanged. A token-level state still evolves at every
    position using the frozen Merge(e_t, s_(t-1)) + RMSNorm path, but the expensive
    shared decoder stages no longer sit inside that serial recurrence. Instead,
    their self-attention, prefix cross-attention, and FFN execute over the complete
    carrier-state sequence in parallel with explicit causal/local masks.

    No parameters are added or removed relative to RecurrentLoopedTransformer.
    """

    @staticmethod
    def _split_heads(
        x: torch.Tensor, n_heads: int, head_dim: int
    ) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, head_dim).transpose(1, 2)

    def _carrier_sequence(self, memory: torch.Tensor) -> torch.Tensor:
        b, t, _ = memory.shape
        state = self.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
        states: list[torch.Tensor] = []
        for token_index in range(t):
            merged = torch.cat((memory[:, token_index, :], state), dim=-1)
            state = self.merge_norm(self.merge(merged))
            states.append(state)
        return torch.stack(states, dim=1)

    def _parallel_local_self_attention(
        self, stage_index: int, x: torch.Tensor
    ) -> torch.Tensor:
        attn = self.stages[stage_index].self_attn
        q_raw, k_raw, v_raw = attn.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q_raw, attn.n_heads, attn.head_dim)
        k = self._split_heads(k_raw, attn.n_heads, attn.head_dim)
        v = self._split_heads(v_raw, attn.n_heads, attn.head_dim)

        t = x.size(1)
        q_pos = torch.arange(t, device=x.device)
        k_pos = torch.arange(t, device=x.device)
        causal = k_pos[None, :] <= q_pos[:, None]
        local = k_pos[None, :] >= (
            q_pos[:, None] - self.cfg.swa_window + 1
        )
        mask = causal & local

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, t, attn.d_model)
        return attn.out(y)

    def _parallel_prefix_cross_attention(
        self,
        stage_index: int,
        x: torch.Tensor,
        memory_kv: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cross = self.stages[stage_index].cross_attn
        q = self._split_heads(cross.q(x), cross.n_heads, cross.head_dim)
        k, v = memory_kv

        t = x.size(1)
        q_pos = torch.arange(t, device=x.device)
        k_pos = torch.arange(t, device=x.device)
        causal = k_pos[None, :] <= q_pos[:, None]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, t, cross.d_model)
        return cross.out(y)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        memory = self.encode(tokens)
        memory_kv = [
            stage.cross_attn.precompute(memory) for stage in self.stages
        ]

        hidden = self._carrier_sequence(memory)

        for stage_index, stage in enumerate(self.stages):
            hidden = hidden + self._parallel_local_self_attention(
                stage_index, stage.self_norm(hidden)
            )
            hidden = hidden + self._parallel_prefix_cross_attention(
                stage_index,
                stage.cross_norm(hidden),
                memory_kv[stage_index],
            )
            hidden = hidden + stage.ff(stage.ff_norm(hidden))

        self.last_cache_lengths = tuple(
            min(t, self.cfg.swa_window) for _ in self.stages
        )
        return self.lm_head(self.output_norm(hidden))
