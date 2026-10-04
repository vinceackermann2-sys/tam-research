from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer


class ScanRecurrentLoopedTransformer(RecurrentLoopedTransformer):
    """Parameter-neutral token-state RLT with a parallelizable gated recurrence.

    The frozen token RLT's useful state is updated at every token, but its nonlinear
    transition forces a Python/token serial chain. This research variant keeps one
    recurrent state per token while changing the state transition to an elementwise
    affine recurrence:

        s_t = g_t * s_(t-1) + (1-g_t) * c_t

    where c_t and g_t are both functions of the causal encoder memory at token t.
    Because (g, u) affine transforms compose associatively, all token states can be
    evaluated with differentiable prefix tensor operations instead of a token loop.

    No parameters are added or removed. The two halves of the frozen RLT merge
    matrix are repurposed as candidate and gate projections respectively.
    """

    GATE_FLOOR = 0.5
    GATE_CEILING = 0.99

    @staticmethod
    def _split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, head_dim).transpose(1, 2)

    def _scan_states(self, memory: torch.Tensor) -> torch.Tensor:
        d = self.cfg.d_model
        weight = self.merge.weight
        candidate = torch.tanh(F.linear(memory, weight[:, :d]))
        gate_logits = F.linear(memory, weight[:, d:])
        gate = self.GATE_FLOOR + (
            self.GATE_CEILING - self.GATE_FLOOR
        ) * torch.sigmoid(gate_logits)

        # Evaluate the associative affine recurrence in fp32 for numerical range,
        # then return states in the model activation dtype. With gate >= 0.5 and
        # seq_len <= 128, prefix products remain representable in fp32.
        g = gate.float()
        u = ((1.0 - gate) * candidate).float()
        prefix_product = torch.cumprod(g, dim=1)
        prefix_sum = torch.cumsum(u / prefix_product, dim=1)
        s0 = self.start_state.float().view(1, 1, -1)
        states = prefix_product * (s0 + prefix_sum)
        return states.to(memory.dtype)

    def _parallel_self_attention(self, stage_index: int, x: torch.Tensor) -> torch.Tensor:
        attn = self.stages[stage_index].self_attn
        q_raw, k_raw, v_raw = attn.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q_raw, attn.n_heads, attn.head_dim)
        k = self._split_heads(k_raw, attn.n_heads, attn.head_dim)
        v = self._split_heads(v_raw, attn.n_heads, attn.head_dim)

        t = x.size(1)
        q_pos = torch.arange(t, device=x.device)
        k_pos = torch.arange(t, device=x.device)
        causal = k_pos[None, :] <= q_pos[:, None]
        local = k_pos[None, :] >= (q_pos[:, None] - self.cfg.swa_window + 1)
        mask = causal & local

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, t, attn.d_model)
        return attn.out(y)

    def _parallel_cross_attention(
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
        mask = k_pos[None, :] <= q_pos[:, None]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, t, cross.d_model)
        return cross.out(y)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        memory = self.encode(tokens)
        states = self._scan_states(memory)

        # Preserve a direct current-token path in addition to the recurrent state.
        hidden = self.merge_norm(memory + states)
        memory_kv = [
            stage.cross_attn.precompute(memory) for stage in self.stages
        ]

        for stage_index, stage in enumerate(self.stages):
            hidden = hidden + self._parallel_self_attention(
                stage_index, stage.self_norm(hidden)
            )
            hidden = hidden + self._parallel_cross_attention(
                stage_index,
                stage.cross_norm(hidden),
                memory_kv[stage_index],
            )
            hidden = hidden + stage.ff(stage.ff_norm(hidden))

        self.last_cache_lengths = tuple(
            min(t, self.cfg.swa_window) for _ in self.stages
        )
        return self.lm_head(self.output_norm(hidden))
