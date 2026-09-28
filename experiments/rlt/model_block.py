from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer


class BlockRecurrentLoopedTransformer(RecurrentLoopedTransformer):
    """Block-recurrent RLT research variant.

    This preserves the frozen RLT encoder, parameters, encoder/decoder weight sharing,
    decoder SWA cache, cross-attention, tied LM head, and full BPTT. The deliberate
    architectural change is the recurrence granularity: one recurrent state is carried
    between causal token blocks instead of between individual tokens. Tokens inside a
    block are processed in parallel with explicit causal/local masks.

    No parameters are added or removed relative to RecurrentLoopedTransformer with the
    same RLTConfig.
    """

    def __init__(self, cfg: RLTConfig, *, block_size: int = 8):
        super().__init__(cfg)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)

    @staticmethod
    def _split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, head_dim).transpose(1, 2)

    def _self_attention_block(
        self,
        stage_index: int,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn = self.stages[stage_index].self_attn
        q_raw, k_raw, v_raw = attn.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q_raw, attn.n_heads, attn.head_dim)
        k_new = self._split_heads(k_raw, attn.n_heads, attn.head_dim)
        v_new = self._split_heads(v_raw, attn.n_heads, attn.head_dim)

        if cache is None:
            old_k = k_new[:, :, :0, :]
            old_v = v_new[:, :, :0, :]
        else:
            old_k, old_v = cache

        k_all = torch.cat((old_k, k_new), dim=2)
        v_all = torch.cat((old_v, v_new), dim=2)
        old_len = old_k.size(2)
        q_len = q.size(2)

        # Crop keys that are outside the window even for the first query in the
        # block. For block_size=1 this makes the SDPA input shape exactly match
        # the frozen token-step implementation before attention is called.
        crop_start = max(0, old_len - self.cfg.swa_window + 1)
        if crop_start:
            k_all = k_all[:, :, crop_start:, :]
            v_all = v_all[:, :, crop_start:, :]
        total_len = k_all.size(2)

        q_pos = old_len + torch.arange(q_len, device=x.device) - crop_start
        k_pos = torch.arange(total_len, device=x.device)
        causal = k_pos[None, :] <= q_pos[:, None]
        local = k_pos[None, :] >= (q_pos[:, None] - self.cfg.swa_window + 1)
        mask = causal & local

        y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, q_len, attn.d_model)
        y = attn.out(y)

        if total_len > self.cfg.swa_window:
            new_k = k_all[:, :, -self.cfg.swa_window :, :]
            new_v = v_all[:, :, -self.cfg.swa_window :, :]
        else:
            new_k, new_v = k_all, v_all
        return y, (new_k, new_v)

    def _cross_attention_block(
        self,
        stage_index: int,
        x: torch.Tensor,
        memory_kv: tuple[torch.Tensor, torch.Tensor],
        *,
        block_start: int,
    ) -> torch.Tensor:
        cross = self.stages[stage_index].cross_attn
        q = self._split_heads(cross.q(x), cross.n_heads, cross.head_dim)
        q_len = q.size(2)
        prefix_end = block_start + q_len
        k = memory_kv[0][:, :, :prefix_end, :]
        v = memory_kv[1][:, :, :prefix_end, :]

        q_global = block_start + torch.arange(q_len, device=x.device)
        k_global = torch.arange(prefix_end, device=x.device)
        mask = k_global[None, :] <= q_global[:, None]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b = x.size(0)
        y = y.transpose(1, 2).contiguous().view(b, q_len, cross.d_model)
        return cross.out(y)

    def _decode_block(
        self,
        stage_index: int,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        memory_kv: tuple[torch.Tensor, torch.Tensor],
        *,
        block_start: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        stage = self.stages[stage_index]
        local, new_cache = self._self_attention_block(
            stage_index, stage.self_norm(x), cache
        )
        x = x + local
        x = x + self._cross_attention_block(
            stage_index,
            stage.cross_norm(x),
            memory_kv,
            block_start=block_start,
        )
        x = x + stage.ff(stage.ff_norm(x))
        return x, new_cache

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        memory = self.encode(tokens)
        memory_kv = [
            stage.cross_attn.precompute(memory) for stage in self.stages
        ]

        state = self.start_state.to(memory.dtype).view(1, -1).expand(b, -1)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.stages
        ]
        logits_blocks: list[torch.Tensor] = []

        for block_start in range(0, t, self.block_size):
            block_end = min(t, block_start + self.block_size)
            block_memory = memory[:, block_start:block_end, :]
            q_len = block_end - block_start

            state_expanded = state[:, None, :].expand(-1, q_len, -1)
            merged = torch.cat((block_memory, state_expanded), dim=-1)
            hidden = self.merge_norm(self.merge(merged))

            for stage_index in range(len(self.stages)):
                hidden, caches[stage_index] = self._decode_block(
                    stage_index,
                    hidden,
                    caches[stage_index],
                    memory_kv[stage_index],
                    block_start=block_start,
                )

            state = hidden[:, -1, :]
            logits_blocks.append(self.lm_head(self.output_norm(hidden)))

        self.last_cache_lengths = tuple(
            0 if cache is None else int(cache[0].size(2)) for cache in caches
        )
        return torch.cat(logits_blocks, dim=1)
