from __future__ import annotations

import torch

from experiments.rlt.model import RLTConfig
from experiments.rlt.model_block import BlockRecurrentLoopedTransformer


class PrefixProxyBlockRecurrentLoopedTransformer(BlockRecurrentLoopedTransformer):
    """Parameter-neutral block RLT with a causal intra-block state proxy.

    Plain block recurrence uses the same incoming recurrent state for every token
    inside a block. This variant restores a cheap causal progression without
    serial heavy decoder work: token i receives the incoming block state plus
    the running mean of encoded memory from earlier tokens in the same block.

    The proxy is parameter-free and computed with cumsum, so attention/FFN work
    remains parallel within each block. block_size=1 falls back to the exact
    frozen token-recurrent RLT path inherited from model_block.py.
    """

    def __init__(self, cfg: RLTConfig, *, block_size: int = 4):
        super().__init__(cfg, block_size=block_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.block_size == 1:
            return super().forward(tokens)

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

            # Causal, parallel proxy for the state that would have evolved through
            # earlier tokens in the block. The first token sees exactly the
            # incoming recurrent state; token i sees a running mean of positions
            # [0, i) added in the same d_model state space.
            inclusive = torch.cumsum(block_memory, dim=1)
            prior_sum = inclusive - block_memory
            counts = torch.arange(
                q_len, device=block_memory.device, dtype=block_memory.dtype
            ).view(1, q_len, 1)
            prior_mean = prior_sum / counts.clamp_min(1.0)
            proxy_state = state[:, None, :] + prior_mean

            merged = torch.cat((block_memory, proxy_state), dim=-1)
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
