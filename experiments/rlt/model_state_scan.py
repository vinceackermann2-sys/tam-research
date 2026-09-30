from __future__ import annotations

import torch

from experiments.rlt.model import RLTConfig
from experiments.rlt.model_block import BlockRecurrentLoopedTransformer


class StateScanBlockRecurrentLoopedTransformer(BlockRecurrentLoopedTransformer):
    """Parameter-neutral block RLT with a cheap per-token recurrent state scan.

    Expensive decoder self-attention, cross-attention, and FFN execute in parallel
    over each causal token block. Before that decoder block, the existing RLT
    Merge(memory_t, state_(t-1)) projection is applied sequentially token-by-token
    inside the block. This restores a per-token recurrent information path without
    putting the expensive decoder stack back on the serial critical path.

    The decoded final token of each block becomes the carry state for the next block,
    so expensive decoder information still feeds future blocks. No parameters are
    added or removed relative to the frozen RLT.

    block_size=1 dispatches through the inherited exact frozen-RLT fallback.
    """

    def __init__(self, cfg: RLTConfig, *, block_size: int = 4):
        super().__init__(cfg, block_size=block_size)

    def _scan_merge(
        self,
        block_memory: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        scanned: list[torch.Tensor] = []
        scan_state = state
        q_len = block_memory.size(1)
        for local_index in range(q_len):
            merged = torch.cat(
                (block_memory[:, local_index, :], scan_state),
                dim=-1,
            )
            scan_state = self.merge_norm(self.merge(merged))
            scanned.append(scan_state.unsqueeze(1))
        return torch.cat(scanned, dim=1)

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

            # Cheap token-level recurrence using the frozen RLT merge path.
            hidden = self._scan_merge(block_memory, state)

            # Expensive work remains block-parallel and causally masked.
            for stage_index in range(len(self.stages)):
                hidden, caches[stage_index] = self._decode_block(
                    stage_index,
                    hidden,
                    caches[stage_index],
                    memory_kv[stage_index],
                    block_start=block_start,
                )

            # Decoder information feeds the next block's token-level state scan.
            state = hidden[:, -1, :]
            logits_blocks.append(self.lm_head(self.output_norm(hidden)))

        self.last_cache_lengths = tuple(
            0 if cache is None else int(cache[0].size(2)) for cache in caches
        )
        return torch.cat(logits_blocks, dim=1)
