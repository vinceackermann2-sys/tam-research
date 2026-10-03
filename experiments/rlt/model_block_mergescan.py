from __future__ import annotations

import torch

from experiments.rlt.model import RLTConfig
from experiments.rlt.model_block import BlockRecurrentLoopedTransformer


class MergeScanBlockRecurrentLoopedTransformer(BlockRecurrentLoopedTransformer):
    """Parameter-neutral block RLT with a cheap causal merge-state scan.

    The expensive decoder stack remains block-parallel. Before each block decode,
    the existing learned Merge(memory_t, state) transformation is scanned across
    tokens inside the block. This gives every token a recurrent state prior while
    avoiding token-by-token self-attention/cross-attention/FFN execution.

    No parameters are added or removed relative to the frozen RLT.
    """

    def __init__(self, cfg: RLTConfig, *, block_size: int = 4):
        super().__init__(cfg, block_size=block_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Preserve the exact frozen implementation as the block-size-1 anchor.
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

            # Cheap sequential recurrence only through the existing merge path.
            # Token i receives a state that already contains tokens < i from
            # this block, while the expensive decoder remains block-parallel.
            scan_state = state
            priors: list[torch.Tensor] = []
            for local_index in range(q_len):
                merged = torch.cat(
                    (block_memory[:, local_index, :], scan_state), dim=-1
                )
                scan_state = self.merge_norm(self.merge(merged))
                priors.append(scan_state)

            hidden = torch.stack(priors, dim=1)

            for stage_index in range(len(self.stages)):
                hidden, caches[stage_index] = self._decode_block(
                    stage_index,
                    hidden,
                    caches[stage_index],
                    memory_kv[stage_index],
                    block_start=block_start,
                )

            # Preserve the original RLT notion that recurrent state is the
            # decoder output, but only at the block boundary.
            state = hidden[:, -1, :]
            logits_blocks.append(self.lm_head(self.output_norm(hidden)))

        self.last_cache_lengths = tuple(
            0 if cache is None else int(cache[0].size(2)) for cache in caches
        )
        return torch.cat(logits_blocks, dim=1)
