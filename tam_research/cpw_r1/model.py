from __future__ import annotations

import torch
import torch.nn as nn

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.models import CausalSelfAttention, parameter_count

RETRIEVAL_BLOCK_INDEX = 14
RETRIEVAL_INNER = 48
RETRIEVAL_HEADS = 1
SEQUENCE_MIXER_PARAMETERS = 49_152
MODEL_PARAMETERS = 21_745_408


class CompetitiveRetrievalMixer(nn.Module):
    """One narrow *ordinary causal softmax attention* mixer, not a novel primitive.

    The 4*(256*48) QKV/output weights exactly match the replaced 256->96->256
    local sequence mixer. Workspace remains inert. A token at t never reads >t.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=RETRIEVAL_HEADS,
            inner=RETRIEVAL_INNER,
        )

    def forward(
        self, x: torch.Tensor, workspace: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attention(x), workspace


class CPWR1ResearchLM(SparseWorldCPWResearchLM):
    """Fourteen unchanged local predictors and one final narrow attention mixer."""

    def __init__(self, cfg: CPWV1Config = CPWV1Config()):
        if cfg.n_layers <= RETRIEVAL_BLOCK_INDEX:
            raise ValueError("CPW-R1 requires at least 15 blocks")
        if cfg.d_model != 256 or cfg.sequence_predictor_rank != 96:
            raise ValueError("frozen CPW-R1 parameter match requires d=256/rank=96")

        super().__init__("sequence_only", cfg)
        self.blocks[RETRIEVAL_BLOCK_INDEX].mixer = CompetitiveRetrievalMixer(
            cfg.d_model
        )
        # The replaced module otherwise receives a different PyTorch default
        # initialization from all the other frozen research branches.
        self.blocks[RETRIEVAL_BLOCK_INDEX].mixer.apply(self._init_weights)
        self.retrieval_layers = (RETRIEVAL_BLOCK_INDEX,)

    def query_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project *only* the final query token for matched memory benchmarks."""
        _, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        workspace = torch.zeros_like(x)
        for block in self.blocks:
            x, workspace = block(x, workspace)
        return self.lm_head(self.norm(x[:, -1]))


def cpw_r1_parameter_count() -> int:
    return parameter_count(CPWR1ResearchLM())
