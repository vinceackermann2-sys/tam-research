from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.models import diagonal_affine_scan, parameter_count

from .protocol import (
    D_MODEL,
    FF_MULT,
    MAX_SEQ_LEN,
    MEMORY_PREDICTOR_RANK,
    N_LAYERS,
    SEQUENCE_PREDICTOR_RANK,
    VOCAB_SIZE,
    WORKSPACE_MIX,
    WORLD_STATE_SIZE,
)


@dataclass(frozen=True)
class CPWV1Config:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    max_seq_len: int = MAX_SEQ_LEN
    ff_mult: int = FF_MULT
    world_state_size: int = WORLD_STATE_SIZE
    sequence_predictor_rank: int = SEQUENCE_PREDICTOR_RANK
    memory_predictor_rank: int = MEMORY_PREDICTOR_RANK
    workspace_mix: float = WORKSPACE_MIX
    architecture: str = "cpwv1"


class RecurrentWorldPredictor(nn.Module):
    def __init__(self, d_model: int, state_size: int):
        super().__init__()
        self.candidate = nn.Linear(d_model, state_size, bias=False)
        self.keep = nn.Linear(d_model, state_size, bias=False)
        self.out = nn.Linear(state_size, d_model, bias=False)
        self.last_state_norm: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        update = (1.0 - keep) * candidate
        state = diagonal_affine_scan(keep, update)
        self.last_state_norm = state.float().norm(dim=-1).mean().detach()
        return self.out(state)


class LowRankPredictor(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))


class CPWV1Mixer(nn.Module):
    """Attention-free multi-timescale predictive mixer.

    WORLD sees the causal current prefix through a recurrent state.
    SEQUENCE predicts from the immediately previous latent state.
    MEMORY predicts from the running prefix summary ending before this token.
    The three signals are fused with a parameter-free arithmetic mean.
    """

    def __init__(self, cfg: CPWV1Config):
        super().__init__()
        self.cfg = cfg
        self.world = RecurrentWorldPredictor(
            cfg.d_model,
            cfg.world_state_size,
        )
        self.sequence = LowRankPredictor(
            cfg.d_model,
            cfg.sequence_predictor_rank,
        )
        self.memory = LowRankPredictor(
            cfg.d_model,
            cfg.memory_predictor_rank,
        )

    def forward(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape != workspace.shape:
            raise ValueError("workspace must match token-state shape")

        context = x + self.cfg.workspace_mix * workspace
        world = self.world(context)

        previous = torch.cat(
            (torch.zeros_like(context[:, :1]), context[:, :-1]),
            dim=1,
        )
        sequence = self.sequence(previous)

        prefix = torch.cumsum(context, dim=1)
        denom = torch.arange(
            1,
            context.size(1) + 1,
            device=context.device,
            dtype=context.dtype,
        ).view(1, -1, 1)
        prefix = prefix / denom
        previous_prefix = torch.cat(
            (torch.zeros_like(prefix[:, :1]), prefix[:, :-1]),
            dim=1,
        )
        memory = self.memory(previous_prefix)

        mixed = (world + sequence + memory) / 3.0
        next_workspace = (
            (1.0 - self.cfg.workspace_mix) * workspace
            + self.cfg.workspace_mix * world
        )
        return mixed, next_workspace


class CPWV1Block(nn.Module):
    def __init__(self, cfg: CPWV1Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = CPWV1Mixer(cfg)
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed, workspace = self.mixer(self.norm1(x), workspace)
        x = x + mixed
        x = x + self.ff(self.norm2(x))
        return x, workspace


class CPWV1ResearchLM(nn.Module):
    def __init__(self, cfg: CPWV1Config = CPWV1Config()):
        super().__init__()
        if cfg.architecture != "cpwv1":
            raise ValueError("CPWV1ResearchLM requires architecture='cpwv1'")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(CPWV1Block(cfg) for _ in range(cfg.n_layers))
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        workspace = torch.zeros_like(x)
        for block in self.blocks:
            x, workspace = block(x, workspace)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def router_stats(self) -> dict[str, object] | None:
        norms = [
            block.mixer.world.last_state_norm
            for block in self.blocks
            if block.mixer.world.last_state_norm is not None
        ]
        if not norms:
            return None
        vals = [float(v.float().cpu()) for v in norms]
        return {
            "mean": {
                "active_predictors": 3.0,
                "attention_present": 0.0,
                "router_present": 0.0,
                "world_state_norm": sum(vals) / len(vals),
            },
            "per_layer_world_state_norm": vals,
        }


def cpwv1_parameter_count() -> int:
    return parameter_count(CPWV1ResearchLM(CPWV1Config()))
