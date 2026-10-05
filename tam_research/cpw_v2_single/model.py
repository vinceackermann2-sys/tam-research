from __future__ import annotations

import torch
import torch.nn as nn

from tam_research.cpw_v1.model import CPWV1Config, LowRankPredictor
from tam_research.cpw_v1_fast.model import TritonScanWorldPredictor
from tam_research.models import parameter_count


SINGLE_ARMS = {"sequence_only", "memory_only", "world_only"}


class SinglePredictorMixer(nn.Module):
    def __init__(self, cfg: CPWV1Config, arm: str):
        super().__init__()
        if arm not in SINGLE_ARMS:
            raise ValueError(f"unknown single-predictor arm: {arm}")
        self.cfg = cfg
        self.arm = arm
        self.world = (
            TritonScanWorldPredictor(cfg.d_model, cfg.world_state_size)
            if arm == "world_only"
            else None
        )
        self.sequence = (
            LowRankPredictor(cfg.d_model, cfg.sequence_predictor_rank)
            if arm == "sequence_only"
            else None
        )
        self.memory = (
            LowRankPredictor(cfg.d_model, cfg.memory_predictor_rank)
            if arm == "memory_only"
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.world is not None:
            context = x + self.cfg.workspace_mix * workspace
            world = self.world(context)
            next_workspace = (
                (1.0 - self.cfg.workspace_mix) * workspace
                + self.cfg.workspace_mix * world
            )
            return world, next_workspace

        context = x
        if self.sequence is not None:
            previous = torch.cat(
                (torch.zeros_like(context[:, :1]), context[:, :-1]),
                dim=1,
            )
            return self.sequence(previous), workspace

        if self.memory is not None:
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
            return self.memory(previous_prefix), workspace

        raise RuntimeError("single predictor missing")


class SinglePredictorBlock(nn.Module):
    def __init__(self, cfg: CPWV1Config, arm: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = SinglePredictorMixer(cfg, arm)
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


class SinglePredictorCPWResearchLM(nn.Module):
    def __init__(
        self,
        arm: str,
        cfg: CPWV1Config = CPWV1Config(),
    ):
        super().__init__()
        if arm not in SINGLE_ARMS:
            raise ValueError(f"unknown single-predictor arm: {arm}")
        self.arm = arm
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            SinglePredictorBlock(cfg, arm) for _ in range(cfg.n_layers)
        )
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
            raise ValueError("sequence exceeds max_seq_len")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        workspace = torch.zeros_like(x)
        for block in self.blocks:
            x, workspace = block(x, workspace)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def router_stats(self) -> dict[str, object] | None:
        norms: list[float] = []
        if self.arm == "world_only":
            for block in self.blocks:
                world = block.mixer.world
                if world is not None and world.last_state_norm is not None:
                    norms.append(float(world.last_state_norm.float().cpu()))
        return {
            "mean": {
                "active_predictors": 1.0,
                "world_present": 1.0 if self.arm == "world_only" else 0.0,
                "sequence_present": 1.0 if self.arm == "sequence_only" else 0.0,
                "memory_present": 1.0 if self.arm == "memory_only" else 0.0,
                "world_state_norm": sum(norms) / len(norms) if norms else 0.0,
            }
        }


def single_parameter_count(arm: str) -> int:
    return parameter_count(SinglePredictorCPWResearchLM(arm))
