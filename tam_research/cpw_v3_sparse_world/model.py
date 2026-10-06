from __future__ import annotations

import torch
import torch.nn as nn

from tam_research.cpw_v1.model import CPWV1Config, LowRankPredictor
from tam_research.cpw_v1_fast.model import TritonScanWorldPredictor
from tam_research.models import parameter_count


ARM_WORLD_LAYERS: dict[str, tuple[int, ...]] = {
    "sequence_only": (),
    "world_last1": (14,),
    "world_2": (7, 14),
    "world_3": (4, 9, 14),
}


class SparseWorldMixer(nn.Module):
    """Exactly one temporal predictor per block.

    Most blocks use the cheap one-step SEQUENCE predictor. Selected blocks swap
    it parameter-for-parameter for the recurrent WORLD predictor, creating a
    full-prefix causal state path without adding model parameters.
    """

    def __init__(self, cfg: CPWV1Config, *, use_world: bool):
        super().__init__()
        self.cfg = cfg
        self.use_world = bool(use_world)
        self.world = (
            TritonScanWorldPredictor(cfg.d_model, cfg.world_state_size)
            if self.use_world
            else None
        )
        self.sequence = (
            None
            if self.use_world
            else LowRankPredictor(cfg.d_model, cfg.sequence_predictor_rank)
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

        if self.sequence is None:
            raise RuntimeError("sparse-world mixer has no predictor")
        previous = torch.cat(
            (torch.zeros_like(x[:, :1]), x[:, :-1]),
            dim=1,
        )
        return self.sequence(previous), workspace


class SparseWorldBlock(nn.Module):
    def __init__(self, cfg: CPWV1Config, *, use_world: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = SparseWorldMixer(cfg, use_world=use_world)
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


class SparseWorldCPWResearchLM(nn.Module):
    def __init__(
        self,
        arm: str,
        cfg: CPWV1Config = CPWV1Config(),
    ):
        super().__init__()
        if arm not in ARM_WORLD_LAYERS:
            raise ValueError(f"unknown CPW-v3 arm: {arm}")
        world_layers = ARM_WORLD_LAYERS[arm]
        if world_layers and max(world_layers) >= cfg.n_layers:
            raise ValueError("configured WORLD layer exceeds model depth")

        self.arm = arm
        self.cfg = cfg
        self.world_layers = tuple(world_layers)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            SparseWorldBlock(cfg, use_world=(i in self.world_layers))
            for i in range(cfg.n_layers)
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
    def router_stats(self) -> dict[str, object]:
        norms: list[float] = []
        for block in self.blocks:
            world = block.mixer.world
            if world is not None and world.last_state_norm is not None:
                norms.append(float(world.last_state_norm.float().cpu()))
        return {
            "mean": {
                "active_predictors": 1.0,
                "world_layers": float(len(self.world_layers)),
                "sequence_layers": float(self.cfg.n_layers - len(self.world_layers)),
                "world_state_norm": sum(norms) / len(norms) if norms else 0.0,
            },
            "world_layer_indices": list(self.world_layers),
        }


def sparse_world_parameter_count(arm: str) -> int:
    return parameter_count(SparseWorldCPWResearchLM(arm))
