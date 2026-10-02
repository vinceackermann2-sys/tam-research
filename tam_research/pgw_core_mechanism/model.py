from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.models import parameter_count
from tam_research.pgw_v1.model import ChunkLocalAttention, PGWV1Config
from tam_research.pgw_v2_mechanism.model import (
    PGWControlConfig,
    RoutingControlWorkspace,
)

from .protocol import (
    CHUNK_SIZE,
    D_MODEL,
    FF_MULT,
    FULL_LOCAL_ATTN_INNER,
    LOCAL_ATTN_INNER,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
    WORKSPACE_LAYERS,
)


@dataclass(frozen=True)
class ChunkLocal256Config(PGWV1Config):
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    max_seq_len: int = MAX_SEQ_LEN
    ff_mult: int = FF_MULT
    architecture: str = "chunk_local_256"
    local_attn_inner: int = FULL_LOCAL_ATTN_INNER
    chunk_size: int = CHUNK_SIZE
    workspace_layers: tuple[int, ...] = ()


@dataclass(frozen=True)
class PGWCoreConfig(PGWControlConfig):
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    max_seq_len: int = MAX_SEQ_LEN
    ff_mult: int = FF_MULT
    architecture: str = "pgw_core_control"
    local_attn_inner: int = LOCAL_ATTN_INNER
    chunk_size: int = CHUNK_SIZE
    workspace_layers: tuple[int, ...] = WORKSPACE_LAYERS
    control_mode: str = "no_workspace"
    core_mode: str = "predictor_reset"


class ChunkLocalOnlyBlock(nn.Module):
    def __init__(self, cfg: ChunkLocal256Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = ChunkLocalAttention(cfg)
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class ChunkLocal256ResearchLM(nn.Module):
    """Transformer-shaped model whose attention is causal only within 64-token chunks."""

    def __init__(self, cfg: ChunkLocal256Config):
        super().__init__()
        if cfg.architecture != "chunk_local_256":
            raise ValueError("ChunkLocal256ResearchLM requires architecture='chunk_local_256'")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(ChunkLocalOnlyBlock(cfg) for _ in range(cfg.n_layers))
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
            raise ValueError("sequence length exceeds max_seq_len")
        if t % self.cfg.chunk_size:
            raise ValueError("sequence length must be divisible by chunk_size")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def router_stats(self) -> None:
        return None


class CoreControlWorkspace(RoutingControlWorkspace):
    """No-workspace PGW core with predictor carry removed or predictor disabled."""

    def __init__(
        self,
        cfg: PGWCoreConfig,
        *,
        layer_index: int,
        core_mode: str,
    ):
        if core_mode not in {"predictor_reset", "local_only"}:
            raise ValueError(f"unsupported core_mode={core_mode!r}")
        super().__init__(
            cfg,
            layer_index=layer_index,
            control_mode="no_workspace",
        )
        self.core_mode = core_mode

    def forward(
        self,
        x: torch.Tensor,
        local: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape != local.shape:
            raise ValueError("workspace input and local output must have equal shapes")
        b, t, d = x.shape
        if t % self.chunk_size:
            raise ValueError("sequence length must be divisible by chunk_size")

        workspace = self.workspace_init.to(dtype=x.dtype)[None, :, :].expand(
            b, -1, -1
        )

        if self.core_mode == "local_only":
            zero = torch.zeros((), device=x.device, dtype=torch.float32)
            self.last_stats = {
                "selected_fraction": 0.0,
                "surprise_mean": zero,
                "surprise_std": zero,
                "workspace_norm": workspace.float().norm(dim=-1).mean().detach(),
            }
            return local

        outputs: list[torch.Tensor] = []
        surprise_records: list[torch.Tensor] = []

        for start in range(0, t, self.chunk_size):
            end = start + self.chunk_size
            x_chunk = x[:, start:end, :]
            local_chunk = local[:, start:end, :]
            context = x_chunk + local_chunk

            # Reset at every chunk boundary: no recurrent/carry information
            # from the preceding completed chunk can enter this predictor.
            zero_previous = torch.zeros(b, 1, d, device=x.device, dtype=x.dtype)
            previous = torch.cat((zero_previous, context[:, :-1, :]), dim=1)
            prediction = self.predict_up(F.gelu(self.predict_down(previous)))
            outputs.append(local_chunk + prediction / math.sqrt(2.0))

            target_norm = F.layer_norm(context.detach().float(), (d,))
            prediction_norm = F.layer_norm(prediction.detach().float(), (d,))
            surprise_records.append(
                (target_norm - prediction_norm).square().mean(dim=-1)
            )

        all_surprise = torch.cat(surprise_records, dim=1)
        self.last_stats = {
            "selected_fraction": 0.0,
            "surprise_mean": all_surprise.mean().detach(),
            "surprise_std": all_surprise.std(unbiased=False).detach(),
            "workspace_norm": workspace.float().norm(dim=-1).mean().detach(),
        }
        return torch.cat(outputs, dim=1)


class PGWCoreMixer(nn.Module):
    def __init__(self, cfg: PGWCoreConfig, *, layer_index: int):
        super().__init__()
        self.local = ChunkLocalAttention(cfg)
        self.workspace = (
            CoreControlWorkspace(
                cfg,
                layer_index=layer_index,
                core_mode=cfg.core_mode,
            )
            if layer_index in cfg.workspace_layers
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        if self.workspace is None:
            return local
        return self.workspace(x, local)


class PGWCoreBlock(nn.Module):
    def __init__(self, cfg: PGWCoreConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = PGWCoreMixer(cfg, layer_index=layer_index)
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class PGWCoreResearchLM(nn.Module):
    def __init__(self, cfg: PGWCoreConfig):
        super().__init__()
        if cfg.architecture != "pgw_core_control":
            raise ValueError("PGWCoreResearchLM requires architecture='pgw_core_control'")
        if cfg.core_mode not in {"predictor_reset", "local_only"}:
            raise ValueError("unsupported PGW core control mode")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            PGWCoreBlock(cfg, i) for i in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            if block.mixer.workspace is not None:
                nn.init.normal_(
                    block.mixer.workspace.workspace_init,
                    mean=0.0,
                    std=0.02,
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if t % self.cfg.chunk_size:
            raise ValueError("sequence length must be divisible by chunk_size")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def router_stats(self) -> dict[str, object] | None:
        rows = [
            block.mixer.workspace.last_stats
            for block in self.blocks
            if block.mixer.workspace is not None
            and block.mixer.workspace.last_stats is not None
        ]
        if not rows:
            return None

        def scalar(value: object) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.float().cpu())
            return float(value)

        keys = (
            "selected_fraction",
            "surprise_mean",
            "surprise_std",
            "workspace_norm",
        )
        per_layer = [{key: scalar(row[key]) for key in keys} for row in rows]
        return {
            "mean": {
                key: sum(row[key] for row in per_layer) / len(per_layer)
                for key in keys
            },
            "per_layer": per_layer,
        }


def chunk_local_256_parameter_count() -> int:
    return parameter_count(ChunkLocal256ResearchLM(ChunkLocal256Config()))


def core_control_parameter_count(mode: str) -> int:
    return parameter_count(PGWCoreResearchLM(PGWCoreConfig(core_mode=mode)))
