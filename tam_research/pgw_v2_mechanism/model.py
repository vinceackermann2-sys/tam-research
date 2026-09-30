from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.models import parameter_count
from tam_research.pgw_v1.model import ChunkLocalAttention
from tam_research.pgw_v2.model import (
    PGWV2Config,
    TokenConditionedPredictiveEventWorkspace,
)

from .protocol import FIXED_RANDOM_POSITIONS, RECENCY_POSITIONS


@dataclass(frozen=True)
class PGWControlConfig(PGWV2Config):
    architecture: str = "pgw_control"
    control_mode: str = "fixed_random"


class RoutingControlWorkspace(TokenConditionedPredictiveEventWorkspace):
    """PGW-v2 workspace with preregistered routing controls."""

    def __init__(
        self,
        cfg: PGWControlConfig,
        *,
        layer_index: int,
        control_mode: str,
    ):
        super().__init__(cfg)
        if control_mode not in {"fixed_random", "recency", "no_workspace"}:
            raise ValueError(f"unsupported control_mode={control_mode!r}")
        self.layer_index = int(layer_index)
        self.control_mode = control_mode
        if control_mode == "fixed_random":
            if self.layer_index not in FIXED_RANDOM_POSITIONS:
                raise ValueError("fixed-random control used outside frozen workspace layers")
            frozen = FIXED_RANDOM_POSITIONS[self.layer_index]
            if len(frozen) != self.selected_events:
                raise ValueError("fixed-random routing count mismatch")
            if any(p < 0 or p >= self.chunk_size for p in frozen):
                raise ValueError("fixed-random position outside chunk")
        if control_mode == "recency":
            if len(RECENCY_POSITIONS) != self.selected_events:
                raise ValueError("recency routing count mismatch")

    def _control_indices(
        self,
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.control_mode == "fixed_random":
            positions = FIXED_RANDOM_POSITIONS[self.layer_index]
        elif self.control_mode == "recency":
            positions = RECENCY_POSITIONS
        else:
            raise RuntimeError("no-workspace control has no routed indices")
        base = torch.tensor(positions, device=device, dtype=torch.long)
        return base[None, :].expand(batch, -1)

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
        previous_context = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        outputs: list[torch.Tensor] = []
        surprise_records: list[torch.Tensor] = []
        selected_total = 0

        for start in range(0, t, self.chunk_size):
            end = start + self.chunk_size
            x_chunk = x[:, start:end, :]
            local_chunk = local[:, start:end, :]
            pre_read = x_chunk + local_chunk

            if self.control_mode == "no_workspace" or start == 0:
                token_read = torch.zeros_like(pre_read)
            else:
                token_read = self._read_workspace(workspace, pre_read)

            context = pre_read + token_read
            previous = torch.cat(
                (previous_context[:, None, :], context[:, :-1, :]),
                dim=1,
            )
            prediction = self.predict_up(F.gelu(self.predict_down(previous)))
            outputs.append(
                local_chunk + (prediction + token_read) / math.sqrt(2.0)
            )

            target_norm = F.layer_norm(context.detach().float(), (d,))
            prediction_norm = F.layer_norm(prediction.detach().float(), (d,))
            surprise = (target_norm - prediction_norm).square().mean(dim=-1)
            surprise_records.append(surprise)

            if self.control_mode != "no_workspace":
                indices = self._control_indices(batch=b, device=x.device)
                gather = indices.unsqueeze(-1).expand(-1, -1, d)
                selected = torch.gather(context, dim=1, index=gather)
                workspace = self._update_workspace(
                    workspace,
                    self.event_in(selected),
                )
                selected_total += b * self.selected_events

            previous_context = context[:, -1, :]

        all_surprise = torch.cat(surprise_records, dim=1)
        self.last_stats = {
            "selected_fraction": selected_total / max(b * t, 1),
            "surprise_mean": all_surprise.mean().detach(),
            "surprise_std": all_surprise.std(unbiased=False).detach(),
            "workspace_norm": workspace.float().norm(dim=-1).mean().detach(),
        }
        return torch.cat(outputs, dim=1)


class PGWControlMixer(nn.Module):
    def __init__(self, cfg: PGWControlConfig, *, layer_index: int):
        super().__init__()
        self.local = ChunkLocalAttention(cfg)
        self.workspace = (
            RoutingControlWorkspace(
                cfg,
                layer_index=layer_index,
                control_mode=cfg.control_mode,
            )
            if layer_index in cfg.workspace_layers
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        if self.workspace is None:
            return local
        return self.workspace(x, local)


class PGWControlBlock(nn.Module):
    def __init__(self, cfg: PGWControlConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = PGWControlMixer(cfg, layer_index=layer_index)
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class PGWControlResearchLM(nn.Module):
    def __init__(self, cfg: PGWControlConfig):
        super().__init__()
        if cfg.architecture != "pgw_control":
            raise ValueError("PGWControlResearchLM requires architecture='pgw_control'")
        if cfg.control_mode not in {"fixed_random", "recency", "no_workspace"}:
            raise ValueError("unsupported PGW mechanism control")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            PGWControlBlock(cfg, i) for i in range(cfg.n_layers)
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
        keys = (
            "selected_fraction",
            "surprise_mean",
            "surprise_std",
            "workspace_norm",
        )

        def scalar(value: object) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.float().cpu())
            return float(value)

        per_layer = [{key: scalar(row[key]) for key in keys} for row in rows]
        return {
            "mean": {
                key: sum(row[key] for row in per_layer) / len(per_layer)
                for key in keys
            },
            "per_layer": per_layer,
        }


def control_parameter_count(mode: str) -> int:
    cfg = PGWControlConfig(control_mode=mode)
    return parameter_count(PGWControlResearchLM(cfg))
