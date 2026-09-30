from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.models import parameter_count
from tam_research.pgw_v1.model import (
    ChunkLocalAttention,
    PredictiveEventWorkspace,
)

from .protocol import (
    CHUNK_SIZE,
    D_MODEL,
    LOCAL_ATTN_INNER,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    PREDICTOR_RANK,
    SELECTED_EVENTS,
    VOCAB_SIZE,
    WORKSPACE_DIM,
    WORKSPACE_FF_RANK,
    WORKSPACE_HEADS,
    WORKSPACE_LAYERS,
    WORKSPACE_SLOTS,
)


@dataclass(frozen=True)
class PGWV2Config:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    max_seq_len: int = MAX_SEQ_LEN
    ff_mult: int = 4
    architecture: str = "pgwv2"
    local_attn_inner: int = LOCAL_ATTN_INNER
    chunk_size: int = CHUNK_SIZE
    workspace_layers: tuple[int, ...] = WORKSPACE_LAYERS
    workspace_dim: int = WORKSPACE_DIM
    workspace_slots: int = WORKSPACE_SLOTS
    workspace_heads: int = WORKSPACE_HEADS
    predictor_rank: int = PREDICTOR_RANK
    workspace_ff_rank: int = WORKSPACE_FF_RANK
    selected_events: int = SELECTED_EVENTS


class TokenConditionedPredictiveEventWorkspace(PredictiveEventWorkspace):
    """PGW-v1 workspace with token-specific reads from prior workspace slots."""

    def _read_workspace(
        self,
        workspace: torch.Tensor,
        query_source: torch.Tensor,
    ) -> torch.Tensor:
        b, tokens, _ = query_source.shape
        slots = workspace.size(1)
        h = self.workspace_heads
        dh = self.workspace_head_dim
        w = self.workspace_dim

        # Reuse the event-space projection as the token read query. No new
        # trainable parameter is introduced relative to PGW-v1.
        q = self.event_in(query_source).view(b, tokens, h, dh).transpose(1, 2)
        k = self.workspace_k(workspace).view(b, slots, h, dh).transpose(1, 2)
        v = self.workspace_v(workspace).view(b, slots, h, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        read = torch.matmul(weights, v)
        read = read.transpose(1, 2).contiguous().view(b, tokens, w)
        return self.broadcast(read)

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

            if start == 0:
                token_read = torch.zeros_like(pre_read)
            else:
                # Critically, workspace contains only earlier COMPLETED chunks.
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

            indices = torch.topk(
                surprise,
                k=self.selected_events,
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            gather = indices.unsqueeze(-1).expand(-1, -1, d)
            selected = torch.gather(context, dim=1, index=gather)

            # Write only after the current chunk output/read has been fixed.
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


class PGWV2Mixer(nn.Module):
    def __init__(self, cfg: PGWV2Config, *, use_workspace: bool):
        super().__init__()
        self.local = ChunkLocalAttention(cfg)
        self.workspace = (
            TokenConditionedPredictiveEventWorkspace(cfg)
            if use_workspace
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        if self.workspace is None:
            return local
        return self.workspace(x, local)


class PGWV2Block(nn.Module):
    def __init__(self, cfg: PGWV2Config, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = PGWV2Mixer(
            cfg,
            use_workspace=layer_index in cfg.workspace_layers,
        )
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class PGWV2ResearchLM(nn.Module):
    def __init__(self, cfg: PGWV2Config):
        super().__init__()
        if cfg.architecture != "pgwv2":
            raise ValueError("PGWV2ResearchLM only supports architecture='pgwv2'")
        if any(index < 0 or index >= cfg.n_layers for index in cfg.workspace_layers):
            raise ValueError("workspace layer index outside model depth")
        if len(set(cfg.workspace_layers)) != len(cfg.workspace_layers):
            raise ValueError("workspace layer indices must be unique")

        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            PGWV2Block(cfg, layer_index)
            for layer_index in range(cfg.n_layers)
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
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
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


def pgwv2_parameter_count() -> int:
    return parameter_count(PGWV2ResearchLM(PGWV2Config()))


def pgwv2_workspace_parameter_count() -> int:
    return parameter_count(
        TokenConditionedPredictiveEventWorkspace(PGWV2Config())
    )
