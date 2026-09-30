from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.models import parameter_count

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
class PGWV1Config:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    max_seq_len: int = MAX_SEQ_LEN
    ff_mult: int = 4
    architecture: str = "pgwv1"
    local_attn_inner: int = LOCAL_ATTN_INNER
    chunk_size: int = CHUNK_SIZE
    workspace_layers: tuple[int, ...] = WORKSPACE_LAYERS
    workspace_dim: int = WORKSPACE_DIM
    workspace_slots: int = WORKSPACE_SLOTS
    workspace_heads: int = WORKSPACE_HEADS
    predictor_rank: int = PREDICTOR_RANK
    workspace_ff_rank: int = WORKSPACE_FF_RANK
    selected_events: int = SELECTED_EVENTS


class ChunkLocalAttention(nn.Module):
    """Vectorized non-overlapping causal attention inside fixed-size chunks."""

    def __init__(self, cfg: PGWV1Config):
        super().__init__()
        if cfg.local_attn_inner % cfg.n_heads:
            raise ValueError("local_attn_inner must be divisible by n_heads")
        self.d_model = cfg.d_model
        self.inner = cfg.local_attn_inner
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.local_attn_inner // cfg.n_heads
        self.chunk_size = cfg.chunk_size
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.local_attn_inner, bias=False)
        self.out = nn.Linear(cfg.local_attn_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("ChunkLocalAttention expects [batch,time,d_model]")
        b, t, _ = x.shape
        if t % self.chunk_size:
            raise ValueError("sequence length must be divisible by chunk_size")
        chunks = t // self.chunk_size

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape_heads(value: torch.Tensor) -> torch.Tensor:
            value = value.view(
                b,
                chunks,
                self.chunk_size,
                self.n_heads,
                self.head_dim,
            )
            value = value.permute(0, 1, 3, 2, 4).contiguous()
            return value.view(
                b * chunks,
                self.n_heads,
                self.chunk_size,
                self.head_dim,
            )

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.view(
            b,
            chunks,
            self.n_heads,
            self.chunk_size,
            self.head_dim,
        )
        y = y.permute(0, 1, 3, 2, 4).contiguous()
        y = y.view(b, t, self.inner)
        return self.out(y)


class PredictiveEventWorkspace(nn.Module):
    """Sparse delayed cross-chunk workspace for one PGW-v1 workspace block."""

    def __init__(self, cfg: PGWV1Config):
        super().__init__()
        d = cfg.d_model
        w = cfg.workspace_dim
        if w % cfg.workspace_heads:
            raise ValueError("workspace_dim must be divisible by workspace_heads")
        if not 0 < cfg.selected_events <= cfg.chunk_size:
            raise ValueError("selected_events must be within chunk_size")

        self.d_model = d
        self.workspace_dim = w
        self.workspace_slots = cfg.workspace_slots
        self.workspace_heads = cfg.workspace_heads
        self.workspace_head_dim = w // cfg.workspace_heads
        self.chunk_size = cfg.chunk_size
        self.selected_events = cfg.selected_events

        self.predict_down = nn.Linear(d, cfg.predictor_rank, bias=False)
        self.predict_up = nn.Linear(cfg.predictor_rank, d, bias=False)

        self.event_in = nn.Linear(d, w, bias=False)
        self.workspace_q = nn.Linear(w, w, bias=False)
        self.workspace_k = nn.Linear(w, w, bias=False)
        self.workspace_v = nn.Linear(w, w, bias=False)
        self.workspace_out = nn.Linear(w, w, bias=False)
        self.broadcast = nn.Linear(w, d, bias=False)
        self.workspace_init = nn.Parameter(torch.empty(cfg.workspace_slots, w))
        self.workspace_ff_down = nn.Linear(w, cfg.workspace_ff_rank, bias=False)
        self.workspace_ff_up = nn.Linear(cfg.workspace_ff_rank, w, bias=False)

        self.last_stats: dict[str, object] | None = None

    def _update_workspace(
        self,
        workspace: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        b, slots, w = workspace.shape
        event_count = events.size(1)
        h = self.workspace_heads
        dh = self.workspace_head_dim

        q = self.workspace_q(workspace).view(b, slots, h, dh).transpose(1, 2)
        k = self.workspace_k(events).view(b, event_count, h, dh).transpose(1, 2)
        v = self.workspace_v(events).view(b, event_count, h, dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(b, slots, w)

        workspace = F.layer_norm(
            workspace + self.workspace_out(context),
            (w,),
        )
        workspace = F.layer_norm(
            workspace
            + self.workspace_ff_up(F.gelu(self.workspace_ff_down(workspace))),
            (w,),
        )
        return workspace

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
            if start == 0:
                broadcast = torch.zeros(
                    b, 1, d, device=x.device, dtype=x.dtype
                )
            else:
                broadcast = self.broadcast(workspace.mean(dim=1)).unsqueeze(1)

            x_chunk = x[:, start:end, :]
            local_chunk = local[:, start:end, :]
            context = x_chunk + local_chunk + broadcast

            previous = torch.cat(
                (previous_context[:, None, :], context[:, :-1, :]),
                dim=1,
            )
            prediction = self.predict_up(F.gelu(self.predict_down(previous)))
            outputs.append(local_chunk + (prediction + broadcast) / math.sqrt(2.0))

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


class PGWV1Mixer(nn.Module):
    def __init__(self, cfg: PGWV1Config, *, use_workspace: bool):
        super().__init__()
        self.local = ChunkLocalAttention(cfg)
        self.workspace = PredictiveEventWorkspace(cfg) if use_workspace else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        if self.workspace is None:
            return local
        return self.workspace(x, local)


class PGWV1Block(nn.Module):
    def __init__(self, cfg: PGWV1Config, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = PGWV1Mixer(
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


class PGWV1ResearchLM(nn.Module):
    def __init__(self, cfg: PGWV1Config):
        super().__init__()
        if cfg.architecture != "pgwv1":
            raise ValueError("PGWV1ResearchLM only supports architecture='pgwv1'")
        if any(index < 0 or index >= cfg.n_layers for index in cfg.workspace_layers):
            raise ValueError("workspace layer index outside model depth")
        if len(set(cfg.workspace_layers)) != len(cfg.workspace_layers):
            raise ValueError("workspace layer indices must be unique")

        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            PGWV1Block(cfg, layer_index)
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


def pgwv1_parameter_count() -> int:
    return parameter_count(PGWV1ResearchLM(PGWV1Config()))


def pgwv1_local_attention_parameter_count() -> int:
    return parameter_count(ChunkLocalAttention(PGWV1Config()))


def pgwv1_workspace_parameter_count() -> int:
    return parameter_count(PredictiveEventWorkspace(PGWV1Config()))
