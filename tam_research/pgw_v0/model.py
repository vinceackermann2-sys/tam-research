from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.models import diagonal_affine_scan, parameter_count

from .protocol import (
    AUTO_RANK,
    D_MODEL,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    SEGMENT_SIZE,
    SELECTED_EVENTS,
    STATE_SIZE,
    VOCAB_SIZE,
    WORKSPACE_DIM,
    WORKSPACE_HEADS,
    WORKSPACE_SLOTS,
)


@dataclass(frozen=True)
class PGWConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    max_seq_len: int = MAX_SEQ_LEN
    ff_mult: int = 4
    architecture: str = "pgw"
    state_size: int = STATE_SIZE
    workspace_dim: int = WORKSPACE_DIM
    workspace_slots: int = WORKSPACE_SLOTS
    workspace_heads: int = WORKSPACE_HEADS
    segment_size: int = SEGMENT_SIZE
    selected_events: int = SELECTED_EVENTS
    auto_rank: int = AUTO_RANK


class PGWMixer(nn.Module):
    """Predictive recurrent substrate + surprise-selected delayed global workspace."""

    def __init__(self, cfg: PGWConfig):
        super().__init__()
        d = cfg.d_model
        s = cfg.state_size
        w = cfg.workspace_dim
        if w % cfg.workspace_heads:
            raise ValueError("workspace_dim must be divisible by workspace_heads")
        if not 0 < cfg.selected_events <= cfg.segment_size:
            raise ValueError("selected_events must be within segment_size")

        self.d_model = d
        self.state_size = s
        self.workspace_dim = w
        self.workspace_slots = cfg.workspace_slots
        self.workspace_heads = cfg.workspace_heads
        self.workspace_head_dim = w // cfg.workspace_heads
        self.segment_size = cfg.segment_size
        self.selected_events = cfg.selected_events

        self.candidate = nn.Linear(d, s, bias=False)
        self.keep = nn.Linear(d, s, bias=True)
        self.predict = nn.Linear(s, d, bias=False)

        self.auto_down = nn.Linear(d, cfg.auto_rank, bias=False)
        self.auto_up = nn.Linear(cfg.auto_rank, d, bias=False)

        self.event_in = nn.Linear(d, w, bias=False)
        self.workspace_q = nn.Linear(w, w, bias=False)
        self.workspace_k = nn.Linear(w, w, bias=False)
        self.workspace_v = nn.Linear(w, w, bias=False)
        self.workspace_out = nn.Linear(w, w, bias=False)
        self.broadcast = nn.Linear(w, d, bias=False)
        self.workspace_init = nn.Parameter(torch.empty(cfg.workspace_slots, w))

        self.last_stats: dict[str, object] | None = None

    def _scan_with_initial(
        self,
        keep: torch.Tensor,
        innovation: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        local = diagonal_affine_scan(keep, innovation)
        prefix_keep = torch.cumprod(keep, dim=1)
        return local + prefix_keep * initial[:, None, :]

    def _update_workspace(
        self,
        workspace: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        b, slots, w = workspace.shape
        event_count = events.shape[1]
        h = self.workspace_heads
        dh = self.workspace_head_dim

        q = self.workspace_q(workspace).view(b, slots, h, dh).transpose(1, 2)
        k = self.workspace_k(events).view(b, event_count, h, dh).transpose(1, 2)
        v = self.workspace_v(events).view(b, event_count, h, dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(b, slots, w)
        updated = workspace + self.workspace_out(context)
        return F.layer_norm(updated, (w,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("PGWMixer expects [batch,time,d_model]")
        b, t, d = x.shape
        dtype = x.dtype
        device = x.device

        state_prev = torch.zeros(b, self.state_size, device=device, dtype=dtype)
        workspace = self.workspace_init.to(dtype=dtype)[None, :, :].expand(b, -1, -1)
        outputs: list[torch.Tensor] = []
        surprise_records: list[torch.Tensor] = []
        selected_total = 0

        for start in range(0, t, self.segment_size):
            end = min(start + self.segment_size, t)
            x_segment = x[:, start:end, :]

            if start == 0:
                broadcast = torch.zeros(b, 1, d, device=device, dtype=dtype)
            else:
                broadcast = self.broadcast(workspace.mean(dim=1)).unsqueeze(1)

            substrate_input = x_segment + broadcast
            candidate = torch.tanh(self.candidate(substrate_input))
            keep = torch.sigmoid(self.keep(substrate_input))
            innovation = (1.0 - keep) * candidate
            state = self._scan_with_initial(keep, innovation, state_prev)

            previous_states = torch.cat(
                (state_prev[:, None, :], state[:, :-1, :]),
                dim=1,
            )
            prediction = self.predict(previous_states)
            automatic = prediction + self.auto_up(F.gelu(self.auto_down(prediction)))
            outputs.append((automatic + broadcast) / math.sqrt(2.0))

            target = F.layer_norm(x_segment.detach().float(), (d,))
            predicted = F.layer_norm(prediction.detach().float(), (d,))
            surprise = (target - predicted).square().mean(dim=-1)
            surprise_records.append(surprise)

            event_count = min(self.selected_events, end - start)
            indices = torch.topk(
                surprise,
                k=event_count,
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            gather = indices.unsqueeze(-1).expand(-1, -1, d)
            selected = torch.gather(x_segment, dim=1, index=gather)
            workspace = self._update_workspace(workspace, self.event_in(selected))
            selected_total += b * event_count
            state_prev = state[:, -1, :]

        all_surprise = torch.cat(surprise_records, dim=1)
        self.last_stats = {
            "selected_fraction": selected_total / max(b * t, 1),
            "surprise_mean": all_surprise.mean().detach(),
            "surprise_std": all_surprise.std(unbiased=False).detach(),
            "workspace_norm": workspace.float().norm(dim=-1).mean().detach(),
        }
        return torch.cat(outputs, dim=1)


class PGWBlock(nn.Module):
    def __init__(self, cfg: PGWConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = PGWMixer(cfg)
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class PGWResearchLM(nn.Module):
    def __init__(self, cfg: PGWConfig):
        super().__init__()
        if cfg.architecture != "pgw":
            raise ValueError("PGWResearchLM only supports architecture='pgw'")
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(PGWBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.mixer.auto_up.weight)
            nn.init.normal_(block.mixer.workspace_init, mean=0.0, std=0.02)

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
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def router_stats(self) -> dict[str, object] | None:
        rows = [
            block.mixer.last_stats
            for block in self.blocks
            if block.mixer.last_stats is not None
        ]
        if not rows:
            return None
        keys = ("selected_fraction", "surprise_mean", "surprise_std", "workspace_norm")

        def scalar(value: object) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.float().cpu())
            return float(value)

        per_layer = [
            {key: scalar(row[key]) for key in keys}
            for row in rows
        ]
        return {
            "mean": {
                key: sum(row[key] for row in per_layer) / len(per_layer)
                for key in keys
            },
            "per_layer": per_layer,
        }


def pgw_parameter_count() -> int:
    return parameter_count(PGWResearchLM(PGWConfig()))


def pgw_mixer_parameter_count() -> int:
    return parameter_count(PGWMixer(PGWConfig()))
