from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.models import (
    CausalSelfAttention,
    diagonal_affine_scan,
    parameter_count,
)

from .protocol import (
    ATTENTION_INNER,
    D_MODEL,
    FF_MULT,
    MAX_SEQ_LEN,
    MEMORY_HORIZON,
    MEMORY_PREDICTOR_RANK,
    N_HEADS,
    N_LAYERS,
    ROUTER_EXPERTS,
    ROUTER_TOP_K,
    SEQUENCE_PREDICTOR_RANK,
    VOCAB_SIZE,
    WORKSPACE_MIX,
    WORLD_STATE_SIZE,
)


@dataclass(frozen=True)
class CPWV0Config:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    max_seq_len: int = MAX_SEQ_LEN
    ff_mult: int = FF_MULT
    attention_inner: int = ATTENTION_INNER
    world_state_size: int = WORLD_STATE_SIZE
    sequence_predictor_rank: int = SEQUENCE_PREDICTOR_RANK
    memory_predictor_rank: int = MEMORY_PREDICTOR_RANK
    router_experts: int = ROUTER_EXPERTS
    router_top_k: int = ROUTER_TOP_K
    memory_horizon: int = MEMORY_HORIZON
    workspace_mix: float = WORKSPACE_MIX
    architecture: str = "cpwv0"


def _norm_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = pred.size(-1)
    p = F.layer_norm(pred.float(), (d,))
    t = F.layer_norm(target.detach().float(), (d,))
    return F.mse_loss(p, t)


class RecurrentWorldPredictor(nn.Module):
    """Causal compressed world state: s_t = keep_t*s_(t-1) + update_t."""

    def __init__(self, d_model: int, state_size: int):
        super().__init__()
        self.candidate = nn.Linear(d_model, state_size, bias=False)
        self.keep = nn.Linear(d_model, state_size, bias=False)
        self.out = nn.Linear(state_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        update = (1.0 - keep) * candidate
        state = diagonal_affine_scan(keep, update)
        return self.out(state), state


class LowRankPredictor(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))


class CPWMixer(nn.Module):
    """Four fixed-function predictors with a learned sparse thalamic router.

    The reference implementation computes all branches and routes top-2. The
    routing is sparse information flow, not yet a sparse-kernel wall-clock claim.
    """

    def __init__(self, cfg: CPWV0Config):
        super().__init__()
        self.cfg = cfg
        self.attention = CausalSelfAttention(
            cfg.d_model,
            cfg.n_heads,
            inner=cfg.attention_inner,
        )
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
        self.router = nn.Linear(
            cfg.d_model,
            cfg.router_experts,
            bias=False,
        )
        self.last_stats: dict[str, torch.Tensor | float] | None = None
        self.last_aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape != workspace.shape:
            raise ValueError("workspace must match the token-state shape")

        context = x + self.cfg.workspace_mix * workspace
        attention = self.attention(context)
        world, world_state = self.world(context)

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

        branches = torch.stack(
            (attention, world, sequence, memory),
            dim=2,
        )
        router_logits = self.router(context)
        top_values, top_indices = torch.topk(
            router_logits,
            k=self.cfg.router_top_k,
            dim=-1,
        )
        top_weights = torch.softmax(top_values.float(), dim=-1).to(context.dtype)
        gather_index = top_indices.unsqueeze(-1).expand(
            -1, -1, -1, context.size(-1)
        )
        selected = torch.gather(branches, 2, gather_index)
        mixed = (selected * top_weights.unsqueeze(-1)).sum(dim=2)

        mask = torch.zeros_like(router_logits)
        mask.scatter_(-1, top_indices, 1.0)
        full_probs = torch.softmax(router_logits.float(), dim=-1)
        entropy = -(full_probs * full_probs.clamp_min(1e-9).log()).sum(-1)

        aux_terms: list[torch.Tensor] = []
        if context.size(1) > 1:
            aux_terms.append(_norm_mse(sequence[:, 1:], context[:, 1:]))
            aux_terms.append(_norm_mse(world[:, :-1], context[:, 1:]))
        h = self.cfg.memory_horizon
        if context.size(1) > h:
            aux_terms.append(_norm_mse(memory[:, :-h], context[:, h:]))
        aux = (
            torch.stack(aux_terms).mean()
            if aux_terms
            else context.float().sum() * 0.0
        )

        next_workspace = (
            (1.0 - self.cfg.workspace_mix) * workspace
            + self.cfg.workspace_mix * world
        )
        self.last_aux_loss = aux
        self.last_stats = {
            "active_fraction": mask.mean().detach(),
            "router_entropy": entropy.mean().detach(),
            "attention_usage": mask[..., 0].mean().detach(),
            "world_usage": mask[..., 1].mean().detach(),
            "sequence_usage": mask[..., 2].mean().detach(),
            "memory_usage": mask[..., 3].mean().detach(),
            "world_state_norm": world_state.float().norm(dim=-1).mean().detach(),
        }
        return mixed, next_workspace


class CPWBlock(nn.Module):
    def __init__(self, cfg: CPWV0Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = CPWMixer(cfg)
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


class CPWV0ResearchLM(nn.Module):
    def __init__(self, cfg: CPWV0Config = CPWV0Config()):
        super().__init__()
        if cfg.architecture != "cpwv0":
            raise ValueError("CPWV0ResearchLM requires architecture='cpwv0'")
        if cfg.router_experts != 4:
            raise ValueError("CPW-v0 has exactly four fixed-function branches")
        if not 0 < cfg.router_top_k <= cfg.router_experts:
            raise ValueError("invalid router_top_k")

        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(CPWBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._aux_loss: torch.Tensor | None = None
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
        aux: list[torch.Tensor] = []
        for block in self.blocks:
            x, workspace = block(x, workspace)
            if block.mixer.last_aux_loss is not None:
                aux.append(block.mixer.last_aux_loss)
        self._aux_loss = (
            torch.stack(aux).mean()
            if aux
            else x.float().sum() * 0.0
        )
        return self.lm_head(self.norm(x))

    def auxiliary_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            raise RuntimeError("forward must run before auxiliary_loss")
        return self._aux_loss

    @torch.no_grad()
    def router_stats(self) -> dict[str, object] | None:
        rows = [
            block.mixer.last_stats
            for block in self.blocks
            if block.mixer.last_stats is not None
        ]
        if not rows:
            return None

        def scalar(v: object) -> float:
            if isinstance(v, torch.Tensor):
                return float(v.float().cpu())
            return float(v)

        keys = (
            "active_fraction",
            "router_entropy",
            "attention_usage",
            "world_usage",
            "sequence_usage",
            "memory_usage",
            "world_state_norm",
        )
        per_layer = [{k: scalar(row[k]) for k in keys} for row in rows]
        return {
            "mean": {
                k: sum(row[k] for row in per_layer) / len(per_layer)
                for k in keys
            },
            "per_layer": per_layer,
        }


def cpwv0_parameter_count() -> int:
    return parameter_count(CPWV0ResearchLM(CPWV0Config()))
