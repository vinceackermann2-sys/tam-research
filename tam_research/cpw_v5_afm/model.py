from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.cpw_v1.model import CPWV1Config, LowRankPredictor
from tam_research.models import parameter_count

from .protocol import (
    AFM_EPS,
    AFM_GATE_OFFSET,
    AFM_KEY_RANK,
    AFM_LAYER,
    AFM_VALUE_RANK,
)


def exclusive_fast_weight_read(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    *,
    eps: float = AFM_EPS,
) -> torch.Tensor:
    """Vectorized causal fast-weight read.

    The state visible at position t contains writes from positions < t only.
    Accumulation is float32 for numerical stability under BF16 autocast.
    """
    weighted_k = (gate * k).float()
    values = v.float()
    writes = weighted_k.unsqueeze(-1) * values.unsqueeze(-2)

    inclusive_memory = torch.cumsum(writes, dim=1)
    inclusive_mass = torch.cumsum(weighted_k, dim=1)
    memory_before = inclusive_memory - writes
    mass_before = inclusive_mass - weighted_k

    qf = q.float()
    numerator = torch.einsum("btk,btkv->btv", qf, memory_before)
    denominator = torch.einsum("btk,btk->bt", qf, mass_before).unsqueeze(-1)
    read = numerator / denominator.clamp_min(eps)
    read = torch.where(denominator > eps, read, torch.zeros_like(read))
    return read.to(v.dtype)


def exclusive_fast_weight_read_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    *,
    eps: float = AFM_EPS,
) -> torch.Tensor:
    """Slow recurrence used only by zero-GPU equivalence contracts."""
    b, t, key_rank = q.shape
    value_rank = v.size(-1)
    memory = torch.zeros(
        b,
        key_rank,
        value_rank,
        device=q.device,
        dtype=torch.float32,
    )
    mass = torch.zeros(
        b,
        key_rank,
        device=q.device,
        dtype=torch.float32,
    )
    outputs: list[torch.Tensor] = []

    for index in range(t):
        q_now = q[:, index].float()
        numerator = torch.einsum("bk,bkv->bv", q_now, memory)
        denominator = torch.einsum("bk,bk->b", q_now, mass).unsqueeze(-1)
        read = numerator / denominator.clamp_min(eps)
        read = torch.where(denominator > eps, read, torch.zeros_like(read))
        outputs.append(read)

        weighted_k = (gate[:, index] * k[:, index]).float()
        value = v[:, index].float()
        memory = memory + weighted_k.unsqueeze(-1) * value.unsqueeze(-2)
        mass = mass + weighted_k

    return torch.stack(outputs, dim=1).to(v.dtype)


class AssociativeFastMemoryPredictor(nn.Module):
    """Parameter-matched causal key->value fast-weight memory."""

    def __init__(
        self,
        d_model: int,
        *,
        key_rank: int = AFM_KEY_RANK,
        value_rank: int = AFM_VALUE_RANK,
        gate_offset: float = AFM_GATE_OFFSET,
    ):
        super().__init__()
        self.key_rank = int(key_rank)
        self.value_rank = int(value_rank)
        self.gate_offset = float(gate_offset)

        self.q_proj = nn.Linear(d_model, self.key_rank, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_rank, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_rank, bias=False)
        self.out_proj = nn.Linear(self.value_rank, d_model, bias=False)
        self.gate_prev = nn.Linear(d_model, 1, bias=False)
        self.gate_cur = nn.Linear(d_model, 1, bias=False)

        self.last_gate_mean: torch.Tensor | None = None
        self.last_read_norm: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous = torch.cat(
            (torch.zeros_like(x[:, :1]), x[:, :-1]),
            dim=1,
        )
        q = F.elu(self.q_proj(x)) + 1.0
        k = F.elu(self.k_proj(previous)) + 1.0
        v = self.v_proj(x)

        logits = (
            self.gate_prev(previous)
            + self.gate_cur(x)
            - self.gate_offset
        )
        gate = torch.sigmoid(logits)
        valid_write = (
            torch.arange(x.size(1), device=x.device)
            .view(1, -1, 1)
            .ne(0)
            .to(x.dtype)
        )
        gate = gate * valid_write

        read = exclusive_fast_weight_read(q, k, v, gate)
        out = self.out_proj(read)

        self.last_gate_mean = gate.float().mean().detach()
        self.last_read_norm = out.float().norm(dim=-1).mean().detach()
        return out


class AFMMixer(nn.Module):
    def __init__(self, cfg: CPWV1Config, *, use_afm: bool):
        super().__init__()
        self.use_afm = bool(use_afm)
        self.afm = (
            AssociativeFastMemoryPredictor(cfg.d_model)
            if self.use_afm
            else None
        )
        self.sequence = (
            None
            if self.use_afm
            else LowRankPredictor(cfg.d_model, cfg.sequence_predictor_rank)
        )

    def forward(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.afm is not None:
            return self.afm(x), workspace
        if self.sequence is None:
            raise RuntimeError("AFM mixer has no predictor")
        previous = torch.cat(
            (torch.zeros_like(x[:, :1]), x[:, :-1]),
            dim=1,
        )
        return self.sequence(previous), workspace


class AFMBlock(nn.Module):
    def __init__(self, cfg: CPWV1Config, *, use_afm: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mixer = AFMMixer(cfg, use_afm=use_afm)
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


class AFMCPWResearchLM(nn.Module):
    def __init__(
        self,
        cfg: CPWV1Config = CPWV1Config(),
        *,
        afm_layer: int = AFM_LAYER,
    ):
        super().__init__()
        if not 0 <= afm_layer < cfg.n_layers:
            raise ValueError("AFM layer is outside model depth")
        self.cfg = cfg
        self.afm_layer = int(afm_layer)

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            AFMBlock(cfg, use_afm=(index == self.afm_layer))
            for index in range(cfg.n_layers)
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
    def memory_stats(self) -> dict[str, float]:
        afm = self.blocks[self.afm_layer].mixer.afm
        if afm is None:
            raise RuntimeError("configured AFM layer is missing")
        return {
            "afm_layer": float(self.afm_layer),
            "gate_mean": (
                float(afm.last_gate_mean.cpu())
                if afm.last_gate_mean is not None
                else 0.0
            ),
            "read_norm": (
                float(afm.last_read_norm.cpu())
                if afm.last_read_norm is not None
                else 0.0
            ),
        }


def afm_mixer_parameter_count() -> int:
    return parameter_count(AssociativeFastMemoryPredictor(256))


def afm_parameter_count() -> int:
    return parameter_count(AFMCPWResearchLM())
