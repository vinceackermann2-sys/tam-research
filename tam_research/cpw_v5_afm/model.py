from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.cpw_v1.model import CPWV1Config, LowRankPredictor
from tam_research.models import parameter_count

from .protocol import AFM_LAYER, MEMORY_RANK


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


def associative_scan_vectorized(
    key_features: torch.Tensor,
    values: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal additive key/value fast-memory scan.

    key_features: positive shared key/query features [B,T,R]
    values: value features [B,T,R]
    gates: write strength [B,T,1]

    At token t, the write key comes from token t-1 while the write value comes
    from token t. The returned read at t therefore uses only information at or
    before t and can learn transitions such as previous-key -> current-value.
    """
    if key_features.shape != values.shape:
        raise ValueError("key_features and values must have identical shape")
    if gates.shape != key_features.shape[:-1] + (1,):
        raise ValueError("gates must be [B,T,1]")

    write_keys = _shift_right(key_features)
    writes = (
        gates.unsqueeze(-1)
        * write_keys.unsqueeze(-1)
        * values.unsqueeze(-2)
    )
    matrix = torch.cumsum(writes, dim=1)

    key_mass = torch.cumsum(gates * write_keys, dim=1)
    numerator = torch.einsum("btr,btrs->bts", key_features, matrix)
    denominator = torch.einsum(
        "btr,btr->bt", key_features, key_mass
    ).unsqueeze(-1)
    reads = numerator / denominator.clamp_min(1e-4)
    return reads, matrix[:, -1]


def associative_scan_reference(
    key_features: torch.Tensor,
    values: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal recurrent reference used only for zero-GPU equivalence tests."""
    b, t, rank = key_features.shape
    matrix = torch.zeros(
        b, rank, rank, dtype=key_features.dtype, device=key_features.device
    )
    mass = torch.zeros(
        b, rank, dtype=key_features.dtype, device=key_features.device
    )
    reads: list[torch.Tensor] = []
    previous_key = torch.zeros(
        b, rank, dtype=key_features.dtype, device=key_features.device
    )
    for i in range(t):
        gate = gates[:, i]
        value = values[:, i]
        matrix = matrix + (
            gate.unsqueeze(-1)
            * previous_key.unsqueeze(-1)
            * value.unsqueeze(-2)
        )
        mass = mass + gate * previous_key
        query = key_features[:, i]
        numerator = torch.einsum("br,brs->bs", query, matrix)
        denominator = (query * mass).sum(dim=-1, keepdim=True).clamp_min(1e-4)
        reads.append(numerator / denominator)
        previous_key = query
    return torch.stack(reads, dim=1), matrix


class AssociativeFastMemoryPredictor(nn.Module):
    """Fixed-size causal associative memory with shared key/query projection."""

    def __init__(self, d_model: int, rank: int = MEMORY_RANK):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.key_proj = nn.Linear(d_model, rank, bias=False)
        self.value_proj = nn.Linear(d_model, rank, bias=False)
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        self.gate_prev = nn.Linear(d_model, 1, bias=False)
        self.gate_cur = nn.Linear(d_model, 1, bias=False)
        self.last_memory_norm: torch.Tensor | None = None
        self.last_mean_gate: torch.Tensor | None = None

    @property
    def state_size(self) -> int:
        return self.rank * self.rank + self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ELU+1 gives positive linear-attention features and a stable
        # query-dependent normalizer. The same key_proj is used for source
        # keys and later queries.
        key_features = F.elu(self.key_proj(x)) + 1.0
        values = self.value_proj(x)
        previous = _shift_right(x)
        gates = torch.sigmoid(
            self.gate_prev(previous) + self.gate_cur(x)
        )
        reads, final_matrix = associative_scan_vectorized(
            key_features, values, gates
        )
        self.last_memory_norm = final_matrix.float().norm(dim=(-2, -1)).mean().detach()
        self.last_mean_gate = gates.float().mean().detach()
        return self.out_proj(reads)


class AFMBlock(nn.Module):
    def __init__(self, cfg: CPWV1Config, *, use_afm: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.use_afm = bool(use_afm)
        self.afm = (
            AssociativeFastMemoryPredictor(cfg.d_model, MEMORY_RANK)
            if self.use_afm
            else None
        )
        self.sequence = (
            None
            if self.use_afm
            else LowRankPredictor(cfg.d_model, cfg.sequence_predictor_rank)
        )
        ff = cfg.ff_mult * cfg.d_model
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.afm is not None:
            mixed = self.afm(h)
        else:
            if self.sequence is None:
                raise RuntimeError("AFM block has no temporal mixer")
            mixed = self.sequence(_shift_right(h))
        x = x + mixed
        x = x + self.ff(self.norm2(x))
        return x


class AFMCPWResearchLM(nn.Module):
    """Fourteen frozen-style local sequence blocks plus one AFM block."""

    def __init__(self, cfg: CPWV1Config = CPWV1Config()):
        super().__init__()
        if AFM_LAYER >= cfg.n_layers:
            raise ValueError("AFM layer exceeds model depth")
        self.cfg = cfg
        self.afm_layers = (AFM_LAYER,)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            AFMBlock(cfg, use_afm=(i == AFM_LAYER))
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

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden(tokens))

    @torch.no_grad()
    def memory_stats(self) -> dict[str, float | int]:
        afm = self.blocks[AFM_LAYER].afm
        if afm is None:
            raise RuntimeError("configured AFM layer is missing")
        return {
            "afm_layers": 1,
            "afm_layer_index": AFM_LAYER,
            "memory_rank": afm.rank,
            "memory_state_scalars": afm.state_size,
            "memory_norm": (
                float(afm.last_memory_norm.cpu())
                if afm.last_memory_norm is not None
                else 0.0
            ),
            "mean_write_gate": (
                float(afm.last_mean_gate.cpu())
                if afm.last_mean_gate is not None
                else 0.0
            ),
        }


def afm_parameter_count() -> int:
    return parameter_count(AFMCPWResearchLM())
