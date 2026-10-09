"""PGW-v3 Stage-0 structural mechanism; no LM, benchmark, or training authority.

Only completed chunks write the bounded workspace; later chunks can query it.
The route modes instantiate IDENTICAL parameters but differ in active paths.
Consequently equal instantiated count DOES NOT imply fair active-parameter parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTE_MODES = frozenset(
    {"hybrid", "utility_only", "surprise_only", "fixed_random", "recency", "no_workspace"}
)


@dataclass(frozen=True)
class PGWV3Stage0Config:
    d_model: int = 32
    key_width: int = 8
    value_width: int = 16
    predictor_rank: int = 8
    chunk_size: int = 8
    workspace_slots: int = 4
    selected_events: int = 2
    route_mode: str = "hybrid"
    surprise_weight: float = 0.5
    utility_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.route_mode not in ROUTE_MODES:
            raise ValueError("unrecognized PGW-v3 Stage-0 routing mode")
        if min(
            self.d_model, self.key_width, self.value_width,
            self.predictor_rank, self.chunk_size, self.workspace_slots,
        ) < 1:
            raise ValueError("dimensions must be strictly positive")
        if not 0 < self.selected_events <= self.chunk_size:
            raise ValueError("selected_events must be within chunk")
        if self.utility_temperature <= 0 or not math.isfinite(self.utility_temperature):
            raise ValueError("utility_temperature must be positive and finite")
        if not math.isfinite(self.surprise_weight) or self.surprise_weight < 0:
            raise ValueError("surprise_weight must be nonnegative and finite")


class UtilityAddressedWorkspace(nn.Module):
    """Isolated key/value event memory for a pre-existing causal chunk substrate.

    Input/output: [batch, time, d_model]. Output is *workspace read only*;
    callers may add it to their own frozen local-attention outputs later.
    No baseline Transformer, language trainer, or historical PGW module is changed.
    """

    def __init__(self, cfg: PGWV3Stage0Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.predict_down = nn.Linear(d, cfg.predictor_rank, bias=False)
        self.predict_up = nn.Linear(cfg.predictor_rank, d, bias=False)
        self.utility = nn.Linear(d, 1, bias=False)
        self.event_key = nn.Linear(d, cfg.key_width, bias=False)
        self.event_value = nn.Linear(d, cfg.value_width, bias=False)
        self.query_key = nn.Linear(d, cfg.key_width, bias=False)
        self.write_gate = nn.Linear(d, 1, bias=True)
        self.read_out = nn.Linear(cfg.value_width, d, bias=False)
        self.initial_keys = nn.Parameter(torch.empty(cfg.workspace_slots, cfg.key_width))
        self.initial_values = nn.Parameter(torch.zeros(cfg.workspace_slots, cfg.value_width))
        nn.init.normal_(self.initial_keys, mean=0.0, std=0.05)
        self.last_stats: dict[str, object] | None = None

    def _scores(self, chunk: torch.Tensor) -> torch.Tensor:
        """Scores use this completed chunk only; no labels or future chunks."""
        b, t, d = chunk.shape
        previous = torch.cat((chunk.new_zeros(b, 1, d), chunk[:, :-1]), dim=1)
        prediction = self.predict_up(F.gelu(self.predict_down(previous)))
        discrepancy = (
            F.layer_norm(chunk.detach().float(), (d,))
            - F.layer_norm(prediction.detach().float(), (d,))
        ).square().mean(dim=-1)
        # Stop gradient through surprise, as in historical PGW-v2.
        surprise = (discrepancy - discrepancy.mean(dim=1, keepdim=True)) / (
            discrepancy.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        utility = self.utility(chunk).squeeze(-1)
        mode = self.cfg.route_mode
        if mode == "surprise_only":
            return surprise
        if mode == "utility_only":
            return utility
        return utility + self.cfg.surprise_weight * surprise

    def _indices(self, scores: torch.Tensor) -> torch.Tensor:
        batch, t = scores.shape
        mode = self.cfg.route_mode
        k = self.cfg.selected_events
        if mode == "recency":
            return torch.arange(t - k, t, device=scores.device).expand(batch, -1)
        if mode == "fixed_random":
            # A deterministic, input-independent, local PRNG: no global RNG use.
            picks = sorted(random.Random(135400 + t * 100 + k).sample(range(t), k))
            return torch.tensor(picks, dtype=torch.long, device=scores.device).expand(batch, -1)
        return torch.topk(scores, k=k, dim=1, sorted=True).indices

    def _read(
        self, chunk: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        query = self.query_key(chunk)
        weights = F.softmax(
            torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(self.cfg.key_width),
            dim=-1,
        )
        return self.read_out(torch.matmul(weights, values))

    def _write(
        self,
        event: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable straight-through *slot* selection, gated key/value overwrite."""
        key = self.event_key(event)
        value = self.event_value(event)
        scores = torch.einsum("bd,bsd->bs", key, keys) / math.sqrt(self.cfg.key_width)
        probability = F.softmax(scores, dim=-1)
        hard = F.one_hot(scores.argmax(dim=-1), num_classes=self.cfg.workspace_slots)
        assignment = hard.to(probability.dtype) + probability - probability.detach()
        gate = torch.sigmoid(self.write_gate(event))
        amount = assignment.unsqueeze(-1) * gate.unsqueeze(1)
        keys = keys + amount * (key.unsqueeze(1) - keys)
        values = values + amount * (value.unsqueeze(1) - values)
        return keys, values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.cfg.d_model:
            raise ValueError("expected [batch,time,d_model]")
        batch, length, _ = x.shape
        if length == 0 or length % self.cfg.chunk_size:
            raise ValueError("time must be a positive multiple of chunk_size")
        k, chunk_len = self.cfg.selected_events, self.cfg.chunk_size
        keys = self.initial_keys.to(x.dtype).unsqueeze(0).expand(batch, -1, -1)
        values = self.initial_values.to(x.dtype).unsqueeze(0).expand(batch, -1, -1)
        outputs: list[torch.Tensor] = []
        selected_positions: list[torch.Tensor] = []
        for start in range(0, length, chunk_len):
            chunk = x[:, start : start + chunk_len]
            if self.cfg.route_mode == "no_workspace":
                outputs.append(torch.zeros_like(chunk))
                continue
            # Crucial causal lag: READ first, WRITE only after output is fixed.
            read = self._read(chunk, keys, values)
            outputs.append(torch.zeros_like(read) if start == 0 else read)
            scores = self._scores(chunk)
            indices = self._indices(scores)
            selected_positions.append(indices.detach())
            soft = F.softmax(scores / self.cfg.utility_temperature, dim=1)
            for i in range(k):
                positions = indices[:, i]
                event = chunk.gather(1, positions[:, None, None].expand(-1, 1, chunk.shape[-1])).squeeze(1)
                if self.cfg.route_mode in ("hybrid", "utility_only"):
                    # Exact-forward weight 1; differentiable surrogate gives
                    # the learned router a gradient despite hard top-k.
                    selected_soft = soft.gather(1, positions[:, None])
                    event = event * (1.0 + selected_soft - selected_soft.detach())
                keys, values = self._write(event, keys, values)
        self.last_stats = {
            "selected_fraction": (0.0 if self.cfg.route_mode == "no_workspace" else k / chunk_len),
            "selected_positions": (
                torch.stack(selected_positions, dim=1) if selected_positions else None
            ),
        }
        return torch.cat(outputs, dim=1)


def instantiated_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
