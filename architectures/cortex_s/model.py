from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CortexSConfig:
    d_model: int = 48
    num_experts: int = 4
    top_k: int = 2
    expert_hidden: int = 96
    internal_ticks: int = 2
    num_ops: int = 3
    num_values: int = 16

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not (1 <= self.top_k <= self.num_experts):
            raise ValueError("top_k must be in [1, num_experts]")
        if self.internal_ticks <= 0:
            raise ValueError("internal_ticks must be positive")


@dataclass
class CortexSState:
    hidden: torch.Tensor


class SparseExpertBlock(nn.Module):
    """Top-k routed expert block.

    This small reference implementation computes all experts before gathering the
    selected outputs. That keeps the implementation simple and testable. A GPU
    systems implementation must replace it with true sparse dispatch before any
    throughput claim is made.
    """

    def __init__(self, cfg: CortexSConfig):
        super().__init__()
        cfg.validate()
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.expert_hidden),
                    nn.SiLU(),
                    nn.Linear(cfg.expert_hidden, cfg.d_model),
                )
                for _ in range(cfg.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError("SparseExpertBlock expects [batch, d_model]")
        router_logits = self.router(x)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_values, dim=-1)

        all_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        selected = torch.gather(
            all_outputs,
            1,
            top_indices[..., None].expand(-1, -1, x.shape[-1]),
        )
        mixed = (selected * weights[..., None]).sum(dim=1)
        return mixed, top_indices


class CortexSCore(nn.Module):
    """Minimal CORTEX-S research core.

    It combines a persistent recurrent workspace with sparse routed internal
    computation. The state persists between calls when the caller supplies the
    returned CortexSState to the next call.
    """

    def __init__(self, cfg: CortexSConfig = CortexSConfig()):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.op_embedding = nn.Embedding(cfg.num_ops, cfg.d_model)
        self.value_embedding = nn.Embedding(cfg.num_values, cfg.d_model)
        self.input_projection = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.recurrent_update = nn.GRUCell(cfg.d_model, cfg.d_model)
        self.pre_expert_norm = nn.LayerNorm(cfg.d_model)
        self.experts = SparseExpertBlock(cfg)
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.state_head = nn.Linear(cfg.d_model, cfg.num_values)
        self.halt_head = nn.Linear(cfg.d_model, 1)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> CortexSState:
        p = next(self.parameters())
        return CortexSState(
            hidden=torch.zeros(
                batch_size,
                self.cfg.d_model,
                device=device if device is not None else p.device,
                dtype=dtype if dtype is not None else p.dtype,
            )
        )

    def encode_observation(self, op: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.input_projection(
            torch.cat([self.op_embedding(op), self.value_embedding(value)], dim=-1)
        )

    def step(
        self,
        op: torch.Tensor,
        value: torch.Tensor,
        state: CortexSState,
        *,
        internal_ticks: Optional[int] = None,
    ) -> tuple[torch.Tensor, CortexSState, list[torch.Tensor], torch.Tensor]:
        ticks = self.cfg.internal_ticks if internal_ticks is None else internal_ticks
        if ticks <= 0:
            raise ValueError("internal_ticks must be positive")

        observation = self.encode_observation(op, value)
        hidden = self.recurrent_update(observation, state.hidden)
        routes: list[torch.Tensor] = []
        for _ in range(ticks - 1):
            expert_delta, top_indices = self.experts(self.pre_expert_norm(hidden))
            hidden = hidden + 0.25 * expert_delta
            routes.append(top_indices)

        normalized = self.output_norm(hidden)
        logits = self.state_head(normalized)
        halt_probability = torch.sigmoid(self.halt_head(normalized)).squeeze(-1)
        return logits, CortexSState(hidden=hidden), routes, halt_probability

    def forward(
        self,
        ops: torch.Tensor,
        values: torch.Tensor,
        *,
        state: Optional[CortexSState] = None,
        internal_ticks: Optional[int] = None,
    ) -> tuple[torch.Tensor, CortexSState, list[torch.Tensor], torch.Tensor]:
        if ops.shape != values.shape or ops.ndim != 2:
            raise ValueError("ops and values must both be [batch, sequence]")
        batch, length = ops.shape
        if state is None:
            state = self.initial_state(batch, device=ops.device)
        if state.hidden.shape != (batch, self.cfg.d_model):
            raise ValueError("state shape does not match batch/config")

        logits = []
        routes: list[torch.Tensor] = []
        halts = []
        for index in range(length):
            step_logits, state, step_routes, halt = self.step(
                ops[:, index],
                values[:, index],
                state,
                internal_ticks=internal_ticks,
            )
            logits.append(step_logits)
            routes.extend(step_routes)
            halts.append(halt)
        return torch.stack(logits, dim=1), state, routes, torch.stack(halts, dim=1)


class FastAssociativeMemory:
    """Non-parametric episodic memory used to test fast learning without weight updates."""

    def __init__(self, key_dim: int, value_dim: int, capacity: int = 256):
        if key_dim <= 0 or value_dim <= 0 or capacity <= 0:
            raise ValueError("dimensions and capacity must be positive")
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._values.clear()

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        key = key.detach().flatten().cpu().float()
        value = value.detach().flatten().cpu().float()
        if key.numel() != self.key_dim or value.numel() != self.value_dim:
            raise ValueError("key/value dimension mismatch")
        self._keys.append(key)
        self._values.append(value)
        if len(self._keys) > self.capacity:
            self._keys.pop(0)
            self._values.pop(0)

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, float]:
        if not self._keys:
            raise RuntimeError("cannot retrieve from empty memory")
        query = query.detach().flatten().cpu().float()
        if query.numel() != self.key_dim:
            raise ValueError("query dimension mismatch")
        keys = torch.stack(self._keys)
        scores = F.cosine_similarity(keys, query[None, :], dim=-1)
        index = int(scores.argmax().item())
        return self._values[index].clone(), float(scores[index].item())


@dataclass(frozen=True)
class ActionRequest:
    capability: str
    amount: float = 0.0
    target: str = ""


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str


class SafetyKernel:
    """Deterministic capability boundary that is intentionally outside the learner."""

    def __init__(
        self,
        allowed_capabilities: Iterable[str],
        *,
        autonomous_amount_limit: float = 0.0,
        allowed_targets: Optional[Iterable[str]] = None,
    ):
        self.allowed_capabilities = frozenset(allowed_capabilities)
        self.autonomous_amount_limit = float(autonomous_amount_limit)
        self.allowed_targets = None if allowed_targets is None else frozenset(allowed_targets)

    def authorize(self, request: ActionRequest) -> SafetyDecision:
        if request.capability not in self.allowed_capabilities:
            return SafetyDecision(False, "capability_not_authorized")
        if request.amount < 0:
            return SafetyDecision(False, "invalid_negative_amount")
        if request.amount > self.autonomous_amount_limit:
            return SafetyDecision(False, "human_approval_required")
        if self.allowed_targets is not None and request.target not in self.allowed_targets:
            return SafetyDecision(False, "target_not_authorized")
        return SafetyDecision(True, "authorized")


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
