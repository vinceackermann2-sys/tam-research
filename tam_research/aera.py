from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AERAConfig:
    d_model: int = 128
    n_experts: int = 8
    top_k_experts: int = 2
    expert_mult: int = 2
    memory_dim: int = 32
    max_reason_steps: int = 4
    halt_threshold: float = 0.65
    fast_memory_lr: float = 0.2
    fast_memory_decay: float = 0.995


@dataclass
class FastMemoryState:
    matrix: torch.Tensor

    def detach(self) -> "FastMemoryState":
        return FastMemoryState(self.matrix.detach())


@dataclass
class AERAState:
    """Per-stream state. Never share this object across unrelated users/examples."""

    stream: torch.Tensor
    memory: FastMemoryState

    def detach(self) -> "AERAState":
        return AERAState(self.stream.detach(), self.memory.detach())


class ComputeController(nn.Module):
    """Cheap controller for experts, memory writes, and adaptive compute."""

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.proj = nn.Linear(d_model, n_experts + 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.proj(x)
        expert_logits = raw[..., : self.n_experts]
        controls = torch.sigmoid(raw[..., self.n_experts :])
        return {
            "expert_logits": expert_logits,
            "difficulty": controls[..., 0:1],
            "novelty": controls[..., 1:2],
            "memory_write": controls[..., 2:3],
            "extra_reasoning": controls[..., 3:4],
        }


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, mult: int):
        super().__init__()
        hidden = d_model * mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseExpertLayer(nn.Module):
    """Reference top-k conditional expert execution.

    Only tokens assigned to an expert are passed through that expert. This is a
    correctness sandbox; production GPU efficiency still requires fused routing
    and grouped GEMMs.
    """

    def __init__(self, cfg: AERAConfig):
        super().__init__()
        if not 1 <= cfg.top_k_experts <= cfg.n_experts:
            raise ValueError("top_k_experts must be in [1, n_experts]")
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k_experts
        self.experts = nn.ModuleList(
            ExpertMLP(cfg.d_model, cfg.expert_mult) for _ in range(cfg.n_experts)
        )
        self.last_counts: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[:-1] != x.shape[:-1] or logits.size(-1) != self.n_experts:
            raise ValueError("router logits shape does not match expert layer")
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        flat_logits = logits.reshape(-1, self.n_experts)
        top_values, top_indices = torch.topk(flat_logits, self.top_k, dim=-1)
        weights = F.softmax(top_values.float(), dim=-1).to(flat.dtype)
        out = torch.zeros_like(flat)
        counts = torch.zeros(self.n_experts, device=x.device, dtype=torch.long)

        for expert_id, expert in enumerate(self.experts):
            assignments = (top_indices == expert_id).nonzero(as_tuple=False)
            counts[expert_id] = assignments.size(0)
            if assignments.numel() == 0:
                continue
            token_index = assignments[:, 0]
            route_index = assignments[:, 1]
            selected = flat.index_select(0, token_index)
            contribution = expert(selected) * weights[token_index, route_index, None]
            out = out.index_add(0, token_index, contribution)

        self.last_counts = counts.detach().cpu()
        return out.view(original_shape)

    def routing_stats(self) -> dict[str, object] | None:
        if self.last_counts is None:
            return None
        total = int(self.last_counts.sum())
        return {
            "assignments": total,
            "per_expert": self.last_counts.tolist(),
            "active_fraction_of_experts_per_token": self.top_k / self.n_experts,
        }


class FastWeightMemory(nn.Module):
    """Session-local associative fast weights with no global parameter update."""

    def __init__(self, cfg: AERAConfig):
        super().__init__()
        self.memory_dim = cfg.memory_dim
        self.lr = cfg.fast_memory_lr
        self.decay = cfg.fast_memory_decay
        self.q = nn.Linear(cfg.d_model, cfg.memory_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.memory_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.memory_dim, bias=False)
        self.out = nn.Linear(cfg.memory_dim, cfg.d_model, bias=False)

    def empty_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> FastMemoryState:
        return FastMemoryState(
            torch.zeros(
                batch_size,
                self.memory_dim,
                self.memory_dim,
                device=device,
                dtype=dtype,
            )
        )

    def read(self, x: torch.Tensor, state: FastMemoryState) -> torch.Tensor:
        query = F.normalize(self.q(x), dim=-1)
        recalled = torch.einsum("btd,bdm->btm", query, state.matrix)
        return self.out(recalled)

    @torch.no_grad()
    def local_update(
        self,
        x: torch.Tensor,
        write_strength: torch.Tensor,
        state: FastMemoryState,
    ) -> FastMemoryState:
        key = F.normalize(self.k(x.detach()), dim=-1)
        value = torch.tanh(self.v(x.detach()))
        strength = write_strength.detach().clamp(0.0, 1.0)
        update = torch.einsum("bti,btj->bij", key * strength, value)
        denom = strength.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        update = update / denom
        matrix = self.decay * state.matrix + self.lr * update
        return FastMemoryState(matrix.detach())


class StreamState(nn.Module):
    """Compact recurrent state for recent/event history."""

    def __init__(self, d_model: int):
        super().__init__()
        self.cell = nn.GRUCell(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = initial
        outputs = []
        for t in range(x.size(1)):
            state = self.cell(x[:, t], state)
            outputs.append(state)
        return torch.stack(outputs, dim=1), state


class AdaptiveLatentReasoner(nn.Module):
    """Variable-step latent reasoning with hard inference-time halting."""

    def __init__(self, cfg: AERAConfig):
        super().__init__()
        self.max_steps = cfg.max_reason_steps
        self.halt_threshold = cfg.halt_threshold
        self.cell = nn.GRUCell(cfg.d_model, cfg.d_model)
        self.halt = nn.Linear(cfg.d_model, 1)
        self.last_steps: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        shape = x.shape
        z = x.reshape(-1, x.size(-1))
        diff = difficulty.reshape(-1)
        active = torch.ones(z.size(0), dtype=torch.bool, device=z.device)
        steps = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        for step in range(self.max_steps):
            idx = active.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break
            current = z.index_select(0, idx)
            updated = self.cell(current, current)
            z = z.index_copy(0, idx, updated)
            steps[idx] += 1
            halt_prob = torch.sigmoid(self.halt(updated)).squeeze(-1)
            required = self.halt_threshold + 0.25 * diff.index_select(0, idx)
            required = required.clamp(max=0.95)
            stop = halt_prob >= required
            if step + 1 == self.max_steps:
                stop = torch.ones_like(stop)
            next_active = active.clone()
            next_active[idx[stop]] = False
            active = next_active

        self.last_steps = steps.detach().cpu()
        return z.view(shape)

    def step_stats(self) -> dict[str, float] | None:
        if self.last_steps is None:
            return None
        values = self.last_steps.float()
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }


class AERACore(nn.Module):
    """Phase-0 AERA mechanism sandbox.

    This is not yet a production language model. It composes a cheap controller,
    true top-k expert execution, stream state, session-local fast memory, and a
    variable-step latent reasoner so each mechanism can be measured/ablated.
    """

    def __init__(self, cfg: AERAConfig = AERAConfig()):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.controller = ComputeController(cfg.d_model, cfg.n_experts)
        self.memory = FastWeightMemory(cfg)
        self.experts = SparseExpertLayer(cfg)
        self.stream = StreamState(cfg.d_model)
        self.reasoner = AdaptiveLatentReasoner(cfg)
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def empty_state(self, x: torch.Tensor) -> AERAState:
        b = x.size(0)
        return AERAState(
            stream=torch.zeros(b, self.cfg.d_model, device=x.device, dtype=x.dtype),
            memory=self.memory.empty_state(b, x.device, x.dtype),
        )

    def forward(
        self,
        events: torch.Tensor,
        state: AERAState | None = None,
        *,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, AERAState]:
        if events.ndim != 3 or events.size(-1) != self.cfg.d_model:
            raise ValueError("events must have shape [batch, time, d_model]")
        if state is None:
            state = self.empty_state(events)

        x = self.norm(events)
        control = self.controller(x)
        recalled = self.memory.read(x, state.memory)
        x = x + recalled

        expert_out = self.experts(x, control["expert_logits"])
        x = x + expert_out

        stream_out, final_stream = self.stream(x, state.stream)
        x = x + stream_out

        difficulty = (control["difficulty"] + control["extra_reasoning"]) * 0.5
        x = x + self.reasoner(x, difficulty)

        memory_state = state.memory
        if update_memory:
            write = (control["memory_write"] * control["novelty"]).clamp(0.0, 1.0)
            memory_state = self.memory.local_update(x, write, memory_state)

        return self.out_norm(x), AERAState(final_stream, memory_state)

    def compute_stats(self) -> dict[str, object]:
        return {
            "experts": self.experts.routing_stats(),
            "reasoning_steps": self.reasoner.step_stats(),
        }
