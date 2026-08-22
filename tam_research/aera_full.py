from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAConfig, AERAState, FastWeightMemory, SparseExpertLayer, StreamState


@dataclass(frozen=True)
class FullAERAConfig:
    vocab_size: int = 257
    d_model: int = 64
    n_stages: int = 2
    n_heads: int = 4
    local_window: int = 32
    max_seq_len: int = 64
    n_experts: int = 8
    top_k_experts: int = 2
    expert_mult: int = 2
    memory_dim: int = 16
    max_reason_steps: int = 4
    fast_memory_lr: float = 0.2
    fast_memory_decay: float = 0.995
    block_size: int = 4

    def mechanism_cfg(self) -> AERAConfig:
        return AERAConfig(
            d_model=self.d_model,
            n_experts=self.n_experts,
            top_k_experts=self.top_k_experts,
            expert_mult=self.expert_mult,
            memory_dim=self.memory_dim,
            max_reason_steps=self.max_reason_steps,
            fast_memory_lr=self.fast_memory_lr,
            fast_memory_decay=self.fast_memory_decay,
        )


@dataclass
class FullAERAState:
    stages: list[AERAState]

    def detach(self) -> "FullAERAState":
        return FullAERAState([s.detach() for s in self.stages])


class LocalCausalAttention(nn.Module):
    """Exact causal attention restricted to a bounded recent window."""

    def __init__(self, d_model: int, n_heads: int, window: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if window < 1:
            raise ValueError("window must be >=1")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        i = torch.arange(t, device=x.device)[:, None]
        j = torch.arange(t, device=x.device)[None, :]
        allowed = (j <= i) & (j >= i - self.window + 1)
        mask = torch.zeros(t, t, device=x.device, dtype=q.dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class FullComputeController(nn.Module):
    """Cheap AERA controller. Signals are decisions/interfaces, not speed claims."""

    CONTROL_NAMES = (
        "difficulty",
        "novelty",
        "memory_read",
        "memory_write",
        "attention_need",
        "retrieval_need",
        "precision_budget",
        "block_generation",
    )

    def __init__(self, cfg: FullAERAConfig):
        super().__init__()
        out = cfg.n_experts + cfg.max_reason_steps + len(self.CONTROL_NAMES)
        self.n_experts = cfg.n_experts
        self.max_reason_steps = cfg.max_reason_steps
        self.proj = nn.Linear(cfg.d_model, out)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.proj(x)
        e = self.n_experts
        r = self.max_reason_steps
        expert_logits = raw[..., :e]
        depth_logits = raw[..., e : e + r]
        control = torch.sigmoid(raw[..., e + r :])
        result: dict[str, torch.Tensor] = {
            "expert_logits": expert_logits,
            "depth_logits": depth_logits,
        }
        for i, name in enumerate(self.CONTROL_NAMES):
            result[name] = control[..., i : i + 1]
        return result


class BudgetedLatentReasoner(nn.Module):
    """Differentiable soft-depth training + genuinely variable hard-depth inference.

    Soft mode computes all candidate recurrent depths and mixes them according to a
    learned depth distribution. Hard mode chooses argmax depth per event and only
    advances events that still require another transition.
    """

    def __init__(self, d_model: int, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.cell = nn.GRUCell(d_model, d_model)
        self.last_steps: torch.Tensor | None = None
        self.last_expected_steps: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        depth_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if depth_logits.shape != (*x.shape[:-1], self.max_steps):
            raise ValueError("depth_logits shape mismatch")
        flat = x.reshape(-1, x.size(-1))
        logits = depth_logits.reshape(-1, self.max_steps)
        probs = F.softmax(logits.float(), dim=-1).to(flat.dtype)
        depth_values = torch.arange(1, self.max_steps + 1, device=x.device, dtype=flat.dtype)
        expected = (probs * depth_values[None]).sum(dim=-1)
        self.last_expected_steps = expected.detach().cpu()

        if not hard:
            current = flat
            states = []
            for _ in range(self.max_steps):
                current = self.cell(current, current)
                states.append(current)
            stacked = torch.stack(states, dim=1)
            mixed = (stacked * probs[..., None]).sum(dim=1)
            self.last_steps = None
            return mixed.view_as(x)

        chosen = logits.argmax(dim=-1) + 1
        current = flat
        for step in range(1, self.max_steps + 1):
            idx = (chosen >= step).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break
            selected = current.index_select(0, idx)
            updated = self.cell(selected, selected)
            current = current.index_copy(0, idx, updated)
        self.last_steps = chosen.detach().cpu()
        return current.view_as(x)

    def stats(self) -> dict[str, float] | None:
        if self.last_steps is not None:
            v = self.last_steps.float()
            return {"mode": "hard", "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
        if self.last_expected_steps is not None:
            v = self.last_expected_steps.float()
            return {"mode": "soft", "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
        return None


class AERAStage(nn.Module):
    def __init__(self, cfg: FullAERAConfig):
        super().__init__()
        mech = cfg.mechanism_cfg()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.controller = FullComputeController(cfg)
        self.attn = LocalCausalAttention(cfg.d_model, cfg.n_heads, cfg.local_window)
        self.experts = SparseExpertLayer(mech)
        self.stream = StreamState(cfg.d_model)
        self.memory = FastWeightMemory(mech)
        self.reasoner = BudgetedLatentReasoner(cfg.d_model, cfg.max_reason_steps)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.last_controls: dict[str, torch.Tensor] | None = None

    def empty_state(self, x: torch.Tensor) -> AERAState:
        b = x.size(0)
        return AERAState(
            stream=torch.zeros(b, self.cfg.d_model, device=x.device, dtype=x.dtype),
            memory=self.memory.empty_state(b, x.device, x.dtype),
        )

    def forward(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, torch.Tensor]]:
        if state is None:
            state = self.empty_state(events)
        h = self.norm(events)
        control = self.controller(h)
        self.last_controls = {k: v.detach() for k, v in control.items() if k not in {"expert_logits", "depth_logits"}}

        recalled = self.memory.read(h, state.memory)
        h = h + control["memory_read"] * recalled

        # The reference attention path is always computed in training. A future
        # grouped hard-attention kernel is required before claiming savings here.
        attn = self.attn(h)
        h = h + control["attention_need"] * attn

        expert = self.experts(h, control["expert_logits"])
        h = h + expert

        stream_out, final_stream = self.stream(h, state.stream)
        h = h + stream_out

        reasoned = self.reasoner(h, control["depth_logits"], hard=hard)
        h = h + reasoned

        memory_state = state.memory
        if update_memory:
            write = (control["novelty"] * control["memory_write"]).clamp(0.0, 1.0)
            memory_state = self.memory.local_update(h, write, memory_state)

        return self.out_norm(h), AERAState(final_stream, memory_state), control

    def stats(self) -> dict[str, object]:
        controls: dict[str, float] = {}
        if self.last_controls:
            controls = {k: float(v.float().mean().cpu()) for k, v in self.last_controls.items()}
        return {
            "experts": self.experts.routing_stats(),
            "reasoning": self.reasoner.stats(),
            "controls": controls,
        }


class AERATextLM(nn.Module):
    """Integrated pre-100M AERA text reference model.

    Fixed token events are deliberate for the first architecture-isolation tests.
    Variable byte/event patching is tested separately, then introduced as an ablation.
    """

    def __init__(self, cfg: FullAERAConfig = FullAERAConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.local_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.stages = nn.ModuleList(AERAStage(cfg) for _ in range(cfg.n_stages))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.next_event = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.block_offsets = nn.Parameter(torch.zeros(cfg.block_size, cfg.d_model))
        nn.init.normal_(self.block_offsets, std=0.02)

    def empty_state(self, x: torch.Tensor) -> FullAERAState:
        placeholder = self.token_emb(x[:, :1])
        return FullAERAState([stage.empty_state(placeholder) for stage in self.stages])

    def forward(
        self,
        tokens: torch.Tensor,
        state: FullAERAState | None = None,
        *,
        hard: bool = False,
        update_memory: bool = False,
        return_block_logits: bool = False,
    ) -> dict[str, object]:
        if tokens.ndim != 2:
            raise ValueError("tokens must be [batch,time]")
        b, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("chunk exceeds max_seq_len")
        if state is None:
            state = self.empty_state(tokens)
        if len(state.stages) != len(self.stages):
            raise ValueError("state stage count mismatch")

        pos = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.local_pos(pos)[None]
        new_states: list[AERAState] = []
        controls: list[dict[str, torch.Tensor]] = []
        for stage, stage_state in zip(self.stages, state.stages):
            x, new_state, control = stage(
                x,
                stage_state,
                hard=hard,
                update_memory=update_memory,
            )
            new_states.append(new_state)
            controls.append(control)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        result: dict[str, object] = {
            "logits": logits,
            "hidden": hidden,
            "state": FullAERAState(new_states),
            "next_event_prediction": self.next_event(hidden),
            "controls": controls,
        }
        if return_block_logits:
            block_hidden = hidden[..., None, :] + self.block_offsets[None, None, :, :]
            result["block_logits"] = self.lm_head(block_hidden)
        return result

    def compute_stats(self) -> dict[str, object]:
        return {"stages": [stage.stats() for stage in self.stages]}

    def loss(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
        *,
        event_weight: float = 0.1,
        compute_weight: float = 0.01,
        balance_weight: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        logits = output["logits"]
        hidden = output["hidden"]
        event_pred = output["next_event_prediction"]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(hidden, torch.Tensor)
        assert isinstance(event_pred, torch.Tensor)

        if tokens.size(1) < 2:
            raise ValueError("need at least two tokens for LM loss")
        lm = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        target_event = self.token_emb(tokens[:, 1:]).detach()
        event = F.mse_loss(event_pred[:, :-1].float(), target_event.float())

        controls = output["controls"]
        assert isinstance(controls, list)
        expected_steps = []
        balance_terms = []
        for c in controls:
            depth_p = F.softmax(c["depth_logits"].float(), dim=-1)
            depth_values = torch.arange(1, self.cfg.max_reason_steps + 1, device=tokens.device, dtype=depth_p.dtype)
            expected_steps.append((depth_p * depth_values).sum(dim=-1).mean())
            route_p = F.softmax(c["expert_logits"].float(), dim=-1).mean(dim=(0, 1))
            uniform = torch.full_like(route_p, 1.0 / self.cfg.n_experts)
            balance_terms.append(((route_p - uniform) ** 2).mean())
        compute = torch.stack(expected_steps).mean() / self.cfg.max_reason_steps
        balance = torch.stack(balance_terms).mean()
        total = lm + event_weight * event + compute_weight * compute + balance_weight * balance
        return {
            "total": total,
            "next_token": lm,
            "next_event": event,
            "compute": compute,
            "route_balance": balance,
        }


class SurpriseEventPatcher:
    """Deterministic reference patcher driven by externally supplied surprise.

    A learned entropy/surprise predictor can supply the scores. Keeping segmentation
    separate makes patching semantics easy to test before coupling it to an LM.
    """

    def __init__(self, min_patch: int = 1, max_patch: int = 8, threshold: float = 0.6):
        if not 1 <= min_patch <= max_patch:
            raise ValueError("invalid patch bounds")
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.threshold = threshold

    def spans(self, surprise: torch.Tensor) -> list[tuple[int, int]]:
        if surprise.ndim != 1:
            raise ValueError("surprise must be 1D")
        n = surprise.numel()
        spans: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = start + self.min_patch
            while end < n and end - start < self.max_patch:
                # High-surprise next unit begins a new event.
                if float(surprise[end]) >= self.threshold:
                    break
                end += 1
            spans.append((start, min(end, n)))
            start = min(end, n)
        return spans

    def compress(self, x: torch.Tensor, surprise: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        if x.ndim != 2 or x.size(0) != surprise.numel():
            raise ValueError("x must be [time,dim] aligned with surprise")
        spans = self.spans(surprise)
        patches = torch.stack([x[a:b].mean(dim=0) for a, b in spans])
        return patches, spans


class ModalityAdapterBank(nn.Module):
    """Maps heterogeneous modality features into one shared event space."""

    def __init__(self, d_model: int, modality_dims: dict[str, int]):
        super().__init__()
        if not modality_dims:
            raise ValueError("need at least one modality")
        self.names = tuple(modality_dims)
        self.adapters = nn.ModuleDict({name: nn.Linear(dim, d_model) for name, dim in modality_dims.items()})
        self.modality_emb = nn.ParameterDict({name: nn.Parameter(torch.zeros(d_model)) for name in modality_dims})
        for p in self.modality_emb.values():
            nn.init.normal_(p, std=0.02)

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.adapters:
            raise KeyError(name)
        return self.adapters[name](x) + self.modality_emb[name]


@dataclass(frozen=True)
class ReplayRecord:
    session_id: str
    key: int
    value: int
    verified: bool
    priority: float
    provenance: str


@dataclass
class VerifiedReplayBuffer:
    capacity: int = 4096
    records: list[ReplayRecord] = field(default_factory=list)

    def add(self, record: ReplayRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity :]

    def verified_for_session(self, session_id: str) -> list[ReplayRecord]:
        return [r for r in self.records if r.session_id == session_id and r.verified]

    def prioritized_sample(self, n: int, *, seed: int = 0) -> list[ReplayRecord]:
        verified = [r for r in self.records if r.verified]
        if not verified or n <= 0:
            return []
        rng = random.Random(seed)
        weights = [max(r.priority, 1e-6) for r in verified]
        return rng.choices(verified, weights=weights, k=min(n, len(verified)))

    def current_verified_value(self, session_id: str, key: int) -> int | None:
        # Latest verified record wins; unverified records can never overwrite it.
        for r in reversed(self.records):
            if r.session_id == session_id and r.key == key and r.verified:
                return r.value
        return None


class BlockVerifier:
    """Reference verification policy for parallel/block drafts."""

    def __init__(self, min_confidence: float = 0.8):
        self.min_confidence = min_confidence

    def accept_mask(self, confidence: torch.Tensor) -> torch.Tensor:
        if confidence.ndim < 1:
            raise ValueError("confidence must have a block dimension")
        return confidence >= self.min_confidence

    def accepted_per_call(self, confidence: torch.Tensor) -> float:
        return float(self.accept_mask(confidence).float().sum(dim=-1).mean())


def aera_parameter_accounting(model: AERATextLM) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    expert_total = 0
    for stage in model.stages:
        expert_total += sum(p.numel() for p in stage.experts.parameters())
    active_expert = expert_total * model.cfg.top_k_experts / model.cfg.n_experts
    always_active = total - expert_total
    active_estimate = always_active + active_expert
    return {
        "stored_parameters": total,
        "expert_parameters_stored": expert_total,
        "estimated_active_parameters_per_event": int(round(active_estimate)),
        "estimated_active_fraction": active_estimate / total,
    }
