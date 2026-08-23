from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState
from .aera_delta_memory import DeltaFastMemory
from .aera_integrated import BlockDraftHead


@dataclass(frozen=True)
class HardwareAERAConfig:
    """Hardware-aware AERA reference configuration.

    The causal processing unit is a bounded chunk. Exact detail is handled by
    Flash-compatible causal attention inside the chunk. Expensive conditional
    decisions, recurrent state updates, and fast-memory writes happen once per
    chunk rather than once per token.
    """

    vocab_size: int = 257
    d_model: int = 128
    n_stages: int = 2
    n_heads: int = 4
    chunk_size: int = 64
    n_experts: int = 8
    max_active_experts: int = 2
    expert_mult: int = 4
    memory_dim: int = 32
    max_reason_steps: int = 4
    block_size: int = 4
    fast_memory_lr: float = 0.2
    fast_memory_decay: float = 0.999

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.chunk_size < 2:
            raise ValueError("chunk_size must be >=2")
        if self.max_active_experts not in (1, 2):
            raise ValueError("reference supports max_active_experts in {1,2}")
        if self.max_active_experts > self.n_experts:
            raise ValueError("active experts cannot exceed stored experts")
        if self.max_reason_steps < 1:
            raise ValueError("max_reason_steps must be >=1")


@dataclass
class HardwareAERAState:
    stages: list[AERAState]

    def detach(self) -> "HardwareAERAState":
        return HardwareAERAState([s.detach() for s in self.stages])


class FlashChunkAttention(nn.Module):
    """Flash-compatible exact causal attention inside one bounded chunk.

    No explicit T x T mask is created. PyTorch SDPA can select a fused/Flash
    backend on supported CUDA hardware.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class ChunkController(nn.Module):
    """One cheap controller decision per causal processing chunk.

    The *start* decision uses only the first event of the current chunk plus the
    carried state, so routing cannot leak future tokens. An end-of-chunk call may
    decide memory writes and latent depth because it only affects the final token
    and future chunks.
    """

    CONTROL_NAMES = (
        "difficulty",
        "novelty",
        "memory_read",
        "memory_write",
        "retrieval_need",
        "precision_budget",
        "block_generation",
        "state_read",
    )

    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__()
        out = cfg.n_experts + 2 + cfg.max_reason_steps + len(self.CONTROL_NAMES)
        self.n_experts = cfg.n_experts
        self.max_reason_steps = cfg.max_reason_steps
        self.proj = nn.Linear(2 * cfg.d_model, out)

    def forward(self, event: torch.Tensor, stream: torch.Tensor) -> dict[str, torch.Tensor]:
        if event.ndim != 2 or stream.shape != event.shape:
            raise ValueError("controller expects [batch,d_model] event and stream")
        raw = self.proj(torch.cat((event, stream), dim=-1))
        e = self.n_experts
        r = self.max_reason_steps
        result: dict[str, torch.Tensor] = {
            "expert_logits": raw[:, :e],
            "expert_count_logits": raw[:, e : e + 2],
            "depth_logits": raw[:, e + 2 : e + 2 + r],
        }
        controls = torch.sigmoid(raw[:, e + 2 + r :])
        for i, name in enumerate(self.CONTROL_NAMES):
            result[name] = controls[:, i : i + 1]
        return result


class StackedChunkExpertBank(nn.Module):
    """Contiguous top-1/top-2 expert bank for chunk-level conditional compute.

    Stored expert matrices are stacked so selected experts can be gathered and
    executed with batched GEMMs. Soft mode keeps a differentiable probability for
    using the second expert. Hard mode makes the count decision discrete.
    """

    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.expert_mult
        self.n_experts = cfg.n_experts
        self.max_active = cfg.max_active_experts
        self.w1 = nn.Parameter(torch.empty(cfg.n_experts, hidden, cfg.d_model))
        self.w2 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, hidden))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        self.last_counts: torch.Tensor | None = None
        self.last_route_probs: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        count_logits: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        b, t, d = x.shape
        if expert_logits.shape != (b, self.n_experts):
            raise ValueError("expert_logits must be [batch,n_experts]")
        if count_logits.shape != (b, 2):
            raise ValueError("count_logits must be [batch,2]")

        route_probs = F.softmax(expert_logits.float(), dim=-1).to(x.dtype)
        selected_probs, idx = torch.topk(route_probs, self.max_active, dim=-1)
        count_probs = F.softmax(count_logits.float(), dim=-1).to(x.dtype)
        if hard:
            chosen_count = count_logits.argmax(dim=-1) + 1
            second_gate = (chosen_count >= 2).to(x.dtype)
        else:
            # Differentiable expected use of the second expert. This reference
            # computes top-2 in training; production grouped kernels can exploit
            # hard count decisions after the controller is learned.
            chosen_count = 1 + (count_probs[:, 1] >= 0.5).long()
            second_gate = count_probs[:, 1]

        p1 = selected_probs[:, 0]
        if self.max_active == 1:
            weight = torch.ones(b, 1, device=x.device, dtype=x.dtype)
            idx = idx[:, :1]
        else:
            p2 = selected_probs[:, 1]
            denom = (p1 + second_gate * p2).clamp_min(1e-6)
            weight = torch.stack((p1 / denom, second_gate * p2 / denom), dim=-1)

        sw1 = self.w1[idx]
        sw2 = self.w2[idx]
        h = torch.einsum("btd,bkhd->bkth", x, sw1)
        h = F.gelu(h)
        y = torch.einsum("bkth,bkdh->bktd", h, sw2)
        out = (y * weight[:, :, None, None]).sum(dim=1)

        self.last_counts = chosen_count.detach().cpu()
        self.last_route_probs = route_probs.detach().float().mean(dim=0).cpu()
        return out

    def stats(self) -> dict[str, object] | None:
        if self.last_counts is None:
            return None
        counts = self.last_counts.float()
        return {
            "stored_experts": self.n_experts,
            "mean_active_experts": float(counts.mean()),
            "min_active_experts": int(counts.min()),
            "max_active_experts": int(counts.max()),
            "mean_route_probabilities": self.last_route_probs.tolist() if self.last_route_probs is not None else None,
        }


class ChunkLatentReasoner(nn.Module):
    """Adaptive latent reasoning over one chunk/workspace summary."""

    def __init__(self, d_model: int, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.cell = nn.GRUCell(d_model, d_model)
        self.last_steps: torch.Tensor | None = None
        self.last_expected: torch.Tensor | None = None

    def forward(self, summary: torch.Tensor, depth_logits: torch.Tensor, *, hard: bool) -> torch.Tensor:
        if depth_logits.shape != (summary.size(0), self.max_steps):
            raise ValueError("depth_logits shape mismatch")
        probs = F.softmax(depth_logits.float(), dim=-1).to(summary.dtype)
        values = torch.arange(1, self.max_steps + 1, device=summary.device, dtype=summary.dtype)
        self.last_expected = (probs * values[None]).sum(dim=-1).detach().cpu()

        if not hard:
            current = summary
            states = []
            for _ in range(self.max_steps):
                current = self.cell(current, current)
                states.append(current)
            stacked = torch.stack(states, dim=1)
            self.last_steps = None
            return (stacked * probs[:, :, None]).sum(dim=1)

        chosen = depth_logits.argmax(dim=-1) + 1
        current = summary
        for step in range(1, self.max_steps + 1):
            idx = (chosen >= step).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break
            selected = current.index_select(0, idx)
            updated = self.cell(selected, selected)
            current = current.index_copy(0, idx, updated)
        self.last_steps = chosen.detach().cpu()
        return current

    def stats(self) -> dict[str, object] | None:
        if self.last_steps is not None:
            x = self.last_steps.float()
            return {"mode": "hard", "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
        if self.last_expected is not None:
            x = self.last_expected.float()
            return {"mode": "soft", "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
        return None


class HardwareAERAStage(nn.Module):
    """One hardware-aware AERA stage.

    Causality rule:
    * chunk-start decisions can affect every token but only see the first token and
      prior state;
    * chunk-end decisions see the final causal token representation and may affect
      only the final token, persistent state, memory, or future chunks.
    """

    def __init__(self, cfg: HardwareAERAConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.controller = ChunkController(cfg)
        self.state_to_chunk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn = FlashChunkAttention(cfg.d_model, cfg.n_heads)
        self.experts = StackedChunkExpertBank(cfg)
        self.reasoner = ChunkLatentReasoner(cfg.d_model, cfg.max_reason_steps)
        self.stream_cell = nn.GRUCell(cfg.d_model, cfg.d_model)
        self.memory = DeltaFastMemory(
            cfg.d_model,
            cfg.memory_dim,
            lr=cfg.fast_memory_lr,
            decay=cfg.fast_memory_decay,
        )
        self.reason_to_chunk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.last_start_controls: dict[str, torch.Tensor] | None = None
        self.last_end_controls: dict[str, torch.Tensor] | None = None

    def empty_state(self, x: torch.Tensor) -> AERAState:
        b = x.size(0)
        return AERAState(
            stream=torch.zeros(b, self.cfg.d_model, device=x.device, dtype=x.dtype),
            memory=self.memory.empty_state(b, x.device, x.dtype),
        )

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, dict[str, torch.Tensor]]]:
        if events.ndim != 3 or events.size(1) > self.cfg.chunk_size:
            raise ValueError("events must be [batch,time,d_model] within chunk_size")
        if state is None:
            state = self.empty_state(events)

        h = self.norm(events)
        # Start control is causal for every position: first event + prior stream only.
        start_control = self.controller(h[:, 0], state.stream)
        self.last_start_controls = {k: v.detach() for k, v in start_control.items() if "logits" not in k}

        memory_read = self.memory.read(h[:, :1], state.memory).squeeze(1)
        carried = self.state_to_chunk(state.stream)
        context = (
            start_control["state_read"] * carried
            + start_control["memory_read"] * memory_read
        )
        h = h + context[:, None, :]

        h = h + self.attn(h)
        h = h + self.experts(
            h,
            start_control["expert_logits"],
            start_control["expert_count_logits"],
            hard=hard,
        )

        # The last position is a causal summary of the whole chunk after attention.
        end_summary = h[:, -1]
        end_control = self.controller(end_summary, state.stream)
        self.last_end_controls = {k: v.detach() for k, v in end_control.items() if "logits" not in k}
        reasoned = self.reasoner(end_summary, end_control["depth_logits"], hard=hard)

        # Latent reasoning can alter the boundary prediction and future state without
        # leaking future information into earlier token logits.
        last_mask = torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
        last_mask[-1] = 1
        h = h + self.reason_to_chunk(reasoned)[:, None, :] * last_mask[None, :, None]
        h = self.out_norm(h)

        final_stream = self.stream_cell(reasoned, state.stream)
        memory_state = state.memory
        if update_memory:
            write = (end_control["novelty"] * end_control["memory_write"]).clamp(0.0, 1.0)
            memory_state = self.memory.local_update(reasoned[:, None, :], write[:, None, :], state.memory)

        controls = {"start": start_control, "end": end_control}
        return h, AERAState(final_stream, memory_state), controls

    def stats(self) -> dict[str, object]:
        def means(x: dict[str, torch.Tensor] | None) -> dict[str, float]:
            if not x:
                return {}
            return {k: float(v.float().mean()) for k, v in x.items()}

        return {
            "experts": self.experts.stats(),
            "reasoning": self.reasoner.stats(),
            "start_controls": means(self.last_start_controls),
            "end_controls": means(self.last_end_controls),
        }


class HardwareAwareAERATextLM(nn.Module):
    """Causal, chunk-stateful, hardware-aware AERA language-model reference."""

    def __init__(self, cfg: HardwareAERAConfig = HardwareAERAConfig()):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.local_pos = nn.Embedding(cfg.chunk_size, cfg.d_model)
        self.stages = nn.ModuleList(HardwareAERAStage(cfg) for _ in range(cfg.n_stages))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.next_event = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.block_draft = BlockDraftHead(cfg.d_model, cfg.block_size)

    def empty_state(self, tokens: torch.Tensor) -> HardwareAERAState:
        placeholder = self.token_emb(tokens[:, :1])
        return HardwareAERAState([stage.empty_state(placeholder) for stage in self.stages])

    def forward(
        self,
        tokens: torch.Tensor,
        state: HardwareAERAState | None = None,
        *,
        hard: bool = False,
        update_memory: bool = False,
        return_block_logits: bool = False,
    ) -> dict[str, object]:
        if tokens.ndim != 2 or tokens.size(1) < 1:
            raise ValueError("tokens must be nonempty [batch,time]")
        if state is None:
            state = self.empty_state(tokens)
        if len(state.stages) != len(self.stages):
            raise ValueError("state stage count mismatch")

        outputs: list[torch.Tensor] = []
        control_history: list[list[dict[str, dict[str, torch.Tensor]]]] = []
        current_state = state
        for start in range(0, tokens.size(1), self.cfg.chunk_size):
            chunk = tokens[:, start : start + self.cfg.chunk_size]
            pos = torch.arange(chunk.size(1), device=tokens.device)
            x = self.token_emb(chunk) + self.local_pos(pos)[None]
            new_states: list[AERAState] = []
            stage_controls: list[dict[str, dict[str, torch.Tensor]]] = []
            for stage, stage_state in zip(self.stages, current_state.stages):
                x, new_state, controls = stage.forward_chunk(
                    x,
                    stage_state,
                    hard=hard,
                    update_memory=update_memory,
                )
                new_states.append(new_state)
                stage_controls.append(controls)
            outputs.append(x)
            control_history.append(stage_controls)
            current_state = HardwareAERAState(new_states)

        hidden = self.norm(torch.cat(outputs, dim=1))
        result: dict[str, object] = {
            "logits": self.lm_head(hidden),
            "hidden": hidden,
            "state": current_state,
            "next_event_prediction": self.next_event(hidden),
            "controls": control_history,
        }
        if return_block_logits:
            result["block_logits"] = self.block_draft(hidden, self.lm_head)
        return result

    def objective(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
        *,
        event_weight: float = 0.05,
        compute_weight: float = 0.002,
        balance_weight: float = 0.02,
        block_weight: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        logits = output["logits"]
        event_pred = output["next_event_prediction"]
        controls = output["controls"]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(event_pred, torch.Tensor)
        assert isinstance(controls, list)
        if tokens.size(1) < 2:
            raise ValueError("need >=2 tokens")

        lm = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        target_event = self.token_emb(tokens[:, 1:]).detach()
        event = F.mse_loss(event_pred[:, :-1].float(), target_event.float())

        depth_cost = []
        expert_count_cost = []
        route_balance = []
        for chunk_controls in controls:
            for stage_controls in chunk_controls:
                start = stage_controls["start"]
                end = stage_controls["end"]
                count_p = F.softmax(start["expert_count_logits"].float(), dim=-1)
                expert_count_cost.append((count_p * torch.tensor([1.0, 2.0], device=tokens.device)).sum(dim=-1).mean() / 2.0)
                route_p = F.softmax(start["expert_logits"].float(), dim=-1).mean(dim=0)
                uniform = torch.full_like(route_p, 1.0 / self.cfg.n_experts)
                route_balance.append(((route_p - uniform) ** 2).mean())
                depth_p = F.softmax(end["depth_logits"].float(), dim=-1)
                depth = torch.arange(1, self.cfg.max_reason_steps + 1, device=tokens.device, dtype=depth_p.dtype)
                depth_cost.append((depth_p * depth).sum(dim=-1).mean() / self.cfg.max_reason_steps)

        compute = 0.5 * (torch.stack(depth_cost).mean() + torch.stack(expert_count_cost).mean())
        balance = torch.stack(route_balance).mean()

        block = torch.zeros((), device=tokens.device)
        block_logits = output.get("block_logits")
        if isinstance(block_logits, torch.Tensor):
            valid = tokens.size(1) - self.cfg.block_size
            if valid > 0:
                terms = []
                for offset in range(self.cfg.block_size):
                    pred = block_logits[:, :valid, offset]
                    target = tokens[:, offset + 1 : offset + 1 + valid]
                    terms.append(F.cross_entropy(pred.float().reshape(-1, self.cfg.vocab_size), target.reshape(-1)))
                block = torch.stack(terms).mean()

        total = lm + event_weight * event + compute_weight * compute + balance_weight * balance + block_weight * block
        return {
            "total": total,
            "next_token": lm,
            "next_event": event,
            "compute": compute,
            "route_balance": balance,
            "block": block,
        }

    def stats(self) -> dict[str, object]:
        return {"stages": [stage.stats() for stage in self.stages]}


def hardware_parameter_accounting(model: HardwareAwareAERATextLM, mean_active_experts: float = 1.5) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    expert_total = sum(stage.experts.w1.numel() + stage.experts.w2.numel() for stage in model.stages)
    per_expert = expert_total / (model.cfg.n_stages * model.cfg.n_experts)
    active_expert = per_expert * mean_active_experts * model.cfg.n_stages
    always = total - expert_total
    active = always + active_expert
    return {
        "stored_parameters": total,
        "expert_parameters_stored": expert_total,
        "assumed_mean_active_experts": mean_active_experts,
        "estimated_active_parameters_per_chunk": int(round(active)),
        "estimated_active_fraction": active / total,
    }
