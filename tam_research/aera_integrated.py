from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState, ExpertMLP, StreamState
from .aera_delta_memory import DeltaFastMemory
from .aera_full import (
    BudgetedLatentReasoner,
    FullAERAConfig,
    FullAERAState,
    FullComputeController,
    LocalCausalAttention,
)


class TrainableSparseExpertLayer(nn.Module):
    """True top-k expert execution with a task-gradient path to the router.

    The forward pass executes only selected experts. Selected gate probabilities are
    normalized by a *detached* selected-probability sum. Therefore the forward
    weights sum to one (including top-k=1), while gradients still flow through the
    selected probabilities and their full softmax coupling to router logits.

    This is a reference estimator, not a claim of optimal MoE training. A fused
    grouped-GEMM implementation is still required before GPU efficiency claims.
    """

    def __init__(self, cfg: FullAERAConfig):
        super().__init__()
        if not 1 <= cfg.top_k_experts <= cfg.n_experts:
            raise ValueError("top_k_experts must be in [1,n_experts]")
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k_experts
        self.experts = nn.ModuleList(
            ExpertMLP(cfg.d_model, cfg.expert_mult) for _ in range(cfg.n_experts)
        )
        self.last_counts: torch.Tensor | None = None
        self.last_mean_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[:-1] != x.shape[:-1] or logits.size(-1) != self.n_experts:
            raise ValueError("router logits shape mismatch")
        shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        flat_logits = logits.reshape(-1, self.n_experts)
        all_probs = F.softmax(flat_logits.float(), dim=-1).to(flat.dtype)
        selected_probs, top_indices = torch.topk(all_probs, self.top_k, dim=-1)
        denom = selected_probs.sum(dim=-1, keepdim=True).detach().clamp_min(1e-6)
        weights = selected_probs / denom

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
            contribution = expert(selected)
            contribution = contribution * weights[token_index, route_index, None]
            # Autocast is allowed to promote an expert contribution (for example
            # BF16 inputs with an FP32 pointwise result). index_add requires source
            # and destination dtypes to match exactly, so restore the activation
            # dtype at the sparse accumulation boundary. The cast is differentiable
            # and preserves the router task-gradient path.
            contribution = contribution.to(dtype=out.dtype)
            out = out.index_add(0, token_index, contribution)

        self.last_counts = counts.detach().cpu()
        self.last_mean_probs = all_probs.detach().float().mean(dim=0).cpu()
        return out.view(shape)

    def stats(self) -> dict[str, object] | None:
        if self.last_counts is None:
            return None
        total = int(self.last_counts.sum())
        mean_probs = (
            self.last_mean_probs.tolist() if self.last_mean_probs is not None else None
        )
        return {
            "assignments": total,
            "per_expert": self.last_counts.tolist(),
            "active_fraction_of_experts_per_event": self.top_k / self.n_experts,
            "mean_router_probabilities": mean_probs,
        }


class BlockDraftHead(nn.Module):
    """Small offset-conditioned multi-unit draft head.

    The expensive core hidden state is reused. A small shared nonlinear draft network
    receives the hidden state plus an offset embedding and predicts several future
    output units in parallel. Verification remains external to this head.
    """

    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.offset = nn.Embedding(block_size, d_model)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

    def forward(self, hidden: torch.Tensor, lm_head: nn.Linear) -> torch.Tensor:
        b, t, d = hidden.shape
        offset_ids = torch.arange(self.block_size, device=hidden.device)
        offset = self.offset(offset_ids).view(1, 1, self.block_size, d)
        h = hidden[:, :, None, :].expand(b, t, self.block_size, d)
        z = self.net(torch.cat((h, offset.expand(b, t, -1, -1)), dim=-1))
        return lm_head(z)


class IntegratedAERAStage(nn.Module):
    def __init__(self, cfg: FullAERAConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.controller = FullComputeController(cfg)
        self.attn = LocalCausalAttention(cfg.d_model, cfg.n_heads, cfg.local_window)
        self.experts = TrainableSparseExpertLayer(cfg)
        self.stream = StreamState(cfg.d_model)
        self.memory = DeltaFastMemory(
            cfg.d_model,
            cfg.memory_dim,
            lr=cfg.fast_memory_lr,
            decay=cfg.fast_memory_decay,
        )
        self.reasoner = BudgetedLatentReasoner(cfg.d_model, cfg.max_reason_steps)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.last_controls: dict[str, torch.Tensor] | None = None

    def empty_state(self, x: torch.Tensor) -> AERAState:
        return AERAState(
            stream=torch.zeros(
                x.size(0), self.cfg.d_model, device=x.device, dtype=x.dtype
            ),
            memory=self.memory.empty_state(x.size(0), x.device, x.dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, torch.Tensor]]:
        if state is None:
            state = self.empty_state(x)
        h = self.norm(x)
        control = self.controller(h)
        self.last_controls = {
            k: v.detach()
            for k, v in control.items()
            if k not in {"expert_logits", "depth_logits"}
        }

        recalled = self.memory.read(h, state.memory)
        h = h + control["memory_read"] * recalled

        attn = self.attn(h)
        h = h + control["attention_need"] * attn

        h = h + self.experts(h, control["expert_logits"])

        stream_out, final_stream = self.stream(h, state.stream)
        h = h + stream_out

        h = h + self.reasoner(h, control["depth_logits"], hard=hard)

        memory_state = state.memory
        if update_memory:
            write = (control["novelty"] * control["memory_write"]).clamp(0.0, 1.0)
            memory_state = self.memory.local_update(h, write, state.memory)

        return self.out_norm(h), AERAState(final_stream, memory_state), control

    def stats(self) -> dict[str, object]:
        controls: dict[str, float] = {}
        if self.last_controls:
            controls = {
                k: float(v.float().mean().cpu())
                for k, v in self.last_controls.items()
            }
        return {
            "experts": self.experts.stats(),
            "reasoning": self.reasoner.stats(),
            "controls": controls,
        }


class IntegratedAERATextLM(nn.Module):
    """Canonical integrated AERA text core for pre-scale experiments."""

    def __init__(self, cfg: FullAERAConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.local_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.stages = nn.ModuleList(IntegratedAERAStage(cfg) for _ in range(cfg.n_stages))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.next_event = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.block_draft = BlockDraftHead(cfg.d_model, cfg.block_size)

    def empty_state(self, tokens: torch.Tensor) -> FullAERAState:
        placeholder = self.token_emb(tokens[:, :1])
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
        _, t = tokens.shape
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
        out: dict[str, object] = {
            "logits": self.lm_head(hidden),
            "hidden": hidden,
            "state": FullAERAState(new_states),
            "next_event_prediction": self.next_event(hidden),
            "controls": controls,
        }
        if return_block_logits:
            out["block_logits"] = self.block_draft(hidden, self.lm_head)
        return out

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

        lm = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        target_event = self.token_emb(tokens[:, 1:]).detach()
        event = F.mse_loss(event_pred[:, :-1].float(), target_event.float())

        expected_steps = []
        balance_terms = []
        for control in controls:
            depth_prob = F.softmax(control["depth_logits"].float(), dim=-1)
            depth = torch.arange(
                1,
                self.cfg.max_reason_steps + 1,
                device=tokens.device,
                dtype=depth_prob.dtype,
            )
            expected_steps.append((depth_prob * depth).sum(dim=-1).mean())
            route_prob = F.softmax(control["expert_logits"].float(), dim=-1)
            mean_route = route_prob.mean(dim=(0, 1))
            uniform = torch.full_like(mean_route, 1.0 / self.cfg.n_experts)
            balance_terms.append(((mean_route - uniform) ** 2).mean())
        compute = torch.stack(expected_steps).mean() / self.cfg.max_reason_steps
        balance = torch.stack(balance_terms).mean()

        block = torch.zeros((), device=tokens.device)
        block_logits = output.get("block_logits")
        if isinstance(block_logits, torch.Tensor):
            valid = tokens.size(1) - self.cfg.block_size
            if valid > 0:
                terms = []
                for offset in range(self.cfg.block_size):
                    pred = block_logits[:, :valid, offset]
                    target = tokens[:, offset + 1 : offset + 1 + valid]
                    terms.append(
                        F.cross_entropy(
                            pred.float().reshape(-1, self.cfg.vocab_size),
                            target.reshape(-1),
                        )
                    )
                block = torch.stack(terms).mean()

        total = (
            lm
            + event_weight * event
            + compute_weight * compute
            + balance_weight * balance
            + block_weight * block
        )
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


def integrated_parameter_accounting(model: IntegratedAERATextLM) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    expert_total = sum(
        p.numel() for stage in model.stages for p in stage.experts.parameters()
    )
    active_expert = expert_total * model.cfg.top_k_experts / model.cfg.n_experts
    always_active = total - expert_total
    active = always_active + active_expert
    return {
        "stored_parameters": total,
        "expert_parameters_stored": expert_total,
        "estimated_active_parameters_per_event": int(round(active)),
        "estimated_active_fraction": active / total,
    }
