from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v8 import StageRouteGate
from .aera_hardware_core_v19 import TokenwiseFastMemoryStage
from .aera_hardware_core_v23 import (
    BudgetedSparseDualDeltaFastMemoryStage,
    HardwareAwareAERATextLMV23,
    select_budgeted_event_pairs,
    sparse_write_budget,
)

CONTEXT_WINDOW = 8
EPISODIC_CAPACITY = 48
DUPLICATE_SIMILARITY = 0.95
READ_TOP_K = 4
READ_TEMPERATURE = 0.10
MIN_STRENGTH = 1e-4


@dataclass
class ContextualEpisodicMemoryState:
    """Newest-first bounded session-local contextual key/value memory."""

    keys: torch.Tensor
    values: torch.Tensor
    strengths: torch.Tensor
    valid: torch.Tensor

    def detach(self) -> "ContextualEpisodicMemoryState":
        return ContextualEpisodicMemoryState(
            self.keys.detach(),
            self.values.detach(),
            self.strengths.detach(),
            self.valid.detach(),
        )


def causal_contextualize(h: torch.Tensor, *, window: int = CONTEXT_WINDOW) -> torch.Tensor:
    """Add the mean of at most the previous `window` causal events to each event."""
    if h.ndim != 3:
        raise ValueError("h must be [batch,time,dim]")
    if window < 1:
        raise ValueError("window must be positive")
    time = h.size(1)
    if time == 0:
        return h
    prefix = torch.cat((torch.zeros_like(h[:, :1]), h.cumsum(dim=1)), dim=1)
    end = torch.arange(time, device=h.device, dtype=torch.long)
    start = (end - window).clamp_min(0)
    previous_sum = prefix.index_select(1, end) - prefix.index_select(1, start)
    counts = (end - start).clamp_min(1).to(h.dtype).view(1, time, 1)
    previous_mean = previous_sum / counts
    return h + previous_mean


def _gather_slots(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    if x.ndim == 2:
        return x.gather(1, indices)
    raise ValueError("slot tensor must be rank 2 or 3")


class VectorizedContextualEpisodicMemory(nn.Module):
    """V24 bounded contextual KV state with one block update per completed chunk."""

    def __init__(self, source: nn.Module, *, capacity: int = EPISODIC_CAPACITY) -> None:
        super().__init__()
        for name in ("q", "k", "v", "out", "memory_dim"):
            if not hasattr(source, name):
                raise TypeError(f"source memory missing {name}")
        self.memory_dim = int(source.memory_dim)
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        d_model = int(source.q.in_features)
        self.q = nn.Linear(d_model, self.memory_dim, bias=False)
        self.k = nn.Linear(d_model, self.memory_dim, bias=False)
        self.v = nn.Linear(d_model, self.memory_dim, bias=False)
        self.out = nn.Linear(self.memory_dim, d_model, bias=False)
        self.q.load_state_dict(source.q.state_dict())
        self.k.load_state_dict(source.k.state_dict())
        self.v.load_state_dict(source.v.state_dict())
        self.out.load_state_dict(source.out.state_dict())
        self.differentiable_pretraining = bool(
            getattr(source, "differentiable_pretraining", False)
        )

    def set_differentiable_pretraining(self, enabled: bool) -> None:
        self.differentiable_pretraining = bool(enabled)

    def empty_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ContextualEpisodicMemoryState:
        shape = (batch_size, self.capacity, self.memory_dim)
        return ContextualEpisodicMemoryState(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
            strengths=torch.zeros(batch_size, self.capacity, device=device, dtype=dtype),
            valid=torch.zeros(batch_size, self.capacity, device=device, dtype=torch.bool),
        )

    def read(
        self,
        contextual_events: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> torch.Tensor:
        if contextual_events.ndim != 3:
            raise ValueError("contextual_events must be [batch,time,d_model]")
        if state.keys.shape != state.values.shape:
            raise ValueError("episodic key/value state mismatch")
        if state.keys.shape[:2] != state.valid.shape:
            raise ValueError("episodic validity shape mismatch")
        if state.strengths.shape != state.valid.shape:
            raise ValueError("episodic strength shape mismatch")

        query = F.normalize(self.q(contextual_events), dim=-1)
        keys = F.normalize(state.keys, dim=-1)
        similarity = torch.einsum("btd,bsd->bts", query, keys)
        strength_bias = torch.log(
            state.strengths.clamp(MIN_STRENGTH, 1.0)
        )[:, None, :]
        logits = (similarity + strength_bias) / READ_TEMPERATURE
        masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
        top_k = min(READ_TOP_K, self.capacity)
        top_logits, top_indices = torch.topk(masked, k=top_k, dim=-1)
        top_valid = state.valid[:, None, :].expand(
            -1, contextual_events.size(1), -1
        ).gather(-1, top_indices)
        safe_logits = top_logits.masked_fill(~top_valid, -1e9)
        weights = torch.softmax(safe_logits.float(), dim=-1).to(contextual_events.dtype)
        weights = weights * top_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        expanded_values = state.values[:, None, :, :].expand(
            -1, contextual_events.size(1), -1, -1
        )
        gathered_values = expanded_values.gather(
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, self.memory_dim),
        )
        recalled = (weights.unsqueeze(-1) * gathered_values).sum(dim=2)
        return self.out(recalled)

    def _vectorized_update(
        self,
        address_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
        *,
        detach_inputs: bool,
    ) -> ContextualEpisodicMemoryState:
        address = address_source.detach() if detach_inputs else address_source
        payload = payload_source.detach() if detach_inputs else payload_source
        strength = write_strength.detach() if detach_inputs else write_strength
        old_keys = state.keys.detach() if detach_inputs else state.keys
        old_values = state.values.detach() if detach_inputs else state.values
        old_strengths = state.strengths.detach() if detach_inputs else state.strengths
        old_valid = state.valid.detach()

        new_keys = F.normalize(self.k(address), dim=-1)
        new_values = torch.tanh(self.v(payload))
        new_strengths = strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0

        incoming_similarity = torch.einsum("bkd,bjd->bkj", new_keys, new_keys)
        k_count = new_keys.size(1)
        position = torch.arange(k_count, device=new_keys.device)
        later = position[None, :, None] < position[None, None, :]
        shadowed_incoming = (
            incoming_similarity.ge(DUPLICATE_SIMILARITY)
            & new_valid[:, :, None]
            & new_valid[:, None, :]
            & later
        ).any(dim=2)
        new_valid = new_valid & ~shadowed_incoming

        similarity = torch.einsum("bkd,bsd->bks", new_keys, old_keys)
        duplicate_old = (
            similarity.ge(DUPLICATE_SIMILARITY)
            & new_valid[:, :, None]
            & old_valid[:, None, :]
        ).any(dim=1)
        keep_old = old_valid & ~duplicate_old

        new_keys = new_keys.flip(1)
        new_values = new_values.flip(1)
        new_strengths = new_strengths.flip(1)
        new_valid = new_valid.flip(1)

        all_keys = torch.cat((new_keys, old_keys), dim=1)
        all_values = torch.cat((new_values, old_values), dim=1)
        all_strengths = torch.cat((new_strengths, old_strengths), dim=1)
        all_valid = torch.cat((new_valid, keep_old), dim=1)

        total = all_valid.size(1)
        slot_position = torch.arange(
            total, device=all_valid.device, dtype=torch.float32
        )
        priority = all_valid.float() * 2.0 - slot_position[None, :] * 1e-6
        keep_indices = torch.topk(
            priority,
            k=self.capacity,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        next_state = ContextualEpisodicMemoryState(
            keys=_gather_slots(all_keys, keep_indices),
            values=_gather_slots(all_values, keep_indices),
            strengths=_gather_slots(all_strengths, keep_indices),
            valid=_gather_slots(all_valid, keep_indices),
        )
        return next_state.detach() if detach_inputs else next_state

    def update_block(
        self,
        address_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        if address_source.shape != payload_source.shape or address_source.ndim != 3:
            raise ValueError("address/payload sources must match [batch,candidates,d_model]")
        if write_strength.shape != (*address_source.shape[:-1], 1):
            raise ValueError("write_strength must be [batch,candidates,1]")
        if self.differentiable_pretraining:
            return self._vectorized_update(
                address_source,
                payload_source,
                write_strength,
                state,
                detach_inputs=False,
            )
        with torch.no_grad():
            return self._vectorized_update(
                address_source,
                payload_source,
                write_strength,
                state,
                detach_inputs=True,
            )


class VectorizedContextualEpisodicMemoryStage(TokenwiseFastMemoryStage):
    """V23 compute path with contextual episodic read/write state."""

    def __init__(self, source: BudgetedSparseDualDeltaFastMemoryStage) -> None:
        super().__init__(source)
        self.pair_write_gate = source.pair_write_gate
        self.memory = VectorizedContextualEpisodicMemory(source.memory)
        self.last_selected_indices: torch.Tensor | None = None
        self.last_selected_count = 0
        self.last_candidate_count = 0
        self.last_pair_gate: torch.Tensor | None = None
        self.last_pair_strength: torch.Tensor | None = None
        self.last_vectorized_update_calls = 0

    def empty_state(self, x: torch.Tensor) -> AERAState:
        return AERAState(
            stream=torch.zeros(
                x.size(0), self.cfg.d_model, device=x.device, dtype=x.dtype
            ),
            memory=self.memory.empty_state(x.size(0), x.device, x.dtype),
        )

    def _tokenwise_context(
        self,
        h: torch.Tensor,
        state: AERAState,
        start_control: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(state.memory, ContextualEpisodicMemoryState):
            raise TypeError("v24 requires ContextualEpisodicMemoryState")
        contextual = causal_contextualize(h)
        memory_read = self.memory.read(contextual, state.memory)
        carried = self.state_to_chunk(state.stream)
        context = (
            carried[:, None, :]
            + start_control["memory_read"][:, None, :] * memory_read
        )
        return context, memory_read

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, dict[str, torch.Tensor]]]:
        prior_state = state if state is not None else self.empty_state(events)
        if not isinstance(prior_state.memory, ContextualEpisodicMemoryState):
            raise TypeError("v24 stage received non-episodic memory state")

        h_out, next_state, controls = super().forward_chunk(
            events,
            prior_state,
            hard=hard,
            update_memory=False,
        )
        candidate_count = max(int(events.size(1)) - 1, 0)
        self.last_candidate_count = candidate_count
        self.last_selected_count = 0
        self.last_selected_indices = None
        self.last_vectorized_update_calls = 0
        if not update_memory or candidate_count == 0:
            return h_out, next_state, controls

        base_h = self.norm(events)
        contextual = causal_contextualize(base_h)
        address_source = contextual[:, :-1]
        payload_source = contextual[:, 1:]
        pair_features = torch.cat((address_source, payload_source), dim=-1)
        pair_logits = self.pair_write_gate(pair_features)
        pair_gate = torch.sigmoid(pair_logits)
        end_control = controls["end"]
        chunk_strength = (
            end_control["novelty"] * end_control["memory_write"]
        ).clamp(0.0, 1.0)
        write_strength = pair_gate * chunk_strength[:, None, :]
        selected = select_budgeted_event_pairs(
            address_source,
            payload_source,
            write_strength,
            pair_logits,
            differentiable_selector=self.memory.differentiable_pretraining,
        )
        if selected.hard_count != sparse_write_budget(candidate_count):
            raise RuntimeError("v24 sparse write budget mismatch")

        self.last_pair_gate = pair_gate.detach()
        self.last_pair_strength = selected.strength.detach()
        self.last_selected_indices = selected.indices.detach()
        self.last_selected_count = selected.hard_count
        memory_state = self.memory.update_block(
            selected.address,
            selected.payload,
            selected.strength,
            prior_state.memory,
        )
        self.last_vectorized_update_calls = 1
        return h_out, AERAState(next_state.stream, memory_state), controls

    def stats(self) -> dict[str, object]:
        result = super().stats()
        result["event_pair_write_candidates"] = self.last_candidate_count
        result["event_pair_selected_writes"] = self.last_selected_count
        result["event_pair_selected_fraction"] = (
            self.last_selected_count / self.last_candidate_count
            if self.last_candidate_count
            else 0.0
        )
        result["vectorized_memory_update_calls"] = self.last_vectorized_update_calls
        if self.last_selected_indices is not None:
            result["event_pair_selected_indices"] = (
                self.last_selected_indices.cpu().tolist()
            )
        return result


def _as_epi(memory: Any) -> ContextualEpisodicMemoryState:
    if not isinstance(memory, ContextualEpisodicMemoryState):
        raise TypeError("v24 routing requires ContextualEpisodicMemoryState")
    return memory


def _select_epi_state(state: AERAState, idx: torch.Tensor) -> AERAState:
    m = _as_epi(state.memory)
    return AERAState(
        stream=state.stream.index_select(0, idx),
        memory=ContextualEpisodicMemoryState(
            m.keys.index_select(0, idx),
            m.values.index_select(0, idx),
            m.strengths.index_select(0, idx),
            m.valid.index_select(0, idx),
        ),
    )


def _restore_epi_dtype(base: AERAState, update: AERAState) -> AERAState:
    bm = _as_epi(base.memory)
    um = _as_epi(update.memory)
    return AERAState(
        stream=update.stream.to(dtype=base.stream.dtype),
        memory=ContextualEpisodicMemoryState(
            um.keys.to(dtype=bm.keys.dtype),
            um.values.to(dtype=bm.values.dtype),
            um.strengths.to(dtype=bm.strengths.dtype),
            um.valid,
        ),
    )


def _merge_epi_state(
    base: AERAState,
    update: AERAState,
    idx: torch.Tensor,
) -> AERAState:
    update = _restore_epi_dtype(base, update)
    bm = _as_epi(base.memory)
    um = _as_epi(update.memory)
    return AERAState(
        stream=base.stream.index_copy(0, idx, update.stream),
        memory=ContextualEpisodicMemoryState(
            bm.keys.index_copy(0, idx, um.keys),
            bm.values.index_copy(0, idx, um.values),
            bm.strengths.index_copy(0, idx, um.strengths),
            bm.valid.index_copy(0, idx, um.valid),
        ),
    )


def _blend_epi_state(
    base: AERAState,
    update: AERAState,
    gate: torch.Tensor,
    *,
    hard_validity: bool,
) -> AERAState:
    bm = _as_epi(base.memory)
    um = _as_epi(update.memory)
    gs = gate.to(base.stream.dtype)
    g3 = gate[:, None, :].to(bm.keys.dtype)
    g2 = gate.to(bm.strengths.dtype)
    if hard_validity:
        use_update = gate[:, 0].ge(0.5)[:, None]
        valid = torch.where(use_update, um.valid, bm.valid)
    else:
        valid = bm.valid | um.valid
    return AERAState(
        stream=base.stream + gs * (update.stream - base.stream),
        memory=ContextualEpisodicMemoryState(
            bm.keys + g3 * (um.keys - bm.keys),
            bm.values + g3 * (um.values - bm.values),
            bm.strengths + g2 * (um.strengths - bm.strengths),
            valid,
        ),
    )


class HardwareAwareAERATextLMV24(HardwareAwareAERATextLMV23):
    """V23 adaptive compute with vectorized contextual episodic fast memory."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            VectorizedContextualEpisodicMemoryStage(stage) for stage in self.stages
        )
        self.set_memory_pretraining_mode(False)

    def _v24_stages_ready(self) -> bool:
        return bool(self.stages) and all(
            isinstance(stage, VectorizedContextualEpisodicMemoryStage)
            for stage in self.stages
        )

    def set_memory_pretraining_mode(self, enabled: bool) -> None:
        # The v18 constructor intentionally calls this virtual method before the
        # v19-v23 constructors have wrapped their stages. During that inherited
        # initialization window, delegate to the inherited implementation; once
        # v24 stages exist, switch the episodic memories directly.
        if not self._v24_stages_ready():
            super().set_memory_pretraining_mode(enabled)
            return
        for stage in self.stages:
            stage.memory.set_differentiable_pretraining(enabled)

    def memory_pretraining_mode(self) -> bool:
        if not self._v24_stages_ready():
            return super().memory_pretraining_mode()
        flags = [stage.memory.differentiable_pretraining for stage in self.stages]
        if len(set(flags)) != 1:
            raise RuntimeError("v24 memory pretraining flags disagree")
        return flags[0]

    def _route_one_stage(
        self,
        x: torch.Tensor,
        stage: nn.Module,
        stage_state: AERAState,
        router: StageRouteGate,
        *,
        route_mode: str,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, object]]:
        gate, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        prob = torch.sigmoid(logits)

        if route_mode == "hard_sparse":
            run_idx = (gate[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
            if run_idx.numel() == 0:
                return x, stage_state, {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": 0.0,
                    "start": None,
                    "end": None,
                }
            selected_x = x.index_select(0, run_idx)
            selected_state = _select_epi_state(stage_state, run_idx)
            selected_y, selected_new_state, selected_controls = stage.forward_chunk(
                selected_x,
                selected_state,
                hard=True,
                update_memory=update_memory,
            )
            selected_y = selected_y.to(dtype=x.dtype)
            selected_new_state = _restore_epi_dtype(
                selected_state, selected_new_state
            )
            return (
                x.index_copy(0, run_idx, selected_y),
                _merge_epi_state(stage_state, selected_new_state, run_idx),
                {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": float(run_idx.numel() / x.size(0)),
                    "start": selected_controls["start"],
                    "end": selected_controls["end"],
                },
            )

        isolate = bool(
            route_mode == "straight_through"
            and getattr(self, "_isolate_router_task_gradient", False)
        )
        task_gate = gate.detach() if isolate else gate
        processed, processed_state, controls = stage.forward_chunk(
            x,
            stage_state,
            hard=False,
            update_memory=update_memory,
        )
        processed = processed.to(dtype=x.dtype)
        processed_state = _restore_epi_dtype(stage_state, processed_state)
        gate_for_residual = task_gate.to(dtype=x.dtype)
        y = x + gate_for_residual[:, None, :] * (processed - x)
        new_state = _blend_epi_state(
            stage_state,
            processed_state,
            task_gate,
            hard_validity=route_mode == "straight_through",
        )
        info: dict[str, object] = {
            "stage_route_probability": prob,
            "stage_route_gate": task_gate,
            "executed_fraction": 1.0,
            "start": controls["start"],
            "end": controls["end"],
        }
        if isolate:
            info["task_router_gradient_isolated"] = True
        return y, new_state, info


def episodic_state_bytes_per_session(
    *,
    n_stages: int,
    memory_dim: int,
    capacity: int = EPISODIC_CAPACITY,
    element_size: int = 4,
) -> int:
    per_stage = (
        2 * capacity * memory_dim * element_size
        + capacity * element_size
        + capacity
    )
    return n_stages * per_stage


def vectorized_contextual_episodic_protocol() -> dict[str, object]:
    return {
        "version": "aera-v24-vectorized-contextual-episodic-memory",
        "source": "aera-v23",
        "context_window_previous_events": CONTEXT_WINDOW,
        "context_rule": "h_t + mean(previous up to 8 normalized stage events)",
        "capacity_slots_per_stage": EPISODIC_CAPACITY,
        "duplicate_similarity_threshold": DUPLICATE_SIMILARITY,
        "within_incoming_block_newest_wins": True,
        "read_top_k": READ_TOP_K,
        "read_temperature": READ_TEMPERATURE,
        "read_score": "(cosine + log(clamp(strength,1e-4,1))) / temperature",
        "write_budget_changed_from_v23": False,
        "controlled_selected_writes": sparse_write_budget(5),
        "real_language_selected_writes": sparse_write_budget(255),
        "sequential_delta_recurrence": False,
        "inverse_covariance_state": False,
        "vectorized_update_calls_per_completed_stage_chunk": 1,
        "qkvout_dimension_changed": False,
        "extra_learned_parameters": 0,
        "routing_changed": False,
        "stream_changed": False,
        "deployment_local_update_detached": True,
        "base_pretraining_update_differentiable": True,
        "gpu_authorized": False,
    }
