from __future__ import annotations

"""AERA-v25 Factorized Identity-Context Episodic Memory (FICEM).

Preregistered in issue #359 after the valid v24.1 controlled failure #358.
V25 changes only the episodic address representation.  Sparse candidate selection,
vectorized bounded state, payload encoding, recurrent stream and adaptive compute
remain inherited from v24/v23.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v23 import select_budgeted_event_pairs, sparse_write_budget
from .aera_hardware_core_v24 import (
    DUPLICATE_SIMILARITY,
    EPISODIC_CAPACITY,
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
    ContextualEpisodicMemoryState,
    HardwareAwareAERATextLMV24,
    VectorizedContextualEpisodicMemoryStage,
    _gather_slots,
    causal_contextualize,
    episodic_state_bytes_per_session,
)

_EQUAL_SUBSPACE_SCALE = math.sqrt(0.5)


def causal_identity_context(
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current-event identity source, strictly-prior context, and v24 sum.

    `identity` is exactly the current pre-memory event. `context` contains only the
    mean of previous <=8 events. `contextual` is bit-compatible with v24's
    `causal_contextualize` and remains the source for pair selection and payloads.
    """
    contextual = causal_contextualize(h)
    context = contextual - h
    return h, context, contextual


class FactorizedIdentityContextEpisodicMemory(nn.Module):
    """Bounded v24 episodic state with an identity/context-factorized address."""

    def __init__(
        self,
        source: nn.Module,
        *,
        capacity: int = EPISODIC_CAPACITY,
    ) -> None:
        super().__init__()
        for name in ("q", "k", "v", "out", "memory_dim"):
            if not hasattr(source, name):
                raise TypeError(f"source memory missing {name}")
        self.memory_dim = int(source.memory_dim)
        if self.memory_dim % 2:
            raise ValueError("v25 requires an even memory_dim")
        self.identity_dim = self.memory_dim // 2
        self.context_dim = self.memory_dim // 2
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")

        d_model = int(source.q.in_features)
        self.identity_proj = nn.Linear(d_model, self.identity_dim, bias=False)
        self.context_proj = nn.Linear(d_model, self.context_dim, bias=False)
        self.v = nn.Linear(d_model, self.memory_dim, bias=False)
        self.out = nn.Linear(self.memory_dim, int(source.out.out_features), bias=False)
        with torch.no_grad():
            self.identity_proj.weight.copy_(source.q.weight[: self.identity_dim])
            self.context_proj.weight.copy_(source.k.weight[: self.context_dim])
            self.v.weight.copy_(source.v.weight)
            self.out.weight.copy_(source.out.weight)
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
        key_shape = (batch_size, self.capacity, self.memory_dim)
        value_shape = (batch_size, self.capacity, self.memory_dim)
        return ContextualEpisodicMemoryState(
            keys=torch.zeros(key_shape, device=device, dtype=dtype),
            values=torch.zeros(value_shape, device=device, dtype=dtype),
            strengths=torch.zeros(batch_size, self.capacity, device=device, dtype=dtype),
            valid=torch.zeros(batch_size, self.capacity, device=device, dtype=torch.bool),
        )

    def address_factors(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if identity_source.shape != context_source.shape:
            raise ValueError("identity/context sources must have identical shapes")
        identity = F.normalize(self.identity_proj(identity_source), dim=-1)
        context = F.normalize(self.context_proj(context_source), dim=-1)
        combined = torch.cat(
            (
                identity * _EQUAL_SUBSPACE_SCALE,
                context * _EQUAL_SUBSPACE_SCALE,
            ),
            dim=-1,
        )
        # When t=0 has an exact-zero causal context, normalize the remaining
        # identity half rather than reducing the address norm.
        combined = F.normalize(combined, dim=-1)
        return identity, context, combined

    def read(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> torch.Tensor:
        if identity_source.ndim != 3 or context_source.ndim != 3:
            raise ValueError("read sources must be [batch,time,d_model]")
        if identity_source.shape != context_source.shape:
            raise ValueError("identity/context read sources must match")
        if state.keys.shape != state.values.shape:
            raise ValueError("episodic key/value state mismatch")
        if state.keys.shape[:2] != state.valid.shape:
            raise ValueError("episodic validity shape mismatch")
        if state.strengths.shape != state.valid.shape:
            raise ValueError("episodic strength shape mismatch")

        _, _, query = self.address_factors(identity_source, context_source)
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
            -1, identity_source.size(1), -1
        ).gather(-1, top_indices)
        safe_logits = top_logits.masked_fill(~top_valid, -1e9)
        weights = torch.softmax(safe_logits.float(), dim=-1).to(identity_source.dtype)
        weights = weights * top_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        expanded_values = state.values[:, None, :, :].expand(
            -1, identity_source.size(1), -1, -1
        )
        gathered_values = expanded_values.gather(
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, self.memory_dim),
        )
        recalled = (weights.unsqueeze(-1) * gathered_values).sum(dim=2)
        return self.out(recalled)

    def _vectorized_update(
        self,
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
        *,
        detach_inputs: bool,
    ) -> ContextualEpisodicMemoryState:
        identity = identity_source.detach() if detach_inputs else identity_source
        context = context_source.detach() if detach_inputs else context_source
        payload = payload_source.detach() if detach_inputs else payload_source
        strength = write_strength.detach() if detach_inputs else write_strength
        old_keys = state.keys.detach() if detach_inputs else state.keys
        old_values = state.values.detach() if detach_inputs else state.values
        old_strengths = state.strengths.detach() if detach_inputs else state.strengths
        old_valid = state.valid.detach()

        _, _, new_keys = self.address_factors(identity, context)
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

        similarity = torch.einsum("bkd,bsd->bks", new_keys, F.normalize(old_keys, dim=-1))
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
        slot_position = torch.arange(total, device=all_valid.device, dtype=torch.float32)
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
        identity_source: torch.Tensor,
        context_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
    ) -> ContextualEpisodicMemoryState:
        if identity_source.shape != context_source.shape or identity_source.shape != payload_source.shape:
            raise ValueError("identity/context/payload sources must match")
        if identity_source.ndim != 3:
            raise ValueError("write sources must be [batch,candidates,d_model]")
        if write_strength.shape != (*identity_source.shape[:-1], 1):
            raise ValueError("write_strength must be [batch,candidates,1]")
        if self.differentiable_pretraining:
            return self._vectorized_update(
                identity_source,
                context_source,
                payload_source,
                write_strength,
                state,
                detach_inputs=False,
            )
        with torch.no_grad():
            return self._vectorized_update(
                identity_source,
                context_source,
                payload_source,
                write_strength,
                state,
                detach_inputs=True,
            )


class FactorizedIdentityContextEpisodicMemoryStage(VectorizedContextualEpisodicMemoryStage):
    """Exact v24 stage plumbing with only the memory address factorized."""

    def __init__(self, source: VectorizedContextualEpisodicMemoryStage) -> None:
        # Deliberately initialize nn.Module directly: calling the inherited v24
        # constructor would try to reinterpret an already-v24 memory as v23 state.
        nn.Module.__init__(self)
        required = (
            "cfg",
            "norm",
            "controller",
            "state_to_chunk",
            "attn",
            "experts",
            "reasoner",
            "stream_cell",
            "stream_input_norm",
            "reason_to_chunk",
            "out_norm",
            "pair_write_gate",
            "memory",
        )
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(f"v24 source stage missing modules: {missing}")
        self.cfg = source.cfg
        self.norm = source.norm
        self.controller = source.controller
        self.state_to_chunk = source.state_to_chunk
        self.attn = source.attn
        self.experts = source.experts
        self.reasoner = source.reasoner
        self.stream_cell = source.stream_cell
        self.stream_input_norm = source.stream_input_norm
        self.reason_to_chunk = source.reason_to_chunk
        self.out_norm = source.out_norm
        self.pair_write_gate = source.pair_write_gate
        self.memory = FactorizedIdentityContextEpisodicMemory(source.memory)
        self.last_start_controls = getattr(source, "last_start_controls", None)
        self.last_end_controls = getattr(source, "last_end_controls", None)
        self.last_selected_indices: torch.Tensor | None = None
        self.last_selected_count = 0
        self.last_candidate_count = 0
        self.last_pair_gate: torch.Tensor | None = None
        self.last_pair_strength: torch.Tensor | None = None
        self.last_vectorized_update_calls = 0

    def _tokenwise_context(
        self,
        h: torch.Tensor,
        state: AERAState,
        start_control: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(state.memory, ContextualEpisodicMemoryState):
            raise TypeError("v25 requires contextual episodic state")
        identity, causal_context, _ = causal_identity_context(h)
        memory_read = self.memory.read(identity, causal_context, state.memory)
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
            raise TypeError("v25 stage received non-episodic memory state")

        # Use v19's generic stage computation through the inherited method, but
        # suppress its memory write. Dynamic dispatch still uses our factorized read.
        h_out, next_state, controls = super(
            VectorizedContextualEpisodicMemoryStage, self
        ).forward_chunk(
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
        identity, causal_context, contextual = causal_identity_context(base_h)
        contextual_address = contextual[:, :-1]
        payload_source = contextual[:, 1:]
        pair_features = torch.cat((contextual_address, payload_source), dim=-1)
        pair_logits = self.pair_write_gate(pair_features)
        pair_gate = torch.sigmoid(pair_logits)
        end_control = controls["end"]
        chunk_strength = (
            end_control["novelty"] * end_control["memory_write"]
        ).clamp(0.0, 1.0)
        write_strength = pair_gate * chunk_strength[:, None, :]
        selected = select_budgeted_event_pairs(
            contextual_address,
            payload_source,
            write_strength,
            pair_logits,
            differentiable_selector=self.memory.differentiable_pretraining,
        )
        if selected.hard_count != sparse_write_budget(candidate_count):
            raise RuntimeError("v25 sparse write budget mismatch")

        gather = selected.indices.unsqueeze(-1).expand(-1, -1, identity.size(-1))
        selected_identity = identity[:, :-1].gather(1, gather)
        selected_context = causal_context[:, :-1].gather(1, gather)

        self.last_pair_gate = pair_gate.detach()
        self.last_pair_strength = selected.strength.detach()
        self.last_selected_indices = selected.indices.detach()
        self.last_selected_count = selected.hard_count
        memory_state = self.memory.update_block(
            selected_identity,
            selected_context,
            selected.payload,
            selected.strength,
            prior_state.memory,
        )
        self.last_vectorized_update_calls = 1
        return h_out, AERAState(next_state.stream, memory_state), controls


class HardwareAwareAERATextLMV25(HardwareAwareAERATextLMV24):
    """AERA-v24 adaptive compute with factorized identity-context addresses."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            FactorizedIdentityContextEpisodicMemoryStage(stage) for stage in self.stages
        )
        self.set_memory_pretraining_mode(False)


def factorized_identity_context_protocol() -> dict[str, object]:
    return {
        "version": "aera-v25-factorized-identity-context-episodic-memory",
        "source": "aera-v24",
        "identity_source": "current pre-memory normalized event",
        "context_source": "mean previous up to 8 pre-memory normalized events",
        "identity_read_write_projection_shared": True,
        "context_read_write_projection_shared": True,
        "identity_fraction": 0.5,
        "context_fraction": 0.5,
        "combined_address_width_changed": False,
        "capacity_slots_per_stage": EPISODIC_CAPACITY,
        "state_bytes_real_language_four_stage_memory_dim50": episodic_state_bytes_per_session(
            n_stages=4, memory_dim=50
        ),
        "payload_v_out_changed": False,
        "pair_selector_candidate_representation_changed": False,
        "write_budget_changed": False,
        "controlled_selected_writes": sparse_write_budget(5),
        "real_language_selected_writes": sparse_write_budget(255),
        "vectorized_update_calls_per_completed_stage_chunk": 1,
        "duplicate_similarity_threshold": DUPLICATE_SIMILARITY,
        "read_top_k": READ_TOP_K,
        "read_temperature": READ_TEMPERATURE,
        "routing_changed": False,
        "stream_changed": False,
        "experts_changed": False,
        "gpu_authorized": False,
    }
