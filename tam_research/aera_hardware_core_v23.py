from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v22 import (
    DualDeltaFastMemoryState,
    HardwareAwareAERATextLMV22,
    InterferenceCorrectedEventPairFastMemoryStage,
    PretrainableDualDeltaFastMemory,
)


WRITE_BUDGET_DIVISOR = 16
WRITE_BUDGET_MIN = 2
SELECTOR_TEMPERATURE = 1.0
PAIR_GATE_INIT_GAIN = 0.01


def sparse_write_budget(candidate_count: int) -> int:
    """Frozen v23 physical write budget: about one write per 16 candidates."""
    if candidate_count < 0:
        raise ValueError("candidate_count must be nonnegative")
    if candidate_count == 0:
        return 0
    proportional = math.ceil(candidate_count / WRITE_BUDGET_DIVISOR)
    return min(candidate_count, max(WRITE_BUDGET_MIN, proportional))


class SparsePairSelection(NamedTuple):
    address: torch.Tensor
    payload: torch.Tensor
    strength: torch.Tensor
    indices: torch.Tensor
    hard_count: int
    candidate_count: int


def _gather_candidates(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or indices.ndim != 2 or x.size(0) != indices.size(0):
        raise ValueError("expected x=[batch,candidates,dim], indices=[batch,k]")
    gather_index = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
    return x.gather(1, gather_index)


def budgeted_topk_indices(pair_logits: torch.Tensor) -> torch.Tensor:
    """Select the frozen hard write budget and restore causal/chronological order."""
    if pair_logits.ndim == 3 and pair_logits.size(-1) == 1:
        scores = pair_logits[..., 0]
    elif pair_logits.ndim == 2:
        scores = pair_logits
    else:
        raise ValueError("pair_logits must be [batch,candidates] or [batch,candidates,1]")
    k = sparse_write_budget(scores.size(1))
    if k == 0:
        return torch.empty(scores.size(0), 0, dtype=torch.long, device=scores.device)
    selected = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
    # The dual-delta recurrence is order-sensitive. Score chooses *which* writes;
    # original event order chooses *when* they are applied.
    return selected.sort(dim=1).values


def select_budgeted_event_pairs(
    address_source: torch.Tensor,
    payload_source: torch.Tensor,
    write_strength: torch.Tensor,
    pair_logits: torch.Tensor,
    *,
    differentiable_selector: bool,
    temperature: float = SELECTOR_TEMPERATURE,
) -> SparsePairSelection:
    """Hard sparse forward selection with a straight-through soft k-hot surrogate.

    Only the hard top-K candidates are gathered and sent into the sequential
    memory recurrence. During differentiable pretraining, the selected forward
    multiplier remains exactly one while its backward derivative follows a
    softmax mask whose total mass is K. This gives the shared pair gate a learning
    signal without executing the other C-K memory writes.
    """
    if address_source.shape != payload_source.shape or address_source.ndim != 3:
        raise ValueError("address/payload sources must match [batch,candidates,d_model]")
    if write_strength.shape != (*address_source.shape[:-1], 1):
        raise ValueError("write_strength must be [batch,candidates,1]")
    if pair_logits.shape != (*address_source.shape[:-1], 1):
        raise ValueError("pair_logits must be [batch,candidates,1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    candidate_count = address_source.size(1)
    indices = budgeted_topk_indices(pair_logits)
    k = indices.size(1)
    if k == 0:
        return SparsePairSelection(
            address_source[:, :0],
            payload_source[:, :0],
            write_strength[:, :0],
            indices,
            0,
            candidate_count,
        )

    selected_address = _gather_candidates(address_source, indices)
    selected_payload = _gather_candidates(payload_source, indices)
    selected_strength = _gather_candidates(write_strength, indices)

    if differentiable_selector:
        scores = pair_logits[..., 0]
        soft_k_hot = torch.softmax(scores / temperature, dim=1) * float(k)
        soft_selected = soft_k_hot.gather(1, indices)
        # Compute the detached cancellation before adding one so the forward
        # multiplier is bit-exact 1 rather than suffering (1+s)-s rounding.
        straight_through = 1.0 + (soft_selected - soft_selected.detach())
        selected_strength = selected_strength * straight_through.unsqueeze(-1)

    return SparsePairSelection(
        selected_address,
        selected_payload,
        selected_strength,
        indices,
        k,
        candidate_count,
    )


class BudgetedSparseDualDeltaFastMemoryStage(InterferenceCorrectedEventPairFastMemoryStage):
    """V22 dual-delta memory with physically bounded learned event-pair writes."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__(source)
        # V21/v22 used an exact-zero pair-gate initialization. Hard top-k needs
        # deterministic content-dependent exploration from the first step; only
        # this already-existing gate is reinitialized in v23.
        nn.init.xavier_uniform_(self.pair_write_gate.weight, gain=PAIR_GATE_INIT_GAIN)
        nn.init.zeros_(self.pair_write_gate.bias)
        self.last_selected_indices: torch.Tensor | None = None
        self.last_selected_count: int = 0
        self.last_candidate_count: int = 0
        self.last_pair_gate: torch.Tensor | None = None
        self.last_pair_strength: torch.Tensor | None = None

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, dict[str, torch.Tensor]]]:
        prior_state = state if state is not None else self.empty_state(events)
        # Reuse v22/v21's exact compute/read/stream path but suppress the dense
        # completed-chunk write; v23 performs its sparse write below.
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
        if not update_memory or candidate_count == 0:
            return h_out, next_state, controls

        base_h = self.norm(events)
        address_source = base_h[:, :-1]
        payload_source = base_h[:, 1:]
        pair_features = torch.cat((address_source, payload_source), dim=-1)
        pair_logits = self.pair_write_gate(pair_features)
        pair_gate = torch.sigmoid(pair_logits)
        end_control = controls["end"]
        chunk_strength = (
            end_control["novelty"] * end_control["memory_write"]
        ).clamp(0.0, 1.0)
        write_strength = pair_gate * chunk_strength[:, None, :]

        memory = self.memory
        if not isinstance(memory, PretrainableDualDeltaFastMemory):
            raise TypeError("v23 requires PretrainableDualDeltaFastMemory")
        selected = select_budgeted_event_pairs(
            address_source,
            payload_source,
            write_strength,
            pair_logits,
            differentiable_selector=memory.differentiable_pretraining,
        )
        if selected.hard_count != sparse_write_budget(candidate_count):
            raise RuntimeError("v23 sparse write budget mismatch")

        self.last_pair_gate = pair_gate.detach()
        self.last_pair_strength = selected.strength.detach()
        self.last_selected_indices = selected.indices.detach()
        self.last_selected_count = selected.hard_count
        memory_state = self._event_pair_update(
            selected.address,
            selected.payload,
            selected.strength,
            prior_state.memory,
        )
        if not isinstance(memory_state, DualDeltaFastMemoryState):
            raise RuntimeError("v23 sparse update lost dual-delta state")
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
        if self.last_selected_indices is not None:
            result["event_pair_selected_indices"] = self.last_selected_indices.cpu().tolist()
        return result


class HardwareAwareAERATextLMV23(HardwareAwareAERATextLMV22):
    """V22 memory semantics with a fixed physical budget on event-pair writes."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            BudgetedSparseDualDeltaFastMemoryStage(stage) for stage in self.stages
        )


def sparse_dual_delta_memory_protocol() -> dict[str, object]:
    return {
        "version": "aera-v23-budgeted-sparse-dual-delta",
        "source": "aera-v22",
        "dual_delta_equation_changed": False,
        "inverse_covariance_update_changed": False,
        "qkvout_dimension_changed": False,
        "learned_parameter_count_changed": False,
        "read_path_changed": False,
        "candidate_representation_changed": False,
        "routing_changed": False,
        "stream_changed": False,
        "write_budget_divisor": WRITE_BUDGET_DIVISOR,
        "write_budget_min": WRITE_BUDGET_MIN,
        "controlled_candidates": 5,
        "controlled_selected_writes": sparse_write_budget(5),
        "real_language_candidates": 255,
        "real_language_selected_writes": sparse_write_budget(255),
        "selection_score": "existing pair-write gate logit",
        "training_selector": "hard top-k forward + straight-through soft k-hot backward",
        "deployment_selector": "hard top-k",
        "selected_write_order": "original chronological event order",
        "pair_gate_init": f"xavier_uniform_gain_{PAIR_GATE_INIT_GAIN}",
        "extra_learned_parameters": 0,
        "gpu_authorized": False,
    }
