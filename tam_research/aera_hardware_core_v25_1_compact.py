from __future__ import annotations

"""CPU-first stable-compaction candidate for the frozen AERA-v25.1 #381 boundary.

The authoritative v25 write tail concatenates newest writes with prior state, assigns
`valid * 2 - position * 1e-6`, takes a sorted top-k of capacity slots, then gathers
all state fields. For the fixed v25 geometry every valid score is greater than every
invalid score and the position term is strictly decreasing, so that tail is exactly
a stable partition of source slots by validity followed by truncation to capacity.

This module changes only that execution tail. Duplicate detection, newest-first
ordering, projected addresses, values, strengths, validity, routing, learned
parameters, durable state, and all scientific thresholds remain unchanged.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v24 import DUPLICATE_SIMILARITY, ContextualEpisodicMemoryState
from .aera_hardware_core_v25_1 import (
    ExecutionEquivalentFICEMStage,
    ExecutionEquivalentFactorizedIdentityContextMemory,
    HardwareAwareAERATextLMV251,
    execution_equivalent_v25_1_protocol,
)


def _scatter_ranked_slots(
    output: torch.Tensor,
    source: torch.Tensor,
    ranks: torch.Tensor,
    *,
    capacity: int,
) -> torch.Tensor:
    """Scatter retained stable-partition slots; out-of-capacity rows go to a sink.

    `ranks` is a permutation rank over the combined new+old source population.
    Every rank below capacity is unique, so retained destinations have exactly one
    contributing source. Rank >= capacity is redirected to the final sink slot with
    an exact zero source, keeping it away from every durable output destination.
    """
    if ranks.ndim != 2 or source.shape[:2] != ranks.shape:
        raise ValueError("rank/source batch and slot axes must match")
    if output.size(0) != source.size(0) or output.size(1) != capacity + 1:
        raise ValueError("ranked-scatter output must include one sink slot")

    active = ranks < capacity
    destination = torch.where(active, ranks, torch.full_like(ranks, capacity))
    expand_shape = (*destination.shape, *([1] * (source.ndim - 2)))
    destination = destination.view(expand_shape).expand_as(source)
    active_expanded = active.view(expand_shape).expand_as(source)
    contribution = torch.where(active_expanded, source, torch.zeros_like(source))
    return output.scatter_add(1, destination, contribution)


class StableCompactExecutionEquivalentFactorizedIdentityContextMemory(
    ExecutionEquivalentFactorizedIdentityContextMemory
):
    """v25.1 FICEM whose final bounded-state rebuild is a stable fixed-shape scatter."""

    def _stable_compact_state(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_strengths: torch.Tensor,
        new_valid: torch.Tensor,
        old_keys: torch.Tensor,
        old_values: torch.Tensor,
        old_strengths: torch.Tensor,
        keep_old: torch.Tensor,
    ) -> ContextualEpisodicMemoryState:
        if new_keys.shape != new_values.shape:
            raise ValueError("new compact key/value shapes must match")
        if old_keys.shape != old_values.shape:
            raise ValueError("old compact key/value shapes must match")
        if new_keys.size(0) != old_keys.size(0) or new_keys.size(2) != old_keys.size(2):
            raise ValueError("new/old compact batch and feature axes must match")
        if new_strengths.shape != new_valid.shape or new_valid.shape != new_keys.shape[:2]:
            raise ValueError("new compact strength/valid shapes must match slots")
        if old_strengths.shape != keep_old.shape or keep_old.shape != old_keys.shape[:2]:
            raise ValueError("old compact strength/valid shapes must match slots")

        # Stable-partition ranks over conceptual source order [new, old]. Valid
        # entries come first while preserving source order; invalid entries follow,
        # also preserving source order. This is exactly the ordering produced by the
        # authoritative `valid*2 - position*1e-6` sorted top-k tail.
        new_valid_i = new_valid.to(torch.int64)
        old_valid_i = keep_old.to(torch.int64)
        new_invalid_i = (~new_valid).to(torch.int64)
        old_invalid_i = (~keep_old).to(torch.int64)

        new_valid_count = new_valid_i.sum(dim=1, keepdim=True)
        old_valid_count = old_valid_i.sum(dim=1, keepdim=True)
        total_valid = new_valid_count + old_valid_count
        new_invalid_count = new_invalid_i.sum(dim=1, keepdim=True)

        new_valid_rank = new_valid_i.cumsum(dim=1) - 1
        old_valid_rank = new_valid_count + old_valid_i.cumsum(dim=1) - 1
        new_invalid_rank = total_valid + new_invalid_i.cumsum(dim=1) - 1
        old_invalid_rank = (
            total_valid + new_invalid_count + old_invalid_i.cumsum(dim=1) - 1
        )
        new_rank = torch.where(new_valid, new_valid_rank, new_invalid_rank)
        old_rank = torch.where(keep_old, old_valid_rank, old_invalid_rank)

        batch = new_keys.size(0)
        feature = new_keys.size(2)
        # Capacity+1 keeps all truncated positions in an isolated sink destination.
        keys_out = torch.zeros(
            batch,
            self.capacity + 1,
            feature,
            device=new_keys.device,
            dtype=new_keys.dtype,
        )
        values_out = torch.zeros(
            batch,
            self.capacity + 1,
            feature,
            device=new_values.device,
            dtype=new_values.dtype,
        )
        strengths_out = torch.zeros(
            batch,
            self.capacity + 1,
            device=new_strengths.device,
            dtype=new_strengths.dtype,
        )

        keys_out = _scatter_ranked_slots(keys_out, new_keys, new_rank, capacity=self.capacity)
        keys_out = _scatter_ranked_slots(keys_out, old_keys, old_rank, capacity=self.capacity)
        values_out = _scatter_ranked_slots(
            values_out, new_values, new_rank, capacity=self.capacity
        )
        values_out = _scatter_ranked_slots(
            values_out, old_values, old_rank, capacity=self.capacity
        )
        strengths_out = _scatter_ranked_slots(
            strengths_out, new_strengths, new_rank, capacity=self.capacity
        )
        strengths_out = _scatter_ranked_slots(
            strengths_out, old_strengths, old_rank, capacity=self.capacity
        )

        valid_count = total_valid.clamp_max(self.capacity)
        durable_valid = (
            torch.arange(self.capacity, device=new_valid.device)[None, :] < valid_count
        )
        return ContextualEpisodicMemoryState(
            keys=keys_out[:, : self.capacity],
            values=values_out[:, : self.capacity],
            strengths=strengths_out[:, : self.capacity],
            valid=durable_valid,
        )

    def _compact_update_from_projected(
        self,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        old_keys: torch.Tensor,
        old_values: torch.Tensor,
        old_strengths: torch.Tensor,
        old_valid: torch.Tensor,
    ) -> ContextualEpisodicMemoryState:
        new_keys = projected_new_keys
        new_values = torch.tanh(self.v(payload_source))
        new_strengths = write_strength[..., 0].clamp(0.0, 1.0)
        new_valid = new_strengths > 0.0

        # Exact inherited newest-wins duplicate semantics.
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

        similarity = torch.einsum("bkd,bsd->bks", new_keys, normalized_old_keys)
        duplicate_old = (
            similarity.ge(DUPLICATE_SIMILARITY)
            & new_valid[:, :, None]
            & old_valid[:, None, :]
        ).any(dim=1)
        keep_old = old_valid & ~duplicate_old

        # Authoritative v25 reverses incoming writes before bounded-state ordering.
        new_keys = new_keys.flip(1)
        new_values = new_values.flip(1)
        new_strengths = new_strengths.flip(1)
        new_valid = new_valid.flip(1)
        return self._stable_compact_state(
            new_keys,
            new_values,
            new_strengths,
            new_valid,
            old_keys,
            old_values,
            old_strengths,
            keep_old,
        )

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
        normalized_old = F.normalize(old_keys, dim=-1)
        next_state = self._compact_update_from_projected(
            new_keys,
            normalized_old,
            payload,
            strength,
            old_keys,
            old_values,
            old_strengths,
            old_valid,
        )
        return next_state.detach() if detach_inputs else next_state

    def _vectorized_update_from_projected(
        self,
        projected_new_keys: torch.Tensor,
        normalized_old_keys: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: ContextualEpisodicMemoryState,
        *,
        detach_inputs: bool,
    ) -> ContextualEpisodicMemoryState:
        new_keys = projected_new_keys.detach() if detach_inputs else projected_new_keys
        normalized_old = (
            normalized_old_keys.detach() if detach_inputs else normalized_old_keys
        )
        payload = payload_source.detach() if detach_inputs else payload_source
        strength = write_strength.detach() if detach_inputs else write_strength
        old_keys = state.keys.detach() if detach_inputs else state.keys
        old_values = state.values.detach() if detach_inputs else state.values
        old_strengths = state.strengths.detach() if detach_inputs else state.strengths
        old_valid = state.valid.detach()
        next_state = self._compact_update_from_projected(
            new_keys,
            normalized_old,
            payload,
            strength,
            old_keys,
            old_values,
            old_strengths,
            old_valid,
        )
        return next_state.detach() if detach_inputs else next_state


class StableCompactExecutionEquivalentFICEMStage(ExecutionEquivalentFICEMStage):
    """Existing v25.1 stage with only its FICEM state-rebuild implementation replaced."""

    def __init__(self, source: ExecutionEquivalentFICEMStage) -> None:
        super().__init__(source)
        self.memory = StableCompactExecutionEquivalentFactorizedIdentityContextMemory(
            self.memory
        )


class HardwareAwareAERATextLMV251StableCompact(HardwareAwareAERATextLMV251):
    """CPU-gated v25.1 candidate using exact stable-compaction FICEM writes."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            StableCompactExecutionEquivalentFICEMStage(stage) for stage in self.stages
        )
        self.set_memory_pretraining_mode(False)


def stable_compaction_v25_1_protocol() -> dict[str, Any]:
    protocol = dict(execution_equivalent_v25_1_protocol())
    protocol.update(
        {
            "version": "aera-v25.1-execution-equivalent-stable-compaction",
            "stable_compaction": True,
            "write_state_rebuild": "fixed-shape stable validity partition via ranked scatter; no concat/priority-topk/gather tail",
            "write_state_semantics_changed": False,
            "duplicate_semantics_changed": False,
            "incoming_order_changed": False,
            "learned_parameter_count_changed": False,
            "state_dict_schema_changed": False,
            "routing_policy_changed": False,
            "gpu_authorized": False,
            "scientific_training_authorized": False,
            "100m_authorized": False,
        }
    )
    return protocol
