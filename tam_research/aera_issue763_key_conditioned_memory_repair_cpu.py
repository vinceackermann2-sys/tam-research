from __future__ import annotations

"""CPU-only causal-addressing repair prototype for research issue #763.

This module does not modify the frozen #754 scientific harness. It isolates the
mechanical repair diagnosed in #762: writes are aligned to explicit key/value
events and reads occur only after the query key representation is causally
available. Passing these probes is engineering evidence only, not a capability
or scaling result.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

RESEARCH_ISSUE = 763
PARENT_DIAG_ISSUE = 762
FAILED_SCIENTIFIC_ISSUE = 754
FAILED_RESEARCH_ISSUE = 748
SOURCE_MAIN = "d8c832c336338f194a93a302eb87525352e4c59e"
FAILED_SCIENTIFIC_MAIN = "9fbd9c523bbcf938cc3751209320f9bb89aa3b3d"
CONSUMED_SCIENTIFIC_SEED = 17_641
CPU_TEST_SEED = 763_999

FROZEN_WRAPPER_BLOB = "afc939a69633f68ded05eb585c95a599a8f5c981"
FROZEN_BASE_BLOB = "40003b68987d026265b1d53fcb637f18277f9cac"
FROZEN_FIXTURE_BLOB = "981602432b684989f7a5011ff953e6965368c5e5"

# Capability fixture record markers. Tests cross-check these against the frozen
# #746 fixture rather than treating these local constants as new scientific data.
PAD, WRITE, UPDATE, QUERY, ANSWER, RESET, SEP, UNKNOWN = range(8)

GPU_AUTHORIZED = False
SCIENTIFIC_TRAINING_AUTHORIZED = False
SCIENTIFIC_SEED_AUTHORIZED = False
SEEDS_2_3_AUTHORIZED = False
SYSTEMS_OPTIMIZATION_AUTHORIZED = False
ARCHITECTURE_FREEZE_AUTHORIZED = False
SCALING_AUTHORIZED = False
BREAKTHROUGH_PROVEN = False


@dataclass(frozen=True)
class RepairMemoryState:
    matrix: torch.Tensor


@dataclass(frozen=True)
class RepairEvent:
    kind: str
    marker_index: int
    key_index: int | None = None
    value_index: int | None = None
    recall: torch.Tensor | None = None


class EventAlignedDeltaMemory(nn.Module):
    """The established delta-rule transport with event-aligned key/value inputs.

    The transport equation is intentionally the same associative delta rule used
    by the frozen memory family. The CPU prototype simply exposes *which causal
    representation* is used for k/v/q so the addressing defect can be tested in
    isolation. Identity initialization makes deterministic probes possible
    without training or scientific seeds.
    """

    def __init__(self, d_model: int, *, lr: float = 1.0, decay: float = 1.0) -> None:
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be >=2")
        self.d_model = int(d_model)
        self.lr = float(lr)
        self.decay = float(decay)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        eye = torch.eye(d_model)
        with torch.no_grad():
            self.q.weight.copy_(eye)
            self.k.weight.copy_(eye)
            self.v.weight.copy_(eye)
            self.out.weight.copy_(eye)

    def empty_state(self, *, dtype: torch.dtype = torch.float32) -> RepairMemoryState:
        return RepairMemoryState(torch.zeros(self.d_model, self.d_model, dtype=dtype))

    def write_event(
        self,
        key_representation: torch.Tensor,
        value_representation: torch.Tensor,
        state: RepairMemoryState,
        *,
        strength: float = 1.0,
    ) -> RepairMemoryState:
        self._validate_vector(key_representation)
        self._validate_vector(value_representation)
        self._validate_state(state)
        key = F.normalize(self.k(key_representation), dim=-1)
        target = torch.tanh(self.v(value_representation))
        prediction = torch.einsum("i,ij->j", key, state.matrix)
        error = target - prediction
        eta = self.lr * float(strength)
        matrix = self.decay * state.matrix + torch.outer(key * eta, error)
        return RepairMemoryState(matrix)

    def read_key(self, key_representation: torch.Tensor, state: RepairMemoryState) -> torch.Tensor:
        self._validate_vector(key_representation)
        self._validate_state(state)
        query = F.normalize(self.q(key_representation), dim=-1)
        recalled = torch.einsum("i,ij->j", query, state.matrix)
        return self.out(recalled)

    def _validate_vector(self, x: torch.Tensor) -> None:
        if x.ndim != 1 or x.numel() != self.d_model:
            raise ValueError(f"expected vector [{self.d_model}]")

    def _validate_state(self, state: RepairMemoryState) -> None:
        if state.matrix.shape != (self.d_model, self.d_model):
            raise ValueError("memory state shape mismatch")


def _validate_chunk(tokens: Sequence[int], token_representations: torch.Tensor) -> None:
    if token_representations.ndim != 2:
        raise ValueError("token_representations must be [tokens,d_model]")
    if len(tokens) != token_representations.size(0):
        raise ValueError("token/representation length mismatch")


def legacy_chunk_start_read(
    token_representations: torch.Tensor,
    memory: EventAlignedDeltaMemory,
    state: RepairMemoryState,
) -> torch.Tensor:
    """Model the frozen defect: the only read is keyed by token position 0."""
    if token_representations.ndim != 2 or token_representations.size(0) == 0:
        raise ValueError("expected non-empty [tokens,d_model] representations")
    return memory.read_key(token_representations[0], state)


def find_query_key_index(tokens: Sequence[int]) -> int:
    matches = [i for i, token in enumerate(tokens[:-1]) if int(token) == QUERY]
    if len(matches) != 1:
        raise ValueError("expected exactly one QUERY marker")
    return matches[0] + 1


def repaired_query_read(
    tokens: Sequence[int],
    token_representations: torch.Tensor,
    memory: EventAlignedDeltaMemory,
    state: RepairMemoryState,
) -> RepairEvent:
    """Read only when the query-key representation is causally available."""
    _validate_chunk(tokens, token_representations)
    key_index = find_query_key_index(tokens)
    recall = memory.read_key(token_representations[key_index], state)
    return RepairEvent("query", key_index - 1, key_index=key_index, recall=recall)


def apply_capability_chunk(
    tokens: Sequence[int],
    token_representations: torch.Tensor,
    memory: EventAlignedDeltaMemory,
    state: RepairMemoryState,
) -> tuple[RepairMemoryState, tuple[RepairEvent, ...]]:
    """Apply only causal WRITE/UPDATE/RESET events found in one capability chunk.

    A write consumes representations at marker+1 (key) and marker+2 (value).
    Consequently, tokens after the value cannot affect that memory update.
    Queries do not mutate state; their read is returned as an event.
    """
    _validate_chunk(tokens, token_representations)
    events: list[RepairEvent] = []
    current = state
    i = 0
    while i < len(tokens):
        marker = int(tokens[i])
        if marker in {WRITE, UPDATE}:
            if i + 3 >= len(tokens) or int(tokens[i + 3]) != SEP:
                raise ValueError("malformed WRITE/UPDATE record")
            key_index, value_index = i + 1, i + 2
            current = memory.write_event(
                token_representations[key_index],
                token_representations[value_index],
                current,
            )
            events.append(
                RepairEvent(
                    "update" if marker == UPDATE else "write",
                    i,
                    key_index=key_index,
                    value_index=value_index,
                )
            )
            i += 4
            continue
        if marker == RESET:
            current = memory.empty_state(dtype=current.matrix.dtype)
            events.append(RepairEvent("reset", i))
            i += 1
            if i < len(tokens) and int(tokens[i]) == SEP:
                i += 1
            continue
        if marker == QUERY:
            if i + 2 >= len(tokens) or int(tokens[i + 2]) != ANSWER:
                raise ValueError("malformed QUERY record")
            key_index = i + 1
            recall = memory.read_key(token_representations[key_index], current)
            events.append(RepairEvent("query", i, key_index=key_index, recall=recall))
            i += 3
            continue
        i += 1
    return current, tuple(events)


def authority_snapshot() -> dict[str, object]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "parent_diag_issue": PARENT_DIAG_ISSUE,
        "source_main": SOURCE_MAIN,
        "consumed_scientific_seed": CONSUMED_SCIENTIFIC_SEED,
        "cpu_test_seed": CPU_TEST_SEED,
        "gpu_authorized": GPU_AUTHORIZED,
        "scientific_training_authorized": SCIENTIFIC_TRAINING_AUTHORIZED,
        "scientific_seed_authorized": SCIENTIFIC_SEED_AUTHORIZED,
        "seeds_2_3_authorized": SEEDS_2_3_AUTHORIZED,
        "systems_optimization_authorized": SYSTEMS_OPTIMIZATION_AUTHORIZED,
        "architecture_freeze_authorized": ARCHITECTURE_FREEZE_AUTHORIZED,
        "scaling_authorized": SCALING_AUTHORIZED,
        "breakthrough_proven": BREAKTHROUGH_PROVEN,
    }
