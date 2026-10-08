"""Binding-required delayed-associative-recall generator.

This module deliberately does NOT modify the historical CPW-v4 task. Its
purpose is to eliminate the model-free positional/value-range shortcut found
in issue #1301. No GPU runs or scientific seeds are defined here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tam_research.cpw_v4_memory import protocol as frozen

SEQ_LEN = 320
QUERY_POSITION = SEQ_LEN - 1
QUERY_MARKER_POSITION = SEQ_LEN - 2
VALUE_DELAYS = (32, 64, 96, 128, 160, 192, 224, 256)
SCORED_DELAYS = (32, 64, 128, 256)
PAIR_COUNT = 8

KEY_START = frozen.KEY_START
KEY_COUNT = frozen.KEY_COUNT
VALUE_START = frozen.VALUE_START
VALUE_COUNT = frozen.VALUE_COUNT
FILLER_START = frozen.FILLER_START
FILLER_COUNT = frozen.FILLER_COUNT
PAIR_TOKEN = frozen.PAIR_TOKEN
QUERY_TOKEN = frozen.QUERY_TOKEN


@dataclass
class BindingBatch:
    tokens: torch.Tensor
    targets: torch.Tensor
    delays: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    slot_to_pair: torch.Tensor
    source_value_positions: torch.Tensor
    query_pair_indices: torch.Tensor
    query_position: int = QUERY_POSITION


def _ids(
    batch_size: int,
    *,
    start: int,
    population: int,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.stack([
        torch.randperm(population, generator=generator)[:count] + start
        for _ in range(batch_size)
    ]).long()


def _delays(batch_size: int, generator: torch.Generator) -> torch.Tensor:
    values = list(SCORED_DELAYS)
    tiled = (values * ((batch_size + len(values) - 1) // len(values)))[:batch_size]
    result = torch.tensor(tiled, dtype=torch.long)
    return result[torch.randperm(batch_size, generator=generator)]


def make_binding_batch(
    *,
    batch_size: int,
    generator: torch.Generator,
    delay: int | None = None,
) -> BindingBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if delay is not None and delay not in SCORED_DELAYS:
        raise ValueError(f"delay must be one of {SCORED_DELAYS}")
    if not (KEY_COUNT >= PAIR_COUNT and VALUE_COUNT >= PAIR_COUNT):
        raise RuntimeError("insufficient key/value vocabulary")

    tokens = torch.randint(
        FILLER_START,
        FILLER_START + FILLER_COUNT,
        (batch_size, SEQ_LEN),
        generator=generator,
        dtype=torch.long,
    )
    keys = _ids(
        batch_size,
        start=KEY_START,
        population=KEY_COUNT,
        count=PAIR_COUNT,
        generator=generator,
    )
    values = _ids(
        batch_size,
        start=VALUE_START,
        population=VALUE_COUNT,
        count=PAIR_COUNT,
        generator=generator,
    )

    delays = (
        torch.full((batch_size,), int(delay), dtype=torch.long)
        if delay is not None
        else _delays(batch_size, generator)
    )
    targets = torch.empty(batch_size, dtype=torch.long)
    selected_positions = torch.empty(batch_size, dtype=torch.long)
    selected_pairs = torch.empty(batch_size, dtype=torch.long)
    slot_to_pair = torch.empty(batch_size, PAIR_COUNT, dtype=torch.long)

    for row in range(batch_size):
        pair_order = torch.randperm(PAIR_COUNT, generator=generator)
        # slot index -> randomly assigned pair index
        slot_to_pair[row] = pair_order
        for slot_idx, offset in enumerate(VALUE_DELAYS):
            pos = QUERY_POSITION - offset
            pair_idx = int(pair_order[slot_idx])
            tokens[row, pos - 1] = keys[row, pair_idx]
            tokens[row, pos] = values[row, pair_idx]
            tokens[row, pos + 1] = PAIR_TOKEN

        selected_slot = VALUE_DELAYS.index(int(delays[row]))
        selected_pair = int(pair_order[selected_slot])
        selected_pairs[row] = selected_pair
        targets[row] = values[row, selected_pair]
        selected_positions[row] = QUERY_POSITION - int(delays[row])
        tokens[row, QUERY_MARKER_POSITION] = QUERY_TOKEN
        tokens[row, QUERY_POSITION] = keys[row, selected_pair]

    return BindingBatch(
        tokens=tokens,
        targets=targets,
        delays=delays,
        keys=keys,
        values=values,
        slot_to_pair=slot_to_pair,
        source_value_positions=selected_positions,
        query_pair_indices=selected_pairs,
    )


def explicit_key_lookup(tokens: torch.Tensor) -> torch.Tensor:
    """Positive oracle: requires the final query-key token."""
    if tokens.ndim != 2 or tokens.size(1) != SEQ_LEN:
        raise ValueError("unexpected token shape")
    answers = []
    for row in tokens:
        query = int(row[QUERY_POSITION])
        matches = [
            int(row[QUERY_POSITION - delay])
            for delay in VALUE_DELAYS
            if int(row[QUERY_POSITION - delay - 1]) == query
        ]
        if len(matches) != 1:
            raise RuntimeError("expected exactly one key-bound value")
        answers.append(matches[0])
    return torch.tensor(answers, dtype=torch.long, device=tokens.device)


def positional_keyblind_guess(tokens: torch.Tensor) -> torch.Tensor:
    """Query-blind negative control: picks a fixed candidate value position."""
    if tokens.ndim != 2 or tokens.size(1) != SEQ_LEN:
        raise ValueError("unexpected token shape")
    return tokens[:, QUERY_POSITION - SCORED_DELAYS[0]].clone()


def counterfactual_query(
    batch: BindingBatch,
    *,
    new_delay: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same entire prefix, different final key and binding target."""
    if new_delay not in SCORED_DELAYS:
        raise ValueError("counterfactual delay must be scored")
    selected_slot = VALUE_DELAYS.index(new_delay)
    result_tokens = batch.tokens.clone()
    result_targets = torch.empty_like(batch.targets)
    for row in range(result_tokens.size(0)):
        pair_idx = int(batch.slot_to_pair[row, selected_slot])
        result_tokens[row, QUERY_POSITION] = batch.keys[row, pair_idx]
        result_targets[row] = batch.values[row, pair_idx]
    return result_tokens, result_targets
