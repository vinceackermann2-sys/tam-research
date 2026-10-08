"""Deconfounded delayed key->value association data (zero-GPU only).

Scientific issue #1306. This module never alters the frozen CPW-v4 generator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tam_research.cpw_v4_memory.protocol import (
    DISTANCES,
    FILLER_COUNT,
    FILLER_START,
    KEY_COUNT,
    KEY_START,
    PAIR_COUNT,
    PAIR_TOKEN,
    QUERY_TOKEN,
    SEQ_LEN,
    VALUE_COUNT,
    VALUE_START,
)

# Four disjoint, non-overlapping regions for unrelated distractor pairs.
# These never intersect any key/value/separator anchor at 319 - DISTANCES.
DISTRACTOR_KEY_WINDOWS = ((16, 40), (95, 120), (143, 167), (210, 235))
ANCHOR_PAIR_COUNT = len(DISTANCES)


@dataclass(frozen=True)
class BindingRequiredBatch:
    tokens: torch.Tensor
    targets: torch.Tensor
    delays: torch.Tensor
    query_position: int
    anchor_keys: torch.Tensor
    anchor_values: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    distractor_key_positions: torch.Tensor


def _unique_rows(
    *, batch_size: int, first: int, pool: int, n: int, generator: torch.Generator
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(pool, generator=generator)[:n] + first
            for _ in range(batch_size)
        ],
        dim=0,
    ).long()


def make_binding_required_batch(
    *, batch_size: int, generator: torch.Generator, delay: int | None = None
) -> BindingRequiredBatch:
    """Create examples requiring query-key matching, not positional copying.

    The source is generated completely before query choice. Every example has
    four *simultaneously present* candidate VALUES at the four delay offsets.
    Counterfactual queries keep the source exactly unchanged.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if delay is not None and delay not in DISTANCES:
        raise ValueError("unknown delay bucket")
    if PAIR_COUNT != 8 or SEQ_LEN != 320 or ANCHOR_PAIR_COUNT != 4:
        raise AssertionError("the #1306 CPU-only protocol is pinned to 8/320/4")

    query_position = SEQ_LEN - 1
    tokens = torch.randint(
        FILLER_START,
        FILLER_START + FILLER_COUNT,
        (batch_size, SEQ_LEN),
        generator=generator,
        dtype=torch.long,
    )
    keys = _unique_rows(
        batch_size=batch_size, first=KEY_START, pool=KEY_COUNT,
        n=PAIR_COUNT, generator=generator
    )
    values = _unique_rows(
        batch_size=batch_size, first=VALUE_START, pool=VALUE_COUNT,
        n=PAIR_COUNT, generator=generator
    )
    for row in range(batch_size):
        values[row] = values[row, torch.randperm(PAIR_COUNT, generator=generator)]

    # The same 4 target positions are all occupied by valid VALUE tokens,
    # regardless of which of the 4 keys will eventually be queried.
    anchor_value_positions = (query_position - torch.tensor(DISTANCES)).tolist()
    for i, value_pos in enumerate(anchor_value_positions):
        tokens[:, value_pos - 1] = keys[:, i]
        tokens[:, value_pos] = values[:, i]
        tokens[:, value_pos + 1] = PAIR_TOKEN

    distractor_positions = torch.empty((batch_size, PAIR_COUNT - 4), dtype=torch.long)
    for j, (lo, hi) in enumerate(DISTRACTOR_KEY_WINDOWS):
        positions = torch.randint(lo, hi, (batch_size,), generator=generator)
        distractor_positions[:, j] = positions
        row_ix = torch.arange(batch_size)
        tokens[row_ix, positions] = keys[:, ANCHOR_PAIR_COUNT + j]
        tokens[row_ix, positions + 1] = values[:, ANCHOR_PAIR_COUNT + j]
        tokens[row_ix, positions + 2] = PAIR_TOKEN

    # Only now choose the query. Conditional on the source and a key-blind
    # observer, each of the four target VALUEs is equally likely.
    if delay is None:
        query_indices = torch.randint(
            ANCHOR_PAIR_COUNT, (batch_size,), generator=generator
        )
    else:
        query_indices = torch.full(
            (batch_size,), DISTANCES.index(delay), dtype=torch.long
        )
    row_ix = torch.arange(batch_size)
    targets = values[row_ix, query_indices].clone()
    tokens[:, query_position - 1] = QUERY_TOKEN
    tokens[:, query_position] = keys[row_ix, query_indices]

    return BindingRequiredBatch(
        tokens=tokens,
        targets=targets,
        delays=torch.tensor(DISTANCES, dtype=torch.long)[query_indices],
        query_position=query_position,
        anchor_keys=keys[:, :ANCHOR_PAIR_COUNT].clone(),
        anchor_values=values[:, :ANCHOR_PAIR_COUNT].clone(),
        keys=keys,
        values=values,
        distractor_key_positions=distractor_positions,
    )


def counterfactual_anchor_queries(
    batch: BindingRequiredBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,4,T] samples with IDENTICAL prefixes and 4 different queries.

    These four counterfactual targets are all distinct. Any predictor that
    cannot see the final query key can answer at most 1/4 correctly per group.
    """
    variants = batch.tokens[:, None, :].expand(
        -1, ANCHOR_PAIR_COUNT, -1
    ).clone()
    variants[:, :, batch.query_position] = batch.anchor_keys
    return variants, batch.anchor_values.clone()
