from __future__ import annotations

from dataclasses import dataclass

import torch

from .protocol import (
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


@dataclass
class AssociativeBatch:
    tokens: torch.Tensor
    targets: torch.Tensor
    delays: torch.Tensor
    source_value_positions: torch.Tensor
    query_position: int
    keys: torch.Tensor
    values: torch.Tensor
    query_pair_indices: torch.Tensor


def _unique_ids(
    *,
    batch_size: int,
    start: int,
    count: int,
    take: int,
    generator: torch.Generator,
) -> torch.Tensor:
    rows = [
        torch.randperm(count, generator=generator)[:take] + start
        for _ in range(batch_size)
    ]
    return torch.stack(rows, dim=0).long()


def _balanced_delays(
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    base = list(DISTANCES)
    repeated = (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]
    values = torch.tensor(repeated, dtype=torch.long)
    order = torch.randperm(batch_size, generator=generator)
    return values[order]


def make_associative_batch(
    *,
    batch_size: int,
    generator: torch.Generator,
    delay: int | None = None,
    seq_len: int = SEQ_LEN,
) -> AssociativeBatch:
    """Generate fresh random in-context key->value bindings.

    The queried value occurs exactly ``delay`` positions before the final
    queried-key token. Filler uses a disjoint vocabulary region, so neither the
    target value nor any key/value token can leak through filler positions.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if delay is not None and delay not in DISTANCES:
        raise ValueError(f"delay must be one of {DISTANCES}")

    query_position = seq_len - 1
    query_marker_position = query_position - 1

    tokens = torch.randint(
        FILLER_START,
        FILLER_START + FILLER_COUNT,
        (batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
    )
    keys = _unique_ids(
        batch_size=batch_size,
        start=KEY_START,
        count=KEY_COUNT,
        take=PAIR_COUNT,
        generator=generator,
    )
    values = _unique_ids(
        batch_size=batch_size,
        start=VALUE_START,
        count=VALUE_COUNT,
        take=PAIR_COUNT,
        generator=generator,
    )

    # Randomize the value order independently of key order. There is no
    # persistent/global mapping across examples.
    for row in range(batch_size):
        values[row] = values[row, torch.randperm(PAIR_COUNT, generator=generator)]

    query_pair_indices = torch.randint(
        0,
        PAIR_COUNT,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    )
    delays = (
        torch.full((batch_size,), int(delay), dtype=torch.long)
        if delay is not None
        else _balanced_delays(batch_size, generator)
    )
    source_positions = torch.empty(batch_size, dtype=torch.long)
    targets = torch.empty(batch_size, dtype=torch.long)

    for row in range(batch_size):
        q_idx = int(query_pair_indices[row])
        d = int(delays[row])
        selected_value_offset = q_idx * 3 + 1
        block_start = query_position - d - selected_value_offset
        block_end = block_start + PAIR_COUNT * 3

        if block_start < 0 or block_end > query_marker_position:
            raise RuntimeError(
                f"cannot place pair block: delay={d}, q_idx={q_idx}, "
                f"start={block_start}, end={block_end}, seq_len={seq_len}"
            )

        for pair_index in range(PAIR_COUNT):
            pos = block_start + pair_index * 3
            tokens[row, pos] = keys[row, pair_index]
            tokens[row, pos + 1] = values[row, pair_index]
            tokens[row, pos + 2] = PAIR_TOKEN

        source_pos = block_start + selected_value_offset
        source_positions[row] = source_pos
        targets[row] = values[row, q_idx]

        tokens[row, query_marker_position] = QUERY_TOKEN
        tokens[row, query_position] = keys[row, q_idx]

        if query_position - source_pos != d:
            raise RuntimeError("generator failed exact-distance contract")

    return AssociativeBatch(
        tokens=tokens,
        targets=targets,
        delays=delays,
        source_value_positions=source_positions,
        query_position=query_position,
        keys=keys,
        values=values,
        query_pair_indices=query_pair_indices,
    )
