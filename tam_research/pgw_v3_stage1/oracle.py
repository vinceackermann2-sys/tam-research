"""Deterministic PGW-v3 delayed-retrieval data and independent causal oracle.

No torch, no LM, no training, no RNG state, no scientific seeds or GPU usage.
This file freezes the Stage-1 data grammar in issue #1368.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

CHUNK_SIZE = 8
WRITE = 1
READ = 2
DIST = 3
QUERY = 4
KEY_TOKENS = tuple(range(16, 24))
VALUE_TOKENS = tuple(range(32, 64))
FILLER_TOKENS = tuple(range(160, 256))
DELAYS = (1, 2, 4)
OVERWRITES = (0, 1, 2)
SPLITS = ("train", "validation", "test")
NAMESPACE = "PGW_V3_STAGE1_ORACLE_20261009_V1"

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class RetrievalExample:
    split: Split
    index: int
    overwrite_count: int
    interference: bool
    delay_chunks: int
    tokens: tuple[int, ...]
    expected_value: int
    query_key: int
    query_anchor: int
    last_write_chunk: int
    sha256: str


def _entropy(
    split: str, index: int, overwrite_count: int,
    interference: bool, delay_chunks: int, label: str,
) -> int:
    message = (
        f"{NAMESPACE}|{split}|{index}|{overwrite_count}|"
        f"{int(interference)}|{delay_chunks}|{label}"
    )
    return int.from_bytes(hashlib.sha256(message.encode("ascii")).digest(), "big")


def _filler(
    split: str, index: int, overwrites: int,
    interference: bool, delay: int, chunk: int, count: int,
) -> tuple[int, ...]:
    return tuple(
        FILLER_TOKENS[
            _entropy(split, index, overwrites, interference, delay, f"filler:{chunk}:{i}")
            % len(FILLER_TOKENS)
        ]
        for i in range(count)
    )


def make_example(
    *, split: Split, index: int, overwrite_count: int,
    interference: bool, delay_chunks: int,
) -> RetrievalExample:
    """Generate a deterministic *input prefix* with an external READ label.

    Latest query-key WRITE is always in completed chunk #2.
    Exactly delay_chunks complete unrelated chunks follow before READ.
    """
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if overwrite_count not in OVERWRITES:
        raise ValueError("overwrite_count must be 0, 1 or 2")
    if not isinstance(interference, bool):
        raise ValueError("interference must be bool")
    if delay_chunks not in DELAYS:
        raise ValueError("delay_chunks must be 1, 2 or 4")

    def draw(label: str, size: int) -> int:
        return _entropy(
            split, index, overwrite_count, interference, delay_chunks, label
        ) % size

    key = KEY_TOKENS[draw("query_key", len(KEY_TOKENS))]
    other_keys = tuple(value for value in KEY_TOKENS if value != key)

    # Rank values without replacement: older and newer values cannot match.
    ordered_values = sorted(
        VALUE_TOKENS, key=lambda value: (
            draw(f"target_value_rank:{value}", 1 << 256), value
        )
    )
    target_values = ordered_values[: overwrite_count + 1]
    next_target = 0
    chunks: list[tuple[int, ...]] = []

    def unrelated_write(chunk_index: int) -> tuple[int, ...]:
        other_key = other_keys[draw(f"other_key:{chunk_index}", len(other_keys))]
        other_value = VALUE_TOKENS[draw(f"other_value:{chunk_index}", len(VALUE_TOKENS))]
        return (
            WRITE, other_key, other_value,
            *_filler(
                split, index, overwrite_count, interference,
                delay_chunks, chunk_index, CHUNK_SIZE - 3,
            ),
        )

    # Prelude is three fixed-width event chunks. All target writes form
    # a trailing contiguous run ending in chunk #2.
    for chunk_index in range(3):
        if chunk_index >= 2 - overwrite_count:
            chunk = (
                WRITE, key, target_values[next_target],
                *_filler(
                    split, index, overwrite_count, interference,
                    delay_chunks, chunk_index, CHUNK_SIZE - 3,
                ),
            )
            next_target += 1
        else:
            chunk = unrelated_write(chunk_index)
        chunks.append(chunk)

    for chunk_index in range(3, 3 + delay_chunks):
        if interference:
            chunks.append(unrelated_write(chunk_index))
        else:
            chunks.append(
                (
                    DIST,
                    *_filler(
                        split, index, overwrite_count, interference,
                        delay_chunks, chunk_index, CHUNK_SIZE - 1,
                    ),
                )
            )

    read_chunk_index = 3 + delay_chunks
    chunks.append(
        (
            READ, key, QUERY,
            *_filler(
                split, index, overwrite_count, interference,
                delay_chunks, read_chunk_index, CHUNK_SIZE - 3,
            ),
        )
    )
    tokens = tuple(token for chunk in chunks for token in chunk)
    assert all(len(chunk) == CHUNK_SIZE for chunk in chunks)
    assert next_target == overwrite_count + 1
    return RetrievalExample(
        split=split,
        index=index,
        overwrite_count=overwrite_count,
        interference=interference,
        delay_chunks=delay_chunks,
        tokens=tokens,
        # Computed from independent construction, NOT the oracle below.
        expected_value=target_values[-1],
        query_key=key,
        query_anchor=read_chunk_index * CHUNK_SIZE + 2,
        last_write_chunk=2,
        sha256=hashlib.sha256(bytes(tokens)).hexdigest(),
    )


def oracle_latest_value(tokens: tuple[int, ...] | list[int]) -> int:
    """Strict causal reference interpreter, independent of sample metadata.

    Only completed earlier WRITE chunks enter memory; the final READ chunk
    contains a key and a query marker, not the answer token.
    """
    if len(tokens) < 2 * CHUNK_SIZE or len(tokens) % CHUNK_SIZE:
        raise ValueError("expected at least two complete eight-token chunks")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255 for v in tokens):
        raise ValueError("all tokens must be unsigned 8-bit integers")

    memory: dict[int, int] = {}
    seen_read = False
    for start in range(0, len(tokens), CHUNK_SIZE):
        chunk = tokens[start : start + CHUNK_SIZE]
        command = chunk[0]
        if command == WRITE:
            if seen_read:
                raise ValueError("future write after READ is forbidden")
            if chunk[1] not in KEY_TOKENS or chunk[2] not in VALUE_TOKENS:
                raise ValueError("invalid WRITE key/value")
            if any(token not in FILLER_TOKENS for token in chunk[3:]):
                raise ValueError("invalid WRITE filler")
            memory[chunk[1]] = chunk[2]
        elif command == DIST:
            if seen_read:
                raise ValueError("DIST after READ is forbidden")
            if any(token not in FILLER_TOKENS for token in chunk[1:]):
                raise ValueError("invalid DIST filler")
        elif command == READ:
            if seen_read or start != len(tokens) - CHUNK_SIZE:
                raise ValueError("exactly one final READ chunk required")
            if chunk[1] not in KEY_TOKENS or chunk[2] != QUERY:
                raise ValueError("invalid READ key/query")
            if any(token not in FILLER_TOKENS for token in chunk[3:]):
                raise ValueError("invalid READ filler")
            if chunk[1] not in memory:
                raise ValueError("READ cannot access an unwritten key")
            seen_read = True
            return memory[chunk[1]]
        else:
            raise ValueError("invalid command token")
    raise ValueError("missing final READ chunk")


def fixture_panel(*, per_cell: int = 4) -> tuple[RetrievalExample, ...]:
    """All 18 cells across three splits; CPU fixtures, never training seeds."""
    if not isinstance(per_cell, int) or per_cell < 1:
        raise ValueError("per_cell must be a positive integer")
    return tuple(
        make_example(
            split=split, index=index, overwrite_count=overwrites,
            interference=interference, delay_chunks=delay,
        )
        for split in SPLITS
        for overwrites in OVERWRITES
        for interference in (False, True)
        for delay in DELAYS
        for index in range(per_cell)
    )


def assert_panel_disjoint(panel: tuple[RetrievalExample, ...]) -> None:
    """Fail closed on observed exact token-stream duplicate/collision."""
    fingerprints: dict[str, tuple[str, int, int, bool, int]] = {}
    for row in panel:
        identity = (
            row.split, row.index, row.overwrite_count,
            row.interference, row.delay_chunks,
        )
        if row.sha256 != hashlib.sha256(bytes(row.tokens)).hexdigest():
            raise ValueError("stale or invalid sample fingerprint")
        if row.sha256 in fingerprints:
            raise ValueError(f"duplicate sample stream: {fingerprints[row.sha256]} and {identity}")
        fingerprints[row.sha256] = identity
