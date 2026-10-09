"""PGW-v3 Stage-1B: position-deconfounded causal WRITE/READ oracle.

Pure Python, no training, model inference, GPU, RNG state, or scientific seeds.
Frozen design: PGW issue #1374 (parent #1372). Old Stage-1 is unmodified.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Literal

CHUNK_SIZE = 8
EVENT_CHUNKS = 8
WRITE, READ, DIST, QUERY = 1, 2, 3, 4
KEY_TOKENS = tuple(range(16, 24))
VALUE_TOKENS = tuple(range(32, 64))
NOT_FOUND = 64  # External answer only: never appears in an input.
FILLER_TOKENS = tuple(range(160, 256))
POSITIONS = (2, 3, 4, 5, 6)
OVERWRITES = (0, 1, 2)
DELAYS = (1, 2, 4)
SPLITS = ("train", "validation", "test")
NAMESPACE = "PGW_V3_STAGE1B_POSITIONAL_ORACLE_20261009_V1"
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class PositionIndependentExample:
    split: Split
    index: int
    last_position_cell: int
    overwrite_count: int
    interference: bool
    delay_chunks: int
    missing: bool
    tokens: tuple[int, ...]
    expected_value: int
    query_key: int
    query_anchor: int
    last_target_position: int | None
    tracked_write_position: int
    sha256: str


def _entropy(factors: tuple[object, ...], label: str) -> int:
    seed = "|".join(map(str, (NAMESPACE, *factors, label)))
    return int.from_bytes(hashlib.sha256(seed.encode("ascii")).digest(), "big")


def _ranked_unique(domain: tuple[int, ...], factors: tuple[object, ...], label: str) -> list[int]:
    return sorted(domain, key=lambda x: (_entropy(factors, f"{label}:{x}"), x))


def _fill(factors: tuple[object, ...], chunk_index: int, n: int) -> tuple[int, ...]:
    return tuple(
        FILLER_TOKENS[_entropy(factors, f"filler:{chunk_index}:{i}") % len(FILLER_TOKENS)]
        for i in range(n)
    )


def make_example(
    *, split: Split, index: int, last_position: int,
    overwrite_count: int, interference: bool, delay_chunks: int,
    missing: bool,
) -> PositionIndependentExample:
    """Balanced-factor sample; labels are returned separately from input tokens."""
    if split not in SPLITS:
        raise ValueError("invalid split")
    if type(index) is not int or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if type(last_position) is not int or last_position not in POSITIONS:
        raise ValueError("invalid last_position")
    if type(overwrite_count) is not int or overwrite_count not in OVERWRITES:
        raise ValueError("invalid overwrite_count")
    if type(interference) is not bool or type(missing) is not bool:
        raise ValueError("missing and interference must be bool")
    if type(delay_chunks) is not int or delay_chunks not in DELAYS:
        raise ValueError("invalid delay_chunks")
    f = (split, index, last_position, overwrite_count, int(interference), delay_chunks, int(missing))

    query_key = KEY_TOKENS[_entropy(f, "query_key") % len(KEY_TOKENS)]
    keys_except_query = tuple(x for x in KEY_TOKENS if x != query_key)
    tracked = (
        keys_except_query[_entropy(f, "decoy") % len(keys_except_query)]
        if missing else query_key
    )
    # At least three additional keys appear in the event prefix in every case.
    unrelated_pool = tuple(x for x in KEY_TOKENS if x not in (query_key, tracked))
    if not missing:
        unrelated_pool = keys_except_query
    unrelated = _ranked_unique(unrelated_pool, f, "unrelated")[:3]

    # Exactly w older tracked-key writes, all earlier than the tracked L position.
    earlier = _ranked_unique(tuple(range(last_position)), f, "earlier")[:overwrite_count]
    tracked_positions = set(earlier + [last_position])
    assert 7 not in tracked_positions

    # All WRITE values distinct across all prefix/delay events: no accidental
    # stale/current equality or identical answers for two distinct event keys.
    ordered_values = _ranked_unique(VALUE_TOKENS, f, "event_values")
    chunks: list[tuple[int, ...]] = []
    other_cursor = _entropy(f, "other_rotation") % len(unrelated)
    other_index = 0
    for pos in range(EVENT_CHUNKS):
        is_tracked = pos in tracked_positions
        event_key = tracked if is_tracked else unrelated[(other_cursor + other_index) % len(unrelated)]
        if not is_tracked:
            other_index += 1
        chunk = (
            WRITE, event_key, ordered_values[pos],
            *_fill(f, pos, CHUNK_SIZE - 3),
        )
        chunks.append(chunk)

    for i in range(delay_chunks):
        pos = EVENT_CHUNKS + i
        if interference:
            # Never overwrite the target, even after its last event WRITE.
            event_key = unrelated[(other_cursor + pos) % len(unrelated)]
            chunks.append(
                (WRITE, event_key, ordered_values[pos], *_fill(f, pos, CHUNK_SIZE - 3))
            )
        else:
            chunks.append((DIST, *_fill(f, pos, CHUNK_SIZE - 1)))

    read_chunk_idx = EVENT_CHUNKS + delay_chunks
    chunks.append(
        (READ, query_key, QUERY, *_fill(f, read_chunk_idx, CHUNK_SIZE - 3))
    )
    assert all(len(chunk) == CHUNK_SIZE for chunk in chunks)
    tokens = tuple(token for chunk in chunks for token in chunk)
    return PositionIndependentExample(
        split=split,
        index=index,
        last_position_cell=last_position,
        overwrite_count=overwrite_count,
        interference=interference,
        delay_chunks=delay_chunks,
        missing=missing,
        tokens=tokens,
        expected_value=NOT_FOUND if missing else ordered_values[last_position],
        query_key=query_key,
        query_anchor=read_chunk_idx * CHUNK_SIZE + 2,
        last_target_position=None if missing else last_position,
        tracked_write_position=last_position,
        sha256=hashlib.sha256(bytes(tokens)).hexdigest(),
    )


def _parse_events(tokens: tuple[int, ...] | list[int]) -> tuple[list[tuple[int, int]], int]:
    """Independent strict parser. Reads must be final, and labels never occur in input."""
    if len(tokens) < 2 * CHUNK_SIZE or len(tokens) % CHUNK_SIZE != 0:
        raise ValueError("expected two or more complete chunks")
    if any(type(x) is not int or x < 0 or x > 255 for x in tokens):
        raise ValueError("input token must be unsigned 8-bit int")
    writes: list[tuple[int, int]] = []
    read_key: int | None = None
    for start in range(0, len(tokens), CHUNK_SIZE):
        chunk = tokens[start : start + CHUNK_SIZE]
        cmd = chunk[0]
        if cmd == WRITE:
            if chunk[1] not in KEY_TOKENS or chunk[2] not in VALUE_TOKENS:
                raise ValueError("invalid WRITE key/value")
            if any(x not in FILLER_TOKENS for x in chunk[3:]):
                raise ValueError("invalid WRITE filler")
            writes.append((chunk[1], chunk[2]))
        elif cmd == DIST:
            if any(x not in FILLER_TOKENS for x in chunk[1:]):
                raise ValueError("invalid DIST filler")
        elif cmd == READ:
            if start != len(tokens) - CHUNK_SIZE:
                raise ValueError("READ must be final")
            if chunk[1] not in KEY_TOKENS or chunk[2] != QUERY:
                raise ValueError("invalid READ key/marker")
            if any(x not in FILLER_TOKENS for x in chunk[3:]):
                raise ValueError("invalid READ filler")
            read_key = chunk[1]
        else:
            raise ValueError("illegal command")
    if read_key is None:
        raise ValueError("missing READ")
    return writes, read_key


def oracle_latest_value_or_not_found(tokens: tuple[int, ...] | list[int]) -> int:
    writes, read_key = _parse_events(tokens)
    state: dict[int, int] = {}
    for key, value in writes:
        state[key] = value
    return state.get(read_key, NOT_FOUND)


def baseline_copy_chunk_2(tokens: tuple[int, ...] | list[int]) -> int:
    """Algorithmic position-copy shortcut, not a trained baseline."""
    writes, _ = _parse_events(tokens)
    if len(tokens) < 3 * CHUNK_SIZE:
        return NOT_FOUND
    chunk = tokens[2 * CHUNK_SIZE : 3 * CHUNK_SIZE]
    return chunk[2] if chunk[0] == WRITE else NOT_FOUND


def baseline_latest_write_any_key(tokens: tuple[int, ...] | list[int]) -> int:
    """Algorithmic recency shortcut, intentionally ignoring query key."""
    writes, _ = _parse_events(tokens)
    return writes[-1][1] if writes else NOT_FOUND


def baseline_most_frequent_value(tokens: tuple[int, ...] | list[int]) -> int:
    """Algorithmic value-frequency shortcut, ties broken by smallest token ID."""
    writes, _ = _parse_events(tokens)
    if not writes:
        return NOT_FOUND
    counts = Counter(v for _, v in writes)
    return min(counts, key=lambda value: (-counts[value], value))


def fixture_panel(*, per_cell: int = 2) -> tuple[PositionIndependentExample, ...]:
    if type(per_cell) is not int or per_cell < 1:
        raise ValueError("per_cell must be a positive integer")
    return tuple(
        make_example(
            split=split, index=i, last_position=last_position,
            overwrite_count=w, interference=interference,
            delay_chunks=delay, missing=missing,
        )
        for split in SPLITS
        for last_position in POSITIONS
        for w in OVERWRITES
        for interference in (False, True)
        for delay in DELAYS
        for missing in (False, True)
        for i in range(per_cell)
    )


def assert_panel_disjoint(panel: tuple[PositionIndependentExample, ...]) -> None:
    """Observed exact duplicate/fingerprint audit, NOT a global collision guarantee."""
    seen: set[str] = set()
    for row in panel:
        actual = hashlib.sha256(bytes(row.tokens)).hexdigest()
        if actual != row.sha256:
            raise ValueError("stale sample fingerprint")
        if actual in seen:
            raise ValueError("duplicate sample across splits/cells")
        seen.add(actual)
