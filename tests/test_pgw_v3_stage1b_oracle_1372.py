"""PGW-v3 #1374 Stage-1B zero-GPU, no-training position-deconfounding tests."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import random

import pytest

from tam_research.pgw_v3_stage1b.oracle import (
    CHUNK_SIZE, DELAYS, DIST, EVENT_CHUNKS, FILLER_TOKENS,
    KEY_TOKENS, NOT_FOUND, OVERWRITES, POSITIONS, QUERY, READ, SPLITS,
    VALUE_TOKENS, WRITE, assert_panel_disjoint, baseline_copy_chunk_2,
    baseline_latest_write_any_key, baseline_most_frequent_value,
    fixture_panel, make_example, oracle_latest_value_or_not_found,
)


def _chunks(tokens):
    return [tokens[i:i + CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE)]


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("last_position", POSITIONS)
@pytest.mark.parametrize("overwrite_count", OVERWRITES)
@pytest.mark.parametrize("interference", (False, True))
@pytest.mark.parametrize("delay_chunks", DELAYS)
@pytest.mark.parametrize("missing", (False, True))
def test_every_factorial_cell_has_valid_causal_oracle(
    split, last_position, overwrite_count, interference, delay_chunks, missing,
):
    row = make_example(
        split=split, index=0, last_position=last_position,
        overwrite_count=overwrite_count, interference=interference,
        delay_chunks=delay_chunks, missing=missing,
    )
    chunks = _chunks(row.tokens)
    assert len(chunks) == EVENT_CHUNKS + delay_chunks + 1
    assert len(row.tokens) == CHUNK_SIZE * (EVENT_CHUNKS + delay_chunks + 1)
    assert all(len(c) == CHUNK_SIZE for c in chunks)
    assert row.query_anchor == (EVENT_CHUNKS + delay_chunks) * CHUNK_SIZE + 2
    assert chunks[-1][:3] == (READ, row.query_key, QUERY)
    assert all(x in FILLER_TOKENS for x in chunks[-1][3:])
    assert row.expected_value not in chunks[-1]
    assert NOT_FOUND not in row.tokens
    assert all(c[0] == WRITE for c in chunks[:EVENT_CHUNKS])
    assert chunks[7][1] != row.query_key
    assert row.tracked_write_position == last_position

    if missing:
        assert row.expected_value == NOT_FOUND
        assert row.last_target_position is None
        assert all(c[1] != row.query_key for c in chunks[:-1] if c[0] == WRITE)
    else:
        assert row.last_target_position == last_position
        target_indices = [i for i, c in enumerate(chunks[:-1]) if c[0] == WRITE and c[1] == row.query_key]
        assert len(target_indices) == overwrite_count + 1
        assert target_indices[-1] == last_position
        assert chunks[last_position][2] == row.expected_value

    # Missing and present cases are structurally matched: a tracked key at L
    # always has exactly w+1 prefix WRITE events; missing just changes which
    # key is queried at the end.
    tracked_key = chunks[last_position][1]
    tracked_positions = [
        i for i, c in enumerate(chunks[:EVENT_CHUNKS]) if c[1] == tracked_key
    ]
    assert len(tracked_positions) == overwrite_count + 1
    assert tracked_positions[-1] == last_position
    other_event_keys = {c[1] for c in chunks[:EVENT_CHUNKS] if c[1] != tracked_key}
    assert len(other_event_keys) >= 3

    all_write_values = [c[2] for c in chunks[:-1] if c[0] == WRITE]
    assert len(set(all_write_values)) == len(all_write_values)
    assert set(all_write_values).issubset(VALUE_TOKENS)
    assert all((c[0] == WRITE) == interference for c in chunks[EVENT_CHUNKS:-1])
    assert all(c[1] != row.query_key for c in chunks[EVENT_CHUNKS:-1] if c[0] == WRITE)
    assert oracle_latest_value_or_not_found(row.tokens) == row.expected_value


def test_latest_target_stale_unrelated_filler_and_query_counterfactuals():
    row = make_example(
        split="test", index=3, last_position=5,
        overwrite_count=2, interference=True,
        delay_chunks=4, missing=False,
    )
    before = row.expected_value
    prefix = _chunks(row.tokens)
    old_indices = [i for i in range(5) if prefix[i][1] == row.query_key]
    assert len(old_indices) == 2
    stale_edit = list(row.tokens)
    stale_index = old_indices[0] * CHUNK_SIZE + 2
    stale_edit[stale_index] = next(v for v in VALUE_TOKENS if v != stale_edit[stale_index] and v != before)
    assert oracle_latest_value_or_not_found(stale_edit) == before

    latest_edit = list(row.tokens)
    new_value = next(v for v in VALUE_TOKENS if v != before and v not in stale_edit)
    latest_edit[5 * CHUNK_SIZE + 2] = new_value
    assert oracle_latest_value_or_not_found(latest_edit) == new_value
    assert new_value != before

    other_key_chunk = next(i for i, c in enumerate(prefix[:EVENT_CHUNKS]) if c[1] != row.query_key)
    other_edit = list(row.tokens)
    other_offset = other_key_chunk * CHUNK_SIZE + 2
    other_edit[other_offset] = next(v for v in VALUE_TOKENS if v != other_edit[other_offset])
    assert oracle_latest_value_or_not_found(other_edit) == before

    filler_edit = list(row.tokens)
    filler_edit[-1] = next(v for v in FILLER_TOKENS if v != filler_edit[-1])
    assert oracle_latest_value_or_not_found(filler_edit) == before

    query_other = list(row.tokens)
    new_key = prefix[other_key_chunk][1]
    query_other[row.query_anchor - 1] = new_key
    assert oracle_latest_value_or_not_found(query_other) != before

    unwritten_keys = set(KEY_TOKENS) - {c[1] for c in prefix[:-1] if c[0] == WRITE}
    assert unwritten_keys
    query_unwritten = list(row.tokens)
    query_unwritten[row.query_anchor - 1] = min(unwritten_keys)
    assert oracle_latest_value_or_not_found(query_unwritten) == NOT_FOUND


def test_fixed_naive_algorithmic_heuristics_fail_on_adversarial_present_and_missing_cases():
    # Eight valid WRITE chunks: answer for key16 is value55 from chunk3,
    # whereas position2 gives 42, latest-global gives 47, min value gives 40.
    keys = (16, 17, 18, 16, 17, 19, 17, 18)
    values = (40, 41, 42, 55, 44, 45, 46, 47)
    writes = tuple(
        token
        for key, value in zip(keys, values)
        for token in (WRITE, key, value, *([160] * 5))
    )
    dist = (DIST, *([161] * 7)) * 2
    present = writes + dist + (READ, 16, QUERY, *([162] * 5))
    missing = writes + dist + (READ, 23, QUERY, *([162] * 5))

    assert oracle_latest_value_or_not_found(present) == 55
    assert oracle_latest_value_or_not_found(missing) == NOT_FOUND
    for prefix, answer in ((present, 55), (missing, NOT_FOUND)):
        assert baseline_copy_chunk_2(prefix) == 42
        assert baseline_latest_write_any_key(prefix) == 47
        assert baseline_most_frequent_value(prefix) == 40
        assert baseline_copy_chunk_2(prefix) != answer
        assert baseline_latest_write_any_key(prefix) != answer
        assert baseline_most_frequent_value(prefix) != answer


@pytest.mark.parametrize("bad", (
    "truncated", "single_chunk", "missing_read", "read_before_final",
    "second_read", "bad_write_key", "bad_write_value", "bad_write_filler",
    "bad_dist_filler", "bad_read_key", "bad_query_marker", "bad_read_filler",
    "bad_command", "bool_token", "negative_token", "overflow_token",
))
def test_strict_oracle_rejects_malformed_streams(bad):
    row = make_example(
        split="train", index=11, last_position=4, overwrite_count=1,
        interference=False, delay_chunks=2, missing=False,
    )
    t = list(row.tokens)
    if bad == "truncated":
        t.pop()
    elif bad == "single_chunk":
        t = t[-CHUNK_SIZE:]
    elif bad == "missing_read":
        t[-CHUNK_SIZE] = DIST
    elif bad == "read_before_final":
        t[:CHUNK_SIZE] = t[-CHUNK_SIZE:]
    elif bad == "second_read":
        t[CHUNK_SIZE:2*CHUNK_SIZE] = t[-CHUNK_SIZE:]
    elif bad == "bad_write_key":
        t[1] = 99
    elif bad == "bad_write_value":
        t[2] = 99
    elif bad == "bad_write_filler":
        t[3] = 32
    elif bad == "bad_dist_filler":
        t[EVENT_CHUNKS * CHUNK_SIZE + 1] = 16
    elif bad == "bad_read_key":
        t[-CHUNK_SIZE + 1] = 99
    elif bad == "bad_query_marker":
        t[-CHUNK_SIZE + 2] = 99
    elif bad == "bad_read_filler":
        t[-1] = 64
    elif bad == "bad_command":
        t[0] = 99
    elif bad == "bool_token":
        t[0] = True
    elif bad == "negative_token":
        t[0] = -1
    elif bad == "overflow_token":
        t[0] = 256
    with pytest.raises(ValueError):
        oracle_latest_value_or_not_found(t)


@pytest.mark.parametrize("field,bad", (
    ("split", "wrong"), ("index", -1), ("index", True),
    ("last_position", 7), ("last_position", False), ("overwrite_count", 3),
    ("interference", 1), ("missing", 1), ("delay_chunks", 3),
))
def test_invalid_generator_requests_fail_closed(field, bad):
    kwargs = dict(
        split="train", index=0, last_position=5, overwrite_count=1,
        interference=False, delay_chunks=2, missing=False,
    )
    kwargs[field] = bad
    with pytest.raises(ValueError):
        make_example(**kwargs)


def test_strata_coverage_determinism_and_observed_split_disjointness():
    global_state = random.getstate()
    panel = fixture_panel(per_cell=2)
    assert random.getstate() == global_state
    assert len(panel) == 3 * 5 * 3 * 2 * 3 * 2 * 2
    assert_panel_disjoint(panel)
    assert len({x.sha256 for x in panel}) == len(panel)
    assert {x.last_target_position for x in panel if not x.missing} == set(POSITIONS)
    assert {x.delay_chunks for x in panel} == set(DELAYS)
    assert sum(x.missing for x in panel) * 2 == len(panel)
    assert Counter(x.split for x in panel) == {s: len(panel)//3 for s in SPLITS}
    assert panel == fixture_panel(per_cell=2)
    assert all(x.sha256 == hashlib.sha256(bytes(x.tokens)).hexdigest() for x in panel)

    # All factor dimensions included in the hashed generation namespace.
    a = make_example(
        split="train", index=10, last_position=2, overwrite_count=1,
        interference=False, delay_chunks=1, missing=False,
    )
    b = make_example(
        split="test", index=10, last_position=2, overwrite_count=1,
        interference=False, delay_chunks=1, missing=False,
    )
    assert a.tokens != b.tokens
    assert a == make_example(
        split="train", index=10, last_position=2, overwrite_count=1,
        interference=False, delay_chunks=1, missing=False,
    )


def test_duplicate_and_corrupt_fingerprints_refused():
    sample = make_example(
        split="validation", index=2, last_position=6,
        overwrite_count=0, interference=True, delay_chunks=4, missing=True,
    )
    with pytest.raises(ValueError, match="duplicate"):
        assert_panel_disjoint((sample, sample))
    with pytest.raises(ValueError, match="fingerprint"):
        assert_panel_disjoint((replace(sample, sha256="0"*64),))


@pytest.mark.parametrize("count", (0, -1, True, 1.5))
def test_fixture_panel_invalid_count(count):
    with pytest.raises(ValueError):
        fixture_panel(per_cell=count)
