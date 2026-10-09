"""PGW-v3 Stage-1 #1368: pure-Python deterministic oracle integrity tests.

Only fixtures and causal reference answer checks; no training or model inference.
"""

import dataclasses
import hashlib

import pytest

from tam_research.pgw_v3_stage1.oracle import (
    CHUNK_SIZE, DELAYS, DIST, FILLER_TOKENS, KEY_TOKENS, OVERWRITES,
    QUERY, READ, SPLITS, VALUE_TOKENS, WRITE, assert_panel_disjoint,
    fixture_panel, make_example, oracle_latest_value,
)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("overwrites", OVERWRITES)
@pytest.mark.parametrize("interference", (False, True))
@pytest.mark.parametrize("delay", DELAYS)
def test_grammar_oracle_and_latest_write(split, overwrites, interference, delay):
    e = make_example(
        split=split, index=6, overwrite_count=overwrites,
        interference=interference, delay_chunks=delay,
    )
    assert len(e.tokens) == CHUNK_SIZE * (4 + delay)
    assert len(e.tokens) % CHUNK_SIZE == 0
    assert len(e.sha256) == 64
    assert e.query_anchor == (3 + delay) * CHUNK_SIZE + 2
    assert e.last_write_chunk == 2
    assert e.tokens[e.query_anchor - 2] == READ
    assert e.tokens[e.query_anchor - 1] == e.query_key
    assert e.tokens[e.query_anchor] == QUERY
    assert e.query_key in KEY_TOKENS
    assert e.expected_value in VALUE_TOKENS
    assert oracle_latest_value(e.tokens) == e.expected_value

    chunks = [
        e.tokens[i : i + CHUNK_SIZE]
        for i in range(0, len(e.tokens), CHUNK_SIZE)
    ]
    target_writes = [
        (i, chunk[2])
        for i, chunk in enumerate(chunks[:-1])
        if chunk[0] == WRITE and chunk[1] == e.query_key
    ]
    assert len(target_writes) == overwrites + 1
    assert [i for i, _ in target_writes] == list(range(2 - overwrites, 3))
    assert len({v for _, v in target_writes}) == overwrites + 1
    assert target_writes[-1] == (2, e.expected_value)
    for chunk in chunks[3 : 3 + delay]:
        assert (chunk[0] == WRITE) == interference
        if chunk[0] == WRITE:
            assert chunk[1] != e.query_key
    assert set(chunks[-1][3:]).issubset(FILLER_TOKENS)
    assert e.expected_value not in chunks[-1]


def test_stale_write_replacement_does_not_change_answer_but_latest_does():
    e = make_example(
        split="test", index=17, overwrite_count=2,
        interference=True, delay_chunks=4,
    )
    tokens = list(e.tokens)
    expected = oracle_latest_value(tokens)
    old_value = tokens[2]
    different = next(v for v in VALUE_TOKENS if v != old_value and v != expected)
    tokens[2] = different
    assert oracle_latest_value(tokens) == expected

    updated = next(v for v in VALUE_TOKENS if v != expected)
    tokens[2 + 2 * CHUNK_SIZE] = updated
    assert oracle_latest_value(tokens) == updated
    assert updated != expected


def test_unrelated_value_filler_or_suffix_changes_cannot_change_answer():
    e = make_example(
        split="validation", index=14, overwrite_count=0,
        interference=True, delay_chunks=4,
    )
    expected = e.expected_value
    original = list(e.tokens)
    assert original[0] == WRITE and original[1] != e.query_key
    original[2] = next(v for v in VALUE_TOKENS if v != original[2])
    assert oracle_latest_value(original) == expected

    original[4] = next(v for v in FILLER_TOKENS if v != original[4])
    assert oracle_latest_value(original) == expected

    original[-1] = next(v for v in FILLER_TOKENS if v != original[-1])
    assert oracle_latest_value(original) == expected


def test_query_changes_can_change_answer_only_for_written_key():
    e = make_example(
        split="train", index=21, overwrite_count=0,
        interference=False, delay_chunks=1,
    )
    tokens = list(e.tokens)
    other_key = tokens[1]
    other_value = tokens[2]
    assert other_key != e.query_key
    tokens[e.query_anchor - 1] = other_key
    assert oracle_latest_value(tokens) == (
        # The most recent same-key earlier WRITE (possibly chunk #1).
        next(
            tokens[start + 2]
            for start in reversed(range(0, 3 * CHUNK_SIZE, CHUNK_SIZE))
            if tokens[start] == WRITE and tokens[start + 1] == other_key
        )
    )
    assert other_value in VALUE_TOKENS


@pytest.mark.parametrize("bad", ("short", "missing_read", "read_not_final", "unbound", "bad_key",
    "bad_value", "bad_marker", "bad_filler", "bad_command", "negative_token", "bool_token"))
def test_invalid_streams_rejected(bad):
    e = make_example(
        split="train", index=9, overwrite_count=2,
        interference=False, delay_chunks=1,
    )
    tokens = list(e.tokens)
    if bad == "short":
        tokens = tokens[:-1]
    elif bad == "missing_read":
        tokens[e.query_anchor - 2] = DIST
    elif bad == "read_not_final":
        tokens[:CHUNK_SIZE] = tokens[-CHUNK_SIZE:]
    elif bad == "unbound":
        tokens[e.query_anchor - 1] = next(k for k in KEY_TOKENS if k != e.query_key)
    elif bad == "bad_key":
        tokens[1] = 240
    elif bad == "bad_value":
        tokens[2] = 9
    elif bad == "bad_marker":
        tokens[e.query_anchor] = 9
    elif bad == "bad_filler":
        tokens[3] = 32
    elif bad == "bad_command":
        tokens[0] = 250
    elif bad == "negative_token":
        tokens[0] = -1
    elif bad == "bool_token":
        tokens[0] = True
    with pytest.raises(ValueError):
        oracle_latest_value(tokens)


@pytest.mark.parametrize("field,wrong", [
    ("split", "unknown"), ("index", -1), ("index", True),
    ("overwrite_count", 3), ("interference", 1), ("delay_chunks", 3),
])
def test_bad_sample_requests_rejected(field, wrong):
    kwargs = {
        "split": "train", "index": 0, "overwrite_count": 1,
        "interference": False, "delay_chunks": 2,
    }
    kwargs[field] = wrong
    with pytest.raises(ValueError):
        make_example(**kwargs)


def test_fixture_determinism_split_disjointness_and_global_rng_independence():
    a = fixture_panel(per_cell=8)
    b = tuple(reversed(fixture_panel(per_cell=8)))
    assert len(a) == 3 * 3 * 2 * 3 * 8
    assert a == tuple(reversed(b))
    assert_panel_disjoint(a)
    assert len({item.sha256 for item in a}) == len(a)
    assert {s.split for s in a} == set(SPLITS)
    assert {s.overwrite_count for s in a} == set(OVERWRITES)
    assert {s.delay_chunks for s in a} == set(DELAYS)
    assert all(s.sha256 == hashlib.sha256(bytes(s.tokens)).hexdigest() for s in a)
    assert all(oracle_latest_value(s.tokens) == s.expected_value for s in a)


def test_disjointness_rejects_repeated_and_tampered_samples():
    sample = make_example(
        split="test", index=8, overwrite_count=1,
        interference=True, delay_chunks=4,
    )
    with pytest.raises(ValueError, match="duplicate"):
        assert_panel_disjoint((sample, sample))
    with pytest.raises(ValueError, match="fingerprint"):
        assert_panel_disjoint((dataclasses.replace(sample, sha256="0" * 64),))


def test_no_read_suffix_label_leakage_even_with_future_interference():
    e = make_example(
        split="test", index=99, overwrite_count=1,
        interference=True, delay_chunks=4,
    )
    final_chunk = e.tokens[-CHUNK_SIZE:]
    assert final_chunk[:3] == (READ, e.query_key, QUERY)
    assert all(token in FILLER_TOKENS for token in final_chunk[3:])
    assert e.expected_value not in final_chunk


def test_fixture_panel_requires_positive_cell_count():
    for invalid in (0, -1):
        with pytest.raises(ValueError):
            fixture_panel(per_cell=invalid)
