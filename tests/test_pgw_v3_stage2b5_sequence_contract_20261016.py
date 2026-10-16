"""CPU-only PGW-v3 Stage-2B5 traversal regression (2026-10-16).

The frozen content-set SHA-256 roots are insensitive to row order. This
regression independently pins the generator's *traversal contract* without
changing or republishing the source-bound sealed corpus roots. No optimizer,
model inference, scientific seeds, GPU, or research attempt is used.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import random

import pytest

from tam_research.pgw_v3_stage1b.oracle import (
    DELAYS, OVERWRITES, POSITIONS, SPLITS,
)
from tam_research.pgw_v3_stage2b5.corpus import (
    SPLIT_REPEATS,
    _digest,
    _record_line,
    candidate_corpus_manifest,
)

# PRE-EXISTING roots from issue #1454, NOT generated or updated by this test.
SEALED_ALL_ROOT = (
    "36a40585fe3deb2c4fc268c32eb93092ebc46f7809b772cdbc223d0185d940ef"
)
SEALED_SPLIT_ROOTS = (
    ("train", "0174103c2ef1cf14e5dec0df23498d994062dabd0e4151d02f7d2ba4f055d9a3"),
    ("validation", "c6b4fb7bf1943755510bb24fb98053a0ccf45cc83ecb4b51c821193845668611"),
    ("test", "3b3ab8e8b5d68337ad1608053541838f02369033b6c95e538a7266b330d8c7d3"),
)


def _key(record):
    return (
        record.split, record.index, record.last_position,
        record.overwrite_count, record.interference, record.delay,
        record.missing,
    )


def _expected_keys():
    # Independent declaration of the Stage-2B5 grammar's iteration order.
    return tuple(
        (split, index, position, overwrites, interference, delay, missing)
        for split, repeats in SPLIT_REPEATS
        for position in POSITIONS
        for overwrites in OVERWRITES
        for interference in (False, True)
        for delay in DELAYS
        for missing in (False, True)
        for index in range(repeats)
    )


def _order_digest(records):
    h = hashlib.sha256()
    h.update(b"PGW-V3-STAGE2B5-ORDER-V1\n")
    for position, row in enumerate(records):
        # Include both rank and sample identity; never confuse this new digest
        # with the pre-existing immutable, order-INsensitive corpus seal.
        h.update(str(position).encode("ascii"))
        h.update(b":")
        h.update(row.sample_id.encode("ascii"))
        h.update(b":")
        h.update(_record_line(row).encode("ascii"))
    return h.hexdigest()


@pytest.fixture(scope="module")
def manifest():
    return candidate_corpus_manifest()


def test_ordered_traversal_is_independently_bound_to_original_grammar(manifest):
    before_rng = random.getstate()
    assert tuple(split for split, _ in SPLIT_REPEATS) == SPLITS
    assert manifest.total_samples == len(manifest.records) == 5760
    assert manifest.root_sha256 == SEALED_ALL_ROOT
    assert tuple((x.name, x.root_sha256) for x in manifest.splits) == SEALED_SPLIT_ROOTS
    assert _digest(_record_line(r) for r in manifest.records) == SEALED_ALL_ROOT

    expected = _expected_keys()
    actual = tuple(map(_key, manifest.records))
    assert len(expected) == 5760
    assert actual == expected
    assert len(set(actual)) == len(actual)
    assert len({r.sample_id for r in manifest.records}) == len(actual)
    assert random.getstate() == before_rng


def test_frozen_set_digest_does_not_protect_traversal_order(manifest):
    original = manifest.records
    reordered = (original[1], original[0]) + original[2:]
    assert _key(original[0]) != _key(original[1])
    # Demonstrates a genuine negative control, not a failing suite:
    # a swap is invisible to Stage-2B5's content seal.
    assert _digest(_record_line(r) for r in reordered) == SEALED_ALL_ROOT
    assert _order_digest(original) != _order_digest(reordered)
    assert tuple(map(_key, reordered)) != _expected_keys()
    # No mutation to the source records or immutable root strings.
    assert manifest.records == original


def test_content_modification_still_changes_frozen_set_digest(manifest):
    first = manifest.records[0]
    changed = replace(first, answer=64 if first.answer != 64 else 32)
    forged = (changed,) + manifest.records[1:]
    assert _digest(_record_line(r) for r in forged) != SEALED_ALL_ROOT
