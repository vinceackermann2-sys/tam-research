"""PGW-v3 Stage-2B1: immutable-data INTEGRITY preflight, no training.

This fixture-level audit is NOT a scientific dataset manifest or permission
to optimize an architecture. Source & counts frozen in issue #1394.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random

from tam_research.pgw_v3_stage1b.oracle import (
    DELAYS, NOT_FOUND, OVERWRITES, POSITIONS, QUERY, READ, SPLITS,
    assert_panel_disjoint, baseline_copy_chunk_2,
    baseline_latest_write_any_key, baseline_most_frequent_value,
    fixture_panel, oracle_latest_value_or_not_found,
)

BASELINE_NAMES = (
    "always_not_found",
    "copy_chunk_2",
    "latest_global_write",
    "most_frequent_value",
)
REPEATS_PER_CELL = 4
CELLS_PER_SPLIT = len(POSITIONS) * len(OVERWRITES) * 2 * len(DELAYS) * 2


@dataclass(frozen=True)
class AccuracyCounts:
    correct: int
    present_correct: int
    missing_correct: int


@dataclass(frozen=True)
class SplitManifest:
    split: str
    samples: int
    present: int
    missing: int
    unique_cells: int
    sha256: str
    baselines: tuple[tuple[str, AccuracyCounts], ...]


@dataclass(frozen=True)
class StaticManifest:
    samples: int
    total_input_tokens: int
    sha256: str
    per_split: tuple[SplitManifest, ...]
    per_cell: int


def _record(row) -> str:
    """Canonical row identity; external answer is audit-only, never a feature."""
    columns = [
        row.split, row.index, row.last_position_cell, row.overwrite_count,
        int(row.interference), row.delay_chunks, int(row.missing),
        row.sha256, row.expected_value,
    ]
    return json.dumps(columns, ensure_ascii=True, separators=(",", ":")) + "\n"


def _hash_rows(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("ascii"))
    return digest.hexdigest()


def static_fixture_manifest(*, per_cell: int = REPEATS_PER_CELL) -> StaticManifest:
    """Fully check a bounded deterministic Stage-1B fixture panel once.

    The returned hashes cover exactly the selected fixture samples, NOT a
    future complete trained corpus. All algorithmic heuristics are untrained.
    """
    if type(per_cell) is not int or per_cell < 1 or per_cell > 16:
        raise ValueError("per_cell must be an integer from 1 to 16")
    python_rng_before = random.getstate()
    panel = fixture_panel(per_cell=per_cell)
    assert_panel_disjoint(panel)
    if random.getstate() != python_rng_before:
        raise ValueError("forbidden global Python RNG mutation")

    expected_per_split = CELLS_PER_SPLIT * per_cell
    if len(panel) != expected_per_split * len(SPLITS):
        raise ValueError("incomplete split/cell fixture panel")

    canonical = sorted((_record(row) for row in panel))
    split_reports: list[SplitManifest] = []
    for split in SPLITS:
        rows = [row for row in panel if row.split == split]
        if len(rows) != expected_per_split:
            raise ValueError("incorrect sample count for split")
        cells = Counter(
            (r.last_position_cell, r.overwrite_count,
             r.interference, r.delay_chunks, r.missing)
            for r in rows
        )
        if len(cells) != CELLS_PER_SPLIT or set(cells.values()) != {per_cell}:
            raise ValueError("unbalanced PGW-v3 factorial cell coverage")
        present = sum(not r.missing for r in rows)
        missing = sum(r.missing for r in rows)
        if present != missing or present + missing != len(rows):
            raise ValueError("query presence stratification mismatch")

        hits = {name: [0, 0, 0] for name in BASELINE_NAMES}
        for r in rows:
            if hashlib.sha256(bytes(r.tokens)).hexdigest() != r.sha256:
                raise ValueError("invalid token stream hash")
            if r.tokens[r.query_anchor] != QUERY or r.tokens[r.query_anchor - 2] != READ:
                raise ValueError("invalid final query/read markers")
            if r.expected_value in r.tokens[-8:]:
                raise ValueError("external answer leaked into READ suffix")
            expected = oracle_latest_value_or_not_found(r.tokens)
            if expected != r.expected_value:
                raise ValueError("frozen oracle and external label disagree")
            if r.missing != (expected == NOT_FOUND):
                raise ValueError("missing label does not match oracle")

            predictions = (
                NOT_FOUND,
                baseline_copy_chunk_2(r.tokens),
                baseline_latest_write_any_key(r.tokens),
                baseline_most_frequent_value(r.tokens),
            )
            for name, predicted in zip(BASELINE_NAMES, predictions):
                if predicted == expected:
                    hits[name][0] += 1
                    hits[name][1 if not r.missing else 2] += 1

        if hits["always_not_found"] != [missing, 0, missing]:
            raise ValueError("always-NOT_FOUND 50% shortcut validation failed")
        scores = tuple(
            (name, AccuracyCounts(*hits[name])) for name in BASELINE_NAMES
        )
        split_reports.append(
            SplitManifest(
                split=split,
                samples=len(rows),
                present=present,
                missing=missing,
                unique_cells=len(cells),
                sha256=_hash_rows(sorted(_record(r) for r in rows)),
                baselines=scores,
            )
        )
    return StaticManifest(
        samples=len(panel),
        total_input_tokens=sum(len(r.tokens) for r in panel),
        sha256=_hash_rows(canonical),
        per_split=tuple(split_reports),
        per_cell=per_cell,
    )
