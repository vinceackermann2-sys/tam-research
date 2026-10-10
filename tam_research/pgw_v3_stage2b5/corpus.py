"""Stage-2B5 full candidate corpus identities and pure-Python held-out scoring.

NO MODEL TRAINING, optimizer, GPU, scientific seeds or scientific attempts.
All inputs are regenerated from frozen Stage-1B oracle grammar, never changed.
This is a *candidate* train/validation/test protocol, not run authority.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Iterable

from tam_research.pgw_v3_stage1b.oracle import (
    CHUNK_SIZE, DELAYS, FILLER_TOKENS, NOT_FOUND, OVERWRITES, POSITIONS,
    QUERY, READ, SPLITS, VALUE_TOKENS, WRITE, assert_panel_disjoint,
    baseline_copy_chunk_2, baseline_latest_write_any_key,
    baseline_most_frequent_value, make_example,
    oracle_latest_value_or_not_found,
)

# Intentionally distinct from Stage-2B1's 4/split/cell integrity fixture.
SPLIT_REPEATS = (("train", 16), ("validation", 8), ("test", 8))
CELLS_PER_SPLIT = len(POSITIONS) * len(OVERWRITES) * 2 * len(DELAYS) * 2
TOTAL_SAMPLES = CELLS_PER_SPLIT * sum(n for _, n in SPLIT_REPEATS)
ABSENT_CLASS = NOT_FOUND


@dataclass(frozen=True)
class CorpusRecord:
    sample_id: str
    split: str
    index: int
    last_position: int
    overwrite_count: int
    interference: bool
    delay: int
    missing: bool
    token_sha256: str
    semantic_sha256: str
    answer: int

    @property
    def cell(self) -> tuple[int, int, bool, int, bool]:
        return (
            self.last_position, self.overwrite_count, self.interference,
            self.delay, self.missing,
        )


@dataclass(frozen=True)
class SplitRoot:
    name: str
    sample_count: int
    per_cell: int
    cells: int
    present: int
    missing: int
    root_sha256: str


@dataclass(frozen=True)
class CandidateCorpusManifest:
    root_sha256: str
    records: tuple[CorpusRecord, ...]
    splits: tuple[SplitRoot, ...]
    total_samples: int
    total_input_tokens: int


@dataclass(frozen=True)
class CellMetric:
    cell: tuple[int, int, bool, int, bool]
    total: int
    correct: int


@dataclass(frozen=True)
class EvaluationScore:
    split: str
    total: int
    correct: int
    total_accuracy: float
    present_total: int
    present_correct: int
    present_accuracy: float
    missing_total: int
    missing_correct: int
    missing_accuracy: float
    macro_cell_accuracy: float
    missing_tp: int
    missing_fp: int
    missing_fn: int
    missing_tn: int
    missing_precision: float
    missing_recall: float
    by_delay: tuple[tuple[int, float], ...]
    by_overwrites: tuple[tuple[int, float], ...]
    by_position: tuple[tuple[int, float], ...]
    cells: tuple[CellMetric, ...]


def _ascii_row(values: tuple[object, ...]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":")) + "\n"


def _digest(rows: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for row in sorted(rows):
        hasher.update(row.encode("ascii"))
    return hasher.hexdigest()


def _semantic_trace(tokens: tuple[int, ...]) -> str:
    """Ordered WRITE key/value events + final READ key; ignore filler entropy."""
    if len(tokens) % CHUNK_SIZE:
        raise ValueError("incomplete token stream")
    events: list[tuple[int, int]] = []
    for start in range(0, len(tokens), CHUNK_SIZE):
        chunk = tokens[start : start + CHUNK_SIZE]
        if chunk[0] == WRITE:
            events.append((chunk[1], chunk[2]))
        elif chunk[0] == READ:
            if start != len(tokens) - CHUNK_SIZE or chunk[2] != QUERY:
                raise ValueError("invalid final READ/QUERY")
            return _digest([_ascii_row((tuple(events), chunk[1]))])
    raise ValueError("missing READ")


def _generate(row: CorpusRecord):
    sample = make_example(
        split=row.split, index=row.index, last_position=row.last_position,
        overwrite_count=row.overwrite_count,
        interference=row.interference, delay_chunks=row.delay,
        missing=row.missing,
    )
    if (
        sample.sha256 != row.token_sha256
        or sample.expected_value != row.answer
        or _semantic_trace(sample.tokens) != row.semantic_sha256
    ):
        raise ValueError("corpus row identity changed relative to frozen source")
    return sample


def _record_line(record: CorpusRecord) -> str:
    # Label is metadata in a SHA-bound audit record, NEVER an input feature.
    return _ascii_row((
        record.split, record.index, record.last_position,
        record.overwrite_count, int(record.interference), record.delay,
        int(record.missing), record.token_sha256, record.semantic_sha256,
        record.answer,
    ))


def candidate_corpus_manifest() -> CandidateCorpusManifest:
    """Build complete deterministic 5,760-sample identities, NO training.

    Strict duplicate audits cover both byte-identical streams AND full ordered
    key/value/query traces with filler removed. Semantic uniqueness here does
    not imply distribution-shift independence or generalization of a model.
    """
    if tuple(x for x, _ in SPLIT_REPEATS) != SPLITS:
        raise ValueError("frozen split order changed")
    python_rng_before = random.getstate()
    rows: list[CorpusRecord] = []
    split_roots: list[SplitRoot] = []
    seen_tokens: set[str] = set()
    seen_semantics: set[str] = set()
    seen_ids: set[str] = set()
    token_total = 0
    for split, repeats in SPLIT_REPEATS:
        split_rows = []
        cells: Counter[tuple[int, int, bool, int, bool]] = Counter()
        for last_position in POSITIONS:
            for overwrite_count in OVERWRITES:
                for interference in (False, True):
                    for delay in DELAYS:
                        for missing in (False, True):
                            for index in range(repeats):
                                sample = make_example(
                                    split=split, index=index,
                                    last_position=last_position,
                                    overwrite_count=overwrite_count,
                                    interference=interference,
                                    delay_chunks=delay, missing=missing,
                                )
                                token_sha = hashlib.sha256(bytes(sample.tokens)).hexdigest()
                                if sample.sha256 != token_sha:
                                    raise ValueError("Stage-1B token digest mismatch")
                                if oracle_latest_value_or_not_found(sample.tokens) != sample.expected_value:
                                    raise ValueError("independent oracle disagreement")
                                if sample.expected_value not in VALUE_TOKENS + (NOT_FOUND,):
                                    raise ValueError("illegal external oracle label")
                                if sample.missing != (sample.expected_value == NOT_FOUND):
                                    raise ValueError("oracle presence flag mismatch")
                                if NOT_FOUND in sample.tokens:
                                    raise ValueError("NOT_FOUND external label leaked into input")
                                if sample.query_anchor != len(sample.tokens) - CHUNK_SIZE + 2:
                                    raise ValueError("invalid final READ anchor")
                                if sample.tokens[-CHUNK_SIZE] != READ or sample.tokens[-CHUNK_SIZE+2] != QUERY:
                                    raise ValueError("missing READ/QUERY marker")
                                if sample.expected_value in sample.tokens[-CHUNK_SIZE:]:
                                    raise ValueError("external target appears in final READ suffix")
                                semantic_sha = _semantic_trace(sample.tokens)
                                if token_sha in seen_tokens or semantic_sha in seen_semantics:
                                    raise ValueError("duplicate input or ordered semantic WRITE/READ trace")
                                seen_tokens.add(token_sha)
                                seen_semantics.add(semantic_sha)
                                meta = (
                                    split, index, last_position, overwrite_count,
                                    int(interference), delay, int(missing), token_sha,
                                )
                                sample_id = hashlib.sha256(
                                    _ascii_row(meta).encode("ascii")
                                ).hexdigest()
                                if sample_id in seen_ids:
                                    raise ValueError("duplicate sample identifier")
                                seen_ids.add(sample_id)
                                rec = CorpusRecord(
                                    sample_id=sample_id, split=split, index=index,
                                    last_position=last_position,
                                    overwrite_count=overwrite_count,
                                    interference=interference, delay=delay,
                                    missing=missing, token_sha256=token_sha,
                                    semantic_sha256=semantic_sha,
                                    answer=sample.expected_value,
                                )
                                token_total += len(sample.tokens)
                                cells[rec.cell] += 1
                                rows.append(rec)
                                split_rows.append(rec)
        if len(cells) != CELLS_PER_SPLIT or set(cells.values()) != {repeats}:
            raise ValueError("split factorial cell coverage violated")
        present = sum(not r.missing for r in split_rows)
        absent = sum(r.missing for r in split_rows)
        if present != absent or len(split_rows) != CELLS_PER_SPLIT * repeats:
            raise ValueError("split present/missing balance violated")
        split_roots.append(
            SplitRoot(
                name=split, sample_count=len(split_rows), per_cell=repeats,
                cells=len(cells), present=present, missing=absent,
                root_sha256=_digest(_record_line(row) for row in split_rows),
            )
        )
    if random.getstate() != python_rng_before:
        raise ValueError("global Python RNG changed")
    if len(rows) != TOTAL_SAMPLES:
        raise ValueError("frozen total candidate corpus cardinality changed")
    return CandidateCorpusManifest(
        root_sha256=_digest(_record_line(r) for r in rows),
        records=tuple(rows), splits=tuple(split_roots),
        total_samples=len(rows), total_input_tokens=token_total,
    )


def algorithmic_predictions(
    manifest: CandidateCorpusManifest, *, split: str, rule: str
) -> tuple[tuple[str, int], ...]:
    """No trained model. Generate external label-free baseline predictions."""
    if split not in SPLITS:
        raise ValueError("invalid split")
    if rule not in ("always_not_found", "oracle", "copy_chunk_2",
                    "latest_global_write", "most_frequent_value"):
        raise ValueError("invalid algorithmic audit rule")
    rows = (r for r in manifest.records if r.split == split)
    results = []
    for rec in rows:
        if rule == "always_not_found":
            pred = NOT_FOUND
        else:
            sample = _generate(rec)
            if rule == "oracle":
                pred = oracle_latest_value_or_not_found(sample.tokens)
            elif rule == "copy_chunk_2":
                pred = baseline_copy_chunk_2(sample.tokens)
            elif rule == "latest_global_write":
                pred = baseline_latest_write_any_key(sample.tokens)
            else:
                pred = baseline_most_frequent_value(sample.tokens)
        results.append((rec.sample_id, pred))
    return tuple(results)


def evaluate_complete_predictions(
    manifest: CandidateCorpusManifest, *, split: str,
    predictions: Iterable[tuple[str, int]],
) -> EvaluationScore:
    """Strict held-out metrics on a COMPLETE supplied per-split prediction panel.

    Does NOT run inference. Never sorts or alters supplied model predictions.
    Missing, extra, duplicate, non-integer or out-of-vocab classes fail closed.
    """
    if split not in SPLITS:
        raise ValueError("unknown evaluation split")
    rows = tuple(r for r in manifest.records if r.split == split)
    expected = {r.sample_id: r for r in rows}
    if len(expected) != len(rows) or not rows:
        raise ValueError("empty/duplicated manifest rows")
    seen: dict[str, int] = {}
    for item in predictions:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("predictions must be (sample-id, class) tuples")
        key, prediction = item
        if not isinstance(key, str) or key not in expected or key in seen:
            raise ValueError("unknown or duplicate prediction identity")
        if type(prediction) is not int or prediction not in VALUE_TOKENS + (NOT_FOUND,):
            raise ValueError("predicted class must be a 32..64 token id")
        seen[key] = prediction
    if len(seen) != len(expected):
        raise ValueError("incomplete prediction panel")
    correct = present_correct = absent_correct = 0
    present_total = missing_total = 0
    tp = fp = fn = tn = 0
    cell_total: Counter[tuple[int, int, bool, int, bool]] = Counter()
    cell_correct: Counter[tuple[int, int, bool, int, bool]] = Counter()
    partitions = {
        "delay": {}, "overwrite": {}, "position": {},
    }
    for r in rows:
        prediction = seen[r.sample_id]
        good = prediction == r.answer
        correct += int(good)
        if r.missing:
            missing_total += 1
            absent_correct += int(good)
        else:
            present_total += 1
            present_correct += int(good)
        if prediction == NOT_FOUND:
            tp += int(r.missing)
            fp += int(not r.missing)
        else:
            fn += int(r.missing)
            tn += int(not r.missing)
        cell_total[r.cell] += 1
        cell_correct[r.cell] += int(good)
        for label, factor in (
            ("delay", r.delay),
            ("overwrite", r.overwrite_count),
            ("position", r.last_position),
        ):
            total, successes = partitions[label].get(factor, (0, 0))
            partitions[label][factor] = (total + 1, successes + int(good))
    if len(cell_total) != CELLS_PER_SPLIT or min(cell_total.values()) < 1:
        raise ValueError("evaluation did not cover all frozen factorial cells")
    cell_scores = tuple(
        CellMetric(cell=k, total=cell_total[k], correct=cell_correct[k])
        for k in sorted(cell_total)
    )
    macro = sum(x.correct / x.total for x in cell_scores) / CELLS_PER_SPLIT
    def group_scores(name: str) -> tuple[tuple[int, float], ...]:
        return tuple(
            (k, successes / total)
            for k, (total, successes) in sorted(partitions[name].items())
        )
    return EvaluationScore(
        split=split, total=len(rows), correct=correct,
        total_accuracy=correct / len(rows),
        present_total=present_total, present_correct=present_correct,
        present_accuracy=present_correct / present_total,
        missing_total=missing_total, missing_correct=absent_correct,
        missing_accuracy=absent_correct / missing_total,
        macro_cell_accuracy=macro,
        missing_tp=tp, missing_fp=fp, missing_fn=fn, missing_tn=tn,
        missing_precision=tp / (tp + fp) if tp + fp else 0.0,
        missing_recall=tp / (tp + fn) if tp + fn else 0.0,
        by_delay=group_scores("delay"),
        by_overwrites=group_scores("overwrite"),
        by_position=group_scores("position"),
        cells=cell_scores,
    )
