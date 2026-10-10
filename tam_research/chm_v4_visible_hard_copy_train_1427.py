from __future__ import annotations

"""#1427 NON-LEARNED, TRAIN-only visible-text address -> payload copy reference.

This is NOT a model, not a GPU experiment and not heldout performance.
Model-equivalent inputs are limited to SealedInput(query,memory_text,options).
No structured Episode or hidden role/gold labels enter read_visible_text().
"""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Literal

from .chm_v4_independent_binding_benchmark_1421 import (
    CODES, FAMILIES, POSITIVES, SealedInput, SPLIT_COUNTS, STALE_CODE,
    OLD_100M_STOP, OLD_SCIENTIFIC_SEED_CONSUMED,
    oracle_read, seal_input, train_counterfactual_group,
)

CLASSIFICATION = "CHM_V4_1427_VISIBLE_HARD_COPY_TRAIN_ONLY_RULE"
SCIENTIFIC_RUN_AUTHORIZED = False
GPU_AUTHORIZED = False
HELDOUT_SCORING_AUTHORIZED = False
SOURCE_LINE = re.compile(r"^\[(\d{4})\] (.+)$")
NAME = r"(V4-(?:train|development|test)-[A-Za-z0-9-]+)"
QUERY_NAME = re.compile(r"V4-(?:train|development|test)-E\d{4}\b")
RELATION_GRAMMARS = (
    re.compile(rf"^{NAME} references record {NAME}\.$"),
    re.compile(rf"^Record pointer for {NAME} = {NAME}\.$"),
)
VALUE_GRAMMARS = (
    re.compile(rf"^Code for {NAME} is CODE-(\d+)\.$"),
    re.compile(rf"^{NAME} holds access code CODE-(\d+)\.$"),
)


@dataclass(frozen=True)
class VisibleRecord:
    position: int
    subject: str
    code: int | None
    linked_document: str | None


@dataclass(frozen=True)
class VisibleRead:
    answer_id: int | None
    first_physical_source: int | None
    payload_physical_source: int | None
    action: Literal["copy", "abstain"]
    routing: Literal["latest_direct", "latest_link", "absent"]


def _extract_query_subject(view: SealedInput) -> str:
    subjects = set(QUERY_NAME.findall(view.query))
    if len(subjects) != 1:
        raise ValueError("visible query must have exactly one V4 subject")
    return next(iter(subjects))


def _parse_completed_memory(view: SealedInput) -> tuple[VisibleRecord, ...]:
    if not isinstance(view, SealedInput):
        raise TypeError("only sealed visible query/memory/options may be read")
    expected_options = tuple(f"CODE-{x}" for x in CODES)
    if view.answer_options != expected_options:
        raise ValueError("visible candidate options not frozen eight-way set")
    if not view.query or not view.memory_text or len(view.memory_text) > 12000:
        raise ValueError("missing/unbounded visible query or completed memory")
    rows: list[VisibleRecord] = []
    for line in view.memory_text.splitlines():
        anchor = SOURCE_LINE.fullmatch(line)
        if anchor is None:
            raise ValueError("malformed completed memory source-position line")
        position = int(anchor.group(1))
        text = anchor.group(2)
        found = None
        for pattern in RELATION_GRAMMARS:
            result = pattern.fullmatch(text)
            if result:
                found = VisibleRecord(
                    position, result.group(1), None, result.group(2),
                )
                break
        if found is None:
            for pattern in VALUE_GRAMMARS:
                result = pattern.fullmatch(text)
                if result:
                    found = VisibleRecord(
                        position, result.group(1), int(result.group(2)), None,
                    )
                    break
        if found is None:
            raise ValueError("unsupported or ambiguous visible memory grammar")
        rows.append(found)

    positions = [r.position for r in rows]
    if (not rows or positions != sorted(positions)
        or len(positions) != len(set(positions))
        or any(p < 0 or p >= 128 for p in positions)):
        raise ValueError("duplicate, unsorted or invalid memory source address")
    options = Counter(r.code for r in rows if r.code in CODES)
    if options != Counter(CODES):
        raise ValueError("incomplete/duplicated visible candidate-code bag")
    if any(r.code not in (*CODES, STALE_CODE, None) for r in rows):
        raise ValueError("out-of-vocabulary code in completed memory")
    if sum(r.code == STALE_CODE for r in rows) > 1:
        raise ValueError("multiple unsupported stale record codes")
    return tuple(rows)


def read_visible_text(view: SealedInput) -> VisibleRead:
    """Pure deterministic text algorithm, not learned reasoning or inference.

    No ep.family or oracle roles needed: visible link to subject indicates
    two-hop query, else newest visible direct subject code, else abstain.
    """
    if not isinstance(view, SealedInput):
        raise TypeError("only sealed model-visible input is permitted")
    queried = _extract_query_subject(view)
    memory = _parse_completed_memory(view)
    links = sorted(
        (r for r in memory
         if r.subject == queried and r.linked_document is not None),
        key=lambda r: r.position,
    )
    if links:
        current = links[-1]
        candidates = sorted(
            (r for r in memory
             if r.subject == current.linked_document and r.code is not None),
            key=lambda r: r.position,
        )
        if not candidates:
            raise ValueError("active visible relation has no readable document payload")
        payload = candidates[-1]
        if payload.code not in CODES:
            raise ValueError("active linked document has invalid candidate code")
        return VisibleRead(
            payload.code, current.position, payload.position, "copy",
            "latest_link",
        )

    direct = sorted(
        (r for r in memory
         if r.subject == queried and r.code is not None),
        key=lambda r: r.position,
    )
    if not direct:
        return VisibleRead(None, None, None, "abstain", "absent")
    payload = direct[-1]
    if payload.code not in CODES:
        raise ValueError("latest matched direct record contains noncandidate code")
    return VisibleRead(
        payload.code, payload.position, payload.position, "copy",
        "latest_direct",
    )


def train_only_visible_reference_audit() -> dict[str, object]:
    """Compare after visible reader returns; no heldout cases or trained model."""
    correct: dict[str, int] = {family: 0 for family in FAMILIES}
    first_hits: dict[str, int] = {family: 0 for family in FAMILIES}
    value_hits: dict[str, int] = {family: 0 for family in FAMILIES}
    name_bag = Counter()
    total = 0
    for family in FAMILIES:
        for idx in range(SPLIT_COUNTS["train"]):
            group = train_counterfactual_group(family, idx)
            for ep in group:
                if ep.split != "train":
                    raise RuntimeError("TRAIN-only visible hard-read leaked heldout")
                # NO hidden gold/oracle role or address is passed to the reader.
                visible = seal_input(ep)
                predicted = read_visible_text(visible)
                # EVALUATOR ONLY after prediction; do not feed back into reader.
                gold, relation_pos, value_pos = oracle_read(ep)
                if gold != ep.gold_answer:
                    raise RuntimeError("benchmark oracle/gold mismatch")
                expected_first = (
                    relation_pos if relation_pos is not None else value_pos
                )
                correct[family] += int(predicted.answer_id == gold)
                first_hits[family] += int(
                    predicted.first_physical_source == expected_first
                )
                value_hits[family] += int(
                    predicted.payload_physical_source == value_pos
                )
                name_bag[predicted.routing] += 1
                total += 1

    expected = SPLIT_COUNTS["train"] * 8
    if (total != 2048 or
        any(correct[f] != expected or first_hits[f] != expected
            or value_hits[f] != expected for f in FAMILIES)):
        raise RuntimeError("TRAIN visible hard-copy does not match the evaluator")
    if name_bag != Counter({
        "latest_direct": 2*expected,
        "latest_link": expected,
        "absent": expected,
    }):
        raise RuntimeError("visible rule routing differs from intended train task")
    return {
        "classification": CLASSIFICATION,
        "train_entities_per_family": SPLIT_COUNTS["train"],
        "train_cases": total,
        "train_oracle_answer_correct_by_family": correct,
        "train_first_source_correct_by_family": first_hits,
        "train_payload_source_correct_by_family": value_hits,
        "visible_rule_routing_counts": dict(name_bag),
        "development_test_cases_constructed": False,
        "trained_model_forward_calls": 0,
        "optimizer_steps": 0,
        "gpu_used": False,
        "new_scientific_attempt": False,
        "historical_scientific_stop": OLD_100M_STOP,
        "historical_seed_consumed": OLD_SCIENTIFIC_SEED_CONSUMED,
        "interpretation": (
            "A purpose-built deterministic text parser recovers the synthetic "
            "train answer; this is a benchmark parseability ceiling, NOT learned "
            "retrieval, model generalization, CPW superiority or breakthrough."
        ),
    }


__all__ = (
    "VisibleRecord", "VisibleRead", "read_visible_text",
    "train_only_visible_reference_audit", "CLASSIFICATION",
)
