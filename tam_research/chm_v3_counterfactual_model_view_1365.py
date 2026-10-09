from __future__ import annotations

"""#1365 zero-GPU matched counterfactuals and sealed model-facing inputs.

V2 is NEW and additive. Its eight positive counterfactual variants keep
query, address layout, entity, document names and all distracting facts fixed.
Only the authoritative target answer in completed memory changes. The model
view never includes a gold pointer, gold label or counterfactual index.
"""

from dataclasses import dataclass, fields
import hashlib
import random
import re
from typing import Literal

from .chm_v3_counterfactual_memory_suite_1358 import (
    ANSWER_IDS, MEMORY_LENGTHS, SPLITS,
)

FAMILIES = ("direct", "overwrite", "two_hop", "no_match")
POSITIVE_FAMILIES = FAMILIES[:3]
STALE_CODE = 19001
DECOY_CODE = 19002
ORIGINAL_SCIENTIFIC_SEED_CONSUMED = 2_013_161
GPU_OR_PAID_TRAINING_AUTHORIZED = False
SCIENTIFIC_QUALITY_CLAIM_AUTHORIZED = False


@dataclass(frozen=True)
class Fact:
    position: int
    kind: Literal["value", "relation", "record"]
    subject: str
    code: int | None = None
    document: str | None = None


@dataclass(frozen=True)
class PairedEpisode:
    split: str
    family: str
    entity: str
    variant: int
    memory_length: int
    query: str
    facts: tuple[Fact, ...]
    gold_answer: int | None


@dataclass(frozen=True)
class SealedModelView:
    """ALL fields a future model may receive. No labels/metadata/oracle roles."""
    query: str
    memory_text: str
    answer_options: tuple[str, ...]


def _stable_seed(*parts: object) -> int:
    raw = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _query(entity: str, family: str) -> str:
    if family == "two_hop":
        return f"Which access code does the record for {entity} store?"
    return f"What is the current access code for {entity}?"


def _authoritative_position(ep: PairedEpisode) -> int | None:
    """Select source position using fact semantics, NOT any oracle label."""
    if ep.family == "two_hop":
        links = [f for f in ep.facts if f.kind == "relation" and f.subject == ep.entity]
        if not links:
            return None
        last_link = max(links, key=lambda f: f.position)
        values = [f for f in ep.facts
                  if f.kind == "record" and f.subject == last_link.document]
        return max(values, key=lambda f: f.position).position if values else None
    values = [f for f in ep.facts if f.kind == "value" and f.subject == ep.entity]
    return max(values, key=lambda f: f.position).position if values else None


def oracle_read(ep: PairedEpisode) -> tuple[int | None, int | None, int | None]:
    """Ground truth is used only by validator/evaluator, never in model view."""
    rel_position = None
    if ep.family == "two_hop":
        links = [f for f in ep.facts if f.kind == "relation" and f.subject == ep.entity]
        if links:
            rel_position = max(links, key=lambda f: f.position).position
    pos = _authoritative_position(ep)
    code = next((f.code for f in ep.facts if f.position == pos), None)
    return code, rel_position, pos


def validate_episode(ep: PairedEpisode) -> None:
    if ep.split not in SPLITS or ep.family not in FAMILIES:
        raise ValueError("invalid split or family")
    if ep.memory_length not in MEMORY_LENGTHS:
        raise ValueError("invalid completed-memory length")
    if type(ep.variant) is not int or ep.variant not in range(8):
        raise ValueError("invalid counterfactual variant")
    if ep.query != _query(ep.entity, ep.family) or any(
        f"CODE-{v}" in ep.query for v in (*ANSWER_IDS, STALE_CODE, DECOY_CODE)
    ):
        raise ValueError("target answer or metadata visible in query")
    if not ep.facts or len(ep.facts) > 12:
        raise ValueError("invalid fact count")
    seen: set[int] = set()
    for fact in ep.facts:
        if type(fact.position) is not int or not 0 <= fact.position < ep.memory_length:
            raise ValueError("fact outside completed memory")
        if fact.position in seen:
            raise ValueError("duplicate memory positions")
        seen.add(fact.position)
        if fact.kind == "relation":
            if fact.document is None or fact.code is not None:
                raise ValueError("malformed relation fact")
        elif fact.kind in ("value", "record"):
            if fact.document is not None or fact.code not in (
                *ANSWER_IDS, STALE_CODE, DECOY_CODE,
            ):
                raise ValueError("malformed value fact")
        else:
            raise ValueError("invalid fact kind")
    answer, relation_pos, answer_pos = oracle_read(ep)
    if answer != ep.gold_answer:
        raise ValueError("oracle answer/ground truth mismatch")
    if ep.family == "no_match" and answer is not None:
        raise ValueError("no-match query was accidentally answerable")
    if ep.family in POSITIVE_FAMILIES and (
        answer not in ANSWER_IDS or answer_pos is None
    ):
        raise ValueError("positive answer not in completed memory")
    if ep.family == "two_hop" and relation_pos is None:
        raise ValueError("two-hop relation inaccessible")


def generate_paired_episode(
    split: Literal["train", "development", "test"],
    family: Literal["direct", "overwrite", "two_hop", "no_match"],
    entity_index: int,
    variant: int,
    *, memory_length: int = 128,
) -> PairedEpisode:
    if split not in SPLITS or family not in FAMILIES:
        raise ValueError("invalid split/family")
    if type(entity_index) is not int or not 0 <= entity_index < SPLITS[split]:
        raise ValueError("entity index not in split")
    if type(variant) is not int or not 0 <= variant < 8:
        raise ValueError("variant outside 0..7")
    if memory_length not in MEMORY_LENGTHS:
        raise ValueError("invalid memory length")
    entity = f"V2-{split}-E{entity_index:03d}"
    doc = f"V2-{split}-D{entity_index:03d}"
    decoy_doc = f"V2-{split}-X{entity_index:03d}"
    answer = ANSWER_IDS[(variant + entity_index * 3 + FAMILIES.index(family)) % 8]

    # CRITICAL: Positions and irrelevant facts must not depend on 'variant'.
    rng = random.Random(_stable_seed(
        "CHM_V3_1365_DISTINCT_LAYOUT", split, family, entity_index, memory_length,
    ))
    positions = rng.sample(range(memory_length), 9)
    facts: list[Fact] = []
    if family == "direct":
        facts.append(Fact(positions[0], "value", entity, code=answer))
    elif family == "overwrite":
        old, new = sorted(positions[:2])
        facts.extend((
            Fact(old, "value", entity, code=STALE_CODE),
            Fact(new, "value", entity, code=answer),
        ))
    elif family == "two_hop":
        facts.extend((
            Fact(positions[0], "relation", entity, document=doc),
            Fact(positions[1], "record", doc, code=answer),
            Fact(positions[2], "record", decoy_doc, code=DECOY_CODE),
        ))

    for j in range(3):
        # Distractor text, answer IDs and positions are constant across all 8.
        facts.append(Fact(
            positions[6 + j], "value",
            f"V2-{split}-unrelated-{entity_index:03d}-{j}",
            code=ANSWER_IDS[(j + entity_index + 2) % 8],
        ))
    ep = PairedEpisode(
        split=split, family=family, entity=entity, variant=variant,
        memory_length=memory_length, query=_query(entity, family),
        facts=tuple(sorted(facts, key=lambda f: f.position)),
        gold_answer=None if family == "no_match" else answer,
    )
    validate_episode(ep)
    return ep


def _render_fact(fact: Fact) -> str:
    if fact.kind == "relation":
        return f"[{fact.position:04d}] Relation: {fact.subject} uses record {fact.document}."
    if fact.kind == "record":
        return f"[{fact.position:04d}] Record {fact.subject} stores CODE-{fact.code}."
    return f"[{fact.position:04d}] Access code for {fact.subject} is CODE-{fact.code}."


def seal_model_view(ep: PairedEpisode) -> SealedModelView:
    """Drop labels, role IDs, oracle pointers, split/variant and RNG metadata."""
    validate_episode(ep)
    return SealedModelView(
        query=ep.query,
        memory_text="\n".join(_render_fact(f) for f in sorted(ep.facts, key=lambda r: r.position)),
        answer_options=tuple(f"CODE-{x}" for x in ANSWER_IDS),
    )


def redact_memory_values(view: SealedModelView) -> SealedModelView:
    """Context-blind control retains query, ordering, location and metadata."""
    return SealedModelView(
        query=view.query,
        memory_text=re.sub(r"CODE-\d+", "CODE-REDACTED", view.memory_text),
        answer_options=view.answer_options,
    )


def validate_matched_group(episodes: tuple[PairedEpisode, ...]) -> None:
    if len(episodes) != 8:
        raise ValueError("eight matched counterfactuals required")
    for ep in episodes:
        validate_episode(ep)
    base = episodes[0]
    if {ep.variant for ep in episodes} != set(range(8)):
        raise ValueError("missing or repeated variant")
    if base.family in POSITIVE_FAMILIES and {
        ep.gold_answer for ep in episodes
    } != set(ANSWER_IDS):
        raise ValueError("counterfactual answers not balanced")
    if base.family == "no_match" and any(ep.gold_answer is not None for ep in episodes):
        raise ValueError("no-match label corrupted")
    target_pos = _authoritative_position(base)
    base_view = seal_model_view(base)
    base_masked = redact_memory_values(base_view)
    for ep in episodes:
        if (ep.split, ep.family, ep.entity, ep.memory_length, ep.query) != (
            base.split, base.family, base.entity, base.memory_length, base.query
        ):
            raise ValueError("counterfactual group identity changed")
        if _authoritative_position(ep) != target_pos:
            raise ValueError("authoritative memory position changed")
        if [f.position for f in ep.facts] != [f.position for f in base.facts]:
            raise ValueError("memory address layout changed")
        for record, old in zip(ep.facts, base.facts):
            if (record.position, record.kind, record.subject, record.document) != (
                old.position, old.kind, old.subject, old.document
            ):
                raise ValueError("memory keys/relations changed")
            if record.position != target_pos and record.code != old.code:
                raise ValueError("non-target memory fact changed")
        view = seal_model_view(ep)
        if view.query != base_view.query or view.answer_options != base_view.answer_options:
            raise ValueError("model view query/options changed")
        if redact_memory_values(view) != base_masked:
            raise ValueError("context-blind metadata/layout leaked variant")
        if base.family in POSITIVE_FAMILIES:
            assert target_pos is not None
            token = f"CODE-{ep.gold_answer}"
            target_line = next(line for line in view.memory_text.splitlines()
                               if line.startswith(f"[{target_pos:04d}]"))
            if token not in target_line:
                raise ValueError("authoritative target missing from model view")
    if base.family in POSITIVE_FAMILIES and len({
        seal_model_view(ep).memory_text for ep in episodes
    }) != 8:
        raise ValueError("model views did not respond to changed fact")


def paired_group(
    split: Literal["train", "development", "test"], family: str,
    entity_index: int, *, memory_length: int = 128,
) -> tuple[PairedEpisode, ...]:
    result = tuple(generate_paired_episode(
        split, family, entity_index, i, memory_length=memory_length,
    ) for i in range(8))
    validate_matched_group(result)
    return result


def _choice_from_view(view: SealedModelView) -> int:
    raw = (view.query + "\x1f" + view.memory_text).encode("utf-8")
    return ANSWER_IDS[int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 8]


def negative_control_report(split: Literal["train", "development", "test"]) -> dict[str, object]:
    if split not in SPLITS:
        raise ValueError("invalid split")
    stats: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        n, q_hits, masked_hits, abstain_hits = 0, 0, 0, 0
        for entity_index in range(SPLITS[split]):
            group = paired_group(split, family, entity_index)
            for ep in group:
                view = seal_model_view(ep)
                qonly = SealedModelView(view.query, "", view.answer_options)
                masked = redact_memory_values(view)
                n += 1
                q_hits += int(_choice_from_view(qonly) == ep.gold_answer)
                masked_hits += int(_choice_from_view(masked) == ep.gold_answer)
                abstain_hits += int(ep.gold_answer is None)
        stats[family] = {
            "cases": n, "query_only_accuracy": q_hits / n,
            "layout_only_accuracy": masked_hits / n,
            "always_abstain_correct": abstain_hits,
        }
    return {
        "classification": "CHM_V3_1365_MATCHED_COUNTERFACTUAL_CPU_NEGATIVE_CONTROLS",
        "split": split, "families": stats, "no_gpu": True,
        "old_scientific_stop_unchanged": True, "model_trained": False,
    }


__all__ = (
    "Fact", "PairedEpisode", "SealedModelView", "FAMILIES", "POSITIVE_FAMILIES",
    "STALE_CODE", "DECOY_CODE", "generate_paired_episode", "paired_group",
    "oracle_read", "seal_model_view", "redact_memory_values",
    "validate_episode", "validate_matched_group", "negative_control_report",
)
