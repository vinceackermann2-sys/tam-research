from __future__ import annotations

"""#1358 counterfactual memory benchmark: data and CPU-only oracle.

This generator is intentionally NEW: it does not import the historical GPT-2
aligned-v4 generator, reuse any old science seed, train a model or run inference.
Structured record types are benchmark/oracle metadata, NOT learned language
understanding. A future model must receive a separately specified input encoding.
"""

from dataclasses import dataclass
from collections.abc import Sequence
import hashlib
import random
from typing import Literal

SPLITS = {"train": 8, "development": 4, "test": 4}
FAMILIES = ("direct", "overwrite", "two_hop", "no_match")
POSITIVE_FAMILIES = FAMILIES[:3]
ANSWER_IDS = (18001, 18002, 18003, 18004, 18005, 18006, 18007, 18008)
MEMORY_LENGTHS = (128, 256, 512, 1024)
HISTORICAL_STAGE_C_SCIENTIFIC_SEED_CONSUMED = 2_013_161
MODAL_GPU_AUTHORIZED = False
STAGE_C_SCIENTIFIC_AUTHORITY = False


@dataclass(frozen=True)
class MemoryRecord:
    position: int  # exact position in completed prior memory; never query/current chunk
    role: Literal["value", "relation", "record"]
    subject: str
    answer_id: int | None = None
    document: str | None = None


@dataclass(frozen=True)
class CounterfactualEpisode:
    split: str
    family: str
    entity: str
    counterfactual_index: int
    query: str
    memory_length: int
    records: tuple[MemoryRecord, ...]
    correct_answer_id: int | None


def _stable_seed(*parts: object) -> int:
    # Independent from all frozen 2026 scientific and probe RNG streams.
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _query(entity: str, family: str) -> str:
    if family == "two_hop":
        return f"Which access code does the record for {entity} store?"
    return f"What is the current access code for {entity}?"


def oracle_read(ep: CounterfactualEpisode) -> tuple[int | None, int | None, int | None]:
    """Read-only reference: (answer, relation position, answer position)."""
    records = sorted(ep.records, key=lambda r: r.position)
    if ep.family == "two_hop":
        rels = [r for r in records if r.role == "relation" and r.subject == ep.entity]
        if not rels:
            return None, None, None
        rel = rels[-1]
        values = [r for r in records if r.role == "record" and r.subject == rel.document]
        if not values:
            return None, rel.position, None
        selected = values[-1]
        return selected.answer_id, rel.position, selected.position
    values = [r for r in records if r.role == "value" and r.subject == ep.entity]
    if not values:
        return None, None, None
    selected = values[-1]  # newest authoritative overwrite only
    return selected.answer_id, None, selected.position


def validate_episode(ep: CounterfactualEpisode) -> None:
    if ep.split not in SPLITS or ep.family not in FAMILIES:
        raise ValueError("unknown split/family")
    if ep.memory_length not in MEMORY_LENGTHS:
        raise ValueError("invalid memory length")
    if not 0 <= ep.counterfactual_index < 8:
        raise ValueError("counterfactual index must be 0..7")
    if ep.query != _query(ep.entity, ep.family):
        raise ValueError("query changed or contains target leakage")
    if len(ep.records) == 0 or len(ep.records) > 12:
        raise ValueError("invalid record count")
    positions = [r.position for r in ep.records]
    if len(positions) != len(set(positions)) or any(
        type(x) is not int or not 0 <= x < ep.memory_length for x in positions
    ):
        raise ValueError("out-of-bounds or duplicate completed memory position")
    for r in ep.records:
        if r.role not in ("value", "relation", "record"):
            raise ValueError("unknown record role")
        if r.role == "relation":
            if r.answer_id is not None or not r.document:
                raise ValueError("invalid relation payload")
        elif r.document is not None or r.answer_id not in ANSWER_IDS:
            raise ValueError("invalid value/record payload")
    if ep.correct_answer_id is not None and ep.correct_answer_id not in ANSWER_IDS:
        raise ValueError("answer outside eight candidates")
    answer, relation_pos, value_pos = oracle_read(ep)
    if answer != ep.correct_answer_id:
        raise ValueError("oracle value differs from archived label")
    if ep.family == "no_match" and answer is not None:
        raise ValueError("negative query must have no matching memory fact")
    if ep.family != "no_match" and (answer is None or value_pos is None):
        raise ValueError("positive answer must be reachable from completed memory")
    if ep.family == "two_hop" and relation_pos is None:
        raise ValueError("two-hop relation missing")
    if ep.family != "two_hop" and relation_pos is not None:
        raise ValueError("non-two-hop relation leaked into answer")


def generate_episode(
    split: Literal["train", "development", "test"],
    family: Literal["direct", "overwrite", "two_hop", "no_match"],
    entity_index: int,
    counterfactual_index: int,
    *,
    memory_length: int = 128,
) -> CounterfactualEpisode:
    if split not in SPLITS or family not in FAMILIES:
        raise ValueError("unrecognized counterfactual split/family")
    if (type(entity_index) is not int or not 0 <= entity_index < SPLITS[split]
            or type(counterfactual_index) is not int
            or not 0 <= counterfactual_index < 8):
        raise ValueError("entity index or answer rotation outside frozen bounds")
    if memory_length not in MEMORY_LENGTHS:
        raise ValueError("unsupported memory length")

    entity = f"{split}-E{entity_index:03d}"  # disjoint identity namespace
    # Every entity and family cycles through ALL 8 values across its 8 episodes.
    family_offset = FAMILIES.index(family)
    answer_id = ANSWER_IDS[(counterfactual_index + entity_index * 3 + family_offset) % 8]
    stale_id = ANSWER_IDS[(ANSWER_IDS.index(answer_id) + 1) % 8]
    rng = random.Random(_stable_seed("CHM_V3_1358_NEW_EPISODES", split, family,
                                    entity_index, counterfactual_index, memory_length))
    positions = rng.sample(range(memory_length), 9)
    recs: list[MemoryRecord] = []
    if family == "direct":
        recs.append(MemoryRecord(positions[0], "value", entity, answer_id=answer_id))
    elif family == "overwrite":
        old, new = sorted(positions[:2])
        recs.extend((MemoryRecord(old, "value", entity, answer_id=stale_id),
                     MemoryRecord(new, "value", entity, answer_id=answer_id)))
    elif family == "two_hop":
        document = f"{split}-D{entity_index:03d}-{counterfactual_index:02d}"
        other_doc = document + "-decoy"
        recs.extend((
            MemoryRecord(positions[0], "relation", entity, document=document),
            MemoryRecord(positions[1], "record", document, answer_id=answer_id),
            MemoryRecord(positions[2], "record", other_doc, answer_id=stale_id),
        ))

    # Same distractor count and value vocabulary, independent of target presence.
    # Any matching answer can only be recovered via subject/document binding.
    for j in range(3):
        decoy_id = ANSWER_IDS[(j + counterfactual_index + 2) % 8]
        recs.append(MemoryRecord(
            positions[6 + j], "value", f"{split}-unrelated-{entity_index:03d}-{j}",
            answer_id=decoy_id,
        ))
    ep = CounterfactualEpisode(
        split=split, family=family, entity=entity,
        counterfactual_index=counterfactual_index,
        query=_query(entity, family), memory_length=memory_length,
        records=tuple(sorted(recs, key=lambda r: r.position)),
        correct_answer_id=None if family == "no_match" else answer_id,
    )
    validate_episode(ep)
    return ep


def generate_suite(
    split: Literal["train", "development", "test"], *,
    memory_length: int = 128,
) -> tuple[CounterfactualEpisode, ...]:
    if split not in SPLITS or memory_length not in MEMORY_LENGTHS:
        raise ValueError("unsupported split or memory size")
    result = tuple(
        generate_episode(split, family, entity_index, counterfactual_index,
                         memory_length=memory_length)
        for family in FAMILIES
        for entity_index in range(SPLITS[split])
        for counterfactual_index in range(8)
    )
    for family in POSITIVE_FAMILIES:
        for entity_index in range(SPLITS[split]):
            group = [
                x for x in result
                if x.family == family and x.entity == f"{split}-E{entity_index:03d}"
            ]
            if len(group) != 8 or len({x.query for x in group}) != 1 or {
                x.correct_answer_id for x in group
            } != set(ANSWER_IDS):
                raise RuntimeError("counterfactual binding/balance invariant failed")
    return result


def query_only_baseline(ep: CounterfactualEpisode) -> int:
    """Deterministic memory-free policy, fixed for an identical query."""
    return ANSWER_IDS[_stable_seed("1358_MEMORY_FREE", ep.query) % 8]


def baseline_report(split: Literal["train", "development", "test"]) -> dict[str, object]:
    suite = generate_suite(split)
    by_family: dict[str, dict[str, float | int]] = {}
    for family in FAMILIES:
        rows = [ep for ep in suite if ep.family == family]
        hits = sum(query_only_baseline(ep) == ep.correct_answer_id for ep in rows)
        by_family[family] = {
            "cases": len(rows), "query_only_correct": hits,
            "query_only_accuracy": hits / len(rows),
            "always_abstain_correct": sum(ep.correct_answer_id is None for ep in rows),
        }
    return {
        "classification": "CHM_V3_1358_NEW_COUNTERFACTUAL_CPU_NEGATIVE_CONTROL",
        "split": split, "cases": len(suite), "families": by_family,
        "gpu_used": False, "trained_checkpoint_replayed": False,
        "historical_stage_c_stop_unchanged": True,
    }


__all__ = (
    "MemoryRecord", "CounterfactualEpisode", "FAMILIES", "POSITIVE_FAMILIES",
    "ANSWER_IDS", "MEMORY_LENGTHS", "SPLITS", "generate_episode",
    "generate_suite", "oracle_read", "validate_episode",
    "query_only_baseline", "baseline_report",
)
