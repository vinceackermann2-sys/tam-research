from __future__ import annotations

"""#1421 CHM-v4 new independent-entity binding benchmark, stdlib / zero GPU.

Fresh namespace and RNG domain, independent entities as experimental units.
Implementation CI may generate ONLY TRAIN episodes. DEV/TEST future indices are
reserved in a *name-only* manifest; no heldout cases or scores materialized.
Original CHM-v3 scored groups, Stage-C and scientific seed are immutable.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Literal

Split = Literal["train", "development", "test"]
Family = Literal["direct", "overwrite", "two_hop", "no_match"]
Role = Literal["value", "relation", "record"]
SPLIT_COUNTS = {"train": 64, "development": 16, "test": 32}
FAMILIES = ("direct", "overwrite", "two_hop", "no_match")
POSITIVES = FAMILIES[:3]
CODES = tuple(range(18001, 18009))
STALE_CODE = 19001
MEMORY_LENGTH = 128
STUDY_NAMESPACE = "CHM_V4_NEW_INDEPENDENT_GROUPED_BINDING_1421"
OLD_100M_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
OLD_SCIENTIFIC_SEED_CONSUMED = 2013161
GPU_AUTHORIZED = False
NEW_SCIENTIFIC_ATTEMPT = False
SCORED_HELDOUT_AUTHORIZED = False
# Split-specific RNG keys do not reuse old CHM-v3 seed names or streams.
RNG_NAMESPACE = "CHM-V4-1421-ONLY-TRAIN-STAGE-A"

@dataclass(frozen=True)
class Record:
    position: int
    role: Role
    subject: str
    code: int | None = None
    document: str | None = None

@dataclass(frozen=True)
class Episode:
    split: Split
    family: Family
    entity: str
    entity_index: int
    variant: int
    query: str
    memory_length: int
    records: tuple[Record, ...]
    gold_answer: int | None

@dataclass(frozen=True)
class SealedInput:
    """Exactly model-visible text/candidates; no role, labels or pointers."""
    query: str
    memory_text: str
    answer_options: tuple[str, ...]


def _hash_int(*parts: object) -> int:
    data = "\x1f".join(map(str, (RNG_NAMESPACE, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def entity_name(split: Split, index: int) -> str:
    if split not in SPLIT_COUNTS or type(index) is not int or not (
        0 <= index < SPLIT_COUNTS[split]
    ):
        raise ValueError("invalid split or independent entity index")
    return f"V4-{split}-E{index:04d}"


def prospective_split_manifest() -> dict[str, object]:
    """NAME-ONLY future heldout reservation. Never creates scored episodes."""
    names = {
        split: tuple(entity_name(split, i) for i in range(count))
        for split, count in SPLIT_COUNTS.items()
    }
    if sum(len(set(n)) for n in names.values()) != sum(map(len, names.values())):
        raise RuntimeError("entity namespace overlap")
    return {
        "namespace": STUDY_NAMESPACE,
        "rng_namespace": RNG_NAMESPACE,
        "splits": {
            split: {
                "independent_entities": len(ids),
                "manifest_sha256": hashlib.sha256(
                    "\n".join(ids).encode("utf-8")
                ).hexdigest(),
                # Merely 2 examples reveal notation, not any real heldout case.
                "first_reserved_entity": ids[0],
                "last_reserved_entity": ids[-1],
            }
            for split, ids in names.items()
        },
        "heldout_examples_generated": False,
        "heldout_predictions_made": 0,
        "scored_test_authorized": False,
    }


def _base_layout(split: Split, family: Family, index: int) -> tuple[
    str, str, tuple[int, ...], tuple[str, ...], int, int
]:
    name = entity_name(split, index)
    if family not in FAMILIES:
        raise ValueError("unknown family")
    doc = f"V4-{split}-DOC{index:04d}"
    # No target or decoy identity depends on answer counterfactual variant.
    decoys = tuple(
        f"V4-{split}-DIST{index:04d}-{j:02d}"
        for j in range(8)
    )
    positions = tuple(
        random.Random(_hash_int("memory-positions", split, family, index))
        .sample(range(MEMORY_LENGTH), 11)
    )
    base_answer_index = _hash_int("base-code-rotation", split, family, index) % 8
    template = _hash_int("template", split, family, index) % 2
    return name, doc, positions, decoys, base_answer_index, template


def _query(entity: str, family: Family, template: int) -> str:
    if family == "two_hop":
        return (
            f"Which access code is stored by the record referenced from {entity}?"
            if template else
            f"What access code does the latest record linked from {entity} store?"
        )
    return (
        f"What is the newest access code for {entity}?"
        if template else f"What is the current access code for {entity}?"
    )


def generate_episode(
    split: Split, family: Family, index: int, variant: int,
) -> Episode:
    """Explicit generation can do TRAIN/DEV/TEST; CI calls TRAIN ONLY."""
    if type(variant) is not int or not 0 <= variant < 8:
        raise ValueError("counterfactual variant must be 0..7")
    name, doc, positions, decoys, base, template = _base_layout(
        split, family, index,
    )
    chosen = CODES[(base + variant) % 8]
    baseline = CODES[base]
    # Every positive variant contains each candidate code exactly once,
    # swapping target with a decoy without changing identity, positions or bag.
    values = [value for value in CODES if value != baseline]
    if chosen != baseline and family != "no_match":
        values[values.index(chosen)] = baseline
    records: list[Record] = []
    if family == "direct":
        records.append(Record(positions[0], "value", name, chosen))
        off = positions[1:8]
        for j, (p, code) in enumerate(zip(off, values)):
            records.append(Record(p, "value", decoys[j], code))
    elif family == "overwrite":
        old, current = sorted(positions[:2])
        records.extend((
            Record(old, "value", name, STALE_CODE),
            Record(current, "value", name, chosen),
        ))
        for j, (p, code) in enumerate(zip(positions[2:9], values)):
            records.append(Record(p, "value", decoys[j], code))
    elif family == "two_hop":
        old, current = sorted(positions[:2])
        # Current relation rebinding: old decoy record still exists, while
        # latest relation selects a different document holding the target.
        records.extend((
            Record(old, "relation", name, document=decoys[0]),
            Record(current, "relation", name, document=doc),
            Record(positions[2], "record", doc, chosen),
        ))
        for j, (p, code) in enumerate(zip(positions[3:10], values)):
            records.append(Record(p, "record", decoys[j], code))
    else:
        # Absent query subject, but every candidate code is present elsewhere.
        for j, (p, code) in enumerate(zip(positions[:8], CODES)):
            records.append(Record(p, "value", decoys[j], code))
    ep = Episode(
        split=split, family=family, entity=name, entity_index=index,
        variant=variant, query=_query(name, family, template),
        memory_length=MEMORY_LENGTH,
        records=tuple(sorted(records, key=lambda r: r.position)),
        gold_answer=None if family == "no_match" else chosen,
    )
    validate_episode(ep)
    return ep


def oracle_read(ep: Episode) -> tuple[int | None, int | None, int | None]:
    """Evaluator-only: (answer code, latest relation pos, value source pos)."""
    if ep.family == "two_hop":
        links = [r for r in ep.records
                 if r.role == "relation" and r.subject == ep.entity]
        if not links:
            return None, None, None
        link = max(links, key=lambda r: r.position)
        matches = [r for r in ep.records
                   if r.role == "record" and r.subject == link.document]
        if not matches:
            return None, link.position, None
        value = max(matches, key=lambda r: r.position)
        return value.code, link.position, value.position
    matches = [r for r in ep.records
               if r.role == "value" and r.subject == ep.entity]
    if not matches:
        return None, None, None
    last = max(matches, key=lambda r: r.position)
    return last.code, None, last.position


def _line(record: Record, template: int) -> str:
    prefix = f"[{record.position:04d}]"
    if record.role == "relation":
        if template:
            return f"{prefix} Record pointer for {record.subject} = {record.document}."
        return f"{prefix} {record.subject} references record {record.document}."
    if template:
        return f"{prefix} {record.subject} holds access code CODE-{record.code}."
    return f"{prefix} Code for {record.subject} is CODE-{record.code}."


def seal_input(ep: Episode) -> SealedInput:
    """Never expose metadata or oracle-selected source to model input."""
    # Public prose formatting follows name/layout only, not ep.gold_answer.
    template = _hash_int("template",ep.split,ep.family,ep.entity_index) % 2
    return SealedInput(
        query=ep.query,
        memory_text="\n".join(_line(r, template) for r in ep.records),
        answer_options=tuple(f"CODE-{code}" for code in CODES),
    )


def validate_episode(ep: Episode) -> None:
    if ep.family not in FAMILIES or ep.split not in SPLIT_COUNTS:
        raise ValueError("unknown family/split")
    if ep.entity != entity_name(ep.split, ep.entity_index):
        raise ValueError("mismatched entity")
    if type(ep.variant) is not int or ep.variant not in range(8):
        raise ValueError("invalid variant")
    if ep.memory_length != MEMORY_LENGTH or len(ep.records) not in (8, 9, 10):
        raise ValueError("invalid completed-memory geometry")
    positions = [r.position for r in ep.records]
    if (positions != sorted(positions) or len(positions) != len(set(positions))
        or any(type(i) is not int or i < 0 or i >= MEMORY_LENGTH for i in positions)):
        raise ValueError("bad source positions")
    for r in ep.records:
        if r.role == "relation":
            if r.document is None or r.code is not None:
                raise ValueError("broken relation")
        elif r.role in ("value", "record"):
            if r.document is not None or r.code not in (*CODES, STALE_CODE):
                raise ValueError("invalid code or record")
        else:
            raise ValueError("unknown record kind")
    bag=Counter(r.code for r in ep.records if r.code in CODES)
    if bag != Counter(CODES):
        raise ValueError("candidate multiset balance failed")
    if ep.family == "overwrite" and sum(r.code == STALE_CODE for r in ep.records) != 1:
        raise ValueError("older overwrite value absent")
    if ep.family == "two_hop":
        links=[r for r in ep.records if r.role=="relation" and r.subject==ep.entity]
        if len(links)!=2 or links[0].document==links[1].document:
            raise ValueError("latest relation rebind absent")
        if any(r.role != "record" for r in ep.records if r.code in CODES):
            raise ValueError("two-hop payload not stored in record")
    actual, _, source = oracle_read(ep)
    if actual != ep.gold_answer or (
        (ep.family == "no_match") != (source is None)
    ):
        raise ValueError("oracle/target/absence disagrees")
    if ep.family in POSITIVES and ep.gold_answer not in CODES:
        raise ValueError("positive answer missing")
    if ep.family == "no_match" and any(r.subject == ep.entity for r in ep.records):
        raise ValueError("negative target accidentally present")
    view=seal_input(ep)
    if any(f"CODE-{x}" in view.query for x in (*CODES,STALE_CODE)):
        raise ValueError("target leaked into query")


def train_counterfactual_group(family: Family, entity: int) -> tuple[Episode, ...]:
    """TRAIN-only callable used by tests. Never generate reserved test cases."""
    group=tuple(generate_episode("train",family,entity,i) for i in range(8))
    validate_train_group(group)
    return group


def validate_train_group(group: tuple[Episode, ...]) -> None:
    if len(group)!=8 or any(ep.split!="train" for ep in group):
        raise ValueError("exactly eight TRAIN-only counterfactual cases required")
    base=group[0]
    if {ep.variant for ep in group} != set(range(8)):
        raise ValueError("not all counterfactual rotations")
    layout=tuple((r.position,r.role,r.subject,r.document) for r in base.records)
    base_view=seal_input(base)
    codes_bag=Counter(r.code for r in base.records if r.code in CODES)
    positive=base.family in POSITIVES
    for ep in group:
        validate_episode(ep)
        if (ep.split,ep.family,ep.entity,ep.query,ep.entity_index)!=(
            base.split,base.family,base.entity,base.query,base.entity_index
        ):
            raise ValueError("counterfactual query/identity drift")
        if tuple((r.position,r.role,r.subject,r.document) for r in ep.records)!=layout:
            raise ValueError("counterfactual record layout drift")
        view=seal_input(ep)
        if view.query != base_view.query or view.answer_options != base_view.answer_options:
            raise ValueError("counterfactual model query/candidates changed")
        if Counter(r.code for r in ep.records if r.code in CODES)!=codes_bag:
            raise ValueError("candidate multiset changed")
        changes=[(a,b) for a,b in zip(base.records,ep.records) if a.code!=b.code]
        expected=0 if ep.variant==0 or not positive else 2
        if len(changes)!=expected or any(
            a.role not in ("value","record") or b.role!=a.role for a,b in changes
        ):
            raise ValueError("counterfactual change outside two code values")
    if positive:
        if {ep.gold_answer for ep in group}!=set(CODES):
            raise ValueError("positive group not 8-way balanced")
        if len({seal_input(ep).memory_text for ep in group})!=8:
            raise ValueError("counterfactual models see no changed bindings")
    elif len({seal_input(ep).memory_text for ep in group})!=1:
        raise ValueError("negative group changed across rotations")


def train_only_audit() -> dict[str, object]:
    """No future holdouts instantiated. Analytical blind baseline = 1/8."""
    num_groups=0
    group_code_bags=0
    layouts=0
    templates=Counter()
    for family in FAMILIES:
        for index in range(SPLIT_COUNTS["train"]):
            group=train_counterfactual_group(family,index)
            num_groups+=1
            templates[_hash_int("template","train",family,index)%2]+=1
            group_code_bags+=int(all(Counter(
                r.code for r in ep.records if r.code in CODES
            )==Counter(CODES) for ep in group))
            layouts+=len({tuple(r.position for r in ep.records) for ep in group})==1
    if num_groups!=256 or group_code_bags!=num_groups or layouts!=num_groups:
        raise RuntimeError("TRAIN-only balanced group audit failed")
    manifest=prospective_split_manifest()
    return {
        "classification": STUDY_NAMESPACE+"_TRAIN_ONLY_STATIC",
        "independent_training_entities": SPLIT_COUNTS["train"],
        "train_groups": num_groups,
        "train_episodes": num_groups*8,
        "train_group_bag_invariants": group_code_bags,
        "train_group_layout_invariants": layouts,
        "template_counts": dict(templates),
        "blind_positive_ceiling_per_eight_variants": "1/8",
        "manifest": manifest,
        "heldout_episodes_constructed": 0,
        "model_training_steps": 0,
        "trained_predictions": 0,
        "gpu_used": False,
        "new_scientific_attempt": False,
        "historical_stop": OLD_100M_STOP,
        "historical_seed_consumed": OLD_SCIENTIFIC_SEED_CONSUMED,
    }
