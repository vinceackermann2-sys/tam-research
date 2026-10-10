from __future__ import annotations

"""#1421 CHM-v4: independent-entity, matched-value-bag benchmark, DATA ONLY.

Absolutely no scoring, training, checkpoints, Modal, CUDA, legacy v3 probes,
old scientific seeds, dev/test generation in CI, or model in this module.
A future separate one-shot scorer must freeze the test manifest independently.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Literal

from .chm_v3_counterfactual_model_view_1365 import SealedModelView

Split = Literal["train", "development", "test"]
Family = Literal["direct", "overwrite", "two_hop", "no_match"]
SPLIT_ENTITIES = {"train": 64, "development": 16, "test": 32}
FAMILIES = ("direct", "overwrite", "two_hop", "no_match")
POSITIVES = FAMILIES[:3]
ANSWER_IDS = (18001, 18002, 18003, 18004, 18005, 18006, 18007, 18008)
MEMORY_LENGTH = 256
STALE_NONCANDIDATE = 19999
V4_NAMESPACE = "CHM_V4_1421_NEW_INDEPENDENT_ENTITIES_OCT2026"
RNG_SALT = "CHM_V4_1421_SPLIT_INDEPENDENT_LAYOUT_20261010_V1"
OLD_100M_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
OLD_100M_SCIENTIFIC_SEED_CONSUMED = 2013161
GPU_AUTHORIZED = False
SCORED_HELDOUT_AUTHORIZED = False

QUERY_TEMPLATES = {
    "ordinary": (
        "What is the current access code for {entity}?",
        "Find the latest valid access code for {entity}.",
        "Which access code currently belongs to {entity}?",
    ),
    "two_hop": (
        "What code is stored in the active record linked to {entity}?",
        "Which code belongs to the latest record reference for {entity}?",
        "Find the access code in the current record for {entity}.",
    ),
}
RENDER_TEMPLATES = (
    {
        "value": "[{position:04d}] Access code for {subject} is CODE-{code}.",
        "record": "[{position:04d}] Record {subject} stores CODE-{code}.",
        "relation": "[{position:04d}] Relation: {subject} uses record {document}.",
    },
    {
        "value": "[{position:04d}] Current entry {subject} has access CODE-{code}.",
        "record": "[{position:04d}] In record {subject}, access is CODE-{code}.",
        "relation": "[{position:04d}] Link {subject} points to record {document}.",
    },
    {
        "value": "[{position:04d}] The entry for {subject} lists CODE-{code}.",
        "record": "[{position:04d}] CODE-{code} is listed for record {subject}.",
        "relation": "[{position:04d}] Record association: {subject} -> {document}.",
    },
)


@dataclass(frozen=True)
class V4Fact:
    position: int
    kind: Literal["value", "relation", "record"]
    subject: str
    code: int | None = None
    document: str | None = None


@dataclass(frozen=True)
class V4Episode:
    split: Split
    family: Family
    entity_index: int
    entity: str
    variant: int
    query: str
    memory_length: int
    facts: tuple[V4Fact, ...]
    gold_answer: int | None
    render_style: int


def _stable_int(*parts: object) -> int:
    raw = "\x1f".join(map(str, (RNG_SALT, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _spec(split: Split, family: Family, entity_index: int) -> tuple[str, str, int, int]:
    if split not in SPLIT_ENTITIES or family not in FAMILIES:
        raise ValueError("v4 unknown split/family")
    if type(entity_index) is not int or not 0 <= entity_index < SPLIT_ENTITIES[split]:
        raise ValueError("v4 entity outside frozen split")
    entity = f"V4-{split}-E{entity_index:04d}"
    doc = f"V4-{split}-D{entity_index:04d}-{family}"
    layout_seed = _stable_int("new-layout", split, family, entity_index)
    style = _stable_int("new-render", split, family, entity_index) % len(RENDER_TEMPLATES)
    return entity, doc, layout_seed, style


def oracle_reference(ep: V4Episode) -> tuple[int | None, int | None, int | None]:
    """Evaluator-held reference: (code, first physical position, code position).

    The oracle reads only completed facts, latest relation/update by POSITION,
    not the gold label. Never give it or its role metadata to model.forward().
    """
    values = sorted(
        (f for f in ep.facts if f.kind == "value" and f.subject == ep.entity),
        key=lambda f: f.position,
    )
    if ep.family == "two_hop":
        relations = sorted(
            (f for f in ep.facts if f.kind == "relation" and f.subject == ep.entity),
            key=lambda f: f.position,
        )
        if not relations:
            return None, None, None
        relation = relations[-1]
        records = sorted(
            (f for f in ep.facts
             if f.kind == "record" and f.subject == relation.document),
            key=lambda f: f.position,
        )
        if not records:
            return None, relation.position, None
        return records[-1].code, relation.position, records[-1].position
    if not values:
        return None, None, None
    last = values[-1]
    return last.code, last.position, last.position


def validate_episode(ep: V4Episode) -> None:
    entity, _, _, render_style = _spec(ep.split, ep.family, ep.entity_index)
    if (ep.entity != entity or ep.render_style != render_style
        or ep.memory_length != MEMORY_LENGTH or type(ep.variant) is not int
        or not 0 <= ep.variant < 8):
        raise ValueError("v4 identity/variant/memory/style drift")
    expected_query = _query(ep.entity, ep.family, ep.render_style)
    if ep.query != expected_query:
        raise ValueError("v4 query leaked answer or changed")
    if not 8 <= len(ep.facts) <= 11:
        raise ValueError("v4 incorrect record count")
    positions = [f.position for f in ep.facts]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("v4 unsorted/duplicate memory sources")
    if any(type(p) is not int or not 0 <= p < ep.memory_length for p in positions):
        raise ValueError("v4 physical position outside completed memory")
    for fact in ep.facts:
        if not fact.subject or fact.kind not in ("value", "record", "relation"):
            raise ValueError("v4 invalid fact kind/subject")
        if fact.kind == "relation":
            if fact.document is None or fact.code is not None:
                raise ValueError("v4 malformed relation")
        elif fact.document is not None or fact.code not in (*ANSWER_IDS, STALE_NONCANDIDATE):
            raise ValueError("v4 malformed candidate or stale code")
    bag = Counter(f.code for f in ep.facts if f.code in ANSWER_IDS)
    if bag != Counter(ANSWER_IDS):
        raise ValueError("v4 must have one copy of each candidate code")
    if ep.family == "overwrite":
        stale = [f for f in ep.facts if f.code == STALE_NONCANDIDATE]
        if len(stale) != 1 or stale[0].subject != ep.entity:
            raise ValueError("v4 overwrite must have exactly one obsolete record")
    elif any(f.code == STALE_NONCANDIDATE for f in ep.facts):
        raise ValueError("v4 unexpected stale noncandidate")
    code, first, second = oracle_reference(ep)
    if ep.family in POSITIVES:
        if code != ep.gold_answer or code not in ANSWER_IDS or first is None or second is None:
            raise ValueError("v4 oracle/positive binding mismatch")
        if ep.family == "two_hop" and first == second:
            raise ValueError("v4 two-hop collapsed to a single source")
        if ep.family != "two_hop" and first != second:
            raise ValueError("v4 nonrelation hop address mismatch")
    elif code is not None or ep.gold_answer is not None:
        raise ValueError("v4 no_match accidentally answerable")


def _query(entity: str, family: str, style: int) -> str:
    section = "two_hop" if family == "two_hop" else "ordinary"
    return QUERY_TEMPLATES[section][style].format(entity=entity)


def generate_v4_episode(
    split: Split, family: Family, entity_index: int, variant: int,
) -> V4Episode:
    """Deterministic for ANY split, but CI/tests must call TRAIN split ONLY."""
    entity, doc, seed, style = _spec(split, family, entity_index)
    if type(variant) is not int or not 0 <= variant < 8:
        raise ValueError("v4 variant outside 0..7")
    positions = random.Random(seed).sample(range(MEMORY_LENGTH), 12)
    base_idx = (_stable_int("new-rotation", split, family, entity_index) % 8)
    base_code = ANSWER_IDS[base_idx]
    answer = ANSWER_IDS[(base_idx + variant) % 8]
    distractors = [c for c in ANSWER_IDS if c != base_code]
    if family != "no_match" and answer != base_code:
        swap_index = distractors.index(answer)
        distractors[swap_index] = base_code

    facts: list[V4Fact] = []
    if family == "direct":
        facts.append(V4Fact(positions[0], "value", entity, code=answer))
        remaining_positions = positions[1:8]
    elif family == "overwrite":
        old, current = sorted(positions[:2])
        facts.extend((
            V4Fact(old, "value", entity, code=STALE_NONCANDIDATE),
            V4Fact(current, "value", entity, code=answer),
        ))
        remaining_positions = positions[2:9]
    elif family == "two_hop":
        old_relation, current_relation = sorted(positions[:2])
        previous_doc = f"{doc}-previous"
        facts.extend((
            V4Fact(old_relation, "relation", entity, document=previous_doc),
            V4Fact(current_relation, "relation", entity, document=doc),
            V4Fact(positions[2], "record", doc, code=answer),
        ))
        remaining_positions = positions[3:10]
    else:
        # Negative contains all eight code choices, but ZERO matching names.
        distractors = list(ANSWER_IDS)
        remaining_positions = positions[:8]

    for i, (pos, code) in enumerate(zip(remaining_positions, distractors)):
        if family == "two_hop":
            subject = (f"{doc}-previous" if i == 0
                       else f"V4-{split}-R{entity_index:04d}-{i:02d}-{family}")
            facts.append(V4Fact(pos, "record", subject, code=code))
        else:
            subject = f"V4-{split}-U{entity_index:04d}-{i:02d}-{family}"
            facts.append(V4Fact(pos, "value", subject, code=code))

    episode = V4Episode(
        split=split, family=family, entity_index=entity_index, entity=entity,
        variant=variant, query=_query(entity, family, style),
        memory_length=MEMORY_LENGTH,
        facts=tuple(sorted(facts, key=lambda f: f.position)),
        gold_answer=None if family == "no_match" else answer,
        render_style=style,
    )
    validate_episode(episode)
    return episode


def seal_v4_view(ep: V4Episode) -> SealedModelView:
    """Model only sees ordinary query and serialized memory, NOT oracle/labels."""
    validate_episode(ep)
    template = RENDER_TEMPLATES[ep.render_style]
    memory = "\n".join(template[f.kind].format(
        position=f.position, subject=f.subject, code=f.code, document=f.document,
    ) for f in ep.facts)
    return SealedModelView(
        query=ep.query, memory_text=memory,
        answer_options=tuple(f"CODE-{code}" for code in ANSWER_IDS),
    )


def code_bag(view: SealedModelView) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in re.findall(r"CODE-(\d+)", view.memory_text)
                        if int(x) in ANSWER_IDS))


def redacted_view(view: SealedModelView) -> SealedModelView:
    return SealedModelView(
        view.query,
        re.sub(r"CODE-\d+", "CODE-REDACTED", view.memory_text),
        view.answer_options,
    )


def validate_v4_group(episodes: tuple[V4Episode, ...]) -> dict[str, object]:
    if len(episodes) != 8 or {e.variant for e in episodes} != set(range(8)):
        raise ValueError("v4 exactly eight variants per entity required")
    episodes = tuple(sorted(episodes, key=lambda e: e.variant))
    base = episodes[0]
    for ep in episodes:
        validate_episode(ep)
        if (ep.split, ep.family, ep.entity, ep.query, ep.render_style) != (
            base.split, base.family, base.entity, base.query, base.render_style,
        ):
            raise ValueError("v4 group identity/query changed")
    layout = tuple((f.position, f.kind, f.subject, f.document) for f in base.facts)
    ref_answer_pos = oracle_reference(base)[2]
    sealed = seal_v4_view(base)
    expected_bag = code_bag(sealed)
    if expected_bag != tuple(sorted(ANSWER_IDS)):
        raise ValueError("v4 base candidate-value bag mismatch")
    for ep in episodes:
        if tuple((f.position, f.kind, f.subject, f.document)
                 for f in ep.facts) != layout:
            raise ValueError("v4 group layout/source identity shortcut")
        view = seal_v4_view(ep)
        if (view.query != sealed.query or redacted_view(view) != redacted_view(sealed)
            or code_bag(view) != expected_bag):
            raise ValueError("v4 query/layout/multiset leaked target")
        if oracle_reference(ep)[2] != ref_answer_pos:
            raise ValueError("v4 authoritative address changed")
        diff = [idx for idx, (old, now) in enumerate(zip(base.facts, ep.facts))
                if old.code != now.code]
        if ep.family in POSITIVES:
            if len(diff) != (0 if ep.variant == 0 else 2):
                raise ValueError("v4 positive must swap target and one decoy")
            if ep.variant and not any(ep.facts[i].position == ref_answer_pos
                                      for i in diff):
                raise ValueError("v4 authoritative fact missing from swap")
        elif diff or view != sealed:
            raise ValueError("v4 negative changed across rotations")
    if base.family in POSITIVES:
        if {e.gold_answer for e in episodes} != set(ANSWER_IDS):
            raise ValueError("v4 positive answers not eight-way balanced")
        if len({seal_v4_view(e).memory_text for e in episodes}) != 8:
            raise ValueError("v4 positive variants not visibly distinguishable")
    elif any(e.gold_answer is not None for e in episodes):
        raise ValueError("v4 no-match target is not absent")
    return {
        "split": base.split, "family": base.family, "entity": base.entity,
        "cases": 8, "independent_entity_clusters": 1,
        "query_layout_codebag_invariant": True,
        "positives_each_answer_once": base.family in POSITIVES,
        "no_match_is_unanswerable": base.family == "no_match",
        "oracle_position_is_variant_invariant": True,
    }


def v4_train_group(family: Family, entity_index: int) -> tuple[V4Episode, ...]:
    """Hard TRAIN-only entrypoint used by repository CI and CPU static audit."""
    episodes = tuple(generate_v4_episode("train", family, entity_index, v)
                     for v in range(8))
    validate_v4_group(episodes)
    return episodes


def train_only_shortcut_audit() -> dict[str, object]:
    """All 2048 TRAIN examples; no development/test data instantiated."""
    families: dict[str, Counter[int]] = {family: Counter() for family in POSITIVES}
    oracle_source_positions: dict[str, set[int]] = defaultdict(set)
    sampled_styles: set[int] = set()
    n_records = 0
    for family in FAMILIES:
        for idx in range(SPLIT_ENTITIES["train"]):
            group = v4_train_group(family, idx)
            for ep in group:
                n_records += 1
                sampled_styles.add(ep.render_style)
                if family in POSITIVES:
                    families[family][ep.gold_answer] += 1
                    oracle_source_positions[family].add(oracle_reference(ep)[2])
    if n_records != 2048 or sampled_styles != {0, 1, 2}:
        raise RuntimeError("v4 train cardinality/template drift")
    for family in POSITIVES:
        if families[family] != Counter({x: SPLIT_ENTITIES["train"] for x in ANSWER_IDS}):
            raise RuntimeError("v4 TRAIN answer values not entity-balanced")
        if len(oracle_source_positions[family]) < 16:
            raise RuntimeError("v4 TRAIN target address lacks variety")
    return {
        "classification": "CHM_V4_1421_INDEPENDENT_ENTITY_TRAIN_ONLY_STATIC_AUDIT",
        "training_cases": n_records, "training_entities_per_family": 64,
        "training_8_way_counterfactual_groups": 256,
        "training_gold_frequency_per_candidate_per_positive_family": 64,
        "distinct_target_addresses_per_positive_family": {
            fam: len(pos) for fam, pos in oracle_source_positions.items()
        },
        "render_styles_used": sorted(sampled_styles),
        "scored_test_cases_constructed": False,
        "model_forward_calls": 0, "training_optimizer_steps": 0,
        "gpu_used": False, "new_scientific_attempt": False,
        "original_100m_scientific_classification": OLD_100M_STOP,
        "original_scientific_seed_consumed": OLD_100M_SCIENTIFIC_SEED_CONSUMED,
    }


def declarative_heldout_manifest() -> dict[str, object]:
    """Frozen split description ONLY; this never constructs dev/test episodes.

    SHA-256 is over the DECLARATION, NOT the hidden future payload. The later
    separately authorised first-run scorer must compute its own payload digest.
    """
    data = {
        "version": V4_NAMESPACE, "generator_salt": RNG_SALT,
        "splits": dict(SPLIT_ENTITIES), "families": list(FAMILIES),
        "candidate_codes": list(ANSWER_IDS), "variants_per_entity_family": 8,
        "memory_length": MEMORY_LENGTH, "render_template_count": len(RENDER_TEMPLATES),
        "development_independent_entities_per_family": 16,
        "test_independent_entities_per_family": 32,
        "expected_test_cases": 32 * 4 * 8,
        "expected_development_cases": 16 * 4 * 8,
        "scored_heldout_constructed": False,
        "payload_sha256_computed": False,
    }
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return {**data, "declarative_manifest_sha256":
            hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


__all__ = (
    "V4Fact", "V4Episode", "SPLIT_ENTITIES", "FAMILIES", "POSITIVES",
    "ANSWER_IDS", "V4_NAMESPACE", "oracle_reference", "validate_episode",
    "generate_v4_episode", "seal_v4_view", "code_bag", "redacted_view",
    "validate_v4_group", "v4_train_group", "train_only_shortcut_audit",
    "declarative_heldout_manifest",
)
