from __future__ import annotations

"""#1373 zero-GPU matched-multiset counterfactual memory benchmark.

Within each eight-way positive group only the *binding* of candidate codes
to target/decoy subjects changes: complete query, record layout, identities,
and the multiset of all candidate codes stay byte-for-byte invariant.
No model training, trained checkpoint replay, Modal, CUDA, or old probe reuse.
"""

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import random
import re
from typing import Literal

from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS, MEMORY_LENGTHS, SPLITS
from .chm_v3_counterfactual_model_view_1365 import (
    Fact, PairedEpisode, SealedModelView, STALE_CODE,
    oracle_read, seal_model_view, redact_memory_values, validate_episode,
)

FAMILIES = ("direct", "overwrite", "two_hop", "no_match")
POSITIVES = FAMILIES[:3]
ORIGINAL_STAGE_C_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
ORIGINAL_SCIENTIFIC_SEED_CONSUMED = 2_013_161
GPU_AUTHORIZED = False
SCIENTIFIC_EXECUTION_AUTHORIZED = False


def _hash_int(*items: object) -> int:
    s = "\x1f".join(map(str, items)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def model_code_bag(view: SealedModelView) -> tuple[str, ...]:
    """Bag/multiset baseline input, without subjects, roles or locations."""
    return tuple(sorted(re.findall(r"CODE-\d+", view.memory_text)))


def query_only_view(view: SealedModelView) -> SealedModelView:
    return SealedModelView(view.query, "", view.answer_options)


def generate_episode(
    split: Literal["train", "development", "test"],
    family: Literal["direct", "overwrite", "two_hop", "no_match"],
    entity_index: int,
    variant: int,
    *,
    memory_length: int = 128,
) -> PairedEpisode:
    if split not in SPLITS or family not in FAMILIES:
        raise ValueError("invalid split/family")
    if type(entity_index) is not int or not 0 <= entity_index < SPLITS[split]:
        raise ValueError("invalid split-specific entity index")
    if type(variant) is not int or not 0 <= variant < 8:
        raise ValueError("variant must be 0..7")
    if memory_length not in MEMORY_LENGTHS:
        raise ValueError("invalid completed-memory size")

    entity = f"V3-{split}-E{entity_index:03d}"
    document = f"V3-{split}-D{entity_index:03d}"
    base_index = (entity_index * 3 + FAMILIES.index(family)) % 8
    base_code = ANSWER_IDS[base_index]
    answer = ANSWER_IDS[(base_index + variant) % 8]
    # Initial position-indexed bag contains exactly one copy of every candidate.
    initial_decoys = [code for code in ANSWER_IDS if code != base_code]
    if family != "no_match" and answer != base_code:
        j = initial_decoys.index(answer)
        decoy_codes = initial_decoys.copy()
        decoy_codes[j] = base_code  # SWAP: preserve multiset, not add/drop code
    else:
        decoy_codes = initial_decoys
    if family == "no_match":
        decoy_codes = list(ANSWER_IDS)

    # Every position is independent of variant. No label in layout/metadata.
    seed = _hash_int("CHM_V3_1373_NEW_BINDING_SUITE", split, family,
                     entity_index, memory_length)
    positions = random.Random(seed).sample(range(memory_length), 9)
    facts: list[Fact] = []
    if family == "direct":
        facts.append(Fact(positions[0], "value", entity, code=answer))
        offsets = positions[1:8]
    elif family == "overwrite":
        older, newer = sorted(positions[:2])
        facts.extend((
            Fact(older, "value", entity, code=STALE_CODE),
            Fact(newer, "value", entity, code=answer),
        ))
        offsets = positions[2:9]
    elif family == "two_hop":
        facts.extend((
            Fact(positions[0], "relation", entity, document=document),
            Fact(positions[1], "record", document, code=answer),
        ))
        offsets = positions[2:9]
    else:
        offsets = positions[:8]

    for j, (position, code) in enumerate(zip(offsets, decoy_codes)):
        role = "record" if family == "two_hop" else "value"
        subject = (f"V3-{split}-X{entity_index:03d}-D{j}"
                   if role == "record" else f"V3-{split}-unrelated-{entity_index:03d}-{j}")
        facts.append(Fact(position, role, subject, code=code))
    query = (f"Which access code does the record for {entity} store?"
             if family == "two_hop" else
             f"What is the current access code for {entity}?")
    ep = PairedEpisode(
        split=split, family=family, entity=entity, variant=variant,
        memory_length=memory_length, query=query,
        facts=tuple(sorted(facts, key=lambda f: f.position)),
        gold_answer=None if family == "no_match" else answer,
    )
    validate_episode(ep)
    return ep


def validate_group(episodes: tuple[PairedEpisode, ...]) -> None:
    if len(episodes) != 8:
        raise ValueError("exactly eight counterfactuals required")
    for ep in episodes:
        validate_episode(ep)
    base = episodes[0]
    if {ep.variant for ep in episodes} != set(range(8)):
        raise ValueError("missing/duplicate rotation")
    if base.family in POSITIVES and {
        ep.gold_answer for ep in episodes
    } != set(ANSWER_IDS):
        raise ValueError("not eight distinct authoritative answers")
    if base.family == "no_match" and any(ep.gold_answer is not None for ep in episodes):
        raise ValueError("negative group not truly negative")

    original_layout = tuple((f.position, f.kind, f.subject, f.document)
                            for f in base.facts)
    base_view = seal_model_view(base)
    original_redacted = redact_memory_values(base_view)
    original_bag = model_code_bag(base_view)
    if Counter(original_bag) != Counter(f"CODE-{code}" for code in ANSWER_IDS):
        raise ValueError("memory candidate multiset must contain each code once")
    originals = sorted(x.code for x in base.facts if x.code is not None)
    original_target_pos = oracle_read(base)[2]
    for ep in episodes:
        if (ep.split, ep.family, ep.entity, ep.query, ep.memory_length) != (
            base.split, base.family, base.entity, base.query, base.memory_length
        ):
            raise ValueError("paired identity/query changed")
        layout = tuple((f.position, f.kind, f.subject, f.document)
                       for f in ep.facts)
        if layout != original_layout:
            raise ValueError("paired layout/subjects/documents changed")
        view = seal_model_view(ep)
        if view.query != base_view.query or view.answer_options != base_view.answer_options:
            raise ValueError("paired query/answer options changed")
        if redact_memory_values(view) != original_redacted:
            raise ValueError("redacted paired layout drift")
        if model_code_bag(view) != original_bag or sorted(
            x.code for x in ep.facts if x.code is not None
        ) != originals:
            raise ValueError("code multiset/frequency changed")
        if oracle_read(ep)[2] != original_target_pos:
            raise ValueError("authoritative memory address changed")
        if base.family in POSITIVES:
            target = oracle_read(ep)[0]
            if target != ep.gold_answer:
                raise ValueError("binding oracle mismatch")
            changed = [p for p, (a, b) in enumerate(zip(ep.facts, base.facts))
                       if a.code != b.code]
            if ep.variant == 0:
                if changed:
                    raise ValueError("base group changed")
            elif len(changed) != 2:
                raise ValueError("exactly target + one decoy must swap")
            else:
                selected_target = next(i for i, f in enumerate(ep.facts)
                                       if f.position == original_target_pos)
                if selected_target not in changed:
                    raise ValueError("target not participating in swap")
        elif view != base_view:
            raise ValueError("negative episodes unexpectedly vary")
    if base.family in POSITIVES and len({
        seal_model_view(ep).memory_text for ep in episodes
    }) != 8:
        raise ValueError("insufficient counterfactual distinctions")


def paired_group(
    split: Literal["train", "development", "test"],
    family: Literal["direct", "overwrite", "two_hop", "no_match"],
    entity_index: int,
    *,
    memory_length: int = 128,
) -> tuple[PairedEpisode, ...]:
    result = tuple(generate_episode(split, family, entity_index, idx,
                                    memory_length=memory_length) for idx in range(8))
    validate_group(result)
    return result


def _choice(data: str) -> int:
    return ANSWER_IDS[_hash_int("CHM_V3_1373_CONTROL", data) % 8]


def control_report(split: Literal["train", "development", "test"]) -> dict[str, object]:
    if split not in SPLITS:
        raise ValueError("unknown split")
    stats: dict[str, dict[str, float | int]] = {}
    for family in FAMILIES:
        n = query_hits = layout_hits = bag_hits = abstain_hits = 0
        for index in range(SPLITS[split]):
            for ep in paired_group(split, family, index):
                v = seal_model_view(ep)
                n += 1
                query_hits += _choice(v.query) == ep.gold_answer
                layout_hits += _choice(redact_memory_values(v).memory_text) == ep.gold_answer
                bag_hits += _choice("\x1f".join(model_code_bag(v))) == ep.gold_answer
                abstain_hits += ep.gold_answer is None
        stats[family] = {
            "cases": n,
            "query_only_accuracy": query_hits / n,
            "layout_only_accuracy": layout_hits / n,
            "bag_of_codes_only_accuracy": bag_hits / n,
            "always_abstain_correct": abstain_hits,
        }
    return {
        "classification": "CHM_V3_1373_MATCHED_MULTISET_CPU_NEGATIVE_CONTROLS",
        "split": split, "families": stats,
        "trained_model_used": False, "gpu_allocated": False,
        "historical_scientific_classification": ORIGINAL_STAGE_C_STOP,
        "historical_scientific_seed_consumed": True,
    }


__all__ = (
    "FAMILIES", "POSITIVES", "generate_episode", "paired_group",
    "validate_group", "model_code_bag", "query_only_view", "control_report",
)
