from __future__ import annotations

"""CHM-v1 held-out natural-language generator v3.

This is the preregistered pre-result correction recorded on issue #854 after a
zero-credit audit found query-visible lexical answer cues in v2 (for example,
"Birch Observatory" -> "birch").  V3 changes only identifier vocabulary and
balanced answer assignment.  It reuses v2's already-tested encoding, padding,
distance, state-slice, scoring-metadata, and EncodedProbe validation machinery.

No model, training, retrieval, threshold, scientific seed, or execution
authority is changed by this module.
"""

import re
import random
from collections import Counter

from .chm_v1_long_memory_eval import (
    DEFAULT_CASES_PER_FAMILY,
    LOCAL_CONTROL_MAX_DISTANCE,
    LONG_RANGE_MIN_DISTANCE,
    EncodedProbe,
    Encode,
    _VALUES,
    _candidate_ids,
    _encoded,
    _filler,
    _pad_until_distance,
    _separate_two_hop_facts,
    _single_token_values,
)

GENERATOR_VERSION = "chm-v1-heldout-natural-v3"

# Opaque first words deliberately share no probe-value word.  Human-readable
# suffixes keep the prompts ordinary enough for the same language-model gate.
_ENTITIES_V3 = (
    "Vexa Harbor",
    "Torin Observatory",
    "Caldor Junction",
    "Pavo Museum",
    "Soran Station",
    "Meko Library",
    "Drava Works",
    "Quen Depot",
    "Ralo Clinic",
    "Siva Hall",
    "Tano Market",
    "Pexa Theater",
    "Daro Terminal",
    "Feno Archive",
    "Cavo Foundry",
    "Bexa Lodge",
)

_LINKS_V3 = (
    "Qora Ledger",
    "Tiven Index",
    "Pavo Register",
    "Sela Catalog",
    "Navo Dossier",
    "Reka Record",
    "Toma File",
    "Vero Note",
    "Cira Register",
    "Penda Ledger",
    "Rivo File",
    "Leda Index",
)


def _lexical_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", text.lower()))


def _identifier_contains_value(identifier: str, value: str) -> bool:
    """Conservative lexical-cue test required by the v3 amendment."""
    ident = identifier.lower()
    val = value.lower()
    return val in ident or val in _lexical_words(identifier)


def _validate_identifier_vocabulary() -> None:
    for identifier in _ENTITIES_V3 + _LINKS_V3:
        for value in _VALUES:
            if _identifier_contains_value(identifier, value):
                raise RuntimeError(
                    f"v3 identifier/value lexical collision: {identifier!r} / {value!r}"
                )


_validate_identifier_vocabulary()


def _local_case_index(case_id: int) -> int:
    return int(case_id % 100_000)


def _answer_slot(case_id: int, family_offset: int) -> int:
    """Balanced 8-way schedule: every slot occurs 3x over 24 cases."""
    return (_local_case_index(case_id) * 5 + family_offset) % 8


def _assert_answer_not_in_identifiers(answer: str, *identifiers: str) -> None:
    for identifier in identifiers:
        if _identifier_contains_value(identifier, answer):
            raise RuntimeError(
                f"v3 query-visible lexical cue: answer={answer!r}, identifier={identifier!r}"
            )


def _rare_fact(
    case_id: int,
    encode: Encode,
    rng: random.Random,
    values: list[tuple[str, int]],
) -> EncodedProbe:
    idx = _local_case_index(case_id)
    entity = _ENTITIES_V3[idx % len(_ENTITIES_V3)]
    answer_slot = _answer_slot(case_id, 1)
    answer, answer_id = values[answer_slot]
    _assert_answer_not_in_identifiers(answer, entity)

    distractors: list[str] = []
    for offset in range(1, 6):
        other = _ENTITIES_V3[(idx + offset) % len(_ENTITIES_V3)]
        distractor_value = values[(answer_slot + offset) % 8][0]
        distractors.append(
            f"Archive note: the access word for {other} is {distractor_value}."
        )

    fact = f"Archive note: the access word for {entity} is {answer}."
    prefix = fact + " " + " ".join(distractors)
    evidence_end = len(_encoded(encode, fact)) - 1
    query = f"Question: What is the access word for {entity}? Answer:"
    prompt, ids, distance = _pad_until_distance(
        encode,
        prefix=prefix,
        evidence_end=evidence_end,
        query=query,
        rng=rng,
        minimum_distance=LONG_RANGE_MIN_DISTANCE,
    )
    probe = EncodedProbe(
        family="rare_fact",
        case_id=case_id,
        prompt_text=prompt,
        prompt_ids=ids,
        answer_token_id=answer_id,
        candidate_token_ids=_candidate_ids(values),
        evidence_end_token=evidence_end,
        query_token=len(ids) - 1,
        evidence_distance=distance,
        generator_version=GENERATOR_VERSION,
    )
    probe.validate()
    return probe


def _overwrite(
    case_id: int,
    encode: Encode,
    rng: random.Random,
    values: list[tuple[str, int]],
) -> EncodedProbe:
    idx = _local_case_index(case_id)
    entity = _ENTITIES_V3[(idx * 3) % len(_ENTITIES_V3)]
    answer_slot = _answer_slot(case_id, 3)
    answer, answer_id = values[answer_slot]
    stale1, stale1_id = values[(answer_slot + 1) % 8]
    stale2, stale2_id = values[(answer_slot + 2) % 8]
    _assert_answer_not_in_identifiers(answer, entity)

    updates = (
        f"Status update: the access word for {entity} is {stale1}. "
        f"Later correction: the access word for {entity} is {stale2}. "
        f"Final authoritative update: the access word for {entity} is {answer}."
    )
    evidence_end = len(_encoded(encode, updates)) - 1
    query = (
        f"Question: According to the latest update, what is the access word for "
        f"{entity}? Answer:"
    )
    prompt, ids, distance = _pad_until_distance(
        encode,
        prefix=updates,
        evidence_end=evidence_end,
        query=query,
        rng=rng,
        minimum_distance=LONG_RANGE_MIN_DISTANCE,
    )
    probe = EncodedProbe(
        family="overwrite",
        case_id=case_id,
        prompt_text=prompt,
        prompt_ids=ids,
        answer_token_id=answer_id,
        candidate_token_ids=_candidate_ids(values),
        evidence_end_token=evidence_end,
        query_token=len(ids) - 1,
        evidence_distance=distance,
        stale_token_ids=(stale1_id, stale2_id),
        generator_version=GENERATOR_VERSION,
    )
    probe.validate()
    return probe


def _two_hop(
    case_id: int,
    encode: Encode,
    rng: random.Random,
    values: list[tuple[str, int]],
) -> EncodedProbe:
    idx = _local_case_index(case_id)
    entity = _ENTITIES_V3[(idx * 5) % len(_ENTITIES_V3)]
    link = _LINKS_V3[idx % len(_LINKS_V3)]
    answer_slot = _answer_slot(case_id, 5)
    answer, answer_id = values[answer_slot]
    _assert_answer_not_in_identifiers(answer, entity, link)

    first_fact = f"Routing note: the record associated with {entity} is {link}."
    second_fact = f"Registry note: the access word stored in {link} is {answer}."
    prefix, first_end, second_end = _separate_two_hop_facts(
        encode,
        first_fact=first_fact,
        second_fact=second_fact,
        rng=rng,
    )
    query = (
        f"Question: Follow the record associated with {entity}. "
        "What access word does that record store? Answer:"
    )
    prompt, ids, distance = _pad_until_distance(
        encode,
        prefix=prefix,
        evidence_end=second_end,
        query=query,
        rng=rng,
        minimum_distance=LONG_RANGE_MIN_DISTANCE,
    )
    probe = EncodedProbe(
        family="two_hop",
        case_id=case_id,
        prompt_text=prompt,
        prompt_ids=ids,
        answer_token_id=answer_id,
        candidate_token_ids=_candidate_ids(values),
        first_evidence_end_token=first_end,
        evidence_end_token=second_end,
        query_token=len(ids) - 1,
        evidence_distance=distance,
        generator_version=GENERATOR_VERSION,
    )
    probe.validate()
    return probe


def _local_negative(
    case_id: int,
    encode: Encode,
    rng: random.Random,
    values: list[tuple[str, int]],
) -> EncodedProbe:
    idx = _local_case_index(case_id)
    entity = _ENTITIES_V3[(idx * 7) % len(_ENTITIES_V3)]
    answer_slot = _answer_slot(case_id, 7)
    answer, answer_id = values[answer_slot]
    _assert_answer_not_in_identifiers(answer, entity)

    old_context = _filler(rng, 8)
    fact = f"Current desk note: the access word for {entity} is {answer}."
    query = f"Question: What is the access word for {entity}? Answer:"
    prompt = old_context + " " + fact + " " + query
    ids = _encoded(encode, prompt)
    evidence_end = len(_encoded(encode, old_context + " " + fact)) - 1
    distance = (len(ids) - 1) - evidence_end
    probe = EncodedProbe(
        family="local_negative",
        case_id=case_id,
        prompt_text=prompt,
        prompt_ids=ids,
        answer_token_id=answer_id,
        candidate_token_ids=_candidate_ids(values),
        evidence_end_token=evidence_end,
        query_token=len(ids) - 1,
        evidence_distance=distance,
        generator_version=GENERATOR_VERSION,
    )
    probe.validate()
    return probe


def generate_probe_suite(
    encode: Encode,
    *,
    seed: int,
    cases_per_family: int = DEFAULT_CASES_PER_FAMILY,
) -> list[EncodedProbe]:
    if cases_per_family < 1:
        raise ValueError("cases_per_family must be positive")
    values = _single_token_values(encode)[:8]
    if len(values) != 8:
        raise ValueError("v3 requires exactly eight usable candidate slots")

    rng = random.Random(seed)
    suite: list[EncodedProbe] = []
    builders = (_rare_fact, _overwrite, _two_hop, _local_negative)
    for family_offset, builder in enumerate(builders):
        family_start = len(suite)
        for index in range(cases_per_family):
            case_id = family_offset * 100_000 + index
            suite.append(builder(case_id, encode, rng, values))

        # The preregistered scientific envelope is 24/family.  Enforce exact
        # balance whenever a caller requests a complete 8-way cycle multiple.
        if cases_per_family % 8 == 0:
            family = suite[family_start:]
            counts = Counter(probe.answer_token_id for probe in family)
            expected = cases_per_family // 8
            if set(counts) != set(_candidate_ids(values)) or any(
                count != expected for count in counts.values()
            ):
                raise RuntimeError(
                    f"v3 answer schedule lost balance: counts={dict(counts)}, expected={expected}"
                )
    return suite
