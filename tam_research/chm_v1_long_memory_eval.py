from __future__ import annotations

"""Deterministic held-out long-memory probes for CHM-v1 / EIEM issue #854.

The generator emits ordinary text only. Hidden case IDs, family labels,
expected answers, stale-answer sets, and candidate sets are evaluator metadata;
none is appended to model input. The three long-range families verify that the
latest evidence required for the answer ends >512 encoded tokens before the
query. The two-hop family additionally places its first relation and second
remote record in different 512-token local windows so the second stored raw
hidden state cannot locally contextualize both relations. The local negative
control deliberately places its evidence <=128 tokens before the query.

This module contains no tokenizer download and no GPU/run trigger. A future
authorized runner supplies the already-frozen GPT-2 encoder.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import random
from typing import Literal

import torch

from .chm_v1_small_lm import CHMV1EIEMLM, CHMV1LocalLM, EpisodicState, LOCAL_WINDOW

Encode = Callable[[str], Sequence[int]]
Family = Literal["rare_fact", "overwrite", "two_hop", "local_negative"]

LONG_RANGE_MIN_DISTANCE = LOCAL_WINDOW + 1
LOCAL_CONTROL_MAX_DISTANCE = 128
# Endpoint separation >=640 plus a <=128-token second fact guarantees that the
# second fact begins strictly after the first fact's 512-token local horizon.
TWO_HOP_MIN_FACT_ENDPOINT_SEPARATION = LOCAL_WINDOW + LOCAL_CONTROL_MAX_DISTANCE
# The preregistered sparse-read gate is judged at the largest frozen slice:
# exactly 1,024 prior episodic items before the final two-hop query chunk.
SPARSE_READ_GATE_MEMORY_SIZE = 2 * LOCAL_WINDOW
DEFAULT_CASES_PER_FAMILY = 24
GENERATOR_VERSION = "chm-v1-heldout-natural-v2"

_ENTITIES = (
    "Aurora Harbor", "Birch Observatory", "Cedar Junction", "Delta Museum",
    "Ember Station", "Fjord Library", "Granite Works", "Helios Depot",
    "Indigo Clinic", "Juniper Hall", "Kestrel Market", "Lumen Theater",
    "Mosaic Terminal", "Nimbus Archive", "Orchid Foundry", "Pioneer Lodge",
)
_LINKS = (
    "Amber Ledger", "Beryl Index", "Copper Register", "Drift Catalog",
    "Elm Dossier", "Flint Record", "Garnet File", "Harbor Note",
    "Ivory Register", "Jade Ledger", "Keystone File", "Larch Index",
)
_VALUES = (
    "amber", "birch", "cobalt", "delta", "ember", "fjord", "granite",
    "hazel", "indigo", "juniper", "kestrel", "lumen", "mosaic", "north",
    "orchid", "pearl", "quartz", "river", "silver", "tulip", "violet",
    "willow", "xenon", "yellow", "zephyr",
)
_FILLER_SENTENCES = (
    "The committee reviewed ordinary maintenance notes and filed them in chronological order.",
    "A delivery arrived after lunch while the front desk updated the public schedule.",
    "Several visitors crossed the courtyard and discussed the weather before continuing inside.",
    "The weekly report described routine staffing, inventory, cleaning, and transport changes.",
    "A technician inspected the lights, recorded the meter reading, and closed the service panel.",
    "The afternoon bulletin listed community events, road notices, opening hours, and meeting rooms.",
)


@dataclass(frozen=True)
class EncodedProbe:
    family: Family
    case_id: int
    prompt_text: str
    prompt_ids: tuple[int, ...]
    answer_token_id: int
    candidate_token_ids: tuple[int, ...]
    evidence_end_token: int
    query_token: int
    evidence_distance: int
    stale_token_ids: tuple[int, ...] = ()
    first_evidence_end_token: int | None = None
    generator_version: str = GENERATOR_VERSION
    used_for_training: bool = False

    def validate(self) -> None:
        if self.query_token != len(self.prompt_ids) - 1:
            raise ValueError("query_token must be the final prompt token position")
        if self.answer_token_id not in self.candidate_token_ids:
            raise ValueError("answer token must be in candidate set")
        if len(set(self.candidate_token_ids)) != len(self.candidate_token_ids):
            raise ValueError("candidate token IDs must be unique")
        if self.family == "local_negative":
            if not (0 <= self.evidence_distance <= LOCAL_CONTROL_MAX_DISTANCE):
                raise ValueError("local control evidence escaped the <=128-token region")
        elif self.evidence_distance < LONG_RANGE_MIN_DISTANCE:
            raise ValueError("long-range evidence must be strictly >512 tokens from query")

        # Freeze exact episodic-state slices at the scored query. Rare-fact and
        # overwrite are scored in chunk 2 with 512 prior items. Two-hop is scored
        # in chunk 3 with 1,024 prior items.
        if self.family in {"rare_fact", "overwrite"}:
            if self.evidence_end_token >= LOCAL_WINDOW:
                raise ValueError("single-hop long-range evidence escaped first local chunk")
            if not (LOCAL_WINDOW <= self.query_token < 2 * LOCAL_WINDOW):
                raise ValueError("single-hop long-range query must use 512-item memory slice")

        if self.family == "two_hop":
            if self.first_evidence_end_token is None:
                raise ValueError("two-hop probe must record first evidence endpoint")
            separation = self.evidence_end_token - self.first_evidence_end_token
            if separation < TWO_HOP_MIN_FACT_ENDPOINT_SEPARATION:
                raise ValueError("two-hop facts are not in separated local horizons")
            if self.first_evidence_end_token >= LOCAL_WINDOW:
                raise ValueError("two-hop first relation must remain in chunk 1")
            if not (LOCAL_WINDOW <= self.evidence_end_token < SPARSE_READ_GATE_MEMORY_SIZE):
                raise ValueError("two-hop second record must remain in chunk 2")
            if not (
                SPARSE_READ_GATE_MEMORY_SIZE
                <= self.query_token
                < SPARSE_READ_GATE_MEMORY_SIZE + LOCAL_WINDOW
            ):
                raise ValueError("two-hop query must use exact 1024-item memory slice")


def _encoded(encode: Encode, text: str) -> tuple[int, ...]:
    ids = tuple(int(x) for x in encode(text))
    if not ids:
        raise ValueError("encoder returned no tokens")
    return ids


def _single_token_values(encode: Encode) -> list[tuple[str, int]]:
    usable: list[tuple[str, int]] = []
    seen: set[int] = set()
    for value in _VALUES:
        ids = tuple(int(x) for x in encode(" " + value))
        if len(ids) == 1 and ids[0] not in seen:
            usable.append((value, ids[0]))
            seen.add(ids[0])
    if len(usable) < 8:
        raise ValueError("encoder exposes fewer than 8 unique one-token probe values")
    return usable


def _candidate_ids(values: list[tuple[str, int]]) -> tuple[int, ...]:
    return tuple(token for _, token in values[:8])


def _filler(rng: random.Random, sentences: int) -> str:
    return " ".join(rng.choice(_FILLER_SENTENCES) for _ in range(sentences))


def _pad_until_distance(
    encode: Encode,
    *,
    prefix: str,
    evidence_end: int,
    query: str,
    rng: random.Random,
    minimum_distance: int,
) -> tuple[str, tuple[int, ...], int]:
    filler_parts: list[str] = []
    # Re-encode the complete prompt because BPE boundaries can change at joins.
    for _ in range(256):
        filler_parts.append(_filler(rng, 2))
        prompt = prefix + " " + " ".join(filler_parts) + " " + query
        ids = _encoded(encode, prompt)
        distance = (len(ids) - 1) - evidence_end
        if distance >= minimum_distance:
            return prompt, ids, distance
    raise RuntimeError("could not create a long enough held-out probe")


def _separate_two_hop_facts(
    encode: Encode,
    *,
    first_fact: str,
    second_fact: str,
    rng: random.Random,
) -> tuple[str, int, int]:
    first_end = len(_encoded(encode, first_fact)) - 1
    second_fact_len = len(_encoded(encode, second_fact))
    if second_fact_len > LOCAL_CONTROL_MAX_DISTANCE:
        raise ValueError("two-hop second fact exceeds 128 tokens; separation proof invalid")
    filler_parts: list[str] = []
    for _ in range(256):
        filler_parts.append(_filler(rng, 2))
        prefix = first_fact + " " + " ".join(filler_parts) + " " + second_fact
        second_end = len(_encoded(encode, prefix)) - 1
        if second_end - first_end >= TWO_HOP_MIN_FACT_ENDPOINT_SEPARATION:
            return prefix, first_end, second_end
    raise RuntimeError("could not separate two-hop facts into different local horizons")


def _rare_fact(case_id: int, encode: Encode, rng: random.Random, values: list[tuple[str, int]]) -> EncodedProbe:
    entity = _ENTITIES[case_id % len(_ENTITIES)]
    answer, answer_id = values[case_id % 8]
    distractors = []
    for offset in range(1, 6):
        other = _ENTITIES[(case_id + offset) % len(_ENTITIES)]
        value = values[(case_id + offset) % 8][0]
        distractors.append(f"Archive note: the access word for {other} is {value}.")
    fact = f"Archive note: the access word for {entity} is {answer}."
    prefix = fact + " " + " ".join(distractors)
    evidence_end = len(_encoded(encode, fact)) - 1
    query = f"Question: What is the access word for {entity}? Answer:"
    prompt, ids, distance = _pad_until_distance(
        encode, prefix=prefix, evidence_end=evidence_end, query=query, rng=rng,
        minimum_distance=LONG_RANGE_MIN_DISTANCE,
    )
    probe = EncodedProbe(
        family="rare_fact", case_id=case_id, prompt_text=prompt, prompt_ids=ids,
        answer_token_id=answer_id, candidate_token_ids=_candidate_ids(values),
        evidence_end_token=evidence_end, query_token=len(ids) - 1,
        evidence_distance=distance,
    )
    probe.validate()
    return probe


def _overwrite(case_id: int, encode: Encode, rng: random.Random, values: list[tuple[str, int]]) -> EncodedProbe:
    entity = _ENTITIES[(case_id * 3) % len(_ENTITIES)]
    stale1, stale1_id = values[(case_id + 1) % 8]
    stale2, stale2_id = values[(case_id + 2) % 8]
    answer, answer_id = values[(case_id + 3) % 8]
    updates = (
        f"Status update: the access word for {entity} is {stale1}. "
        f"Later correction: the access word for {entity} is {stale2}. "
        f"Final authoritative update: the access word for {entity} is {answer}."
    )
    evidence_end = len(_encoded(encode, updates)) - 1
    query = f"Question: According to the latest update, what is the access word for {entity}? Answer:"
    prompt, ids, distance = _pad_until_distance(
        encode, prefix=updates, evidence_end=evidence_end, query=query, rng=rng,
        minimum_distance=LONG_RANGE_MIN_DISTANCE,
    )
    probe = EncodedProbe(
        family="overwrite", case_id=case_id, prompt_text=prompt, prompt_ids=ids,
        answer_token_id=answer_id, candidate_token_ids=_candidate_ids(values),
        evidence_end_token=evidence_end, query_token=len(ids) - 1,
        evidence_distance=distance, stale_token_ids=(stale1_id, stale2_id),
    )
    probe.validate()
    return probe


def _two_hop(case_id: int, encode: Encode, rng: random.Random, values: list[tuple[str, int]]) -> EncodedProbe:
    entity = _ENTITIES[(case_id * 5) % len(_ENTITIES)]
    link = _LINKS[case_id % len(_LINKS)]
    answer, answer_id = values[(case_id + 4) % 8]
    first_fact = f"Routing note: the record associated with {entity} is {link}."
    second_fact = f"Registry note: the access word stored in {link} is {answer}."
    prefix, first_end, second_end = _separate_two_hop_facts(
        encode, first_fact=first_fact, second_fact=second_fact, rng=rng
    )
    query = f"Question: Follow the record associated with {entity}. What access word does that record store? Answer:"
    prompt, ids, distance = _pad_until_distance(
        encode, prefix=prefix, evidence_end=second_end, query=query, rng=rng,
        minimum_distance=LONG_RANGE_MIN_DISTANCE,
    )
    probe = EncodedProbe(
        family="two_hop", case_id=case_id, prompt_text=prompt, prompt_ids=ids,
        answer_token_id=answer_id, candidate_token_ids=_candidate_ids(values),
        first_evidence_end_token=first_end, evidence_end_token=second_end,
        query_token=len(ids) - 1, evidence_distance=distance,
    )
    probe.validate()
    return probe


def _local_negative(case_id: int, encode: Encode, rng: random.Random, values: list[tuple[str, int]]) -> EncodedProbe:
    entity = _ENTITIES[(case_id * 7) % len(_ENTITIES)]
    answer, answer_id = values[(case_id + 5) % 8]
    old_context = _filler(rng, 8)
    fact = f"Current desk note: the access word for {entity} is {answer}."
    query = f"Question: What is the access word for {entity}? Answer:"
    prompt = old_context + " " + fact + " " + query
    ids = _encoded(encode, prompt)
    prefix_before_fact = old_context + " "
    evidence_end = len(_encoded(encode, prefix_before_fact + fact)) - 1
    distance = (len(ids) - 1) - evidence_end
    probe = EncodedProbe(
        family="local_negative", case_id=case_id, prompt_text=prompt, prompt_ids=ids,
        answer_token_id=answer_id, candidate_token_ids=_candidate_ids(values),
        evidence_end_token=evidence_end, query_token=len(ids) - 1,
        evidence_distance=distance,
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
    values = _single_token_values(encode)
    rng = random.Random(seed)
    suite: list[EncodedProbe] = []
    builders = (_rare_fact, _overwrite, _two_hop, _local_negative)
    for family_offset, builder in enumerate(builders):
        for index in range(cases_per_family):
            case_id = family_offset * 100_000 + index
            suite.append(builder(case_id, encode, rng, values))
    return suite


def _candidate_prediction(logits: torch.Tensor, candidate_ids: tuple[int, ...]) -> int:
    candidates = torch.as_tensor(candidate_ids, device=logits.device, dtype=torch.long)
    chosen = int(torch.argmax(logits.index_select(0, candidates)).item())
    return int(candidate_ids[chosen])


@torch.no_grad()
def score_local_probe(model: CHMV1LocalLM, probe: EncodedProbe) -> int:
    """Score only the bounded 512-token working window at the query."""
    device = next(model.parameters()).device
    ids = probe.prompt_ids[-LOCAL_WINDOW:]
    tokens = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
    logits = model(tokens)[0, -1].float()
    return _candidate_prediction(logits, probe.candidate_token_ids)


@torch.no_grad()
def score_eiem_probe(
    model: CHMV1EIEMLM,
    probe: EncodedProbe,
    *,
    mode: Literal["flat", "indexed"],
) -> tuple[int, torch.Tensor, dict[str, object]]:
    """Process the full prompt in causal <=512 chunks with one isolated state."""
    device = next(model.parameters()).device
    state = EpisodicState(f"probe-{probe.family}-{probe.case_id}-{mode}")
    final_logits: torch.Tensor | None = None
    calls = matches = reads = flat_reads = nodes = 0
    build_seconds = search_seconds = verification_seconds = write_seconds = 0.0
    max_state_payload_bytes = 0
    by_memory_size: dict[int, dict[str, float]] = {}
    ids = probe.prompt_ids

    for start in range(0, len(ids), LOCAL_WINDOW):
        memory_before = len(state)
        chunk = torch.tensor(
            ids[start : start + LOCAL_WINDOW], device=device, dtype=torch.long
        ).unsqueeze(0)
        logits, stats = model.forward_session_chunk(
            chunk, [state], mode=mode, update_memory=True,
            verify_indexed_exactness=mode == "indexed",
        )
        final_logits = logits[0, -1].float()
        calls += stats.calls
        matches += stats.exact_matches
        reads += stats.address_vector_reads
        flat_reads += stats.flat_address_vector_reads
        nodes += stats.directory_nodes_visited
        build_seconds += stats.index_build_seconds
        search_seconds += stats.search_seconds
        verification_seconds += stats.verification_seconds
        write_seconds += stats.write_seconds
        max_state_payload_bytes = max(max_state_payload_bytes, stats.state_payload_bytes)

        if stats.calls:
            bucket = by_memory_size.setdefault(
                memory_before,
                {"calls": 0.0, "reads": 0.0, "flat_reads": 0.0, "nodes": 0.0},
            )
            bucket["calls"] += float(stats.calls)
            bucket["reads"] += float(stats.address_vector_reads)
            bucket["flat_reads"] += float(stats.flat_address_vector_reads)
            bucket["nodes"] += float(stats.directory_nodes_visited)

    if final_logits is None:
        raise RuntimeError("probe had no tokens")
    pred = _candidate_prediction(final_logits, probe.candidate_token_ids)

    memory_size_slices: dict[str, dict[str, float]] = {}
    for size, bucket in sorted(by_memory_size.items()):
        memory_size_slices[str(size)] = {
            "retrieval_calls": bucket["calls"],
            "address_vector_reads": bucket["reads"],
            "flat_address_vector_reads": bucket["flat_reads"],
            "address_read_fraction": bucket["reads"] / max(bucket["flat_reads"], 1.0),
            "directory_nodes_per_retrieval": bucket["nodes"] / max(bucket["calls"], 1.0),
        }

    return pred, final_logits, {
        "retrieval_calls": float(calls),
        "exact_matches": float(matches),
        "exact_match_rate": matches / max(calls, 1),
        "address_vector_reads": float(reads),
        "flat_address_vector_reads": float(flat_reads),
        "address_read_fraction": reads / max(flat_reads, 1),
        "directory_nodes_visited": float(nodes),
        "index_build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "verification_seconds": verification_seconds,
        "write_seconds": write_seconds,
        "max_state_payload_bytes": float(max_state_payload_bytes),
        "memory_size_slices": memory_size_slices,
        "sparse_read_gate_memory_size": SPARSE_READ_GATE_MEMORY_SIZE,
    }


def summarize_probe_predictions(
    probes: Sequence[EncodedProbe],
    local_predictions: Sequence[int],
    eiem_predictions: Sequence[int],
) -> dict[str, dict[str, float]]:
    if not (len(probes) == len(local_predictions) == len(eiem_predictions)):
        raise ValueError("probe/prediction lengths differ")
    out: dict[str, dict[str, float]] = {}
    for family in ("rare_fact", "overwrite", "two_hop", "local_negative"):
        rows = [i for i, probe in enumerate(probes) if probe.family == family]
        if not rows:
            continue
        local_correct = sum(local_predictions[i] == probes[i].answer_token_id for i in rows)
        eiem_correct = sum(eiem_predictions[i] == probes[i].answer_token_id for i in rows)
        stale_total = stale_hits = 0
        if family == "overwrite":
            for i in rows:
                stale_total += 1
                stale_hits += int(eiem_predictions[i] in probes[i].stale_token_ids)
        out[family] = {
            "cases": float(len(rows)),
            "local_accuracy": local_correct / len(rows),
            "eiem_accuracy": eiem_correct / len(rows),
            "absolute_gain": (eiem_correct - local_correct) / len(rows),
            "eiem_stale_error": stale_hits / max(stale_total, 1),
            "minimum_evidence_distance": float(min(probes[i].evidence_distance for i in rows)),
        }
    return out
