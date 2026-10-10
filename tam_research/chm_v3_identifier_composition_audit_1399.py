from __future__ import annotations

"""#1399: TRAIN-only static audit of frozen whole-ID hashing vs ID characters.

No trained-model forward(), optimizer, old heldout episode construction,
checkpoint replay, scoring, CUDA, Modal, or new scientific attempt.
New characters are a REPRESENTATION PROPOSAL, not a demonstrated improvement.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re

from .chm_v3_matched_tiny_models_1377 import (
    TOKEN_PATTERN, MAX_TOKENS, _token_id, encode_sealed_view,
)
from .chm_v3_balanced_train_schedule_1391 import (
    TRAIN_STEPS, balanced_training_episode,
)
from .chm_v3_counterfactual_model_view_1365 import (
    SealedModelView, seal_model_view,
)
from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS

CLASSIFICATION = "CHM_V3_1399_TRAIN_ONLY_STATIC_IDENTIFIER_AUDIT"
ORIGINAL_SCIENCE_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
ORIGINAL_SCIENCE_SEED_CONSUMED = 2013161
GPU_AUTHORIZED = False
NEW_SCIENTIFIC_ATTEMPT = False
MODELS_TRAINED = False
FROZEN_MODEL_BLOB = "ffe14b0701e18493a2bf9b45a1238b5acd5e3eab"
FROZEN_SCHEDULER_BLOB = "6d40dafc7a487cfc8d5a39d5067ff00190e60181"
FROZEN_SEALED_VIEW_BLOB = "5bce89954d7a4fd788d1d1bdd972abf744399715"
FROZEN_BENCHMARK_BLOB = "1b71eede754f7396ad2e7284693d6d5c1b8273b4"
ID_SURFACE = re.compile(r"V3-[A-Za-z0-9-]+\Z")
SOURCE_ANCHOR = re.compile(r"\[([0-9]{4})\]\Z")
CODE_SURFACE = re.compile(r"CODE-([0-9]+)\Z")
# Synthetically invented names only: never use reserved original development/test
# entity indices, prompts or stored scored examples in this audit.
SYNTHETIC_PROBE_NAMES = (
    "V3-train-E900",
    "V3-train-D900",
    "V3-hypothetical-E900",
    "V3-hypothetical-D900",
    "V3-hypothetical-X900-D0",
)


@dataclass(frozen=True)
class StaticEncoding:
    token_ids: tuple[int, ...]
    surface_tokens: tuple[str, ...]
    memory_length: int
    line_anchors: tuple[tuple[int, int], ...]
    code_tokens: tuple[tuple[int, int], ...]
    # Section, exact visible identifier, start token, end token (exclusive).
    identifier_spans: tuple[tuple[str, str, int, int], ...]


def pieces_for_identifier(surface: str, *, mode: str) -> tuple[str, ...]:
    """Tagged chars preserve exact lexical identity, with explicit boundaries."""
    if not ID_SURFACE.fullmatch(surface):
        raise ValueError("not a valid V3 identifier")
    if mode == "whole":
        return (surface,)
    if mode == "character":
        return ("ID_OPEN", *(f"IDCHAR:{c}" for c in surface), "ID_CLOSE")
    raise ValueError("unknown identifier mode")


def _emit_piece(
    part: str, section: str, mode: str,
    pieces: list[str],
    identifiers: list[tuple[str, str, int, int]],
) -> None:
    start = len(pieces)
    if ID_SURFACE.fullmatch(part):
        pieces.extend(pieces_for_identifier(part, mode=mode))
        identifiers.append((section, part, start, len(pieces)))
    else:
        pieces.append(part)


def encode_visible_only(view: SealedModelView, *, mode: str) -> StaticEncoding:
    """Pure tokenizer. No oracle, answer, split metadata or labels."""
    if not isinstance(view, SealedModelView):
        raise TypeError("sealed model view required")
    if mode not in ("whole", "character"):
        raise ValueError("invalid tokenization mode")
    if view.answer_options != tuple(f"CODE-{v}" for v in ANSWER_IDS):
        raise ValueError("answer options drift")
    if not view.query or not view.memory_text or len(view.memory_text) > 12000:
        raise ValueError("invalid model-visible text")

    tokens: list[str] = []
    anchors: list[tuple[int, int]] = []
    codes: list[tuple[int, int]] = []
    identifiers: list[tuple[str, str, int, int]] = []

    for line in view.memory_text.splitlines():
        parts = TOKEN_PATTERN.findall(line)
        if not parts:
            continue
        match = SOURCE_ANCHOR.fullmatch(parts[0])
        if match is None:
            raise ValueError("malformed completed-memory source anchor")
        pos = int(match.group(1))
        if any(a == pos for a, _ in anchors):
            raise ValueError("duplicate completed-memory source anchor")
        anchors.append((pos, len(tokens)))
        for part in parts:
            start = len(tokens)
            _emit_piece(part, "memory", mode, tokens, identifiers)
            code = CODE_SURFACE.fullmatch(part)
            if code is not None:
                codes.append((start, int(code.group(1))))
    boundary = len(tokens)
    if not anchors or not codes:
        raise ValueError("no source records or copyable code payloads")
    tokens.append("QUESTION")
    for part in TOKEN_PATTERN.findall(view.query):
        _emit_piece(part, "query", mode, tokens, identifiers)
    tokens.append("ANSWER")
    if boundary >= len(tokens):
        raise ValueError("missing question token")
    return StaticEncoding(
        token_ids=tuple(_token_id(piece) for piece in tokens),
        surface_tokens=tuple(tokens),
        memory_length=boundary,
        line_anchors=tuple(anchors),
        code_tokens=tuple(codes),
        identifier_spans=tuple(identifiers),
    )


def _identifier_group(enc: StaticEncoding) -> dict[str, list[tuple[str, tuple[int, ...]]]]:
    groups: dict[str, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for section, surface, start, end in enc.identifier_spans:
        groups[surface].append((section, enc.token_ids[start:end]))
    return groups


def verify_structural_parity(
    view: SealedModelView, whole: StaticEncoding, chars: StaticEncoding,
) -> dict[str, bool]:
    original = encode_sealed_view(view)
    if (whole.token_ids != original.token_ids or
        whole.memory_length != original.memory_length or
        whole.line_anchors != original.line_anchors or
        whole.code_tokens != original.code_tokens):
        raise RuntimeError("frozen legacy tokenizer parity FAILED")
    if (len(whole.line_anchors) != len(chars.line_anchors) or
        [x[0] for x in whole.line_anchors] != [x[0] for x in chars.line_anchors]):
        raise RuntimeError("memory source positions changed")
    if [x[1] for x in whole.code_tokens] != [x[1] for x in chars.code_tokens]:
        raise RuntimeError("copyable code values or ordering changed")
    if any(chars.surface_tokens[i] != f"CODE-{code}" for i, code in chars.code_tokens):
        raise RuntimeError("a code token ceased being atomic")
    if (any(chars.surface_tokens[i] != f"[{position:04d}]" for position, i in chars.line_anchors)
        or any(i >= chars.memory_length for _, i in chars.line_anchors)
        or any(i >= chars.memory_length for i, _ in chars.code_tokens)):
        raise RuntimeError("memory boundary or atomic source anchor changed")
    for mode, enc in (("whole", whole), ("character", chars)):
        groups = _identifier_group(enc)
        for surface, occurrences in groups.items():
            expected = tuple(_token_id(x) for x in pieces_for_identifier(surface, mode=mode))
            if any(seq != expected for _, seq in occurrences):
                raise RuntimeError("memory/query identifier encoding disagreement")
        if not enc.surface_tokens[enc.memory_length] == "QUESTION":
            raise RuntimeError("query leaked into stored memory")
    return {
        "original_encoder_exact_parity": True,
        "atomic_copy_codes_unchanged": True,
        "memory_anchor_positions_unchanged": True,
        "query_memory_identifier_repeatability": True,
    }


def _digest(v: object) -> str:
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def audit_training_inputs_only() -> dict[str, object]:
    """No trained model and no old development/test data even instantiated."""
    whole_lengths: list[int] = []
    char_lengths: list[int] = []
    per_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "original_max_tokens": 0,
                 "character_max_tokens": 0, "character_budget_overflows": 0}
    )
    training_identifiers: set[str] = set()
    for step in range(TRAIN_STEPS):
        episode = balanced_training_episode(step)
        if episode.split != "train":
            raise RuntimeError("non-training episode reached static audit")
        view = seal_model_view(episode)
        a = encode_visible_only(view, mode="whole")
        b = encode_visible_only(view, mode="character")
        verify_structural_parity(view, a, b)
        whole_lengths.append(len(a.token_ids))
        char_lengths.append(len(b.token_ids))
        training_identifiers.update(span[1] for span in a.identifier_spans)
        cell = per_family[episode.family]
        cell["count"] += 1
        cell["original_max_tokens"] = max(cell["original_max_tokens"], len(a.token_ids))
        cell["character_max_tokens"] = max(cell["character_max_tokens"], len(b.token_ids))
        cell["character_budget_overflows"] += len(b.token_ids) > MAX_TOKENS

    if len(whole_lengths) != TRAIN_STEPS:
        raise RuntimeError("TRAIN-only sample count changed")
    whole_hashes = [_token_id(s) for s in sorted(training_identifiers)]
    char_sequences = {
        s: tuple(_token_id(t) for t in pieces_for_identifier(s, mode="character"))
        for s in sorted(training_identifiers)
    }
    if len(set(char_sequences.values())) != len(char_sequences):
        raise RuntimeError("character identifier alias detected on TRAIN inputs")
    seen_atoms = set(
        atom for identifier in training_identifiers
        for atom in pieces_for_identifier(identifier, mode="character")
    )
    novel_probe = []
    for name in SYNTHETIC_PROBE_NAMES:
        if name in training_identifiers:
            raise RuntimeError("hypothetical probe overlaps original training IDs")
        atoms = pieces_for_identifier(name, mode="character")
        novel_probe.append({
            "identifier": name,
            "whole_token_seen_in_training": name in training_identifiers,
            "character_piece_count": len(atoms),
            "character_pieces_seen_in_training": sum(x in seen_atoms for x in atoms),
            "character_piece_coverage": sum(x in seen_atoms for x in atoms) / len(atoms),
        })
    probe_hashes = [
        tuple(_token_id(t) for t in pieces_for_identifier(s, mode="character"))
        for s in SYNTHETIC_PROBE_NAMES
    ]
    if len(set(probe_hashes)) != len(probe_hashes):
        raise RuntimeError("distinct hypothetical identities aliased")
    report: dict[str, object] = {
        "classification": CLASSIFICATION,
        "historical_science_status": ORIGINAL_SCIENCE_STOP,
        "historical_science_seed_consumed": ORIGINAL_SCIENCE_SEED_CONSUMED,
        "frozen_model_blob": FROZEN_MODEL_BLOB,
        "frozen_scheduler_blob": FROZEN_SCHEDULER_BLOB,
        "input_domain": "TRAIN-only; invented synthetic names, no scored heldout",
        "training_episodes_audited": TRAIN_STEPS,
        "unique_training_identifier_surfaces": len(training_identifiers),
        "whole_identifier_hash_collision_count": len(whole_hashes) - len(set(whole_hashes)),
        "distinct_character_identifier_sequences": len(char_sequences),
        "training_character_atom_inventory_size": len(seen_atoms),
        "input_token_cap": MAX_TOKENS,
        "original_input_max_tokens": max(whole_lengths),
        "character_input_max_tokens": max(char_lengths),
        "original_input_min_tokens": min(whole_lengths),
        "character_input_min_tokens": min(char_lengths),
        "original_token_count_total": sum(whole_lengths),
        "character_token_count_total": sum(char_lengths),
        "character_budget_overflow_episodes": sum(x > MAX_TOKENS for x in char_lengths),
        "per_family": dict(sorted(per_family.items())),
        "hypothetical_identifier_coverage": novel_probe,
        "legacy_encoder_exact_parity_for_all_train": True,
        "payload_anchor_and_memory_query_boundaries_preserved": True,
        "gpu_used": False,
        "models_trained": False,
        "new_scientific_attempt": False,
        "interpretation": (
            "Static lexical/token-budget evidence only. Hashing complete names causes "
            "unseen embedding rows, subject to collisions; it does not prove a causal "
            "reason for prior learned-model failure. Character expansion may exceed "
            "the frozen token budget; this is a proposal, never silently truncate."
        ),
    }
    report["sha256_before_digest_field"] = _digest(report)
    return report


__all__ = (
    "StaticEncoding", "pieces_for_identifier", "encode_visible_only",
    "verify_structural_parity", "audit_training_inputs_only",
    "SYNTHETIC_PROBE_NAMES", "CLASSIFICATION", "MAX_TOKENS",
)
