from __future__ import annotations

"""#1407 TRAIN-only audit: model-visible query-anchored local identifier aliases.

No trained model, no dev/test, no old heldout, no GPU or Modal. This module
builds tokens using ONLY SealedModelView and literal text equality, never
oracle subject roles, gold pointer or target-answer metadata. It does NOT
modify the existing Transformer/pointer tokenizer or use it for inference.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib

from .chm_v3_identifier_composition_audit_1399 import (
    ID_SURFACE, SOURCE_ANCHOR, CODE_SURFACE, encode_visible_only,
)
from .chm_v3_matched_tiny_models_1377 import (
    TOKEN_PATTERN, _token_id, MAX_TOKENS, encode_sealed_view,
)
from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from .chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, paired_group,
)
from .chm_v3_counterfactual_model_view_1365 import (
    SealedModelView, seal_model_view,
)
from .chm_v3_balanced_train_schedule_1391 import TRAIN_STEPS, balanced_training_episode

MAX_ALIASES = 32
QUERY_ANCHOR = "ID_ALIAS_Q"
GPU_AUTHORIZED = False
NEW_SCIENTIFIC_ATTEMPT = False
ORIGINAL_100M_SCIENTIFIC_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
OLD_SCIENTIFIC_SEED_CONSUMED = 2_013_161
ORIGINAL_BALANCED_SCORED_TEST_INDEX_CONSUMED = 2
ORIGINAL_BALANCED_SCORED_DEV_INDEX_CONSUMED = 1


@dataclass(frozen=True)
class AliasEncoding:
    """A text-derived input representation. Alias dictionary is audit ONLY."""
    token_ids: tuple[int, ...]
    surfaces: tuple[str, ...]
    memory_length: int
    line_anchors: tuple[tuple[int, int], ...]
    code_tokens: tuple[tuple[int, int], ...]
    # Exact original strings and alias names are NOT model metadata.
    alias_dictionary_evaluator_only: tuple[tuple[str, str], ...]


def alias_sealed_input(view: SealedModelView) -> AliasEncoding:
    if not isinstance(view, SealedModelView):
        raise TypeError("only sealed model input accepted")
    if view.answer_options != tuple(f"CODE-{x}" for x in ANSWER_IDS):
        raise ValueError("answer options differ from frozen candidate set")
    if not view.query or not view.memory_text or len(view.memory_text) > 12000:
        raise ValueError("invalid model-visible text")

    query_parts = TOKEN_PATTERN.findall(view.query)
    distinct_query_ids = sorted({s for s in query_parts if ID_SURFACE.fullmatch(s)})
    if len(distinct_query_ids) != 1:
        raise ValueError("one and only one query identifier is required")
    query_id = distinct_query_ids[0]

    # Query equality anchoring is derived ONLY from visible text, not oracle.
    alias: dict[str, str] = {query_id: QUERY_ANCHOR}
    source_lines: list[list[str]] = []
    original_positions: list[int] = []
    for raw_line in view.memory_text.splitlines():
        parts = TOKEN_PATTERN.findall(raw_line)
        if not parts:
            continue
        match = SOURCE_ANCHOR.fullmatch(parts[0])
        if match is None:
            raise ValueError("malformed memory source position")
        position = int(match.group(1))
        if position in original_positions:
            raise ValueError("duplicate memory source position")
        original_positions.append(position)
        source_lines.append(parts)
        for part in parts:
            if ID_SURFACE.fullmatch(part) and part not in alias:
                if len(alias) >= MAX_ALIASES:
                    raise ValueError("identifier alias table capacity exceeded")
                alias[part] = f"ID_ALIAS_{len(alias):03d}"
    if not source_lines:
        raise ValueError("missing completed memory")

    model_tokens: list[str] = []
    anchors: list[tuple[int, int]] = []
    codes: list[tuple[int, int]] = []
    for position, parts in zip(original_positions, source_lines):
        anchors.append((position, len(model_tokens)))
        for part in parts:
            new = alias.get(part, part)
            if CODE_SURFACE.fullmatch(part):
                codes.append((len(model_tokens), int(part[5:])))
            model_tokens.append(new)
    memory_length = len(model_tokens)
    model_tokens.append("QUESTION")
    for part in query_parts:
        model_tokens.append(alias.get(part, part))
    model_tokens.append("ANSWER")
    if not codes or len(model_tokens) > MAX_TOKENS:
        raise ValueError("alias input has no codes or violates context budget")

    result = AliasEncoding(
        token_ids=tuple(_token_id(s) for s in model_tokens),
        surfaces=tuple(model_tokens), memory_length=memory_length,
        line_anchors=tuple(anchors), code_tokens=tuple(codes),
        alias_dictionary_evaluator_only=tuple(sorted(alias.items())),
    )
    original = encode_sealed_view(view)
    if (result.memory_length != original.memory_length
        or result.line_anchors != original.line_anchors
        or result.code_tokens != original.code_tokens
        or len(result.token_ids) != len(original.token_ids)):
        raise RuntimeError("source position/code/boundary parity failure")
    # Boundaries and CODE payloads are exactly source-preserving.
    if result.surfaces[memory_length] != "QUESTION":
        raise RuntimeError("memory/query boundary lost")
    if any(result.surfaces[i] != f"CODE-{code}" for i, code in codes):
        raise RuntimeError("code payload changed")
    if len(set(result.token_ids[i] for i,s in enumerate(result.surfaces)
               if s.startswith("ID_ALIAS_"))) != len({
                   s for s in result.surfaces if s.startswith("ID_ALIAS_")
               }):
        raise RuntimeError("distinct aliases collided in SHA hash bucket")
    return result


def validate_train_pair_group(family: str, entity: int) -> dict[str, object]:
    """Only 8 TRAIN cases, no heldout and no trained model calls."""
    if family not in FAMILIES or type(entity) is not int or not 0 <= entity < 8:
        raise ValueError("invalid TRAIN group")
    group = paired_group("train", family, entity)
    enc = [alias_sealed_input(seal_model_view(ep)) for ep in group]
    one = enc[0]
    if len(enc) != 8:
        raise RuntimeError("not eight paired cases")
    if any(e.alias_dictionary_evaluator_only != one.alias_dictionary_evaluator_only
           or e.line_anchors != one.line_anchors
           or e.memory_length != one.memory_length for e in enc):
        raise RuntimeError("alias mapping or memory layout changed across rotations")
    if any(sorted(code for _,code in e.code_tokens) !=
           sorted(code for _,code in one.code_tokens) for e in enc):
        raise RuntimeError("counterfactual answer-code bag changed")
    baseline_surfaces = one.surfaces
    variant_token_positions = []
    for i,e in enumerate(enc):
        diff = [p for p,(x,y) in enumerate(zip(baseline_surfaces,e.surfaces)) if x != y]
        if len(e.surfaces) != len(baseline_surfaces):
            raise RuntimeError("counterfactual input token count differs")
        if family in POSITIVES:
            if len(diff) != (0 if i == 0 else 2):
                raise RuntimeError("positive should swap exactly 2 candidate-code values")
            if any(not CODE_SURFACE.fullmatch(e.surfaces[p]) for p in diff):
                raise RuntimeError("non-answer canonicalized token changed")
        elif diff:
            raise RuntimeError("negative no-match paired input unexpectedly changed")
        variant_token_positions.append(diff)
    return {
        "family": family, "entity": entity, "variants": 8,
        "positive_labels_balanced": (
            set(ep.gold_answer for ep in group) == set(ANSWER_IDS)
            if family in POSITIVES else all(ep.gold_answer is None for ep in group)
        ),
        "identical_alias_mapping_across_counterfactuals": True,
        "same_query_and_layout": True,
        "max_swapped_code_token_positions": max(map(len,variant_token_positions)),
        "original_token_budget_parity": True,
    }


def training_only_static_report() -> dict[str, object]:
    """Never generate or inspect previous heldout development/test entities."""
    sizes: list[int] = []
    alias_cardinality: list[int] = []
    unique_alias_tokens: set[str] = set()
    for step in range(TRAIN_STEPS):
        ep = balanced_training_episode(step)
        if ep.split != "train":
            raise RuntimeError("non-training episode in TRAIN-only audit")
        view = seal_model_view(ep)
        local = alias_sealed_input(view)
        old = encode_visible_only(view, mode="whole")
        if len(local.token_ids) != len(old.token_ids):
            raise RuntimeError("alias budget differs from old hash budget")
        sizes.append(len(local.token_ids))
        alias_cardinality.append(len(local.alias_dictionary_evaluator_only))
        unique_alias_tokens.update(
            s for s in local.surfaces if s.startswith("ID_ALIAS_")
        )
    groups = [
        validate_train_pair_group(fam, ent)
        for fam in FAMILIES for ent in range(8)
    ]
    if len(sizes) != 256 or len(groups) != 32:
        raise RuntimeError("TRAIN-only audit incomplete")
    if not all(g["positive_labels_balanced"] for g in groups):
        raise RuntimeError("training answer balance failed")
    return {
        "classification": "CHM_V3_1407_QUERY_ANCHORED_ALIAS_STATIC_TRAIN_ONLY",
        "old_100m_scientific_status": ORIGINAL_100M_SCIENTIFIC_STOP,
        "old_scientific_seed_consumed": OLD_SCIENTIFIC_SEED_CONSUMED,
        "scored_heldout_inputs_constructed": False,
        "model_forward_calls": 0, "optimizer_updates": 0,
        "training_episodes": len(sizes), "counterfactual_groups": len(groups),
        "original_tokens_sum": sum(sizes),
        "alias_tokens_sum": sum(sizes),
        "original_max_tokens": max(sizes),
        "alias_max_tokens": max(sizes),
        "alias_cardinality_min": min(alias_cardinality),
        "alias_cardinality_max": max(alias_cardinality),
        "distinct_alias_symbols": len(unique_alias_tokens),
        "alias_mapping_invariant_groups": len(groups),
        "gpu_used": False, "new_scientific_attempt": False,
        "interpretation": (
            "Query-anchored local aliases preserve token count and text-derived "
            "identity equality; this introduces a shared inductive bias for "
            "all future model arms. It does NOT show any learning, generalization, "
            "new retrieval, natural-language understanding or breakthrough."
        ),
    }


__all__ = (
    "AliasEncoding", "alias_sealed_input", "validate_train_pair_group",
    "training_only_static_report", "QUERY_ANCHOR",
)
