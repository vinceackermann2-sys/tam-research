from __future__ import annotations

"""#1399 immutable TRAIN-only static identifier tokenizer checks."""

from collections import Counter
import inspect
import math

import pytest

import tam_research.chm_v3_identifier_composition_audit_1399 as audit
from tam_research.chm_v3_balanced_train_schedule_1391 import balanced_training_episode
from tam_research.chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from tam_research.chm_v3_counterfactual_model_view_1365 import (
    SealedModelView, seal_model_view,
)
from tam_research.chm_v3_matched_tiny_models_1377 import encode_sealed_view


def test_exact_old_tokenization_and_unchanged_copy_payload_coordinates():
    for step in (0, 1, 31, 128, 255):
        ep = balanced_training_episode(step)
        assert ep.split == "train"
        view = seal_model_view(ep)
        legacy = audit.encode_visible_only(view, mode="whole")
        character = audit.encode_visible_only(view, mode="character")
        old = encode_sealed_view(view)
        assert legacy.token_ids == old.token_ids
        assert legacy.memory_length == old.memory_length
        assert legacy.line_anchors == old.line_anchors
        assert legacy.code_tokens == old.code_tokens
        evidence = audit.verify_structural_parity(view, legacy, character)
        assert all(evidence.values())
        assert len(character.token_ids) > len(legacy.token_ids)
        assert tuple(v for _, v in character.code_tokens) == tuple(v for _, v in old.code_tokens)
        assert character.surface_tokens[character.memory_length] == "QUESTION"


def test_synthetic_only_repeated_ids_are_equal_and_non_aliasing():
    a = "V3-hypothetical-E900"
    b = "V3-hypothetical-E901"
    v = SealedModelView(
        query=f"What is the current access code for {a}?",
        memory_text=(
            f"[0001] Access code for {a} is CODE-18001.\n"
            f"[0007] Access code for {b} is CODE-18002."
        ),
        answer_options=tuple(f"CODE-{x}" for x in ANSWER_IDS),
    )
    x = audit.encode_visible_only(v, mode="whole")
    y = audit.encode_visible_only(v, mode="character")
    assert audit.verify_structural_parity(v, x, y) == {
        "original_encoder_exact_parity": True,
        "atomic_copy_codes_unchanged": True,
        "memory_anchor_positions_unchanged": True,
        "query_memory_identifier_repeatability": True,
    }
    for enc in (x, y):
        seqs = {}
        for section, surface, start, end in enc.identifier_spans:
            seqs.setdefault(surface, set()).add(enc.token_ids[start:end])
        assert all(len(s) == 1 for s in seqs.values())
        assert seqs[a] != seqs[b]
        assert Counter(s for _, s, _, _ in enc.identifier_spans)[a] == 2
    assert y.surface_tokens[y.code_tokens[0][0]] == "CODE-18001"
    assert y.surface_tokens[y.code_tokens[1][0]] == "CODE-18002"


def test_train_only_audit_256_episodes_and_no_score_or_gpu():
    result = audit.audit_training_inputs_only()
    assert result["classification"] == audit.CLASSIFICATION
    assert result["historical_science_status"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert result["historical_science_seed_consumed"] == 2013161
    assert result["training_episodes_audited"] == 256
    assert sum(v["count"] for v in result["per_family"].values()) == 256
    assert all(v["count"] == 64 for v in result["per_family"].values())
    assert result["original_input_max_tokens"] <= audit.MAX_TOKENS
    assert result["character_input_max_tokens"] >= result["original_input_max_tokens"]
    assert result["character_budget_overflow_episodes"] == sum(
        v["character_budget_overflows"] for v in result["per_family"].values()
    )
    assert result["unique_training_identifier_surfaces"] > 16
    assert result["legacy_encoder_exact_parity_for_all_train"] is True
    assert result["payload_anchor_and_memory_query_boundaries_preserved"] is True
    assert result["gpu_used"] is False
    assert result["models_trained"] is False
    assert result["new_scientific_attempt"] is False
    assert len(result["sha256_before_digest_field"]) == 64
    assert len(result["hypothetical_identifier_coverage"]) == len(audit.SYNTHETIC_PROBE_NAMES)
    assert all(not p["whole_token_seen_in_training"]
               and 0 <= p["character_piece_coverage"] <= 1
               for p in result["hypothetical_identifier_coverage"])
    assert audit.audit_training_inputs_only() == result


def test_fail_closed_bad_view_or_mode_and_invalid_identity():
    options = tuple(f"CODE-{x}" for x in ANSWER_IDS)
    v = SealedModelView("Q V3-synthetic-E001?", "[0000] Access code for V3-synthetic-E001 is CODE-18001.", options)
    with pytest.raises(ValueError):
        audit.encode_visible_only(v, mode="invalid")
    with pytest.raises(ValueError):
        audit.pieces_for_identifier("not-a-V3-id", mode="character")
    with pytest.raises(ValueError):
        audit.encode_visible_only(SealedModelView(v.query, v.memory_text, ("CODE-1",)), mode="whole")


def test_source_never_scores_retrains_or_reads_old_heldout():
    source = inspect.getsource(audit)
    assert audit.GPU_AUTHORIZED is False and audit.NEW_SCIENTIFIC_ATTEMPT is False
    assert audit.MODELS_TRAINED is False
    for forbidden in (
        "heldout_cases(", "predict_from_sealed_view(", "train_balanced_cpu_models(",
        "torch.optim", "torch.cuda", "modal.App(", "torch.load(",
        "generate_episode(\"test\"", "generate_episode(\"development\"",
    ):
        assert forbidden not in source
    assert 'episode.split != "train"' in source
