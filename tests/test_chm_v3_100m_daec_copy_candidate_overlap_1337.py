from __future__ import annotations

import json
from pathlib import Path
import math

import pytest

from tam_research.chm_v3_100m_daec_copy_candidate_overlap import (
    ARCHIVED_JOB_ID,
    ARCHIVED_RUN_ID,
    ARCHIVED_SCIENTIFIC_SEED,
    FROZEN_COPY_TOKEN_COUNTS,
    REFERENCE_GPT2_CANDIDATES,
    analyze_archive,
    eight_way_candidate_nll,
    extract_archived_result,
    frozen_candidate_ids,
)
from tam_research.chm_v1_long_memory_eval import _VALUES


# Exact sparse GPT-2 public vocabulary entries, not model-generated guesses.
# See https://github.com/SungjoonPark/OpenAI-GPT2/blob/master/gpt2/models/117M/encoder.json
def reference_gpt2_sparse_encode(text: str) -> list[int]:
    for word, token_id in REFERENCE_GPT2_CANDIDATES:
        if text == " " + word:
            return [token_id]
    return [8, 9]


def test_public_gpt2_vocab_candidate_ids_have_zero_archived_copy_overlap() -> None:
    names_and_ids = frozen_candidate_ids(reference_gpt2_sparse_encode)
    assert names_and_ids == REFERENCE_GPT2_CANDIDATES
    assert names_and_ids == (
        ("amber", 36505), ("delta", 25979), ("granite", 41013),
        ("mosaic", 47076), ("north", 5093), ("pearl", 43836),
        ("quartz", 47969), ("river", 7850),
    )
    candidates = {token for _, token in names_and_ids}
    assert candidates.isdisjoint(FROZEN_COPY_TOKEN_COUNTS)
    report = analyze_archive(frozen_synthetic_archive(), reference_gpt2_sparse_encode)
    assert report["copies_inside_eight_candidates"] == 0
    assert report["copies_outside_eight_candidates"] == 384


def frozen_toy_encode(text: str) -> list[int]:
    words = _VALUES[:8]
    for index, value in enumerate(words):
        if text == " " + value:
            return [1000 + index]
    return [8, 9]


def frozen_synthetic_archive() -> dict:
    copies = []
    for token_id, count in FROZEN_COPY_TOKEN_COUNTS.items():
        copies.extend([token_id] * count)
    assert len(copies) == 384
    rows = []
    for family_idx, family in enumerate(("rare_fact", "overwrite", "two_hop", "local_negative")):
        for i in range(128):
            copied = None if family == "local_negative" else copies[family_idx * 128 + i]
            rows.append({
                "family": family,
                "case_id": family_idx * 100_000 + i,
                "daec_trace": {
                    "memory_count": 0 if copied is None else (1024 if family == "two_hop" else 512),
                    "copied_token_id": copied,
                },
                "daec_copied_token_id": copied,
                "daec_copy_matches_answer": False,
            })
    return {
        "source_sha": "eb4f0e36ee6c4095e550f26cb7a52c20c5116aa9",
        "scientific_seed": 2013161,
        "scientific_seed_consumed": True,
        "classification": "CHM_V3_100M_DAEC_STAGE_C_STOP",
        "passed": False,
        "status": "COMPLETE",
        "integrity": {
            "probe_count": 512,
            "training_tokens_per_model": 33554432,
            "optimizer_steps_per_model": 2048,
        },
        "probe_rows": rows,
    }


def test_candidate_set_is_derived_from_frozen_generator_values() -> None:
    candidate_pairs = frozen_candidate_ids(frozen_toy_encode)
    assert candidate_pairs == tuple((word, 1000 + idx) for idx,word in enumerate(_VALUES[:8]))


def test_actual_archived_stage_c_copy_rows_have_zero_candidate_overlap() -> None:
    # This subset was parsed from frozen scientific run 37760374218,
    # job 113255052797, RESULT.json (not regenerated or retrained).
    archived_path = (
        Path(__file__).resolve().parent
        / "fixtures/chm_v3_1337_archived_copy_rows.json"
    )
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    assert archived["evidence_kind"] == "READ_ONLY_SUBSET_OF_AUTHORITATIVE_STAGE_C_RESULT_JSON"
    assert archived["archived_run_id"] == 37760374218
    assert archived["archived_job_id"] == 113255052797
    assert archived["terminal_comment_id"] == 6058454735
    assert len(archived["probe_rows"]) == 512
    assert "hard_flat_evaluation_parity_verified" not in archived["integrity"]

    report = analyze_archive(archived, reference_gpt2_sparse_encode)
    assert report["copies_inside_eight_candidates"] == 0
    assert report["copies_outside_eight_candidates"] == 384
    assert report["copy_hits_on_authoritative_answers"] == 0
    assert report["scientific_seed_consumed"] is True
    assert report["scientific_classification_unchanged"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert report["historical_parity_proof_status"] == "missing"
    assert [x["token_id"] for x in report["gpt2_candidates"]] == [
        36505, 25979, 41013, 47076, 5093, 43836, 47969, 7850,
    ]
    assert {
        x["token_id"]: x["count"] for x in report["archived_copied_token_counts"]
    } == FROZEN_COPY_TOKEN_COUNTS
    assert report["per_family"]["local_negative"]["memory_probe_count"] == 0
    for family in ("rare_fact", "overwrite", "two_hop"):
        assert report["per_family"][family]["memory_probe_count"] == 128
        assert report["per_family"][family]["copied_candidate_count"] == 0


def test_outside_candidate_copy_mass_cancels_from_eight_way_normalization() -> None:
    candidates = [0.021,0.055,0.032,0.011,0.05,0.03,0.091,0.040]
    base = eight_way_candidate_nll(candidates,correct_index=5,gate=0,copied_candidate_index=None)
    for gate in (0.01,0.25,0.5,0.95):
        changed = eight_way_candidate_nll(
            candidates, correct_index=5, gate=gate, copied_candidate_index=None,
        )
        assert changed == pytest.approx(base,abs=1e-12)
    wrong = eight_way_candidate_nll(candidates,correct_index=5,gate=.5,copied_candidate_index=2)
    correct = eight_way_candidate_nll(candidates,correct_index=5,gate=.5,copied_candidate_index=5)
    assert wrong > base > correct


def test_result_parser_refuses_missing_and_duplicate_markers() -> None:
    original = frozen_synthetic_archive()
    line = "2026-10-08Z prefix CHM_V3_100M_DAEC_STAGE_C_RESULT=" + json.dumps(original)
    assert extract_archived_result(line)["scientific_seed"] == ARCHIVED_SCIENTIFIC_SEED
    with pytest.raises(ValueError,match="exactly one"):
        extract_archived_result("")
    with pytest.raises(ValueError,match="exactly one"):
        extract_archived_result(line+"\n"+line)


def test_frozen_archive_candidate_join_is_read_only_and_provenance_bound() -> None:
    original = frozen_synthetic_archive()
    prior = json.dumps(original,sort_keys=True)
    report = analyze_archive(original,reference_gpt2_sparse_encode)
    assert json.dumps(original,sort_keys=True) == prior
    assert report["archived_run_id"] == ARCHIVED_RUN_ID
    assert report["archived_job_id"] == ARCHIVED_JOB_ID
    assert report["scientific_classification_unchanged"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert report["scientific_seed_consumed"] is True
    assert report["new_scientific_attempt"] is False
    assert report["gpu_used"] is False
    assert report["copies_inside_eight_candidates"] == 0
    assert report["copies_outside_eight_candidates"] == 384
    assert report["copy_hits_on_authoritative_answers"] == 0
    assert sum(x["memory_probe_count"] for x in report["per_family"].values()) == 384
    assert report["historical_parity_proof_status"] == "missing"
    assert len(report["report_sha256"]) == 64


def test_candidate_join_detects_bad_scientific_provenance_or_copy_distribution() -> None:
    original = frozen_synthetic_archive()
    for field,value in (
        ("source_sha", "0"*40),
        ("scientific_seed", 2013162),
        ("scientific_seed_consumed", False),
        ("classification", "CHM_V3_100M_DAEC_POSITIVE_DEVELOPMENT_SCREEN"),
        ("passed", True),
    ):
        with pytest.raises(ValueError):
            analyze_archive({**original, field:value},frozen_toy_encode)
    bad = frozen_synthetic_archive()
    bad["probe_rows"][0]["daec_copied_token_id"] = 111
    with pytest.raises(ValueError,match="frequencies"):
        analyze_archive(bad,frozen_toy_encode)
    bad = frozen_synthetic_archive()
    bad["integrity"]["hard_flat_evaluation_parity_verified"] = True
    with pytest.raises(ValueError,match="absence"):
        analyze_archive(bad,frozen_toy_encode)


def test_only_cpu_forensic_surface_no_training_or_gpu() -> None:
    import inspect
    import tam_research.chm_v3_100m_daec_copy_candidate_overlap as helper
    source = inspect.getsource(helper)
    assert "modal.App(" not in source
    assert "torch.cuda" not in source
    assert "optimizer.step(" not in source
    assert "run_scientific" not in source
