from __future__ import annotations

from collections import Counter
import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.chm_v1_100m_scale import (
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    FIRST_SCREEN_TOKEN_BUDGET,
)
from tam_research.chm_v1_100m_stage_c_eval import (
    ALL_FAMILIES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CASES_PER_FAMILY,
    EXPECTED_PER_CANDIDATE_PER_FAMILY,
    GENERATOR_VERSION,
    LONG_RANGE_FAMILIES,
    PROBE_SEED,
    TOTAL_PROBES,
    VALIDATION_SEED,
    VALIDATION_TOKENS,
    aligned_local_token_ids,
    candidate_score,
    classify_stage_c,
    eiem_flat_final_logits,
    generate_aligned_probe_suite,
    protocol_manifest,
    stratified_paired_bootstrap,
    validate_protocol_manifest,
)
from tam_research.chm_v1_batched_eval import forward_session_chunk_batched_transport
from tam_research.chm_v1_small_lm import EpisodicState, LOCAL_WINDOW
from tam_research.models import ModelConfig, ResearchLM


class StableWordEncoder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str) -> list[int]:
        pieces = re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text.lower())
        out: list[int] = []
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab) + 100
            out.append(self.vocab[piece])
        return out


class TinyEIEM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResearchLM(
            ModelConfig(
                vocab_size=257,
                d_model=16,
                n_layers=1,
                n_heads=2,
                max_seq_len=LOCAL_WINDOW,
                ff_mult=2,
                architecture="transformer",
            )
        )
        self.query_address = nn.Linear(16, 4, bias=False)
        self.key_address = nn.Linear(16, 4, bias=False)
        self.memory_gate_logit = nn.Parameter(torch.full((16,), -2.0))

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return hidden + torch.sigmoid(self.memory_gate_logit).to(hidden.dtype) * memory


def _integrity() -> dict[str, object]:
    return {
        "local_trainable_parameters": EXPECTED_LOCAL_PARAMETERS,
        "eiem_trainable_parameters": EXPECTED_EIEM_PARAMETERS,
        "training_tokens_per_model": FIRST_SCREEN_TOKEN_BUDGET,
        "backbone_initialization_identical": True,
        "byte_identical_training_stream": True,
        "matched_optimizer_schedule": True,
        "finite_losses": True,
        "finite_parameters": True,
        "no_cross_session_state_aliasing": True,
        "no_future_self_leakage": True,
        "generator_version": GENERATOR_VERSION,
        "probe_seed": PROBE_SEED,
        "validation_seed": VALIDATION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cases_per_family": CASES_PER_FAMILY,
        "probe_count": TOTAL_PROBES,
        "validation_tokens": VALIDATION_TOKENS,
    }


def _passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(ALL_FAMILIES):
        for index in range(CASES_PER_FAMILY):
            long_range = family in LONG_RANGE_FAMILIES
            rows.append(
                {
                    "family": family,
                    "case_id": family_index * 100_000 + index,
                    "generator_version": GENERATOR_VERSION,
                    "local_correct": False if long_range else True,
                    "eiem_correct": True,
                    "local_candidate_nll": 1.0,
                    "eiem_candidate_nll": 0.8 if long_range else 1.0,
                    "local_stale_choice": family == "overwrite" and index < 64,
                    "eiem_stale_choice": family == "overwrite" and index < 32,
                }
            )
    return rows


def test_stage_c_manifest_is_exact_and_grants_zero_execution_authority() -> None:
    manifest = validate_protocol_manifest()
    assert manifest == protocol_manifest()
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["total_probes"] == 512
    assert manifest["long_range_probes"] == 384
    assert manifest["bootstrap_resamples"] == 10_000
    assert manifest["validation_tokens"] == 1_048_576
    assert manifest["scientific_seed_reserved_not_authorized"] == 977_001
    assert manifest["gpu_authorized"] is False
    assert manifest["modal_authorized"] is False
    assert manifest["training_authorized"] is False
    assert manifest["scientific_execution_authorized"] is False


def test_v4_suite_is_balanced_held_out_and_uses_chunk_aligned_local_context() -> None:
    suite = generate_aligned_probe_suite(StableWordEncoder())
    assert len(suite) == TOTAL_PROBES
    assert {probe.generator_version for probe in suite} == {GENERATOR_VERSION}
    assert all(probe.used_for_training is False for probe in suite)

    for family in ALL_FAMILIES:
        family_rows = [probe for probe in suite if probe.family == family]
        assert len(family_rows) == CASES_PER_FAMILY
        counts = Counter(probe.answer_token_id for probe in family_rows)
        assert len(counts) == 8
        assert set(counts.values()) == {EXPECTED_PER_CANDIDATE_PER_FAMILY}

    for probe in suite:
        aligned = aligned_local_token_ids(probe)
        start = (probe.query_token // LOCAL_WINDOW) * LOCAL_WINDOW
        assert aligned == probe.prompt_ids[start : probe.query_token + 1]
        assert 1 <= len(aligned) <= LOCAL_WINDOW
        assert aligned[-1] == probe.prompt_ids[probe.query_token]


def test_v4_probe_and_bootstrap_seed_guards_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="probe seed"):
        generate_aligned_probe_suite(StableWordEncoder(), seed=PROBE_SEED + 1)
    rows = _passing_rows()
    with pytest.raises(RuntimeError, match="bootstrap seed"):
        stratified_paired_bootstrap(rows, seed=BOOTSTRAP_SEED + 1)
    with pytest.raises(RuntimeError, match="exactly"):
        stratified_paired_bootstrap(rows, resamples=BOOTSTRAP_RESAMPLES - 1)


def test_final_token_only_flat_eiem_matches_full_flat_chunk_evaluator() -> None:
    torch.manual_seed(986)
    model = TinyEIEM().eval()
    prompt = tuple((index * 17 + 3) % 257 for index in range(700))

    state = EpisodicState("full-flat-reference")
    reference = None
    with torch.no_grad():
        for start in range(0, len(prompt), LOCAL_WINDOW):
            chunk = torch.tensor(prompt[start : start + LOCAL_WINDOW], dtype=torch.long).unsqueeze(0)
            logits, _ = forward_session_chunk_batched_transport(
                model,
                chunk,
                [state],
                mode="flat",
                update_memory=True,
                verify_indexed_exactness=False,
            )
            reference = logits[0, -1]
    assert reference is not None
    optimized = eiem_flat_final_logits(model, prompt)
    torch.testing.assert_close(optimized, reference, rtol=0.0, atol=0.0)


def test_candidate_score_is_candidate_restricted_and_tracks_stale_choice() -> None:
    logits = torch.full((32,), -10.0)
    candidates = tuple(range(8, 16))
    logits[11] = 4.0
    score = candidate_score(
        logits,
        candidate_token_ids=candidates,
        answer_token_id=11,
        stale_token_ids=(12, 13),
    )
    assert score["predicted_token_id"] == 11
    assert score["correct"] is True
    assert score["candidate_nll"] >= 0.0
    assert score["stale_choice"] is False

    logits[12] = 5.0
    stale = candidate_score(
        logits,
        candidate_token_ids=candidates,
        answer_token_id=11,
        stale_token_ids=(12, 13),
    )
    assert stale["correct"] is False
    assert stale["stale_choice"] is True


def test_stratified_bootstrap_is_deterministic_and_positive_for_uniform_wins() -> None:
    rows = _passing_rows()
    first = stratified_paired_bootstrap(rows)
    second = stratified_paired_bootstrap(rows)
    assert first == second
    assert first["accuracy_gain"]["p2_5"] > 0.0
    assert first["candidate_nll_benefit"]["p2_5"] > 0.0


def test_stage_c_positive_screen_requires_every_preregistered_gate() -> None:
    result = classify_stage_c(
        integrity=_integrity(),
        rows=_passing_rows(),
        local_language_nll=4.0,
        eiem_language_nll=4.01,
    )
    assert result["passed"] is True
    assert result["classification"] == "CHM_V1_100M_STAGE_C_POSITIVE_SCALING_SCREEN"
    assert result["stop_reasons"] == []
    assert result["metrics"]["aggregate_long_range_accuracy_gain"] == pytest.approx(1.0)
    assert result["metrics"]["aggregate_long_range_candidate_nll_benefit"] == pytest.approx(0.2)
    assert result["metrics"]["local_overwrite_stale_choice_rate"] == pytest.approx(0.5)
    assert result["metrics"]["eiem_overwrite_stale_choice_rate"] == pytest.approx(0.25)
    assert result["stage_d_automatically_authorized"] is False


def test_stage_c_stops_on_integrity_or_noninferiority_failure_without_moving_thresholds() -> None:
    integrity = _integrity()
    integrity["training_tokens_per_model"] = FIRST_SCREEN_TOKEN_BUDGET - 1
    result = classify_stage_c(
        integrity=integrity,
        rows=_passing_rows(),
        local_language_nll=4.0,
        eiem_language_nll=4.031,
    )
    assert result["passed"] is False
    assert result["classification"] == "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH"
    assert "training_token_count_mismatch" in result["stop_reasons"]
    assert "ordinary_language_nll_regression_above_gate" in result["stop_reasons"]


def test_stage_c_rejects_wrong_evaluator_version_before_classification() -> None:
    rows = _passing_rows()
    rows[0]["generator_version"] = "wrong-version"
    with pytest.raises(ValueError, match="wrong evaluator version"):
        classify_stage_c(
            integrity=_integrity(),
            rows=rows,
            local_language_nll=4.0,
            eiem_language_nll=4.0,
        )
