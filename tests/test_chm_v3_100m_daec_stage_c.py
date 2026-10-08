from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

import tam_research.chm_v3_100m_daec_stage_c as stage_c
from tam_research.chm_v1_100m_stage_c_eval import (
    ALL_FAMILIES,
    CASES_PER_FAMILY,
    GENERATOR_VERSION,
    TOTAL_PROBES,
)
from tam_research.chm_v3_100m_daec import DecoderAlignedEpisodicCopy


def _integrity() -> dict[str, object]:
    return {
        "local_trainable_parameters": 101_803_520,
        "raw_trainable_parameters": 101_836_800,
        "daec_trainable_parameters": 101_853_697,
        "training_tokens_per_model": 33_554_432,
        "optimizer_steps_per_model": 2_048,
        "scientific_seed": 2_013_161,
        "generator_version": GENERATOR_VERSION,
        "probe_seed": 977301,
        "cases_per_family": CASES_PER_FAMILY,
        "probe_count": TOTAL_PROBES,
        "validation_seed": 977302,
        "validation_tokens": 1_048_576,
        "paired_backbone_initialization_identical": True,
        "raw_daec_address_initialization_identical": True,
        "raw_daec_legacy_gate_initialization_identical": True,
        "daec_initialization_exact": True,
        "hard_flat_evaluation_parity_verified": True,
        "byte_identical_training_stream": True,
        "matched_optimizer_schedule": True,
        "finite_losses": True,
        "finite_parameters": True,
        "no_cross_session_state_aliasing": True,
        "no_future_self_leakage": True,
    }


def _rows(
    *,
    daec_nll: float = 1.70,
    raw_nll: float = 1.85,
    local_nll: float = 2.00,
    local_correct: int = 48,
    raw_correct: int = 56,
    daec_correct: int = 72,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(ALL_FAMILIES):
        for case_id in range(CASES_PER_FAMILY):
            rows.append(
                {
                    "family": family,
                    "case_id": family_index * 1000 + case_id,
                    "generator_version": GENERATOR_VERSION,
                    "evidence_distance": 600,
                    "local_correct": case_id < local_correct,
                    "raw_correct": case_id < raw_correct,
                    "daec_correct": case_id < daec_correct,
                    "local_candidate_nll": local_nll,
                    "raw_candidate_nll": raw_nll,
                    "daec_candidate_nll": daec_nll,
                    "local_stale_choice": False,
                    "raw_stale_choice": False,
                    "daec_stale_choice": False,
                }
            )
    return rows


def test_manifest_freezes_seed_and_grants_no_execution_authority() -> None:
    manifest = stage_c.validate_protocol_manifest()
    assert manifest["issue"] == 1316
    assert manifest["classification"] == "CHM_V3_100M_DAEC_STAGE_C_PREREGISTRATION_NO_GPU_AUTHORITY"
    assert stage_c.STOP_CLASSIFICATION == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert manifest["scientific_seed_reserved"] == 2_013_161
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["training_tokens_per_model"] == 33_554_432
    assert manifest["optimizer_steps_per_model"] == 2_048
    assert manifest["warmup_steps"] == 40
    assert manifest["retrieval_hops"] == 2
    assert manifest["daec_gate_init"] == -4.0
    assert manifest["retrieval_temperature"] == 0.10
    assert manifest["training_stream_seed"] == 2_023_161
    assert manifest["bootstrap_gate"] is False
    assert manifest["gpu_authorized"] is False
    assert manifest["modal_authorized"] is False
    assert manifest["training_launch_authorized"] is False
    assert manifest["stage_d_authorized"] is False
    assert manifest["multi_seed_replication_authorized"] is False


def test_positive_synthetic_screen_passes_all_frozen_gates() -> None:
    result = stage_c.classify_stage_c(
        integrity=_integrity(),
        rows=_rows(),
        local_language_nll=5.0,
        raw_language_nll=5.01,
        daec_language_nll=5.02,
    )
    assert result["passed"] is True
    assert result["classification"] == stage_c.POSITIVE_CLASSIFICATION
    assert result["stop_reasons"] == []
    aggregate = result["metrics"]["aggregate_long_range"]
    assert aggregate["daec_vs_local"]["accuracy_gain"] > 0.05
    assert aggregate["daec_vs_local"]["candidate_nll_benefit"] > 0.05
    assert aggregate["daec_vs_raw"]["accuracy_gain"] > 0.0
    assert aggregate["daec_vs_raw"]["candidate_nll_benefit"] > 0.05


def test_daec_vs_raw_mechanism_gate_can_stop_otherwise_good_screen() -> None:
    result = stage_c.classify_stage_c(
        integrity=_integrity(),
        rows=_rows(daec_nll=1.81, raw_nll=1.85),
        local_language_nll=5.0,
        raw_language_nll=5.01,
        daec_language_nll=5.02,
    )
    assert result["passed"] is False
    assert result["classification"] == stage_c.STOP_CLASSIFICATION
    assert "daec_vs_raw_aggregate_candidate_nll_benefit_below_gate" in result["stop_reasons"]


def test_integrity_drift_stops_screen_without_changing_thresholds() -> None:
    integrity = _integrity()
    integrity["byte_identical_training_stream"] = False
    integrity["scientific_seed"] = 999999
    result = stage_c.classify_stage_c(
        integrity=integrity,
        rows=_rows(),
        local_language_nll=5.0,
        raw_language_nll=5.01,
        daec_language_nll=5.02,
    )
    assert result["passed"] is False
    assert "integrity_byte_identical_training_stream_failed" in result["stop_reasons"]
    assert "integrity_scientific_seed_mismatch" in result["stop_reasons"]


def test_ordinary_language_and_local_negative_regression_are_independent_stops() -> None:
    rows = _rows()
    for row in rows:
        if row["family"] == "local_negative":
            row["daec_correct"] = False
            row["daec_candidate_nll"] = 2.20
    result = stage_c.classify_stage_c(
        integrity=_integrity(),
        rows=rows,
        local_language_nll=5.0,
        raw_language_nll=5.0,
        daec_language_nll=5.04,
    )
    assert result["passed"] is False
    assert "daec_ordinary_language_nll_regression_above_gate" in result["stop_reasons"]
    assert "daec_local_negative_accuracy_regression_above_gate" in result["stop_reasons"]
    assert "daec_local_negative_candidate_nll_regression_above_gate" in result["stop_reasons"]


def test_overwrite_stale_guard_is_exactly_local_referenced() -> None:
    rows = _rows()
    overwrite = [row for row in rows if row["family"] == "overwrite"]
    for row in overwrite[:64]:
        row["local_stale_choice"] = True
    for row in overwrite[:60]:
        row["daec_stale_choice"] = True
    result = stage_c.classify_stage_c(
        integrity=_integrity(),
        rows=rows,
        local_language_nll=5.0,
        raw_language_nll=5.0,
        daec_language_nll=5.0,
    )
    assert result["passed"] is False
    assert "daec_overwrite_stale_state_guard_failed" in result["stop_reasons"]
    assert result["metrics"]["overwrite"]["stale_rule"] == "daec<=0.80*local"


def test_row_validation_rejects_wrong_count_and_generator_version() -> None:
    with pytest.raises(ValueError, match="exactly"):
        stage_c.classify_stage_c(
            integrity=_integrity(),
            rows=_rows()[:-1],
            local_language_nll=5.0,
            raw_language_nll=5.0,
            daec_language_nll=5.0,
        )

    rows = _rows()
    rows[0]["generator_version"] = "wrong"
    with pytest.raises(ValueError, match="wrong evaluator version"):
        stage_c.classify_stage_c(
            integrity=_integrity(),
            rows=rows,
            local_language_nll=5.0,
            raw_language_nll=5.0,
            daec_language_nll=5.0,
        )


def test_triplet_probe_row_scores_all_three_models_without_redefining_candidates() -> None:
    from tam_research.chm_v1_long_memory_eval import EncodedProbe

    probe = EncodedProbe(
        family="rare_fact",
        case_id=1,
        prompt_text="synthetic",
        prompt_ids=(1, 2, 3),
        answer_token_id=4,
        candidate_token_ids=(4, 5, 6, 7, 8, 9, 10, 11),
        evidence_end_token=0,
        query_token=2,
        evidence_distance=2,
        generator_version=GENERATOR_VERSION,
    )
    local = torch.zeros(32)
    raw = torch.zeros(32)
    daec = torch.zeros(32)
    local[5] = 1.0
    raw[6] = 1.0
    daec[4] = 1.0
    row = stage_c.triplet_probe_row(
        probe=probe,
        local_logits=local,
        raw_logits=raw,
        daec_logits=daec,
    )
    assert row["local_correct"] is False
    assert row["raw_correct"] is False
    assert row["daec_correct"] is True
    assert row["daec_candidate_nll"] < row["local_candidate_nll"]


def test_stage_c_module_has_no_training_launcher_or_modal_surface() -> None:
    source = inspect.getsource(stage_c)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)

    assert "modal" not in imported_roots
    assert "backward" not in calls
    assert "save" not in calls
    assert "load" not in calls
    assert "step" not in calls
    assert "zero_grad" not in calls
    assert not any(name.startswith("train") for name in calls)

def test_hard_flat_no_memory_is_exact_local_log_probability() -> None:
    from torch import nn
    import torch.nn.functional as F
    from tam_research.chm_v1_small_lm import _hidden

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(17, 8)
            self.pos_emb = nn.Embedding(512, 8)
            self.blocks = nn.ModuleList()
            self.norm = nn.LayerNorm(8)
            self.lm_head = nn.Linear(8, 17, bias=False)
            self.lm_head.weight = self.token_emb.weight

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.query_address = nn.Linear(8, 4, bias=False)
            self.key_address = nn.Linear(8, 4, bias=False)
            self.daec = DecoderAlignedEpisodicCopy(d_model=8, address_dim=4)

        def query_for(self, hidden):
            return F.normalize(self.query_address(hidden), dim=-1)

        def key_for(self, hidden):
            return F.normalize(self.key_address(hidden), dim=-1)

    torch.manual_seed(1316)
    model = Model()
    ids = [1, 2, 3, 4]
    out, trace = stage_c.daec_hard_flat_final_logits_with_trace(model, ids)
    tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    expected = F.log_softmax(model.backbone.lm_head(_hidden(model.backbone, tokens))[0, -1].float(), dim=-1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert trace["memory_count"] == 0

    full = [3] * 512 + [4]
    out, trace = stage_c.daec_hard_flat_final_logits_with_trace(model, full)
    assert trace["memory_count"] == 512
    assert trace["copied_token_id"] == 3
    assert 0 <= trace["first_hop_index"] < 512
    assert 0 <= trace["second_hop_index"] < 512
    last = torch.tensor([[4]], dtype=torch.long)
    base = torch.softmax(model.backbone.lm_head(_hidden(model.backbone, last))[0, -1].float(), dim=-1)
    gate = trace["gate"]
    expected = base * (1.0 - gate)
    expected[3] += gate
    torch.testing.assert_close(out.exp(), expected, rtol=3e-6, atol=3e-7)
