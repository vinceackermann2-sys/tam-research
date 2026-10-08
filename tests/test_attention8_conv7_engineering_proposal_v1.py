from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
DOC = ROOT / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.md"


def test_proposal_unconditionally_denies_paid_authority():
    p = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    assert p["stage"] == "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED"
    assert all(value is False for value in p["authority"].values())
    pilot = p["draft_engineering_pilot"]
    assert pilot["engineering_seed"] is None
    assert pilot["strict_max_gpu_spend_usd"] is None
    assert pilot["modal_account"] == "primary"
    assert "no retries" in pilot["attempt_policy"]
    assert p["source"]["scientific_failure_issue"] == 1300
    assert p["source"]["scientific_result_sha256"] == "872daab1193742942ef01fd8dee77080a07272eee4b55aa0b9236cffb584d88d"
    text = DOC.read_text(encoding="utf-8")
    assert "ZERO-GPU DESIGN PROPOSAL" in text
    assert "no Modal runner" in text
    assert "no seed" in text.lower()


def test_three_way_frozen_pilot_and_long_horizon_prefix_math():
    p = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    pilot = p["draft_engineering_pilot"]
    geometry = p["model_geometry"]
    assert pilot["comparators"] == [
        "fresh_transformer", "reduced_attention_dense_v2_attention8", "attention8_causal_depthwise_conv7"
    ]
    assert geometry["transformer_parameters"] == 101_803_520
    assert geometry["attention8_parameters"] == 101_795_328
    assert geometry["attention8_conv7_parameters"] == 101_803_536
    assert geometry["attention8_layers"] == [3, 6, 9, 12, 15, 18, 21, 24]
    assert geometry["conv_mixers"] == 16 and geometry["conv_kernel_size"] == 7
    assert pilot["per_model_optimizer_steps"] == 3052
    assert pilot["per_model_token_exposures"] == pilot["per_model_optimizer_steps"] * pilot["tokens_per_step"]
    assert pilot["full_horizon_token_exposures"] == pilot["full_horizon_optimizer_steps"] * pilot["tokens_per_step"]
    assert pilot["warmup_steps_full_horizon"] == int(pilot["full_horizon_optimizer_steps"] * pilot["warmup_ratio_full_horizon"])
    assert pilot["warmup_steps_full_horizon"] == 610
    assert pilot["lr_schedule"] == "cosine full-horizon prefix"
    assert pilot["final_evaluation_batches"] == 50
    assert pilot["train_sha256"] == "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"


def test_proposal_screen_is_draft_not_scientific_authority():
    p = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    g = p["draft_screen_gates"]
    assert g["transformer_comparison_max_nll_delta"] == 0.015
    assert g["attention8_parent_min_quality_improvement_nll"] == 0.010
    assert g["transformer_min_training_tokens_per_second_ratio"] == 1.03
    assert g["all_three_complete_required"] and g["all_finite_required"]
    assert g["selection_only"] == "engineering viability; does not authorize 2B scientific pair"
    assert p["interpretation"]["engineering_result_not_scientific_replication"] is True
    assert p["interpretation"]["historical_200m_panel_comparison_invalid_without_horizon_match"] is True
