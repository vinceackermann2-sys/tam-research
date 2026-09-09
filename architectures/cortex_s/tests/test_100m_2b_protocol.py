from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures.cortex_s.experiments.scale100m_2b.protocol import (
    BASELINE_TRANSFORMER,
    CALIBRATION_SEED,
    CONSUMED_ENGINEERING_SEEDS,
    CORTEX_100M_CONFIG,
    EXPECTED_CORTEX_PARAMS,
    EXPECTED_TRANSFORMER_PARAMS,
    FULL_BATCH_TOKEN_EXPOSURES,
    HARD_FULL_TIMEOUT_SECONDS,
    MAX_PROJECTED_FULL_SECONDS,
    META_SHA256,
    PHYSICAL_EXPERT_HIDDEN,
    PRODUCTION_MOE_BACKEND,
    REPAIR4_EVIDENCE_ISSUE,
    REPAIR4_EVIDENCE_RUN,
    REPAIR4_MEASURED_SPEEDUP,
    RESERVED_FRESH_SEEDS,
    TOTAL_OPTIMIZER_STEPS,
    TRAIN_SHA256,
    TRAIN_TOKENS,
    USER_CREDIT_ENVELOPE_USD,
    VAL_SHA256,
    V2_PREFLIGHT_ISSUE,
    projected_full_cost_usd,
    validate_protocol,
)
from architectures.cortex_s.grouped_moe import (
    PhysicalPaddedGroupedSparseMoE,
    build_production_grouped_cortex_100m,
    padded_hidden,
)
from architectures.cortex_s.language_model import CortexSLM, CortexSLMConfig, parameter_count

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_100m_parameter_match_is_actual_not_analytical_only() -> None:
    model = CortexSLM(CORTEX_100M_CONFIG)
    actual = parameter_count(model)
    result = validate_protocol(actual_cortex_params=actual)
    assert actual == EXPECTED_CORTEX_PARAMS
    assert EXPECTED_TRANSFORMER_PARAMS == int(BASELINE_TRANSFORMER["parameters"])
    assert result["parameter_gap_fraction"] <= 0.0003
    assert result["executed_expert_fraction"] == 0.25
    assert result["attention_layers"] == 4
    assert result["production_moe_backend"] == PRODUCTION_MOE_BACKEND
    assert result["physical_expert_hidden"] == 344


def test_grouped_production_builder_is_exact_100m_backend() -> None:
    model = build_production_grouped_cortex_100m()
    assert parameter_count(model) == EXPECTED_CORTEX_PARAMS
    assert len(model.blocks) == 24
    assert all(isinstance(block.moe, PhysicalPaddedGroupedSparseMoE) for block in model.blocks)
    assert all(block.moe.hidden == 338 for block in model.blocks)
    assert all(block.moe.physical_hidden == 344 for block in model.blocks)
    assert padded_hidden(CORTEX_100M_CONFIG.expert_hidden) == PHYSICAL_EXPERT_HIDDEN


def test_matched_full_batch_step_semantics_are_explicit() -> None:
    assert TRAIN_TOKENS == 2_000_000_000
    assert TOTAL_OPTIMIZER_STEPS == 30_518
    assert FULL_BATCH_TOKEN_EXPOSURES == 2_000_027_648
    assert BASELINE_TRANSFORMER["optimizer_steps"] == TOTAL_OPTIMIZER_STEPS
    assert BASELINE_TRANSFORMER["full_batch_token_exposures"] == FULL_BATCH_TOKEN_EXPOSURES
    assert BASELINE_TRANSFORMER["pretrain_tokens"] == TRAIN_TOKENS


def test_full_credit_ceiling_fails_closed_below_user_envelope() -> None:
    projected = projected_full_cost_usd(MAX_PROJECTED_FULL_SECONDS)
    hard = projected_full_cost_usd(HARD_FULL_TIMEOUT_SECONDS)
    assert projected["total_usd_conservative"] < hard["total_usd_conservative"]
    assert hard["total_usd_conservative"] < USER_CREDIT_ENVELOPE_USD


def test_grouped_v3_engineering_seed_is_fresh_and_scientific_seeds_untouched() -> None:
    from architectures.cortex_s.experiments.scale100m_2b.protocol import PAIRED_SEED

    assert CALIBRATION_SEED == 2026090905
    assert CALIBRATION_SEED not in CONSUMED_ENGINEERING_SEEDS
    assert PAIRED_SEED not in RESERVED_FRESH_SEEDS
    assert CALIBRATION_SEED not in RESERVED_FRESH_SEEDS
    assert PAIRED_SEED != CALIBRATION_SEED
    assert 910001 in CONSUMED_ENGINEERING_SEEDS
    assert 2026090904 in CONSUMED_ENGINEERING_SEEDS


def test_repair4_is_recorded_as_engineering_evidence_not_direct_authority() -> None:
    assert V2_PREFLIGHT_ISSUE == 769
    assert REPAIR4_EVIDENCE_ISSUE == 799
    assert REPAIR4_EVIDENCE_RUN == 34339214619
    assert REPAIR4_MEASURED_SPEEDUP > 1.20


def test_small_same_mechanism_forward_backward_is_finite() -> None:
    cfg = CortexSLMConfig(
        vocab_size=257,
        d_model=64,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        state_size=16,
        num_experts=8,
        top_k=2,
        expert_hidden=42,
        attention_every=2,
    )
    torch.manual_seed(20260908)
    model = CortexSLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    targets = torch.randint(0, cfg.vocab_size, (2, 12))
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    assert math.isfinite(float(loss))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_real_100m_forward_is_finite_on_cpu() -> None:
    torch.manual_seed(20260908)
    model = build_production_grouped_cortex_100m().eval()
    tokens = torch.randint(0, CORTEX_100M_CONFIG.vocab_size, (1, 4))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (1, 4, CORTEX_100M_CONFIG.vocab_size)
    assert torch.isfinite(logits).all()
    # Grouped counts stay tensors in the hot path but final reporting must be JSON-safe.
    json.dumps(model.router_stats())


def test_fingerprint_v2_diagnostic_is_cpu_only_and_non_authorizing() -> None:
    source = (REPO_ROOT / "modal_cortex_s_100m_2b_fingerprint_v2_app.py").read_text(
        encoding="utf-8"
    )
    assert "gpu=" not in source
    assert '"authorizes_h100": False' in source
    assert '"authorizes_full_2b": False' in source
    assert '"gpu_allocated": False' in source
    assert "fingerprint-v2" in source


def test_fingerprint_v2_workflow_has_unique_single_purpose_trigger() -> None:
    workflow = (
        REPO_ROOT / ".github/workflows/modal-cortex-s-100m-2b-fingerprint-v2.yml"
    ).read_text(encoding="utf-8")
    assert "[modal-cortex-s-100m-2b-fingerprint-v2]" in workflow
    assert "phase == 'fingerprint-v2'" in workflow
    assert "--detach" not in workflow
    assert "modal_cortex_s_100m_2b_fingerprint_v2_app.py" in workflow


def test_v3_grouped_paid_launcher_uses_fresh_single_use_namespaces() -> None:
    source = (REPO_ROOT / "modal_cortex_s_100m_2b_app.py").read_text(encoding="utf-8")
    assert TRAIN_SHA256 in source
    assert VAL_SHA256 in source
    assert META_SHA256 in source
    assert 'APP_NAME = "cortex-s-v0-100m-2b-v3-grouped"' in source
    assert 'PREFLIGHT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v3-grouped"' in source
    assert 'RUN_ROOT = "/vol/cortex-s-v0/100m-2b/paired-seed8100-v3-grouped"' in source
    assert 'phase == "preflight-v3-grouped"' in source
    assert 'phase == "full-v3-grouped"' in source
    assert "run_h100_grouped_calibration" in source
    assert "train_full_grouped_2b" in source
    assert "volume.commit()" in source
    assert '"automatic_resume_authorized": False' in source


def test_v3_grouped_workflow_cannot_accept_consumed_v1_or_v2_triggers() -> None:
    workflow = (REPO_ROOT / ".github/workflows/modal-cortex-s-100m-2b.yml").read_text(
        encoding="utf-8"
    )
    assert "[modal-cortex-s-100m-2b-preflight-v3-grouped]" in workflow
    assert "[modal-cortex-s-100m-2b-full-v3-grouped]" in workflow
    assert "preflight-v3-grouped" in workflow
    assert "full-v3-grouped" in workflow
    assert "workflow_dispatch" not in workflow
    assert "[modal-cortex-s-100m-2b-preflight-v2]" not in workflow
    assert "[modal-cortex-s-100m-2b-full-v2]" not in workflow
    assert "'[modal-cortex-s-100m-2b-preflight]'" not in workflow
    assert "'[modal-cortex-s-100m-2b-full]'" not in workflow
