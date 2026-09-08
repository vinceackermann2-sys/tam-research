from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures.cortex_s.experiments.scale100m_2b.protocol import (
    BASELINE_TRANSFORMER,
    CORTEX_100M_CONFIG,
    EXPECTED_CORTEX_PARAMS,
    EXPECTED_TRANSFORMER_PARAMS,
    HARD_FULL_TIMEOUT_SECONDS,
    MAX_PROJECTED_FULL_SECONDS,
    RESERVED_FRESH_SEEDS,
    TRAIN_TOKENS,
    USER_CREDIT_ENVELOPE_USD,
    projected_full_cost_usd,
    validate_protocol,
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


def test_full_credit_ceiling_fails_closed_below_user_envelope() -> None:
    projected = projected_full_cost_usd(MAX_PROJECTED_FULL_SECONDS)
    hard = projected_full_cost_usd(HARD_FULL_TIMEOUT_SECONDS)
    assert projected["total_usd_conservative"] < hard["total_usd_conservative"]
    assert hard["total_usd_conservative"] < USER_CREDIT_ENVELOPE_USD
    assert TRAIN_TOKENS == 2_000_000_000


def test_fresh_replication_seeds_are_not_paired_or_calibration_seeds() -> None:
    from architectures.cortex_s.experiments.scale100m_2b.protocol import (
        CALIBRATION_SEED,
        PAIRED_SEED,
    )

    assert PAIRED_SEED not in RESERVED_FRESH_SEEDS
    assert CALIBRATION_SEED not in RESERVED_FRESH_SEEDS
    assert PAIRED_SEED != CALIBRATION_SEED


def test_small_same_mechanism_forward_backward_is_finite() -> None:
    # Same mechanisms, reduced widths. This catches autograd/routing/state failures
    # cheaply while the separate 100M test above locks the real parameterization.
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
    model = CortexSLM(CORTEX_100M_CONFIG).eval()
    tokens = torch.randint(0, CORTEX_100M_CONFIG.vocab_size, (1, 4))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (1, 4, CORTEX_100M_CONFIG.vocab_size)
    assert torch.isfinite(logits).all()


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
