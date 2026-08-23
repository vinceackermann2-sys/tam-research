from __future__ import annotations

import torch

from tam_research.aera_v14_adaptivity import (
    BATCHES,
    BATCH_SIZE,
    CHUNK_SIZE,
    EXPECTED_CHUNKS,
    SEED,
    spearman_rho,
    summarize_adaptivity,
)


def test_frozen_checkpoint_diagnostic_scope() -> None:
    assert SEED == 8271
    assert CHUNK_SIZE == 256
    assert BATCH_SIZE == 8
    assert BATCHES == 64
    assert EXPECTED_CHUNKS == 1024


def test_spearman_detects_positive_ordering() -> None:
    x = torch.arange(32, dtype=torch.float32)
    y = x.clone()
    assert spearman_rho(x, y) > 0.99


def test_monotonic_difficulty_compute_profile_passes() -> None:
    difficulty = torch.arange(128, dtype=torch.float32)
    compute = torch.cat(
        [
            torch.zeros(32),
            torch.ones(32),
            torch.full((32,), 2.0),
            torch.full((32,), 3.0),
        ]
    )
    summary = summarize_adaptivity(difficulty, compute)
    assert summary["pass"] is True
    assert summary["difficulty_compute_spearman_rho"] >= 0.20
    assert summary["hardest_minus_easiest_optional_stages"] >= 0.25


def test_constant_budget_policy_fails_adaptivity() -> None:
    difficulty = torch.arange(128, dtype=torch.float32)
    compute = torch.ones(128)
    summary = summarize_adaptivity(difficulty, compute)
    assert summary["pass"] is False
    assert summary["checks"]["at_least_two_budget_bins_ge_5pct"] is False
    assert summary["checks"]["spearman_rho_ge_0_20"] is False


def test_inverse_budget_policy_fails_adaptivity() -> None:
    difficulty = torch.arange(128, dtype=torch.float32)
    compute = torch.cat(
        [
            torch.full((32,), 3.0),
            torch.full((32,), 2.0),
            torch.ones(32),
            torch.zeros(32),
        ]
    )
    summary = summarize_adaptivity(difficulty, compute)
    assert summary["pass"] is False
    assert summary["difficulty_compute_spearman_rho"] < 0.0
