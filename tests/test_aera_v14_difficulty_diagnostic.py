from __future__ import annotations

import numpy as np

from tam_research.aera_v14_difficulty_diagnostic import (
    EXPECTED_CHUNKS,
    spearman_rho,
    summarize_adaptivity,
)


def test_spearman_handles_tied_compute_counts() -> None:
    x = np.arange(10, dtype=np.float64)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.float64)
    assert spearman_rho(x, y) > 0.9


def test_monotonic_difficulty_policy_passes_frozen_thresholds() -> None:
    n = EXPECTED_CHUNKS
    difficulty = np.linspace(0.0, 1.0, n, dtype=np.float64)
    # Deliberately discrete and tied, like optional-stage counts from hard routing.
    compute = np.zeros(n, dtype=np.float64)
    compute[difficulty >= 0.25] = 1.0
    compute[difficulty >= 0.60] = 2.0
    compute[difficulty >= 0.90] = 3.0
    result = summarize_adaptivity(difficulty, compute)
    assert result["adaptivity_pass"] is True
    assert result["checks"]["difficulty_quintile_compute_monotonic"] is True


def test_constant_budget_policy_fails_even_with_correct_mean_compute() -> None:
    n = EXPECTED_CHUNKS
    difficulty = np.linspace(0.0, 1.0, n, dtype=np.float64)
    # Mean optional compute ~=1.18 stages/chunk, close to the observed v14 budget,
    # but it is unrelated to difficulty and therefore must not count as adaptivity.
    compute = np.tile(np.asarray([1.0, 1.0, 1.0, 2.0], dtype=np.float64), n // 4)
    result = summarize_adaptivity(difficulty, compute)
    assert result["adaptivity_pass"] is False
    assert result["checks"]["spearman_rho_ge_0_15"] is False


def test_reverse_difficulty_policy_fails() -> None:
    n = EXPECTED_CHUNKS
    difficulty = np.linspace(0.0, 1.0, n, dtype=np.float64)
    compute = np.zeros(n, dtype=np.float64)
    compute[difficulty <= 0.75] = 1.0
    compute[difficulty <= 0.40] = 2.0
    compute[difficulty <= 0.10] = 3.0
    result = summarize_adaptivity(difficulty, compute)
    assert result["adaptivity_pass"] is False
    assert result["spearman_rho"] < 0.0
