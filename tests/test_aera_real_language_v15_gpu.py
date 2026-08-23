from __future__ import annotations

import pytest

from tam_research import aera_real_language_v15_gpu as v15g


def test_v15_gpu_protocol_is_frozen_to_fresh_seed() -> None:
    assert v15g.SEED == 8291
    assert v15g.TARGET_RATES.tolist() == pytest.approx([0.5, 1.0 / 3.0, 1.0 / 6.0])


def test_v15_wrong_seed_is_refused_before_gpu_training(monkeypatch) -> None:
    called = False

    def fake_install() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(v15g, "_install_v15_harness", fake_install)
    with pytest.raises(ValueError, match="frozen to seed 8291"):
        v15g.train_matched_pair(data_dir="unused", run_dir="unused", seed=8292)
    assert called is False


def test_v15_target_rate_mae_threshold_rejects_v14_collapse_profile() -> None:
    v14_rates = [0.556, 0.027, 0.0]
    target = v15g.TARGET_RATES.tolist()
    mae = sum(abs(a - b) for a, b in zip(v14_rates, target)) / 3
    assert mae > 0.12
    assert not all(rate >= 0.05 for rate in v14_rates)
