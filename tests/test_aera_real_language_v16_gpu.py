from __future__ import annotations

import pytest

from tam_research import aera_real_language_v16_gpu as v16g


def test_v16_gpu_protocol_is_frozen_to_fresh_seed() -> None:
    assert v16g.SEED == 8311
    assert v16g.TARGET_RATES.tolist() == pytest.approx([0.5, 1.0 / 3.0, 1.0 / 6.0])


def test_v16_wrong_seed_is_refused_before_gpu_training(monkeypatch) -> None:
    called = False

    def fake_install() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(v16g, "_install_v16_harness", fake_install)
    with pytest.raises(ValueError, match="frozen to seed 8311"):
        v16g.train_matched_pair(data_dir="unused", run_dir="unused", seed=8312)
    assert called is False


def test_v16_hierarchy_threshold_rejects_v15_collapse_profile() -> None:
    v15_rates = [0.444, 0.006, 0.0]
    target = v16g.TARGET_RATES.tolist()
    mae = sum(abs(a - b) for a, b in zip(v15_rates, target)) / 3
    assert mae > 0.12
    assert not all(rate >= 0.05 for rate in v15_rates)
