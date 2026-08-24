from __future__ import annotations

import pytest

from tam_research import aera_real_language_v17_gpu as v17g


def test_v17_gpu_protocol_is_frozen_to_fresh_seed() -> None:
    assert v17g.SEED == 8331
    assert v17g.EVAL_SEED == 98_331
    assert v17g.TARGET_RATES.tolist() == pytest.approx([0.5, 1.0 / 3.0, 1.0 / 6.0])


def test_v17_wrong_seed_is_refused_before_gpu_training(monkeypatch) -> None:
    called = False

    def fake_install() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(v17g, "_install_v17_harness", fake_install)
    with pytest.raises(ValueError, match="frozen to seed 8331"):
        v17g.train_matched_pair(data_dir="unused", run_dir="unused", seed=8332)
    assert called is False


def test_v17_hierarchy_threshold_rejects_v16_deep_stage_collapse() -> None:
    v16_rates = [0.475, 0.225, 0.0]
    target = v17g.TARGET_RATES.tolist()
    mae = sum(abs(a - b) for a, b in zip(v16_rates, target)) / 3
    assert mae <= 0.12
    assert not all(rate >= 0.05 for rate in v16_rates)


def test_v17_adaptivity_eval_scope_is_frozen() -> None:
    assert v17g.ADAPTIVITY_BATCH_SIZE == 8
    assert v17g.ADAPTIVITY_BATCHES == 64
    assert v17g.EXPECTED_CHUNKS == 1024
